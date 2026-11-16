// A/B preference pairs: pre-built sheets. New rounds are same-model samples of the live
// chat checkpoint (condboost / full_064M.pt). Older matchups stay in the pool for admin.
import crypto from 'node:crypto';
import fs from 'node:fs';
import { AB } from './abdata.js';

const SWAP = JSON.parse(fs.readFileSync(new URL('../../tools/ab_prompt_swap.json', import.meta.url), 'utf8'));
// These prompts are not offered as new rounds. Their existing pairs and votes stay as stored.
export const RETIRED_PROMPTS = new Set(SWAP.swap.map((x) => x.old));
export const REPLACEMENT_PROMPTS = SWAP.swap.map((x) => x.new);

export const PAIRS = AB.pairs;           // [{id, prompt, kind:'ab'|'abcal', ck:[left,right], seeds:[l,r], truth?:'A'|'B', mode?:'single'|'rerank_best_of_8'|..., pick?:{...}}]
// Sampling mode of a pair: its own tag, else its matchup's, else 'single' (one plain sample per side; every pool so far).
const M_MODE = new Map((AB.matchups || []).map((m) => [m.key, m.mode || 'single']));
export const pairMode = (pr) => (pr && (pr.mode || M_MODE.get(pr.m))) || 'single';
export const AB_SHA = AB.sha;
export const AN = PAIRS.length;
export const AB_INDEX = new Map(PAIRS.map((p, i) => [p.id, i]));
export const AB_COUNTS = new Map();
export const AB_BLOCK = 8;
export const AB_FAST_MS = 500;
export const CHOICES = ['A', 'B', 'both_good', 'both_bad'];
const vh = (n) => crypto.createHash('sha256').update(PAIRS.slice(0, n).map((p) => p.id).join(',')).digest('base64url').slice(0, 10);
export const APV = vh(AN);
// Pairs are only ever appended, so a done-bitset from an older pool version (a prefix) stays valid.
const OLD_APV = new Set((AB.versions || []).map(vh));
export const MATCHUPS = AB.matchups || [];
export const AB_MODELS = AB.models || {};
// Serving order (config): 'newest' = unlabelled pairs of the newest matchup first, older matchups only once those run out;
// 'balanced' = least-labelled first across all matchups. Env AB_SERVE overrides the value baked into abdata.js.
export const AB_SERVE = process.env.AB_SERVE || AB.serve || 'newest';
export const NEWEST_MATCHUP = MATCHUPS.length ? MATCHUPS[MATCHUPS.length - 1].key : null;
// Live chat model is the condboost checkpoint (worker model_key "v3", file full_064M.pt).
// New rounds prefer that same-model matchup, not the last entry (currently condboost vs v2ft).
export const LIVE_MODEL = 'condboost';
const sameLive = MATCHUPS.filter((m) => m.a === LIVE_MODEL && m.b === LIVE_MODEL);
export const LIVE_MATCHUP = process.env.AB_LIVE_MATCHUP || (sameLive.length ? sameLive[sameLive.length - 1].key : NEWEST_MATCHUP);

export function abDoneSet(p) {
  if (p.apv !== APV && !OLD_APV.has(p.apv)) return new Set();
  const bits = Buffer.from(p.ad || '', 'base64url');
  const s = new Set();
  for (let i = 0; i < bits.length * 8 && i < AN; i++) if (bits[i >> 3] & (1 << (i & 7))) s.add(i);
  return s;
}
export function abEncodeDone(set, p) {
  p.apv = APV;
  const bits = Buffer.alloc(Math.ceil(AN / 8));
  for (const i of set) bits[i >> 3] |= 1 << (i & 7);
  p.ad = bits.toString('base64url');
}
const pub = (i) => ({ pair_id: PAIRS[i].id, prompt: PAIRS[i].prompt, img: `/ab/${PAIRS[i].id}.png` });
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
// Retired prompts stay in the pool (votes still resolve) but are not handed out.
// Same-model pairs with one shared seed would show two identical sides; never serve those.
function servablePair(p) {
  if (!p || RETIRED_PROMPTS.has(p.prompt)) return false;
  const seeds = p.seeds;
  if (Array.isArray(seeds) && seeds.length >= 2 && seeds[0] === seeds[1]) return false;
  return true;
}

// Block of 8 pairs over different prompts, least-labelled first; ~1 hidden calibration pair per 6 early on, then 1 per 16.
export function pickAB(p, seen = null) {
  const done = abDoneSet(p);
  if (seen) for (const i of seen) done.add(i);
  let calDone = 0;
  for (const i of done) if (PAIRS[i].kind === 'abcal') calDone++;
  const n = p.an || 0;
  const wantCal = calDone < 6 ? Math.floor(n / 6) + 1 - calDone : Math.floor(n / 16) + 1 - calDone;
  const free = [], calFree = [];
  for (let i = 0; i < AN; i++) {
    if (done.has(i) || !servablePair(PAIRS[i])) continue;
    (PAIRS[i].kind === 'abcal' ? calFree : free).push(i);
  }
  const newest = AB_SERVE === 'newest' ? LIVE_MATCHUP : null;
  // Prompts this labeller has already seen in the served matchup: spread new people
  // across every prompt before repeating one, and never re-serve a pair they have done.
  const seenPrompts = new Set();
  for (const i of done) if (PAIRS[i].kind === 'ab' && (!newest || PAIRS[i].m === newest)) seenPrompts.add(PAIRS[i].prompt);
  const sk = new Map();
  for (const i of free) {
    const c = AB_COUNTS.get(i) || 0;
    const isNew = !newest || PAIRS[i].m === newest;
    const unseenPrompt = isNew && !seenPrompts.has(PAIRS[i].prompt);
    // tier 0: current/newest matchup, prompt not yet shown to this labeller
    // tier 1e3: current/newest matchup, prompt already seen
    // tier 1e6: older matchups (only once the current one is exhausted)
    const tier = isNew ? (unseenPrompt ? 0 : 1e3) : 1e6;
    sk.set(i, tier + c + Math.random() * 0.9);
  }
  free.sort((x, y) => sk.get(x) - sk.get(y));
  const out = [], prompts = new Set();
  const nCal = Math.max(0, Math.min(2, wantCal, calFree.length));
  shuffle(calFree);
  for (let k = 0; k < nCal; k++) { out.push(calFree[k]); prompts.add(PAIRS[calFree[k]].prompt); }
  for (const i of free) { if (out.length >= AB_BLOCK) break; if (prompts.has(PAIRS[i].prompt)) continue; out.push(i); prompts.add(PAIRS[i].prompt); }
  for (const i of free) { if (out.length >= AB_BLOCK) break; if (!out.includes(i)) out.push(i); }
  if (!out.length) for (const i of calFree) { if (out.length >= AB_BLOCK) break; if (!out.includes(i)) out.push(i); }
  return shuffle(out).map(pub);
}
