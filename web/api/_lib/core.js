// Shared server helpers: signed labeller tokens, pool index, block picking, rate limits.
import crypto from 'node:crypto';
import { POOL } from './pooldata.js';

const SECRET = process.env.TOKEN_SECRET || '';
export const MAX_BATCH = 8;
export const GRID_ENABLED = process.env.GRID_ENABLED === '1'; // grid mode is off for visitors unless explicitly enabled
export const BLOCK = 4;
export const FAST_MS = 1000;          // below this a round is stored but flagged

export const ROUNDS = POOL.rounds;
export const POOL_SHA = POOL.sha;   // checkpoint id -> sha256 for every cell's ck    // [{id, prompt, kind, ck:[...40], seeds:[...40], truth?:[0/1..]}]
export const N = ROUNDS.length;
export const INDEX = new Map(ROUNDS.map((r, i) => [r.id, i]));
const BY_PROMPT = new Map();
for (const [i, r] of ROUNDS.entries()) {
  if (!BY_PROMPT.has(r.prompt)) BY_PROMPT.set(r.prompt, { pool: [], cal: [] });
  BY_PROMPT.get(r.prompt)[r.kind === 'cal' ? 'cal' : 'pool'].push(i);
}
export const PROMPTS = [...BY_PROMPT.keys()];
// Pool version: if the pool is ever rebuilt, old tokens' done-bitsets are ignored instead of misread.
const pvOf = (n) => crypto.createHash('sha256').update(ROUNDS.slice(0, n).map((r) => r.id).join(',')).digest('base64url').slice(0, 10);
export const PV = pvOf(ROUNDS.length);
// Rounds are only ever appended (tools/build_pool_append.py records the old lengths in POOL.versions),
// so a done-bitset from an older version (a prefix) stays valid.
const OLD_PV = new Set((POOL.versions || []).map(pvOf));

const b64u = (buf) => Buffer.from(buf).toString('base64url');
const hmac = (s) => crypto.createHmac('sha256', SECRET).update(s).digest('base64url');

export function assertConfigured() {
  if (SECRET.length < 32) throw Object.assign(new Error('server not configured'), { status: 500 });
}

export function sign(payload) {
  const body = b64u(JSON.stringify(payload));
  return body + '.' + hmac('tok.' + body);
}
export function verify(token) {
  if (typeof token !== 'string' || token.length > 4000) return null;
  const [body, mac] = token.split('.');
  if (!body || !mac) return null;
  const want = hmac('tok.' + body);
  const a = Buffer.from(mac), b = Buffer.from(want);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try {
    const p = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
    if (p.v !== 1 || typeof p.lid !== 'string') return null;
    return p;
  } catch { return null; }
}

export function doneSet(p) {
  if (p.pv !== PV && !OLD_PV.has(p.pv)) return new Set();
  const bits = Buffer.from(p.done || '', 'base64url');
  const s = new Set();
  for (let i = 0; i < bits.length * 8 && i < N; i++) if (bits[i >> 3] & (1 << (i & 7))) s.add(i);
  return s;
}
export function encodeDone(set, p) {
  if (p) p.pv = PV;
  const bits = Buffer.alloc(Math.ceil(N / 8));
  for (const i of set) bits[i >> 3] |= 1 << (i & 7);
  return bits.toString('base64url');
}

export function cleanName(raw) {
  if (typeof raw !== 'string') return null;
  const n = raw.normalize('NFKC').replace(/[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e]/g, '').replace(/\s+/g, ' ').trim();
  if (n.length < 2 || n.length > 30) return null;
  if (!/^[\p{L}\p{N} ._'\-]+$/u.test(n)) return null;
  return n;
}

// Names that could pose as the owner/admin are refused for NEW registrations (existing tokens keep working).
const RESERVED = ['owner', 'example', 'admin'];
const ownerName = (process.env.OWNER_NAME || '').trim().toLowerCase();
if (ownerName && !RESERVED.includes(ownerName)) RESERVED.push(ownerName);
function lev(a, b) {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]; d[0] = i;
    for (let j = 1; j <= b.length; j++) { const t = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = t; }
  }
  return d[b.length];
}
export function normName(n) {
  return String(n).normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
    .replace(/[0]/g, 'o').replace(/[1!|]/g, 'i').replace(/[^a-z0-9]/g, '');
}
export function reservedName(n) {
  const x = normName(n);
  if (!x) return false;
  if (/the owner|Owner/.test(x)) return true;
  return RESERVED.some((r) => lev(x, r) <= 2);
}

export function newToken(name) {
  return { v: 1, pv: PV, lid: crypto.randomBytes(8).toString('hex'), name, done: '', n: 0, hw: 0, hc: 0, t: Date.now() };
}

// ---- per-instance soft state (no IPs stored anywhere) ----
export const COUNTS = new Map(); // round index -> label count (approximate, refreshed from store)
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

export function pickBlock(p, lastPrompt, seen = null) {
  const done = doneSet(p);
  if (seen) for (const i of seen) done.add(i); // server-side: rounds already stored for this labeller
  let calDone = 0;
  for (const i of done) if (ROUNDS[i].kind === 'cal') calDone++;
  const pCal = calDone < 8 ? 0.35 : calDone < 20 ? 0.15 : 0.05;
  const avail = new Map();
  for (const [pr, g] of BY_PROMPT) {
    const pool = g.pool.filter((i) => !done.has(i));
    const cal = g.cal.filter((i) => !done.has(i));
    if (pool.length) avail.set(pr, { pool, cal });
  }
  if (!avail.size) return [];
  let prompts = [...avail.keys()];
  if (prompts.length > 1 && lastPrompt) prompts = prompts.filter((x) => x !== lastPrompt);
  const cnt = (i) => COUNTS.get(i) || 0;
  let prompt, useCal = false;
  const calPrompts = prompts.filter((x) => avail.get(x).cal.length);
  // Calibration: random, but never fall behind ~1 per 3 blocks for the first 8 (then 1 per 8 blocks).
  const blockNo = Math.floor((p.n || 0) / BLOCK);
  const behind = calDone < 8 ? calDone < Math.floor((blockNo + 1) / 3) : calDone < 8 + Math.floor((blockNo - 24) / 8);
  if (calPrompts.length && (behind || Math.random() < pCal)) {
    prompt = calPrompts[Math.floor(Math.random() * calPrompts.length)];
    useCal = true;
  } else {
    const score = (x) => Math.min(...avail.get(x).pool.map(cnt));
    const best = Math.min(...prompts.map(score));
    const cands = prompts.filter((x) => score(x) === best);
    prompt = cands[Math.floor(Math.random() * cands.length)];
  }
  const a = avail.get(prompt);
  const pool = shuffle([...a.pool]).sort((x, y) => cnt(x) - cnt(y));
  const block = pool.slice(0, useCal ? BLOCK - 1 : BLOCK);
  if (useCal) block.splice(Math.floor(Math.random() * (block.length + 1)), 0, a.cal[Math.floor(Math.random() * a.cal.length)]);
  return block.map((i) => ({ round_id: ROUNDS[i].id, prompt: ROUNDS[i].prompt, sheet: `/s/${ROUNDS[i].id}.png` }));
}

export function deviceClass(req) {
  const ua = String(req.headers['user-agent'] || '');
  return /Mobi|Android|iPhone|iPad/i.test(ua) ? 'mobile' : 'desktop';
}

export function body(req) {
  if (req.body && typeof req.body === 'object') return req.body;
  try { return JSON.parse(req.body || '{}'); } catch { return {}; }
}
export function send(res, code, obj) {
  res.statusCode = code;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(obj));
}
