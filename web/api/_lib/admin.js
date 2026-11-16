import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { ROUNDS } from './core.js';
import { POOL } from './pooldata.js';
import { OWNER } from './ownerdata.js';
import { PAIRS, MATCHUPS, AB_MODELS, pairMode } from './ab.js';
import { chatVotes } from './chat.js';
import { isSessionAdmin } from './auth.js';

const ARCHIVE_ID = '2026-11-12-pre-v3';
const ARCHIVE_DIR = path.join(process.env.DATA_DIR || path.join(process.cwd(), 'data'), 'archive', ARCHIVE_ID);

function readJsonl(p) {
  if (!fs.existsSync(p)) return [];
  const out = [];
  for (const line of fs.readFileSync(p, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { out.push(JSON.parse(line)); } catch { /* torn line */ }
  }
  return out;
}

/** Pre-v3 A/B + chat + grid votes, left in place after the v3 pool reset. Never deletes. */
export function loadArchive() {
  const manPath = path.join(ARCHIVE_DIR, 'manifest.json');
  if (!fs.existsSync(manPath)) return null;
  const manifest = JSON.parse(fs.readFileSync(manPath, 'utf8'));
  const labelsRaw = readJsonl(path.join(ARCHIVE_DIR, 'labels.jsonl'));
  const labels = [], ab = [];
  for (const line of labelsRaw) {
    if (Array.isArray(line.labels)) {
      for (const r of line.labels) {
        const row = { ...r, lid: line.lid, name: line.name, device: line.device };
        (row.type === 'ab' ? ab : labels).push(row);
      }
    } else if (line.type === 'ab') ab.push(line);
    else labels.push(line);
  }
  const chat = readJsonl(path.join(ARCHIVE_DIR, 'chat.jsonl'));
  let pairs = {}, matchups = [], nPairRows = 0, nPrompts = 0;
  const abPath = path.join(ARCHIVE_DIR, 'abdata.json');
  if (fs.existsSync(abPath)) {
    const AB = JSON.parse(fs.readFileSync(abPath, 'utf8'));
    const plist = AB.pairs || [];
    nPairRows = plist.length;
    nPrompts = new Set(plist.filter((p) => p.kind === 'ab').map((p) => p.prompt)).size;
    for (const p of plist) {
      pairs[p.id] = { prompt: p.prompt, kind: p.kind, ck: p.ck, mode: p.mode || null, ...(p.m ? { m: p.m } : {}), ...(p.pool ? { pool: p.pool } : {}), ...(p.truth ? { truth: p.truth } : {}) };
    }
    matchups = (AB.matchups || []).map((m) => ({ ...m, pairs: plist.filter((p) => p.m === m.key).length }));
  }
  return {
    id: ARCHIVE_ID, title: 'Pre-v3 archive (12 Nov 2026)', path: ARCHIVE_DIR,
    manifest, labels, ab, chat, pairs, matchups,
    n_ab: ab.length, n_chat: chat.length, n_grid_labels: labels.length,
    n_pairs: nPairRows, n_prompts: nPrompts,
  };
}

const PW = process.env.ADMIN_PASSWORD || '';
const SECRET = process.env.TOKEN_SECRET || '';
const key = () => crypto.createHmac('sha256', SECRET).update('admin:' + PW).digest();

export function checkPassword(pw) {
  if (PW.length < 16 || typeof pw !== 'string') return false;
  const a = crypto.createHash('sha256').update(pw).digest();
  const b = crypto.createHash('sha256').update(PW).digest();
  return crypto.timingSafeEqual(a, b);
}
export function sessionCookie() {
  const exp = Date.now() + 12 * 3600 * 1000;
  const mac = crypto.createHmac('sha256', key()).update(String(exp)).digest('base64url');
  return `pxadm=${exp}.${mac}; Path=/api/admin; HttpOnly; Secure; SameSite=Strict; Max-Age=${12 * 3600}`;
}
export function isAdmin(req) {
  if (isSessionAdmin(req)) return true;
  if (PW.length < 16) return false;
  const m = /(?:^|;\s*)pxadm=(\d+)\.([A-Za-z0-9_-]+)/.exec(String(req.headers.cookie || ''));
  if (!m || Number(m[1]) < Date.now()) return false;
  const want = crypto.createHmac('sha256', key()).update(m[1]).digest('base64url');
  const a = Buffer.from(m[2]), b = Buffer.from(want);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

// Full joined dataset (admin only): pool + calibration rounds and the owner's imported rounds.
export function buildData(webRows) {
  const rounds = {};
  const meta = {};
  for (const r of ROUNDS) {
    rounds[r.id] = { prompt: r.prompt, kind: r.kind, ck: r.ck, ...(r.truth ? { truth: r.truth } : {}) };
    meta[r.id] = { seeds: r.seeds, sha: POOL.sha, src_id: r.src_id || null };
  }
  const labels = [];
  for (const e of OWNER) {
    const rid = 'owner:' + e.id;
    rounds[rid] = { prompt: e.prompt, kind: 'owner', ck: e.cells.map((c) => c.ckpt) };
    meta[rid] = { seeds: e.cells.map((c) => c.seed), sha: e.cells.map((c) => c.sha), src_id: e.id };
    labels.push({ lid: 'owner', name: 'owner', round_id: rid, works: e.cells.filter((c) => c.works).map((c) => c.idx),
      all_bad: e.all_bad, response_ms: e.response_ms, ts: e.timestamp, flags: [], imported: true });
  }
  const ab = [];
  for (const r of webRows) (r.type === 'ab' ? ab : labels).push(r);
  const pairs = {};
  for (const p of PAIRS) pairs[p.id] = { prompt: p.prompt, kind: p.kind, ck: p.ck, mode: pairMode(p), ...(p.pick ? { pick: p.pick } : {}), ...(p.m ? { m: p.m } : {}), ...(p.pool ? { pool: p.pool } : {}), ...(p.truth ? { truth: p.truth } : {}) };
  const matchups = MATCHUPS.map((m) => ({ ...m, mode: m.mode || 'single', pairs: PAIRS.filter((p) => p.m === m.key).length, steps: [AB_MODELS[m.a]?.step, AB_MODELS[m.b]?.step] }));
  const chat = chatVotes().map((v) => ({ ...v }));
  const livePrompts = new Set(PAIRS.filter((p) => p.kind === 'ab').map((p) => p.prompt)).size;
  const livePool = (MATCHUPS.length ? MATCHUPS[MATCHUPS.length - 1].pool : null) || 'live';
  const newest = MATCHUPS.length ? MATCHUPS[MATCHUPS.length - 1].key : null;
  const compare = newest && newest.includes('v3') ? ['v3', 'v2ft'] : ['final32', 'step252k'];
  return {
    rounds, meta, labels, ab, chat, pairs, matchups, compare,
    live_pool: { key: livePool, newest, pairs: PAIRS.filter((p) => p.kind === 'ab').length, prompts: livePrompts, votes: ab.length },
    archive: loadArchive(),
  };
}
