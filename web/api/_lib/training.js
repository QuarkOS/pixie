// Read-only live training snapshot for the admin panel. The pusher POSTs here;
// nothing in this module starts, stops, or loads a model.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const DIR = path.join(process.env.DATA_DIR || path.join(process.cwd(), 'data'), 'training');
const SNAP = path.join(DIR, 'snapshot.json');
const PNG = path.join(DIR, 'sample.png');
const TOKEN = process.env.TRAINING_INGEST_TOKEN || '';

// Verified L10 (full1p4m-L10) held-out image-NLL. Epochs 1–4 come from the
// metrics copy on the box; epoch 32 is final32 (train_summary 1.356390).
// Missing mid-run epochs are omitted rather than invented.
export const L10_BASELINE = [
  { epoch: 1, step: 9684, dev_nll: 1.702087 },
  { epoch: 2, step: 19367, dev_nll: 1.584705 },
  { epoch: 3, step: 29050, dev_nll: 1.531494 },
  { epoch: 4, step: 38733, dev_nll: 1.496214 },
  { epoch: 32, step: 309859, dev_nll: 1.356390 },
];
export const REF_COOLDOWN_A = 1.346102;
export const REF_FINAL32 = 1.356390;
export const STALE_MS = 15 * 60 * 1000;

export function ingestConfigured() {
  return TOKEN.length >= 32;
}

export function checkIngestToken(req) {
  if (!ingestConfigured()) return false;
  const raw = String(req.headers['authorization'] || '');
  const bearer = raw.toLowerCase().startsWith('bearer ') ? raw.slice(7).trim() : '';
  const header = String(req.headers['x-training-token'] || '').trim();
  const sent = bearer || header;
  if (!sent || sent.length !== TOKEN.length) return false;
  return crypto.timingSafeEqual(Buffer.from(sent), Buffer.from(TOKEN));
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}
function str(x, max = 200) {
  if (typeof x !== 'string') return null;
  const s = x.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '').trim();
  return s.length && s.length <= max ? s : (s.length > max ? s.slice(0, max) : null);
}
function cleanPoints(arr, keys, max = 4000) {
  if (!Array.isArray(arr)) return [];
  const out = [];
  for (const p of arr.slice(0, max)) {
    if (!p || typeof p !== 'object') continue;
    const o = {};
    for (const k of keys) {
      const n = num(p[k]);
      if (n !== null) o[k] = n;
    }
    if (Object.keys(o).length) out.push(o);
  }
  return out;
}
function cleanNames(arr, max = 20) {
  if (!Array.isArray(arr)) return [];
  return arr.map((x) => str(x, 120)).filter(Boolean).slice(0, max);
}

export function sanitizeSnapshot(b) {
  if (!b || typeof b !== 'object') return null;
  const total = num(b.total_steps) ?? 282732;
  const step = num(b.step);
  const status = ['running', 'finished', 'stale', 'unknown'].includes(b.status) ? b.status : 'unknown';
  return {
    run: str(b.run, 80) || 'v2-L12-full',
    status,
    step,
    total_steps: total,
    epoch: num(b.epoch),
    steps_per_sec: num(b.steps_per_sec),
    eta_sec: num(b.eta_sec),
    eta_vienna: str(b.eta_vienna, 80),
    train_nll: num(b.train_nll),
    best_dev_nll: num(b.best_dev_nll),
    last_dev_nll: num(b.last_dev_nll),
    last_dev_epoch: num(b.last_dev_epoch),
    vram_mib: num(b.vram_mib),
    lr: num(b.lr),
    train_curve: cleanPoints(b.train_curve, ['step', 'epoch', 'train_nll']),
    dev_curve: cleanPoints(b.dev_curve, ['step', 'epoch', 'dev_nll']),
    checkpoints: {
      best: str(b.checkpoints?.best, 120),
      last: str(b.checkpoints?.last, 120),
      rolling: cleanNames(b.checkpoints?.rolling),
      perm: cleanNames(b.checkpoints?.perm),
    },
    sample_name: str(b.sample_name, 200),
    pushed_at: str(b.pushed_at, 80),
    source_mtime: str(b.source_mtime, 80),
  };
}

export function saveSnapshot(snap, pngBuf) {
  fs.mkdirSync(DIR, { recursive: true, mode: 0o750 });
  const tmp = SNAP + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(snap), { mode: 0o640 });
  fs.renameSync(tmp, SNAP);
  if (pngBuf && Buffer.isBuffer(pngBuf) && pngBuf.length >= 24 && pngBuf[0] === 0x89 && pngBuf[1] === 0x50) {
    const t = PNG + '.tmp';
    fs.writeFileSync(t, pngBuf, { mode: 0o640 });
    fs.renameSync(t, PNG);
  }
}

export function loadSnapshot() {
  try {
    const raw = JSON.parse(fs.readFileSync(SNAP, 'utf8'));
    const st = fs.statSync(SNAP);
    const received_at = st.mtime.toISOString();
    const age_ms = Date.now() - st.mtimeMs;
    return { ...raw, received_at, age_ms, stale: age_ms > STALE_MS, has_sample: fs.existsSync(PNG) };
  } catch {
    return null;
  }
}

export function samplePath() {
  return fs.existsSync(PNG) ? PNG : null;
}

export function references() {
  return { l10_baseline: L10_BASELINE, cooldown_a: REF_COOLDOWN_A, final32: REF_FINAL32 };
}

// ----- run history (compare view). Pushed by tools/push_runs.py; read-only stats only. -----
const RUNS_FILE = path.join(DIR, 'runs.json');
const SPLITS = ['old', 'iid', 'probe', 'iidrestored'];
const FAMILIES = ['v1', 'v2', 'v3'];
const RSTATUS = ['running', 'done', 'stopped'];

function cleanTuples(arr, max = 1200) {
  if (!Array.isArray(arr)) return [];
  const out = [];
  for (const p of arr.slice(0, max)) {
    if (!Array.isArray(p) || p.length < 4) continue;
    const t = [num(p[0]), num(p[1]), num(p[2]), num(p[3])];
    if (t[0] === null || t[3] === null) continue;
    out.push(t);
  }
  return out;
}

function sanitizeRun(r) {
  if (!r || typeof r !== 'object') return null;
  const id = str(r.id, 80);
  if (!id || !/^[A-Za-z0-9._-]+$/.test(id)) return null;
  const dev = {};
  const best = {};
  for (const s of SPLITS) {
    const pts = cleanTuples(r.dev?.[s], 600);
    if (pts.length) dev[s] = pts;
    const b = num(r.best?.[s]);
    if (b !== null) best[s] = b;
  }
  return {
    id,
    label: str(r.label, 120) || id,
    family: FAMILIES.includes(r.family) ? r.family : 'v3',
    arch: str(r.arch, 160),
    params: num(r.params),
    parent: str(r.parent, 80),
    status: RSTATUS.includes(r.status) ? r.status : 'stopped',
    start: str(r.start, 40),
    finish: str(r.finish, 40),
    first_step: num(r.first_step),
    last_step: num(r.last_step),
    total_steps: num(r.total_steps),
    images: num(r.images),
    hours: num(r.hours),
    steps_per_sec: num(r.steps_per_sec),
    eta_sec: num(r.eta_sec),
    best,
    splits: Object.keys(dev),
    train: cleanTuples(r.train, 1200),
    dev,
    source_mtime: str(r.source_mtime, 40),
  };
}

export function sanitizeRuns(b) {
  if (!b || typeof b !== 'object' || !Array.isArray(b.runs)) return null;
  const runs = b.runs.slice(0, 40).map(sanitizeRun).filter(Boolean);
  let baselines = null;
  if (b.baselines && typeof b.baselines === 'object' && b.baselines.rows && typeof b.baselines.rows === 'object') {
    const rows = {};
    for (const [k, v] of Object.entries(b.baselines.rows).slice(0, 20)) {
      if (!/^[A-Za-z0-9_-]{1,40}$/.test(k) || !v || typeof v !== 'object') continue;
      const row = { label: str(v.label, 60) || k, step: num(v.step) };
      for (const s of SPLITS) { const n = num(v[s]); if (n !== null) row[s] = n; }
      rows[k] = row;
    }
    baselines = { rows, time: str(b.baselines.time, 40) };
  }
  return { runs, baselines, pushed_at: str(b.pushed_at, 40) };
}

export function saveRuns(data) {
  fs.mkdirSync(DIR, { recursive: true, mode: 0o750 });
  const tmp = RUNS_FILE + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(data), { mode: 0o640 });
  fs.renameSync(tmp, RUNS_FILE);
}

export function loadRuns() {
  try {
    const raw = JSON.parse(fs.readFileSync(RUNS_FILE, 'utf8'));
    const st = fs.statSync(RUNS_FILE);
    return { ...raw, received_at: st.mtime.toISOString(), age_ms: Date.now() - st.mtimeMs };
  } catch {
    return null;
  }
}
