// Singleton Flywheel v2 + editor + Comparison Grid handlers for the staged app.
// Imported from server.mjs at startup so job-done hooks are installed
// before any request is served.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadPalette } from './palette.mjs';
import { createFlywheel } from './flywheel.mjs';
import { createFlywheelHandlers, readJsonBody } from './handlers.mjs';
import { createGridManager, createGridHandlers } from './grid.mjs';
import { getJob, chatVotes, setOnJobPersisted, enqueueInternal, liveQueueStats, workerOnline } from '../api/_lib/chat.js';
import { parseCond, workerCond, publicCond } from '../api/_lib/cond.js';
import { requireIdent } from '../api/_lib/auth.js';
import { isAdmin } from '../api/_lib/admin.js';
import { getTrustTier, actorFromIdent } from '../api/_lib/trust.js';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC_DIR = path.join(ROOT, 'public');
const DATA_DIR = process.env.DATA_DIR || path.join(ROOT, 'data');
const PALETTE_PATH = path.join(PUBLIC_DIR, 'palette4096.json');

export const palette = loadPalette(PALETTE_PATH);

function toFlywheelJob(job) {
  if (!job) return job;
  const candidates = Array.isArray(job.candidates)
    ? job.candidates.map((c) => ({ ...c, img: c.img || c.png }))
    : job.candidates;
  return {
    ...job,
    job_id: job.job_id || job.id,
    candidates,
    ts: job.ts || (job.finished ? new Date(job.finished).toISOString() : undefined),
  };
}

function toGridJob(job) {
  if (!job) return null;
  const fw = toFlywheelJob(job);
  const images = Array.isArray(job.results)
    ? job.results.map((r, i) => ({ index: i, idx: r.idx, img: r.img || r.png, score: r.score, seed: r.seed }))
    : undefined;
  return {
    ...fw,
    status: job.status,
    error: job.error || null,
    images,
  };
}

const inner = createFlywheel({
  dataDir: DATA_DIR,
  palette,
  holdoutPrompts: [],
});

export const flywheel = new Proxy(inner, {
  get(target, prop, receiver) {
    const val = Reflect.get(target, prop, receiver);
    if (prop === 'persistJobCandidates' && typeof val === 'function') {
      return (job) => val.call(target, toFlywheelJob(job));
    }
    return val;
  },
});

export const flywheelHandlers = createFlywheelHandlers({
  flywheel,
  resolveToken: async (token, req) => actorFromIdent(requireIdent(req, token)),
  isAdmin: (req) => isAdmin(req),
  getChatJob: (jobId) => toFlywheelJob(getJob(jobId)),
  readChatJsonl: () => chatVotes(),
  getTrustTier: (lid) => getTrustTier(lid),
});

export const gridManager = createGridManager({
  dataDir: DATA_DIR,
  submitChatJob: async ({ prompt, palette, style, seed, n_samples, grid_id, row, col }) => {
    if (!workerOnline()) {
      return { status: 'error', error: 'The sprite generator is offline right now.' };
    }
    let cond;
    try {
      cond = parseCond({ palette, style });
    } catch (e) {
      return { status: 'error', error: e.message || 'Invalid palette/style' };
    }
    const packed = { ...cond, worker: workerCond(cond), public: publicCond(cond) };
    const r = enqueueInternal({
      prompt,
      cond: packed,
      seed,
      n_samples,
      source: 'admin_grid',
      grid_id,
      row,
      col,
    });
    if (r.error) return { status: 'error', error: r.error };
    if (r.busy) {
      return { status: 'error', error: r.reason === 'queue' ? 'Chat queue is full' : 'A grid job is already in flight' };
    }
    return { job_id: r.job_id || r.id };
  },
  getChatJob: (jobId) => {
    const j = getJob(jobId);
    if (!j) return null;
    // Keep admin_grid jobs from looking abandoned while the manager polls.
    j.seen = Date.now();
    return toGridJob(j);
  },
  // Live public chat only — never count the in-flight grid job against itself.
  getQueueStats: () => liveQueueStats(),
});

export const gridHandlers = createGridHandlers({
  gridManager,
  isAdmin: (req) => isAdmin(req),
});

setOnJobPersisted((job) => {
  try { flywheel.persistJobCandidates(job); }
  catch (e) { console.error('flywheel persistJobCandidates', e && e.message); }
});

export { getTrustTier, actorFromIdent, readJsonBody, DATA_DIR, PUBLIC_DIR, toFlywheelJob };

if (!fs.existsSync(PALETTE_PATH)) {
  throw new Error('missing ' + PALETTE_PATH);
}
