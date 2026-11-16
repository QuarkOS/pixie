// "Chat" generation: in-memory FIFO job queue (1 GPU job at a time) served to a pull-based worker,
// plus the append-only chat preference votes (DATA_DIR/chat.jsonl, separate from labels.jsonl).
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const DIR = process.env.DATA_DIR || path.join(process.cwd(), 'data');
const FILE = path.join(DIR, 'chat.jsonl');
export const WORKER_KEY = process.env.CHAT_WORKER_KEY || '';
export const PROMPT_MAX = Number(process.env.CHAT_PROMPT_MAX || 60);
export const QUEUE_MAX = Number(process.env.CHAT_QUEUE_MAX || 12);      // waiting jobs (not counting the running one)
export const JOB_TIMEOUT_MS = Number(process.env.CHAT_JOB_TIMEOUT_MS || 60000);
export const N_SAMPLES = 20, N_SHOW = 4;
const ABANDON_MS = 45000;        // queued job whose page stopped polling is dropped
const KEEP_DONE_MS = 60 * 60 * 1000;
const WORKER_STALE_MS = 60000;

const jobs = new Map();          // id -> job
const waiting = [];              // FIFO of job ids
let running = null;              // job id
let avgMs = 3000;                // EMA of job duration (dispatch -> result)
let workerSeen = 0, workerInfo = null;
const pollers = new Set();       // pending worker long-polls
// Browser long-polls on /api/chat/status?wait=1. Woken when a job leaves queued/running.
export const STATUS_WAIT_MS = Number(process.env.CHAT_STATUS_WAIT_MS || 25000);
export const MAX_STATUS_WAITERS = Number(process.env.CHAT_STATUS_WAITERS || 4);
const statusWaiters = new Map();     // job id -> Set<entry>
const statusWaitersByLid = new Map(); // lid -> open waiters

function decWaiter(lid) {
  const n = (statusWaitersByLid.get(lid) || 1) - 1;
  if (n <= 0) statusWaitersByLid.delete(lid);
  else statusWaitersByLid.set(lid, n);
}
function wakeJob(id) {
  const set = statusWaiters.get(id);
  if (!set) return;
  statusWaiters.delete(id);
  for (const entry of set) entry.done('ready');
}
// Resolves 'ready' | 'timeout' | 'cancel' | 'capped' | 'missing'.
// Does not hold when the job is already finished, or when this lid is at the waiter cap
// (the caller then returns the same immediate payload as a non-waiting status call).
export function waitForJob(id, lid, holdMs = STATUS_WAIT_MS, onCancel = null) {
  const j = jobs.get(id);
  if (!j || j.lid !== lid) return Promise.resolve('missing');
  if (j.status !== 'queued' && j.status !== 'running') return Promise.resolve('ready');
  const n = statusWaitersByLid.get(lid) || 0;
  if (!(MAX_STATUS_WAITERS > 0) || n >= MAX_STATUS_WAITERS) return Promise.resolve('capped');
  return new Promise((resolve) => {
    let settled = false;
    const entry = {};
    const done = (reason) => {
      if (settled) return;
      settled = true;
      clearTimeout(entry.timer);
      const set = statusWaiters.get(id);
      if (set) { set.delete(entry); if (!set.size) statusWaiters.delete(id); }
      decWaiter(lid);
      resolve(reason);
    };
    entry.done = done;
    entry.timer = setTimeout(() => done('timeout'), holdMs);
    if (entry.timer.unref) entry.timer.unref();
    let set = statusWaiters.get(id);
    if (!set) { set = new Set(); statusWaiters.set(id, set); }
    set.add(entry);
    statusWaitersByLid.set(lid, n + 1);
    if (onCancel) onCancel(() => done('cancel'));
    const now = jobs.get(id);
    if (!now || now.lid !== lid) done('missing');
    else if (now.status !== 'queued' && now.status !== 'running') done('ready');
  });
}

export function cleanPrompt(s) {
  if (typeof s !== 'string') return null;
  const p = s.normalize('NFKC').replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim();
  if (p.length < 2 || p.length > PROMPT_MAX) return null;
  return p;
}
export const workerOnline = () => Date.now() - workerSeen < WORKER_STALE_MS || !!pollers.size;
export const getWorkerInfo = () => workerInfo;

function position(id) { const i = waiting.indexOf(id); return i < 0 ? 0 : i + 1; }
function etaMs(id) {
  const j = jobs.get(id); if (!j) return null;
  if (j.status === 'running') return Math.max(500, avgMs - (Date.now() - j.started));
  const pos = position(id);
  const cur = running && jobs.get(running) ? Math.max(500, avgMs - (Date.now() - jobs.get(running).started)) : 0;
  return cur + pos * avgMs;
}
export function queueInfo() { return { waiting: waiting.length, running: running ? 1 : 0, avg_ms: Math.round(avgMs), queue_max: QUEUE_MAX, worker_online: workerOnline() }; }

export function inflightFor(lids) {
  sweep();
  const set = new Set(lids);
  for (const j of jobs.values()) if (set.has(j.lid) && (j.status === 'queued' || j.status === 'running')) return j;
  return null;
}
let onJobPersisted = null;
export function setOnJobPersisted(fn) { onJobPersisted = typeof fn === 'function' ? fn : null; }
function persistJobSafe(j) {
  if (!j || !onJobPersisted) return;
  try { onJobPersisted(j); } catch (e) { console.error('flywheel persist', e && e.message); }
}

export function liveQueueStats() {
  sweep();
  let waitingN = 0, runningN = 0;
  for (const j of jobs.values()) {
    if (j.source === 'admin_grid') continue;
    if (j.status === 'queued') waitingN++;
    else if (j.status === 'running') runningN++;
  }
  return { waiting: waitingN, running: runningN };
}

// Internal enqueue used by the admin Comparison Grid. Max one admin_grid job
// (queued or running) at a time so live chat is never starved. These jobs
// skip the user lid in-flight cap and the 45s abandon timer (the grid
// manager holds the job; no browser tab is polling jobView).
export function enqueueInternal(opts = {}) {
  sweep();
  const source = opts.source || 'internal';
  const prompt = typeof opts.prompt === 'string'
    ? opts.prompt.normalize('NFKC').replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim()
    : '';
  if (prompt.length < 2 || prompt.length > 200) return { error: 'Invalid prompt' };
  if (source === 'admin_grid') {
    let liveWaiting = 0;
    for (const id of waiting) {
      const j = jobs.get(id);
      if (j && j.source !== 'admin_grid') liveWaiting++;
    }
    if (liveWaiting >= QUEUE_MAX) return { busy: true, reason: 'queue' };
    for (const j of jobs.values()) {
      if (j.source === 'admin_grid' && (j.status === 'queued' || j.status === 'running')) {
        return { busy: true, reason: 'in_flight' };
      }
    }
  } else if (waiting.length >= QUEUE_MAX) {
    return { busy: true, reason: 'queue' };
  }
  let n = N_SAMPLES;
  if (Number.isInteger(opts.n_samples) && opts.n_samples >= 1 && opts.n_samples <= 8) n = opts.n_samples;
  const show = Math.min(N_SHOW, n);
  const seed = (Number.isInteger(opts.seed) && Number.isSafeInteger(opts.seed)) ? opts.seed : null;
  if (opts.seed != null && (seed == null || seed < 0 || seed > 2147483647)) return { error: 'Invalid seed (expected 0..2147483647)' };
  const id = crypto.randomBytes(9).toString('base64url');
  jobs.set(id, {
    id, lid: 'admin-grid', name: 'Admin grid', prompt, cond: opts.cond || null,
    source, seed, n, show, grid_id: opts.grid_id || null,
    status: 'queued', created: Date.now(), seen: Date.now(), started: 0, finished: 0,
    results: null, all: null, candidates: null, model: null, gen_ms: null, error: null, voted: false,
  });
  waiting.push(id);
  dispatch();
  return { id, job_id: id };
}

export function enqueue(lid, name, prompt, opts = {}) {
  sweep();
  const capLids = opts.accountLids && opts.accountLids.length ? opts.accountLids : [lid];
  if (inflightFor(capLids)) return { busy: true, reason: 'in_flight' };
  if (waiting.length >= QUEUE_MAX) return { busy: true, reason: 'queue' };
  const id = crypto.randomBytes(9).toString('base64url');
  const session_chain_id = opts.session_chain_id ? String(opts.session_chain_id).slice(0, 80) : null;
  const prev_job_id = opts.prev_job_id ? String(opts.prev_job_id).slice(0, 80) : null;
  jobs.set(id, { id, lid, name, prompt, cond: opts.cond || null, session_chain_id, prev_job_id, status: 'queued', created: Date.now(), seen: Date.now(), started: 0, finished: 0, results: null, all: null, candidates: null, model: null, gen_ms: null, error: null, voted: false });
  waiting.push(id);
  dispatch();
  return { id };
}
export function jobView(id, lid) {
  sweep();
  const j = jobs.get(id);
  if (!j || j.lid !== lid) return null;
  j.seen = Date.now();
  const v = { job_id: j.id, status: j.status, prompt: j.prompt };
  if (j.cond && !j.cond.default) v.cond = j.cond.public || null;
  if (j.status === 'queued') { v.position = position(id); v.eta_s = Math.ceil(etaMs(id) / 1000); }
  if (j.status === 'running') { v.position = 0; v.eta_s = Math.ceil(etaMs(id) / 1000); }
  if (j.status === 'done') {
    v.images = j.results.map((r, k) => ({ index: k, img: r.png })); v.gen_ms = j.gen_ms; v.voted = j.voted;
    // grid mode: all N_SAMPLES ranked candidates (absent for old workers / old jobs)
    if (j.candidates && j.candidates.length) v.candidates = j.candidates.map((c) => ({ idx: c.idx, img: c.png, score: c.score, rank: c.rank }));
  }
  if (j.status === 'error') v.error = j.error;
  return v;
}
export function getJob(id) { return jobs.get(id) || null; }

function finish(j, status, error) {
  j.status = status; j.error = error || null; j.finished = Date.now();
  if (running === j.id) running = null;
  wakeJob(j.id);
}
function sweep() {
  const now = Date.now();
  if (running) { const r = jobs.get(running); if (!r) running = null; else if (now - r.started > JOB_TIMEOUT_MS) { finish(r, 'error', 'generation timed out, please try again'); console.log('chat job timeout', r.id); } }
  for (let i = waiting.length - 1; i >= 0; i--) { const j = jobs.get(waiting[i]); if (!j || (j.source !== 'admin_grid' && now - j.seen > ABANDON_MS)) { if (j) finish(j, 'error', 'abandoned'); waiting.splice(i, 1); } }
  for (const [id, j] of jobs) {
    if ((j.status === 'done' || j.status === 'error') && now - j.finished > KEEP_DONE_MS) {
      if (j.status === 'done' && j.source !== 'admin_grid') persistJobSafe(j);
      jobs.delete(id);
    }
  }
}
setInterval(() => { sweep(); dispatch(); }, 1000).unref();

// hand the next job to one waiting worker long-poll (concurrency 1: only when nothing is running)
function dispatch() {
  if (running || !waiting.length || !pollers.size) return;
  const p = pollers.values().next().value; pollers.delete(p);
  const id = waiting.shift(); const j = jobs.get(id);
  j.status = 'running'; j.started = Date.now(); running = id;
  const n = Number.isInteger(j.n) ? j.n : N_SAMPLES;
  const show = Number.isInteger(j.show) ? j.show : N_SHOW;
  const job = { id, prompt: j.prompt, n, show, timeout_ms: JOB_TIMEOUT_MS };
  if (j.source === 'admin_grid') job.source = 'admin_grid';
  if (j.cond && j.cond.worker) job.cond = j.cond.worker;
  if (Number.isInteger(j.seed) && Number.isSafeInteger(j.seed)) job.seed = j.seed;
  clearTimeout(p.t); p.resolve(job);
}
// onCancel receives a function that withdraws this poll (called when the worker's HTTP connection drops before a job arrives)
export function workerPoll(info, holdMs = 20000, onCancel = null) {
  workerSeen = Date.now(); if (info) workerInfo = info;
  sweep();
  return new Promise((resolve) => {
    const p = { resolve: (job) => { workerSeen = Date.now(); resolve(job); } };
    p.t = setTimeout(() => { pollers.delete(p); workerSeen = Date.now(); resolve(null); }, holdMs);
    if (onCancel) onCancel(() => { if (pollers.delete(p)) { clearTimeout(p.t); resolve(null); } });
    pollers.add(p); dispatch();
  });
}
// a job was dispatched but the worker connection was already gone: put it back at the head of the queue
export function requeue(id) { const j = jobs.get(id); if (!j || j.status !== 'running' || running !== id) return; j.status = 'queued'; j.started = 0; running = null; waiting.unshift(id); dispatch(); }
export function workerResult(b) {
  workerSeen = Date.now();
  const j = jobs.get(b.id);
  if (!j || j.status !== 'running') return false;   // timed out / unknown: drop late results
  if (!b.ok) { finish(j, 'error', 'generation failed, please try again'); dispatch(); return true; }
  const show = Number.isInteger(j.show) ? j.show : N_SHOW;
  const nCap = Number.isInteger(j.n) ? j.n : N_SAMPLES;
  const res = (b.results || []).slice(0, show);
  if (res.length !== show || !res.every((r) => typeof r.png === 'string' && r.png.startsWith('data:image/png;base64,') && r.png.length < 200000)) { finish(j, 'error', 'bad worker result'); dispatch(); return true; }
  j.results = res.map((r) => ({ idx: r.idx, png: r.png, seed: r.seed, score: r.score, fill: r.fill, ncolors: r.ncolors }));
  j.all = (b.all || []).slice(0, nCap).map((r) => ({ idx: r.idx, seed: r.seed, score: r.score }));
  j.sampling = cleanSampling(b.sampling);
  j.model = b.model || null; j.scorer = b.scorer || 'feat_linear';
  j.candidates = cleanCandidates(b.candidates, nCap);
  const dur = Date.now() - j.started; j.gen_ms = Number.isFinite(b.ms) ? Math.round(b.ms) : dur;
  avgMs = avgMs * 0.7 + dur * 0.3;
  finish(j, 'done');
  if (j.source !== 'admin_grid') persistJobSafe(j);
  dispatch();
  return true;
}

function cleanSampling(s) {
  if (!s || s.scheme !== 'torch-gumbel-batch-v1' || !Number.isInteger(s.job_seed) || s.job_seed < 0 || s.job_seed > 2147483647
    || !Number.isInteger(s.batch_size) || s.batch_size < 1 || s.batch_size > 64
    || !Number.isFinite(s.cfg_scale) || !Number.isFinite(s.temperature) || s.temperature <= 0
    || !Number.isInteger(s.top_k) || s.top_k < 0 || s.top_k > 4097
    || typeof s.graph !== 'boolean' || typeof s.blank_cache !== 'boolean' || typeof s.tf32 !== 'boolean'
    || s.dtype !== 'bfloat16' || typeof s.torch !== 'string' || s.torch.length > 80
    || !(s.cuda === null || typeof s.cuda === 'string' && s.cuda.length <= 80)) return null;
  return { scheme: s.scheme, job_seed: s.job_seed, batch_size: s.batch_size, cfg_scale: s.cfg_scale,
    temperature: s.temperature, top_k: s.top_k, graph: s.graph, blank_cache: s.blank_cache,
    torch: s.torch, cuda: s.cuda, dtype: s.dtype, tf32: s.tf32 };
}

// optional worker field: [{idx, png, score, rank, seed}] for all ranked candidates; invalid -> dropped (top-4 still works)
function cleanCandidates(c, nCap = N_SAMPLES) {
  if (!Array.isArray(c) || !c.length) return null;
  const out = c.slice(0, nCap).map((x) => ({ idx: Number(x && x.idx), png: x && x.png, score: Number(x && x.score), rank: Number(x && x.rank), seed: x && x.seed }));
  const ok = out.every((x) => Number.isInteger(x.idx) && x.idx >= 0 && Number.isInteger(x.rank) && x.rank >= 1 && Number.isFinite(x.score)
    && typeof x.png === 'string' && x.png.startsWith('data:image/png;base64,') && x.png.length < 200000);
  if (!ok || new Set(out.map((x) => x.idx)).size !== out.length) return null;
  return out.sort((a, b) => a.rank - b.rank);
}

// ---- votes ----
let votes = null;
function loadVotes() {
  if (votes) return votes;
  votes = [];
  if (fs.existsSync(FILE)) for (const line of fs.readFileSync(FILE, 'utf8').split('\n')) { if (!line.trim()) continue; try { votes.push(JSON.parse(line)); } catch { /* torn line */ } }
  return votes;
}
export function chatVotes() { return loadVotes(); }
export function hasVote(jobId) { return loadVotes().some((v) => v.job_id === jobId); }
export function addVote(rec) { loadVotes(); fs.appendFileSync(FILE, JSON.stringify(rec) + '\n', { mode: 0o600 }); votes.push(rec); }
export function purgeVotes(dropLids) {
  loadVotes();
  const keep = votes.filter((v) => !dropLids.has(v.lid));
  if (keep.length === votes.length) return 0;
  const tmp = FILE + '.tmp';
  fs.writeFileSync(tmp, keep.map((v) => JSON.stringify(v)).join('\n') + (keep.length ? '\n' : ''), { mode: 0o600 });
  fs.renameSync(tmp, FILE);
  const n = votes.length - keep.length; votes = keep; return n;
}
