// Long-poll wakes when the worker stores a result. The 800ms client is the previous behaviour.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { encodePng } from '../api/_lib/png.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pixie-wait-'));
process.env.DATA_DIR = dir;
process.env.CHAT_STATUS_WAITERS = '1';
process.env.CHAT_STATUS_WAIT_MS = '4000';

const { enqueue, workerPoll, workerResult, waitForJob, jobView } = await import('../api/_lib/chat.js');

function solid(r, g, b) {
  const rgba = Buffer.alloc(16 * 16 * 4);
  for (let i = 0; i < 16 * 16; i++) {
    rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b; rgba[i * 4 + 3] = 255;
  }
  return 'data:image/png;base64,' + encodePng(rgba, 16, 16).toString('base64');
}
const PNGS = [solid(220, 40, 40), solid(40, 180, 60), solid(40, 80, 220), solid(230, 180, 40)];
function resultBody(id, ms) {
  return {
    id, ok: true, ms,
    results: PNGS.map((png, i) => ({ png, seed: 1000 + i, score: 1 - i * 0.01, fill: 80, ncolors: 4 })),
    model: { key: 'v3', step: 187500 },
    scorer: 'feat_linear',
  };
}

{
  const a = enqueue('lid-a', 'Ada', 'red potion');
  let settled = null;
  const wa = waitForJob(a.id, 'lid-a', 4000).then((r) => { settled = r; return r; });
  assert.equal(await waitForJob(a.id, 'lid-a', 4000), 'capped');
  const t0 = Date.now();
  const dispatched = await workerPoll({ model: 'v3' }, 1000);
  assert.equal(dispatched.id, a.id);
  assert.equal(jobView(a.id, 'lid-a').status, 'running');
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(settled, null);
  setTimeout(() => workerResult(resultBody(a.id, 40)), 40);
  assert.equal(await wa, 'ready');
  assert.ok(Date.now() - t0 < 500, 'long-poll lag ' + (Date.now() - t0));
  assert.equal(jobView(a.id, 'lid-a').status, 'done');
  assert.equal(jobView(a.id, 'lid-a').images.length, 4);
  const b = enqueue('lid-b', 'Bea', 'blue potion');
  let cancelB = null;
  const pending = waitForJob(b.id, 'lid-b', 4000, (c) => { cancelB = c; });
  cancelB();
  assert.equal(await pending, 'cancel');
  assert.equal(jobView(b.id, 'lid-b').status, 'queued');
}

const PORT = 18790;
const KEY = 'k'.repeat(40);
const child = spawn(process.execPath, ['server.mjs'], {
  cwd: path.resolve('.'),
  env: {
    ...process.env,
    DATA_DIR: dir,
    PORT: String(PORT),
    HOST: '127.0.0.1',
    TOKEN_SECRET: 's'.repeat(40),
    CHAT_WORKER_KEY: KEY,
    ADMIN_PASSWORD: 'p'.repeat(20),
    AUTH_DIR: path.join(dir, 'auth'),
    AUTH_TOKEN_GRACE_UNTIL: 'on',
    CHAT_STATUS_WAITERS: '4',
    CHAT_STATUS_WAIT_MS: '25000',
  },
  stdio: ['ignore', 'pipe', 'pipe'],
});
let log = '';
child.stdout.on('data', (c) => { log += c; });
child.stderr.on('data', (c) => { log += c; });
const base = `http://127.0.0.1:${PORT}`;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
async function post(url, body, headers) {
  const r = await fetch(base + url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(headers || {}) },
    body: JSON.stringify(body || {}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((j && j.error) || ('HTTP ' + r.status));
  return j;
}

async function workerLoop(stop) {
  while (!stop.stopped) {
    try {
      const r = await fetch(base + '/api/worker/next', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-worker-key': KEY },
        body: JSON.stringify({ info: { model: 'v3', device: 'cpu', host: 'fake' } }),
      });
      const j = await r.json();
      if (!j.job) continue;
      const t0 = Date.now();
      await sleep(600);
      if (stop.stopped) return;
      await post('/api/worker/result', resultBody(j.job.id, Date.now() - t0), { 'x-worker-key': KEY });
    } catch (e) {
      if (stop.stopped) return;
      throw e;
    }
  }
}

try {
  for (let i = 0; i < 50 && !log.includes('pixel-labeller on'); i++) await sleep(50);
  if (!log.includes('pixel-labeller on')) throw new Error('server did not start: ' + log);
  const stop = { stopped: false };
  const loop = workerLoop(stop);
  for (let i = 0; i < 40; i++) {
    const info = await post('/api/chat/info', {});
    if (info.worker_online) break;
    await sleep(50);
  }
  const info = await post('/api/chat/info', {});
  assert.equal(info.worker_online, true);

  async function trial(mode, n) {
    const reg = await post('/api/register', { name: 'Wait' + mode + n });
    const t0 = performance.now();
    const sub = await post('/api/chat/submit', { token: reg.token, prompt: 'red potion' });
    let v;
    if (mode === 'poll') {
      for (;;) {
        v = await post('/api/chat/status', { token: reg.token, job_id: sub.job_id });
        if (v.status !== 'queued' && v.status !== 'running') break;
        await sleep(800);
      }
    } else {
      const early = await post('/api/chat/status', { token: reg.token, job_id: sub.job_id });
      assert.ok(early.status === 'queued' || early.status === 'running');
      v = await post('/api/chat/status?wait=1', { token: reg.token, job_id: sub.job_id });
    }
    assert.equal(v.status, 'done');
    const wall = performance.now() - t0;
    return { wall, gen: v.gen_ms, lag: wall - v.gen_ms };
  }

  const before = [];
  const after = [];
  for (let n = 0; n < 5; n++) before.push(await trial('poll', n));
  for (let n = 0; n < 5; n++) after.push(await trial('wait', n));
  const mean = (xs, k) => xs.reduce((s, x) => s + x[k], 0) / xs.length;
  const beforeWall = mean(before, 'wall');
  const afterWall = mean(after, 'wall');
  const beforeLag = mean(before, 'lag');
  const afterLag = mean(after, 'lag');
  console.log('MEASURE before_poll_ms ' + before.map((x) => Math.round(x.wall)).join(',') + ' mean ' + Math.round(beforeWall) + ' lag ' + Math.round(beforeLag));
  console.log('MEASURE after_wait_ms ' + after.map((x) => Math.round(x.wall)).join(',') + ' mean ' + Math.round(afterWall) + ' lag ' + Math.round(afterLag));
  console.log('MEASURE gen_ms before ' + before.map((x) => x.gen).join(',') + ' after ' + after.map((x) => x.gen).join(','));
  assert.ok(afterLag < 250, 'long-poll delivery lag ' + afterLag);
  assert.ok(beforeLag > afterLag + 80, 'poll lag ' + beforeLag + ' vs ' + afterLag);
  assert.ok(afterWall < beforeWall, 'after wall ' + afterWall + ' before ' + beforeWall);
  stop.stopped = true;
  console.log('chat_wait_test ok');
} finally {
  child.kill('SIGTERM');
  await new Promise((r) => child.on('exit', r));
  fs.rmSync(dir, { recursive: true, force: true });
}
