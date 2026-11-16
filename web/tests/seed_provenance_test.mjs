import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pixie-seed-'));
process.env.DATA_DIR = dir;
const { enqueueInternal, enqueue, workerPoll, workerResult, getJob, setOnJobPersisted } = await import('../api/_lib/chat.js');
const { normalizeGridInput } = await import('../lib/grid.mjs');
const { createFlywheel } = await import('../lib/flywheel.mjs');
const png = 'data:image/png;base64,fixture';
const sampling = { scheme: 'torch-gumbel-batch-v1', job_seed: 123, batch_size: 3,
  cfg_scale: 2, temperature: .8, top_k: 32, graph: true, blank_cache: true,
  torch: 'fixture', cuda: 'fixture', dtype: 'bfloat16', tf32: false };

assert.ok(enqueueInternal({ source: 'admin_grid', prompt: 'copper golem', seed: -1 }).error);
assert.throws(() => normalizeGridInput({ subjects: ['golem'], phrases: ['copper'], seed: 2**31 }), /seed/);
const a = enqueueInternal({ source: 'admin_grid', prompt: 'copper golem', seed: 123, n_samples: 3 });
const dispatched = await workerPoll({}, 100);
assert.equal(dispatched.id, a.id);
assert.equal(dispatched.source, 'admin_grid');
assert.equal(dispatched.seed, 123);
assert.equal(dispatched.n, 3);
const ranked = [2, 0, 1].map((idx, k) => ({ idx, rank: k + 1, seed: 123, score: 1 - k * .1, png }));
assert.equal(workerResult({ id: a.id, ok: true, results: ranked, all: ranked, candidates: ranked, sampling }), true);
const done = getJob(a.id);
assert.deepEqual(done.sampling, sampling);
assert.deepEqual(done.results.map((r) => r.idx), [2, 0, 1]);
assert.deepEqual(done.all.map((r) => r.idx), [2, 0, 1]);
assert.deepEqual(done.candidates.map((r) => r.idx), [2, 0, 1]);
const flywheel = createFlywheel({ dataDir: path.join(dir, 'archive'), palette: {} });
flywheel.persistJobCandidates({ ...done, job_id: done.id, candidates: ranked.map((c) => ({ ...c, indices: Array(256).fill(c.idx) })) });
const archived = flywheel.readJobCandidates(done.id);
assert.deepEqual(archived.meta.sampling, sampling);
assert.deepEqual(archived.meta.candidates.map((r) => r.idx), [2, 0, 1]);
flywheel.close();

let persisted = null;
setOnJobPersisted((job) => { persisted = job; });
const b = enqueue('fixture', 'fixture', 'copper golem');
await workerPoll({}, 100);
assert.equal(workerResult({ id: b.id, ok: true, results: [ranked[0], ranked[1], ranked[2], ranked[0]],
  all: ranked, sampling: { ...sampling, batch_size: 64 } }), true);
assert.equal(persisted.sampling.batch_size, 64);
assert.equal(persisted.sampling.job_seed, 123);

const c = enqueueInternal({ source: 'admin_grid', prompt: 'copper golem', n_samples: 1 });
await workerPoll({}, 100);
assert.equal(workerResult({ id: c.id, ok: true, results: [ranked[0]], sampling: { ...sampling, job_seed: -1 } }), true);
assert.equal(getJob(c.id).sampling, null);
fs.rmSync(dir, { recursive: true, force: true });
console.log('seed provenance queue tests passed');
