// New A/B rounds are same-model condboost pairs. Retired prompts stay in the pool.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { PAIRS, AB_INDEX, MATCHUPS, LIVE_MATCHUP, NEWEST_MATCHUP, RETIRED_PROMPTS, REPLACEMENT_PROMPTS, pickAB } from '../api/_lib/ab.js';
import { AB } from '../api/_lib/abdata.js';

const abPath = new URL('../api/_lib/abdata.js', import.meta.url);
const before = fs.readFileSync(abPath);

assert.equal(PAIRS.length, 8000);
assert.deepEqual(AB.versions, [4000]);
assert.equal(LIVE_MATCHUP, 'condboost_vs_condboost');
assert.equal(NEWEST_MATCHUP, 'condboost_vs_v2ft');
assert.equal(RETIRED_PROMPTS.size, 20);
assert.equal(REPLACEMENT_PROMPTS.length, 20);
for (const prompt of REPLACEMENT_PROMPTS) assert.equal(PAIRS.some((p) => p.prompt === prompt), false);
for (const key of ['v3_vs_v3', 'v3_vs_v2ft', 'condboost_vs_condboost', 'condboost_vs_v2ft']) {
  assert.ok(MATCHUPS.some((m) => m.key === key), key);
}

const retiredPair = PAIRS.find((p) => p.prompt === 'oak quarterstaff' && p.m === 'v3_vs_v2ft');
assert.ok(retiredPair);
assert.equal(AB_INDEX.get(retiredPair.id) >= 0, true);
assert.deepEqual(PAIRS[AB_INDEX.get(retiredPair.id)].ck.includes('v2ft') || PAIRS[AB_INDEX.get(retiredPair.id)].ck.includes('v3'), true);

const user = { an: 0 };
for (let n = 0; n < 20; n++) {
  const block = pickAB(user);
  assert.equal(block.length, 8);
  for (const row of block) {
    const pr = PAIRS[AB_INDEX.get(row.pair_id)];
    assert.equal(pr.m, LIVE_MATCHUP);
    assert.deepEqual(pr.ck, ['condboost', 'condboost']);
    assert.notEqual(pr.seeds[0], pr.seeds[1]);
    assert.equal(RETIRED_PROMPTS.has(pr.prompt), false);
    assert.equal(row.prompt, pr.prompt);
  }
}

const seen = new Set();
for (let i = 0; i < PAIRS.length; i++) {
  if (PAIRS[i].m === LIVE_MATCHUP && !RETIRED_PROMPTS.has(PAIRS[i].prompt)) seen.add(i);
}
const older = pickAB(user, seen);
assert.equal(older.length, 8);
const olderKeys = new Set();
for (const row of older) {
  const pr = PAIRS[AB_INDEX.get(row.pair_id)];
  assert.notEqual(pr.m, LIVE_MATCHUP);
  assert.equal(RETIRED_PROMPTS.has(pr.prompt), false);
  assert.notEqual(pr.seeds[0], pr.seeds[1]);
  olderKeys.add(pr.m);
}
assert.ok(olderKeys.size >= 1);

const dry = spawnSync('python3', ['tools/append_ab_v3.py', '--prompts', 'tools/ab_prompt_swap.json', '--dry-run'], {
  cwd: fileURLToPath(new URL('..', import.meta.url)),
  encoding: 'utf8',
});
assert.equal(dry.status, 0, dry.stderr || dry.stdout);
const line = dry.stdout.split('\n').find((l) => l.startsWith('PLAN '));
const plan = JSON.parse(line.slice(5));
assert.equal(plan.dry_run, true);
assert.equal(plan.matchup, 'condboost_vs_condboost');
assert.equal(plan.model, 'condboost');
assert.equal(plan.new_pairs, 20 * 18);
assert.equal(plan.existing_pairs, 8000);
assert.equal(plan.versions_prefix, 8000);
assert.deepEqual(plan.example.ck, ['condboost', 'condboost']);
assert.notEqual(plan.example.seeds[0], plan.example.seeds[1]);
assert.equal(plan.prompts.length, 20);
assert.equal(plan.prompts.includes('wizard staff'), true);
assert.equal(plan.prompts.includes('oak quarterstaff'), false);
assert.deepEqual(fs.readFileSync(abPath), before);
assert.equal(crypto.createHash('sha256').update(before).digest('hex'), crypto.createHash('sha256').update(fs.readFileSync(abPath)).digest('hex'));

console.log('ab_serve_test ok');
