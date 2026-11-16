import { assertConfigured, sign, send, body, deviceClass } from './_lib/core.js';
import { requireIdent, identUnauthorized } from './_lib/auth.js';
import { putBatch, ensureCounts, abSeen } from './_lib/store.js';
import { PAIRS, AB_MODELS, AB_INDEX, AB_COUNTS, AB_SHA, AB_FAST_MS, CHOICES, abDoneSet, abEncodeDone, pickAB, pairMode } from './_lib/ab.js';

const MAX = 16;
export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    const answers = Array.isArray(b.answers) ? b.answers : [];
    if (!answers.length) return send(res, 400, { error: 'no answers' });
    if (answers.length > MAX) return send(res, 400, { error: `at most ${MAX} answers per request` });
    const now = Date.now();
    const done = abDoneSet(p);
    const out = [];
    let dup = 0, newlyDone = 0, skippedUnknown = 0;
    await ensureCounts();
    const seen = abSeen(p.lid);
    for (const a of answers) {
      const i = AB_INDEX.get(a && a.pair_id);
      // After a pool reset, phones may still hold leftover answers for old pair ids.
      // Skip those instead of 400, so the client can drop them and load a new pair.
      if (i === undefined) { skippedUnknown++; continue; }
      if (!CHOICES.includes(a.choice)) return send(res, 400, { error: 'choice must be A, B, both_good or both_bad' });
      if (done.has(i)) { dup++; continue; }
      // already stored for this labeller (e.g. an earlier beacon or a retry whose response was lost): count it as done, never store twice
      if (seen.has(i)) { dup++; done.add(i); newlyDone++; continue; }
      const pr = PAIRS[i];
      const rt = Number.isFinite(a.response_ms) ? Math.max(0, Math.min(3600000, Math.round(a.response_ms))) : null;
      const flags = [];
      if (rt === null || rt < AB_FAST_MS) flags.push('fast');
      const winner = a.choice === 'A' ? pr.ck[0] : a.choice === 'B' ? pr.ck[1] : null;
      out.push({ type: 'ab', pair_id: pr.id, matchup: pr.m || null, pool: pr.pool || pr.m || null, mode: pairMode(pr), choice: a.choice, left_model: pr.ck[0], right_model: pr.ck[1],
        left_sha: AB_SHA[pr.ck[0]], right_sha: AB_SHA[pr.ck[1]], left_seed: pr.seeds[0], right_seed: pr.seeds[1], left_step: AB_MODELS[pr.ck[0]]?.step ?? null, right_step: AB_MODELS[pr.ck[1]]?.step ?? null,
        winner_model: pr.kind === 'abcal' ? null : winner, response_ms: rt, ts: new Date(now).toISOString(), flags });
      done.add(i); newlyDone++;
    }
    if (out.length) {
      await putBatch(p.lid, { v: 1, type: 'ab', lid: p.lid, name: p.name, device: deviceClass(req), at: new Date(now).toISOString(), labels: out });
      for (const r of out) { const i = AB_INDEX.get(r.pair_id); AB_COUNTS.set(i, (AB_COUNTS.get(i) || 0) + 1); }
    }
    abEncodeDone(done, p);
    p.an = (p.an || 0) + newlyDone;
    const block = b.want_next === false ? [] : pickAB(p, seen);
    return send(res, 200, { ok: true, saved: out.length, duplicates: dup, skipped_unknown: skippedUnknown, flagged_fast: out.filter((r) => r.flags.includes('fast')).length,
      token: sign(p), done: p.an, block, finished: b.want_next !== false && block.length === 0 });
  } catch (e) {
    console.error('ab submit failed', e && e.message);
    return send(res, e.status || 500, { error: e.status ? e.message : 'could not save, please retry' });
  }
}
