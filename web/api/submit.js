import { assertConfigured, sign, doneSet, encodeDone, INDEX, ROUNDS, MAX_BATCH, FAST_MS,
  COUNTS, send, body, pickBlock, deviceClass, POOL_SHA, PV } from './_lib/core.js';
import { requireIdent, identUnauthorized } from './_lib/auth.js';
import { putBatch, ensureCounts, gridAllowed, gridSeen } from './_lib/store.js';


export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    // grid off: only flushes of rounds that were already shown (want_next:false) are accepted; no new grid rounds are served
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    if (!gridAllowed(p.lid) && b.want_next !== false) return send(res, 410, { error: 'grid mode is turned off, please use A/B' });
    const labels = Array.isArray(b.labels) ? b.labels : [];
    if (!labels.length) return send(res, 400, { error: 'no labels' });
    if (labels.length > MAX_BATCH) return send(res, 400, { error: `at most ${MAX_BATCH} rounds per request` });

    const now = Date.now();
    const hour = Math.floor(now / 3600000);
    if (p.hw !== hour) { p.hw = hour; p.hc = 0; }

    const done = doneSet(p);
    await ensureCounts();
    const seen = gridSeen(p.lid);
    const out = [];
    let dup = 0;
    for (const l of labels) {
      const i = INDEX.get(l && l.round_id);
      if (i === undefined) return send(res, 400, { error: 'unknown round' });
      if (done.has(i)) { dup++; continue; }
      if (seen.has(i)) { dup++; done.add(i); continue; } // already stored for this labeller (retry/beacon): never store twice
      const allBad = l.all_bad === true;
      const works = Array.isArray(l.works) ? l.works : [];
      if (works.length > 40 || !works.every((x) => Number.isInteger(x) && x >= 0 && x < 40)) return send(res, 400, { error: 'bad selection' });
      const uniq = [...new Set(works)].sort((x, y) => x - y);
      if (allBad && uniq.length) return send(res, 400, { error: 'all_bad with selections' });
      const rt = Number.isFinite(l.response_ms) ? Math.max(0, Math.min(3600000, Math.round(l.response_ms))) : null;
      const flags = [];
      if (rt === null || rt < FAST_MS) flags.push('fast');
      if (uniq.length === 40) flags.push('all_clicked');
      const R = ROUNDS[i];
      // provenance stored on the row itself: which checkpoint drew each of the 40 cells, and whether it was a calibration round
      out.push({ round_id: R.id, kind: R.kind, cal: R.kind === 'cal', pool_version: PV,
        ck: R.ck, ck_sha: Object.fromEntries([...new Set(R.ck)].map((c) => [c, POOL_SHA[c] || null])), seeds: R.seeds, works: uniq, all_bad: allBad || uniq.length === 0, all_bad_button: allBad,
        response_ms: rt, ts: new Date(now).toISOString(), flags });
      done.add(i);
    }
    if (out.length) {
      await putBatch(p.lid, { v: 1, lid: p.lid, name: p.name, device: deviceClass(req), at: new Date(now).toISOString(), labels: out });
      for (const r of out) { const i = INDEX.get(r.round_id); COUNTS.set(i, (COUNTS.get(i) || 0) + 1); }
    }
    p.done = encodeDone(done, p);
    p.n = (p.n || 0) + out.length;
    p.hc += out.length;
    await ensureCounts();
    const block = b.want_next === false ? [] : pickBlock(p, typeof b.last_prompt === 'string' ? b.last_prompt : null, seen);
    return send(res, 200, { ok: true, saved: out.length, duplicates: dup, flagged_fast: out.filter((r) => r.flags.includes('fast')).length,
      token: sign(p), done: p.n, block, finished: b.want_next !== false && block.length === 0 });
  } catch (e) {
    console.error('submit failed', e && e.message);
    return send(res, e.status || 500, { error: e.status ? e.message : 'could not save, please retry' });
  }
}
