// POST {token, job_id, choice: 0..3 | -1 | "none", mode?: 'top4'|'grid', candidate_idx?, response_ms} -> {ok, saved}.
// One vote per job; stored in chat.jsonl (kind 'chat'). Old clients (no mode) are logged as mode 'top4'.
// Grid votes: candidate_idx picks one of the job's ranked candidates; choice is its top-4 index (rank-1) or -1 outside the top 4.
import { assertConfigured, send, body, deviceClass } from '../_lib/core.js';
import { requireIdent, identUnauthorized } from '../_lib/auth.js';
import { getJob, hasVote, addVote } from '../_lib/chat.js';
import { saveChatThumb } from '../_lib/share.js';
import { flywheelHandlers, actorFromIdent } from '../../lib/init.mjs';

const FAST_MS = 700;
export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    const j = getJob(String(b.job_id || ''));
    if (!j || j.lid !== p.lid) return send(res, 404, { error: 'unknown or expired request' });
    if (j.status !== 'done') return send(res, 409, { error: 'not finished yet' });
    const none = b.choice === 'none';
    const mode = b.mode === 'grid' ? 'grid' : 'top4';
    let idx = none ? null : Number(b.choice);
    let cand = null;
    if (mode === 'grid' && !none) {
      const ci = Number(b.candidate_idx);
      cand = (j.candidates || []).find((c) => c.idx === ci) || null;
      if (!cand) return send(res, 400, { error: 'unknown candidate' });
      idx = cand.rank <= j.results.length ? cand.rank - 1 : null;   // server-derived top-4 position (null outside the top 4)
    } else if (!none && !(Number.isInteger(idx) && idx >= 0 && idx < j.results.length)) return send(res, 400, { error: 'choice must be 0-3 or "none"' });
    if (!cand && !none && j.candidates) cand = j.candidates.find((c) => c.rank === idx + 1) || null;
    if (j.voted || hasVote(j.id)) { j.voted = true; return send(res, 200, { ok: true, saved: false, duplicate: true }); }
    const rt = Number.isFinite(b.response_ms) ? Math.max(0, Math.min(3600000, Math.round(b.response_ms))) : null;
    const sendMs = Number.isFinite(b.send_ms) ? Math.max(0, Math.min(3600000, Math.round(b.send_ms))) : null;
    const now = new Date().toISOString();
    const chosenPng = none ? null : (cand && cand.png) || (idx != null && j.results[idx] && j.results[idx].png) || null;
    let thumb = null;
    try { if (chosenPng) thumb = saveChatThumb(j.id, chosenPng); } catch (e) { console.error('chat thumb', e && e.message); }
    const voteRecord = { v: 1, type: 'chat', kind: 'chat', job_id: j.id, lid: p.lid, name: p.name, device: deviceClass(req), prompt: j.prompt,
      model: j.model, scorer: j.scorer, sampling: j.sampling || null, n_samples: j.sampling?.batch_size ?? (j.all ? j.all.length : null),
      shown: j.results.map((r, k) => ({ index: k, idx: r.idx, seed: r.seed, score: r.score, fill: r.fill, ncolors: r.ncolors })),
      all: j.all, chosen: idx, none_good: none, chosen_seed: none ? null : (cand ? cand.seed : j.results[idx].seed),
      mode, candidate_idx: cand ? cand.idx : null, chosen_rank: none ? null : (cand ? cand.rank : idx + 1), chosen_score: none ? null : (cand ? cand.score : j.results[idx].score),
      candidates: j.candidates ? j.candidates.map((c) => ({ idx: c.idx, rank: c.rank, seed: c.seed, score: c.score })) : null,
      thumb, cond: j.cond && j.cond.public ? j.cond.public : null,
      response_ms: rt, gen_ms: j.gen_ms, ...(sendMs != null ? { send_ms: sendMs } : {}), ts: now, at: now, flags: rt === null || rt < FAST_MS ? ['fast'] : [] };
    addVote(voteRecord);
    j.voted = true;
    try { flywheelHandlers.onChatVote({ actor: actorFromIdent(ident), body: b, job: j, voteRecord }); }
    catch (e) { console.error('flywheel onChatVote', e && e.message); }
    return send(res, 200, { ok: true, saved: true });
  } catch (e) {
    console.error('chat vote failed', e && e.message);
    return send(res, e.status || 500, { error: e.status ? e.message : 'could not save, please retry' });
  }
}
