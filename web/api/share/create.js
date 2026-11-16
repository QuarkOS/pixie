// POST {token, job_id, index} -> {id, url, prompt}
import { assertConfigured, send, body } from '../_lib/core.js';
import { requireIdent, identUnauthorized } from '../_lib/auth.js';
import { getJob } from '../_lib/chat.js';
import { createShare, PUBLIC_PATH } from '../_lib/share.js';
import { publicCond } from '../_lib/cond.js';
import { flywheel, flywheelHandlers, actorFromIdent } from '../../lib/init.mjs';

function asInt(v) {
  const n = Number(v);
  return Number.isInteger(n) ? n : null;
}

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const j = b.job_id ? getJob(String(b.job_id || '')) : null;
    if (b.job_id && (!j || j.lid !== ident.p.lid)) return send(res, 404, { error: 'unknown or expired request' });
    if (j && j.status !== 'done') return send(res, 409, { error: 'not finished yet' });

    const extra = {};
    if (b.edit_id) {
      const assets = flywheelHandlers.renderEditedShareAssets(String(b.edit_id), (b.prompt || (j && j.prompt) || ''));
      if (!assets) return send(res, 404, { error: 'unknown edit' });
      const recEdit = assets.editRecord;
      if (recEdit && recEdit.lid && recEdit.lid !== ident.p.lid) return send(res, 404, { error: 'unknown edit' });
      extra.png16 = assets.png16;
      extra.ogPng = assets.ogPng;
      extra.edit_id = String(b.edit_id);
      extra.refined = true;
    }

    let row = null;
    const ci = asInt(b.candidate_idx);
    if (!b.edit_id) {
      if (!j) return send(res, 404, { error: 'unknown or expired request' });
      if (ci != null && j.candidates) row = j.candidates.find((c) => c.idx === ci) || null;
      if (!row) {
        const index = asInt(b.index);
        row = (index != null && j.results && j.results[index]) ? j.results[index]
          : (j.candidates || []).find((c) => c.idx === index);
      }
      if (!row || !row.png) return send(res, 400, { error: 'index must be a sprite from this job' });
    }

    const rec = createShare({
      prompt: (j && j.prompt) || b.prompt || '',
      png: row && row.png,
      index: asInt(b.index),
      candidate_idx: ci != null ? ci : (row && Number.isInteger(row.idx) ? row.idx : null),
      job_id: j ? j.id : (b.job_id || null),
      lid: ident.p.lid, name: ident.p.name,
      model: j && j.model,
      cond: (j && j.cond && j.cond.public) ? j.cond.public : publicCond(j && j.cond),
      seed: row && row.seed, score: row && row.score,
      ...extra,
    });
    try { flywheelHandlers.onShareCreate({ actor: actorFromIdent(ident), body: b, job: j, shareRecord: rec }); }
    catch (e) { console.error('flywheel onShareCreate', e && e.message); }
    const base = process.env.AUTH_PUBLIC_URL || '';
    return send(res, 200, { id: rec.id, url: (base ? base.replace(/\/$/, '') : '') + PUBLIC_PATH(rec.id), path: PUBLIC_PATH(rec.id), prompt: rec.prompt, ts: rec.ts, edit_id: rec.edit_id || null, refined: Boolean(rec.refined) });
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'could not create share' });
  }
}
