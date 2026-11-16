// GET/POST {token, job_id, index, size?, bg?, edit_id?, candidate_idx?} -> PNG of a finished job sprite.
// size: 16 | 32 | 64 | 128 | 256 | 512 | 1024 (default 512). bg: transparent | black | white | surface | #rrggbb.
import fs from 'node:fs';
import { assertConfigured, send, body } from '../_lib/core.js';
import { requireIdent, identUnauthorized } from '../_lib/auth.js';
import { getJob } from '../_lib/chat.js';
import { native16FromDataUri, renderSpritePng, parseBg, decodePng } from '../_lib/png.js';
import { flywheel, flywheelHandlers, actorFromIdent } from '../../lib/init.mjs';

function asInt(v) {
  const n = Number(v);
  return Number.isInteger(n) ? n : null;
}

function pickPng(j, index, candidateIdx) {
  const ci = asInt(candidateIdx);
  if (ci != null && j.candidates) {
    const c = j.candidates.find((x) => x.idx === ci);
    if (c && c.png) return { png: c.png, index: c.idx };
  }
  const i = asInt(index);
  if (i == null) return null;
  if (j.results && j.results[i] && j.results[i].png) return { png: j.results[i].png, index: i };
  const c = (j.candidates || []).find((x) => x.idx === i) || (j.candidates || []).find((x) => x.rank === i + 1);
  return c && c.png ? { png: c.png, index: c.idx } : null;
}

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST' && req.method !== 'GET') return send(res, 405, { error: 'POST or GET' });
    const u = new URL(req.url, 'http://x');
    const b = req.method === 'GET' ? Object.fromEntries(u.searchParams) : body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const size = b.size != null && b.size !== '' ? Number(b.size) : 512;
    const bg = parseBg(b.bg);
    let native = null;
    let j = null;
    let nameSrc = 'sprite';
    if (b.edit_id) {
      const rec = flywheel.getEditRecord(String(b.edit_id));
      if (!rec || !fs.existsSync(rec.abs_png_path)) return send(res, 404, { error: 'unknown edit' });
      if (rec.lid && rec.lid !== ident.p.lid) return send(res, 404, { error: 'unknown edit' });
      const im = decodePng(fs.readFileSync(rec.abs_png_path));
      if (im.width !== 16 || im.height !== 16) return send(res, 500, { error: 'could not build png' });
      native = im.rgba;
      nameSrc = rec.prompt || 'sprite';
      if (b.job_id) {
        const maybe = getJob(String(b.job_id || ''));
        if (maybe && maybe.lid === ident.p.lid) j = maybe;
      }
    } else {
      j = getJob(String(b.job_id || ''));
      if (!j || j.lid !== ident.p.lid) return send(res, 404, { error: 'unknown or expired request' });
      if (j.status !== 'done') return send(res, 409, { error: 'not finished yet' });
      const hit = pickPng(j, b.index, b.candidate_idx);
      if (!hit) return send(res, 400, { error: 'index must be a sprite from this job' });
      native = native16FromDataUri(hit.png);
      nameSrc = j.prompt || 'sprite';
    }
    const png = renderSpritePng(native, size, bg);
    const name = (nameSrc || 'sprite').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'sprite';
    try { flywheelHandlers.onChatDownload({ actor: actorFromIdent(ident), body: b, job: j }); }
    catch (e) { console.error('flywheel onChatDownload', e && e.message); }
    res.statusCode = 200;
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', `attachment; filename="${name}-${size}.png"`);
    res.setHeader('Cache-Control', 'no-store');
    return res.end(png);
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'could not build png' });
  }
}
