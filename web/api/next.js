import { assertConfigured, send, body, pickBlock } from './_lib/core.js';
import { requireIdent, identUnauthorized } from './_lib/auth.js';
import { ensureCounts, gridAllowed, gridSeen } from './_lib/store.js';

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    if (!gridAllowed(p.lid)) return send(res, 410, { error: 'grid mode is turned off, please use A/B' });
    await ensureCounts();
    const block = pickBlock(p, typeof b.last_prompt === 'string' ? b.last_prompt : null, gridSeen(p.lid));
    return send(res, 200, { block, done: p.n, finished: block.length === 0 });
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'server error' });
  }
}
