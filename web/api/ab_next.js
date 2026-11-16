import { assertConfigured, send, body } from './_lib/core.js';
import { requireIdent, identUnauthorized } from './_lib/auth.js';
import { ensureCounts, abSeen } from './_lib/store.js';
import { pickAB } from './_lib/ab.js';

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const ident = requireIdent(req, body(req).token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    await ensureCounts();
    const block = pickAB(p, abSeen(p.lid));
    return send(res, 200, { block, done: p.an || 0, finished: block.length === 0 });
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'server error' });
  }
}
