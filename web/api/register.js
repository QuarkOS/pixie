import { assertConfigured, cleanName, reservedName, newToken, sign, send, body, pickBlock, GRID_ENABLED } from './_lib/core.js';
import { ensureCounts } from './_lib/store.js';
import { graceActive } from './_lib/auth.js';

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    if (!graceActive()) return send(res, 410, { error: 'Please log in with email, Google or X.', login: '/login' });
    const name = cleanName(body(req).name);
    if (!name) return send(res, 400, { error: 'Name must be 2-30 characters (letters, numbers, space . _ - \')' });
    if (reservedName(name)) return send(res, 400, { error: 'That name is reserved. Please pick another name.' });
    await ensureCounts();
    const p = newToken(name);
    return send(res, 200, { token: sign(p), labeler_id: p.lid, name, done: 0, block: GRID_ENABLED ? pickBlock(p, null) : [], modes: GRID_ENABLED ? ['grid', 'ab'] : ['ab'] });
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'server error' });
  }
}
