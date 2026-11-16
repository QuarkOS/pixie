import { send } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { loadSnapshot, references } from '../_lib/training.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'GET') return send(res, 405, { error: 'GET only' });
  const snap = loadSnapshot();
  return send(res, 200, {
    ok: true,
    snapshot: snap,
    references: references(),
    server_time: new Date().toISOString(),
  });
}
