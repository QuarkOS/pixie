import { send } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { loadRuns } from '../_lib/training.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'GET') return send(res, 405, { error: 'GET only' });
  return send(res, 200, { ok: true, history: loadRuns(), server_time: new Date().toISOString() });
}
