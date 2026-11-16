import { send, body } from '../_lib/core.js';
import { workerResult } from '../_lib/chat.js';
import { workerAuth } from './next.js';

export default async function handler(req, res) {
  if (!workerAuth(req)) return send(res, 404, { error: 'not found' });
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  const ok = workerResult(body(req));
  return send(res, 200, { ok, accepted: ok });
}
