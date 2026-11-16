// Worker long-poll. Bind it on a private network; the handler rejects public-edge headers.
// Returns {job:{id,prompt,n,show,timeout_ms}} or {job:null}.
import crypto from 'node:crypto';
import { send, body } from '../_lib/core.js';
import { WORKER_KEY, workerPoll, requeue } from '../_lib/chat.js';

export function workerAuth(req) {
  if (req.headers['cf-connecting-ip'] || req.headers['cf-ray']) return false; // never via the public tunnel
  const k = String(req.headers['x-worker-key'] || '');
  if (WORKER_KEY.length < 32 || k.length !== WORKER_KEY.length) return false;
  return crypto.timingSafeEqual(Buffer.from(k), Buffer.from(WORKER_KEY));
}
export default async function handler(req, res) {
  if (!workerAuth(req)) return send(res, 404, { error: 'not found' });
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  const b = body(req);
  const info = b.info && typeof b.info === 'object' ? { model: b.info.model || null, device: b.info.device || null, host: b.info.host || null } : null;
  let cancel = null, closed = false;
  res.on('close', () => { closed = true; if (cancel) cancel(); });
  const job = await workerPoll(info, 20000, (c) => { cancel = c; });
  if (job && (closed || res.destroyed)) { requeue(job.id); return; }
  return send(res, 200, { job });
}
