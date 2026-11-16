// GET /api/admin/chat_thumb?job=...  (admin session) — 64×64 PNG of a saved chat pick.
import { send } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { readChatThumb } from '../_lib/share.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, { error: 'GET only' });
  const u = new URL(req.url, 'http://x');
  const buf = readChatThumb(String(u.searchParams.get('job') || ''));
  if (!buf) return send(res, 404, { error: 'no thumbnail' });
  res.statusCode = 200;
  res.setHeader('Content-Type', 'image/png');
  res.setHeader('Cache-Control', 'private, max-age=31536000, immutable');
  if (req.method === 'HEAD') return res.end();
  return res.end(buf);
}
