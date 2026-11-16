import fs from 'node:fs';
import { send } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { samplePath } from '../_lib/training.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, { error: 'GET only' });
  const fp = samplePath();
  if (!fp) { res.statusCode = 404; res.setHeader('Content-Type', 'text/plain'); return res.end('no sample'); }
  res.setHeader('Content-Type', 'image/png');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  if (req.method === 'HEAD') return res.end();
  fs.createReadStream(fp).pipe(res);
}
