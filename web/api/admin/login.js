import crypto from 'node:crypto';
import { send, body } from '../_lib/core.js';
import { checkPassword, sessionCookie } from '../_lib/admin.js';

const failsBy = new Map(); // per client IP (in memory only)
export default async function handler(req, res) {
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  const now = Date.now();
  const ip = String(req.headers['cf-connecting-ip'] || String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() || req.socket?.remoteAddress || 'x');
  const tag = crypto.createHash('sha256').update('lg' + ip).digest('hex').slice(0, 8); // log tag only, no IP stored
  let fails = failsBy.get(ip);
  if (!fails || now - fails.t > 15 * 60 * 1000) { fails = { t: now, c: 0 }; failsBy.set(ip, fails); }
  if (failsBy.size > 10000) failsBy.clear();
  if (fails.c >= 10) { console.log(`admin login locked client=${tag}`); return send(res, 429, { error: 'too many attempts, wait 15 minutes' }); }
  const pw = typeof body(req).password === 'string' ? body(req).password.trim() : '';
  if (!checkPassword(pw)) {
    fails.c++;
    console.log(`admin login fail client=${tag} fails=${fails.c} len=${pw.length}`);
    await new Promise((r) => setTimeout(r, 800));
    return send(res, 401, { error: 'wrong password' });
  }
  fails.c = 0;
  console.log(`admin login ok client=${tag}`);
  res.setHeader('Set-Cookie', sessionCookie());
  return send(res, 200, { ok: true });
}
