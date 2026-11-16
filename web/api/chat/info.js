// GET/POST -> queue snapshot plus palette/style options for the chat picker.
// Optional token/session adds daily free-quota remaining for unsigned users.
import { send, body } from '../_lib/core.js';
import { requireIdent } from '../_lib/auth.js';
import { queueInfo, PROMPT_MAX } from '../_lib/chat.js';
import { publicCondOptions } from '../_lib/cond.js';
import { peekQuota } from '../_lib/quota.js';
export default async function handler(req, res) {
  const b = body(req) || {};
  const ident = requireIdent(req, b.token) || null;
  const quota = peekQuota(ident, req);
  return send(res, 200, { ...queueInfo(), prompt_max: PROMPT_MAX, cond: publicCondOptions(), quota });
}
