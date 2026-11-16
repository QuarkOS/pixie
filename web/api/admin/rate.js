// Admin-only: hand this device a labeller token for the owner labeller, so the admin can rate from any device.
// Auth = the admin session cookie (server-side); the public register endpoint still refuses reserved names.
import { send, newToken, sign, encodeDone } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { compact, abSeen, gridSeen, overrideMap } from '../_lib/store.js';
import { abEncodeDone } from '../_lib/ab.js';

const OWNER_LID = process.env.OWNER_LID || 'owner';
const OWNER_NAME = process.env.OWNER_NAME || 'owner';
export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  const ov = overrideMap();
  if (ov[OWNER_LID] && ov[OWNER_LID] !== 'owner') return send(res, 409, { error: 'owner labeller is not marked owner' });
  const { rows } = await compact();
  let name = null;
  for (const r of rows) if (r.lid === OWNER_LID && r.name) name = r.name; // latest name used by that id
  name = name || OWNER_NAME;
  const p = newToken(name);
  p.lid = OWNER_LID;
  const seen = abSeen(OWNER_LID);
  abEncodeDone(seen, p);
  p.an = seen.size;
  const gs = gridSeen(OWNER_LID);
  p.done = encodeDone(gs, p); p.n = gs.size;
  console.log('admin rate-as-owner token issued', OWNER_LID);
  return send(res, 200, { token: sign(p), labeler_id: OWNER_LID, name, done: p.an, grid_done: p.n });
}
