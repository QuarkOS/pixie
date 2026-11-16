import { send, body } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { compact, getOverrides, setOverride } from '../_lib/store.js';
import { purgeVotes } from '../_lib/chat.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  const lids = (body(req).lids || []).filter((x) => typeof x === 'string' && /^[0-9a-f]{16}$/.test(x));
  if (!lids.length) return send(res, 400, { error: 'no labeller ids' });
  const { rows } = await compact({ dropLids: new Set(lids) });
  const chatRemoved = purgeVotes(new Set(lids));
  const ov = getOverrides();
  for (const l of lids) if (ov[l]) setOverride(l, 'auto');
  return send(res, 200, { ok: true, removed_lids: lids, rows_left: rows.length, chat_removed: chatRemoved });
}
