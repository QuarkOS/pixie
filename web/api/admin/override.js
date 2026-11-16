import { send, body } from '../_lib/core.js';
import { isAdmin } from '../_lib/admin.js';
import { compact, setOverride, overrideMap } from '../_lib/store.js';
import { STATUSES } from '../_lib/analysis.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  const b = body(req);
  const lid = String(b.lid || ''), status = String(b.status || '');
  if (!/^[0-9a-f]{16}$/.test(lid)) return send(res, 400, { error: 'bad labeler id' });
  if (!STATUSES.includes(status)) return send(res, 400, { error: 'status must be one of ' + STATUSES.join(', ') });
  const { rows } = await compact();
  const row = rows.find((r) => r.lid === lid);
  setOverride(lid, status, row ? row.name : null);
  return send(res, 200, { ok: true, lid, status, overrides: overrideMap() });
}
