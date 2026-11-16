import { send } from '../_lib/core.js';
import { isAdmin, buildData } from '../_lib/admin.js';
import { compact, overrideMap } from '../_lib/store.js';

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  try {
    const { rows, merged } = await compact();
    const d = buildData(rows);
    return send(res, 200, { ...d, overrides: overrideMap(), statuses: ['auto', 'owner', 'trusted', 'excluded'], compare: d.compare || ['v3', 'v2ft'], ck_order: d.compare || ['v3', 'v2ft'], merged, generated: new Date().toISOString() });
  } catch (e) {
    console.error('admin data', e && e.message);
    return send(res, 500, { error: 'could not load labels: ' + (e && e.message) });
  }
}
