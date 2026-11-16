import { send, body } from './_lib/core.js';
import { gridAllowed } from './_lib/store.js';
import { requireIdent } from './_lib/auth.js';
// Modes for this visitor: grid only for GRID_ENABLED=1 or labeller ids marked owner/trusted by the admin (token-signed lid, not the name).
export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const ident = req.method === 'POST' ? requireIdent(req, body(req).token) : null;
  const p = ident && ident.p;
  const grid = gridAllowed(p ? p.lid : null);
  return send(res, 200, { modes: grid ? ['grid', 'ab'] : ['ab'], default: grid ? 'grid' : 'ab' });
}
