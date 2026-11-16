// GET /api/share/:id -> json metadata for a public share.
import { send } from '../_lib/core.js';
import { getShare, PUBLIC_PATH } from '../_lib/share.js';

export default async function handler(req, res, id) {
  const rec = getShare(id);
  if (!rec) return send(res, 404, { error: 'not found' });
  const path_ = PUBLIC_PATH(rec.id);
  return send(res, 200, {
    ok: true,
    id: rec.id, prompt: rec.prompt, ts: rec.ts, cond: rec.cond || null,
    model: rec.model || null,
    author: rec.name || null,
    edit_id: rec.edit_id || null,
    refined: Boolean(rec.refined || rec.edit_id),
    path: path_,
    png_url: path_ + '/png?size=16',
    images: {
      native: path_ + '/png?size=16',
      x512: path_ + '/png?size=512',
      x512_transparent: path_ + '/png?size=512&bg=transparent',
      x512_black: path_ + '/png?size=512&bg=black',
      og: path_ + '/og.png',
    },
  });
}
