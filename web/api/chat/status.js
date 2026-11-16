// POST {token, job_id} -> {job_id, status: queued|running|done|error, position?, eta_s?, images?:[{index,img}], gen_ms?, voted?, error?}
// ?wait=1 holds the request until the job leaves queued/running, the client disconnects, or ~25s pass.
// Without wait, the response is immediate (older clients).
import { assertConfigured, send, body } from '../_lib/core.js';
import { requireIdent, identUnauthorized } from '../_lib/auth.js';
import { jobView, waitForJob, STATUS_WAIT_MS } from '../_lib/chat.js';

function wantsWait(req) {
  try {
    const w = new URL(req.url || '/', 'http://x').searchParams.get('wait');
    return w === '1' || w === 'true';
  } catch { return false; }
}

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    const id = String(b.job_id || '');
    const first = jobView(id, p.lid);
    if (!first) return send(res, 404, { error: 'unknown or expired request, please generate again' });
    if (!wantsWait(req) || (first.status !== 'queued' && first.status !== 'running')) return send(res, 200, first);

    let cancel = null, closed = false;
    const onClose = () => { closed = true; if (cancel) cancel(); };
    res.on('close', onClose);
    await waitForJob(id, p.lid, STATUS_WAIT_MS, (c) => { cancel = c; if (closed) c(); });
    res.off('close', onClose);
    if (closed || res.writableEnded || res.destroyed) return;
    const v = jobView(id, p.lid);
    if (!v) return send(res, 404, { error: 'unknown or expired request, please generate again' });
    return send(res, 200, v);
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'server error' });
  }
}
