import { send, body } from '../_lib/core.js';
import { checkIngestToken, ingestConfigured, sanitizeSnapshot, saveSnapshot } from '../_lib/training.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  if (!ingestConfigured() || !checkIngestToken(req)) return send(res, 404, { error: 'not found' });
  const b = body(req);
  const snap = sanitizeSnapshot(b);
  if (!snap || snap.step === null) return send(res, 400, { error: 'invalid snapshot' });
  let png = null;
  if (typeof b.sample_png_b64 === 'string' && b.sample_png_b64.length) {
    if (b.sample_png_b64.length > 1_800_000) return send(res, 413, { error: 'too large' });
    try { png = Buffer.from(b.sample_png_b64, 'base64'); } catch { png = null; }
    if (png && (png.length < 24 || png[0] !== 0x89 || png[1] !== 0x50)) png = null;
  }
  try {
    snap.received_hint = new Date().toISOString();
    saveSnapshot(snap, png);
  } catch (e) {
    console.error('training ingest write failed', e && e.message);
    return send(res, 500, { error: 'could not store snapshot' });
  }
  return send(res, 200, { ok: true, step: snap.step, has_sample: !!(png && png.length) });
}
