import { send, body } from '../_lib/core.js';
import { checkIngestToken, ingestConfigured, sanitizeRuns, saveRuns } from '../_lib/training.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
  if (!ingestConfigured() || !checkIngestToken(req)) return send(res, 404, { error: 'not found' });
  const data = sanitizeRuns(body(req));
  if (!data || !data.runs.length) return send(res, 400, { error: 'invalid runs' });
  data.received_hint = new Date().toISOString();
  try { saveRuns(data); } catch (e) {
    console.error('runs ingest write failed', e && e.message);
    return send(res, 500, { error: 'could not store runs' });
  }
  return send(res, 200, { ok: true, runs: data.runs.length });
}
