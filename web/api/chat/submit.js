// POST {token, prompt} -> {job_id, status, position, eta_s} | 503 busy / generator offline
import { assertConfigured, send, body } from '../_lib/core.js';
import { requireIdent, identUnauthorized } from '../_lib/auth.js';
import { cleanPrompt, enqueue, jobView, workerOnline, queueInfo, PROMPT_MAX } from '../_lib/chat.js';
import { parseCond, workerCond, publicCond } from '../_lib/cond.js';
import { flywheel, actorFromIdent } from '../../lib/init.mjs';
import { peekQuota, consumeQuota, quotaMessage } from '../_lib/quota.js';

export default async function handler(req, res) {
  try {
    assertConfigured();
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });
    const b = body(req);
    const ident = requireIdent(req, b.token);
    if (!ident) return send(res, 401, identUnauthorized);
    const p = ident.p;
    const prompt = cleanPrompt(b.prompt);
    if (!prompt) return send(res, 400, { error: `Please describe a sprite in 2–${PROMPT_MAX} characters.` });
    const cond = parseCond(b);
    if (!workerOnline()) return send(res, 503, { error: 'The sprite generator is offline right now. Please try again in a few minutes.', code: 'offline', queue: queueInfo() });
    const quotaPeek = peekQuota(ident, req);
    if (quotaPeek.limited && quotaPeek.remaining <= 0) {
      return send(res, 429, { error: quotaMessage(quotaPeek.limit), code: 'quota', quota: { limited: true, limit: quotaPeek.limit, remaining: 0 } });
    }
    const session_chain_id = b.session_chain_id ? String(b.session_chain_id).slice(0, 80) : null;
    const prev_job_id = b.prev_job_id ? String(b.prev_job_id).slice(0, 80) : null;
    const r = enqueue(p.lid, p.name, prompt, {
      accountLids: ident.lids, session_chain_id, prev_job_id,
      cond: { ...cond, worker: workerCond(cond), public: publicCond(cond) },
    });
    if (r.busy && r.reason === 'in_flight') return send(res, 409, { error: 'You already have a sprite generating. Please wait for it to finish.', code: 'in_flight' });
    if (r.busy) return send(res, 503, { error: 'Lots of people are generating right now. Please try again in a minute.', code: 'busy', queue: queueInfo() });
    const quota = consumeQuota(ident, req);
    if (prev_job_id) {
      try {
        const actor = actorFromIdent(ident);
        flywheel.recordEvent({
          job_id: r.id,
          lid: actor.lid,
          user_id: actor.user_id || null,
          trust_tier: actor.trust_tier,
          session_chain_id,
          prev_job_id,
          event_type: 'retry_submit',
          payload: { prompt },
        });
      } catch (e) { console.error('flywheel retry_submit', e && e.message); }
    }
    const view = jobView(r.id, p.lid);
    if (view && quota.limited) view.quota = { limited: true, limit: quota.limit, remaining: quota.remaining };
    return send(res, 200, view);
  } catch (e) {
    return send(res, e.status || 500, { error: e.status ? e.message : 'server error' });
  }
}
