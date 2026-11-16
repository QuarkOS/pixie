// Drop-in HTTP route handlers & existing-route hooks for server.mjs (Node 22+ zero-dependency).
import fs from 'node:fs';
import { VALID_EVENT_TYPES } from './flywheel.mjs';
import { decodePng, upscaleNearestPng, renderOgPng } from './png.mjs';

export function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
  });
  res.end(body);
}

export async function readJsonBody(req, maxBytes = 256 * 1024) {
  if (req._parsedBody !== undefined) return req._parsedBody;
  if (req.body !== undefined && typeof req.body === 'object' && !Buffer.isBuffer(req.body) && !ArrayBuffer.isView(req.body)) {
    req._parsedBody = req.body;
    return req.body;
  }
  const chunks = [];
  let total = 0;
  for await (const ch of req) {
    total += ch.length;
    if (total > maxBytes) {
      const e = new Error('Payload too large');
      e.status = 413;
      throw e;
    }
    chunks.push(ch);
  }
  const text = Buffer.concat(chunks).toString('utf8').trim();
  const parsed = text ? JSON.parse(text) : {};
  req._parsedBody = parsed;
  return parsed;
}

/**
 * Create Flywheel v2 & Pixel Editor v1 route handlers.
 *
 * Expected dependencies from server.mjs:
 * - flywheel: instance from createFlywheel(...)
 * - resolveToken(token, req): returns { ok, lid, user_id, name, trust_tier } or null (for 401)
 * - isAdmin(req): returns boolean (true if valid admin session cookie)
 * - getChatJob(job_id): returns in-memory chat job snapshot (or null if expired)
 * - readChatJsonl(): returns array of parsed chat.jsonl rows (for /api/admin/flywheel)
 * - getTrustTier(lid): returns 'owner'|'trusted'|'authed'|'anon'|'excluded'
 */
export function createFlywheelHandlers({
  flywheel,
  resolveToken,
  isAdmin,
  getChatJob = () => null,
  readChatJsonl = () => [],
  getTrustTier = () => 'anon',
}) {
  // 20 saves / minute per labeller ID for POST /api/chat/edit
  const editRateMap = new Map();
  function checkEditRateLimit(key) {
    const now = Date.now();
    const cutoff = now - 60_000;
    const prev = editRateMap.get(key) || [];
    const recent = prev.filter((t) => t > cutoff);
    if (recent.length >= 20) return false;
    recent.push(now);
    editRateMap.set(key, recent);
    return true;
  }

  async function requireActor(req, body) {
    const actor = await resolveToken(body?.token, req);
    if (!actor || !actor.ok) {
      const e = new Error('Invalid or expired session token.');
      e.status = 401;
      throw e;
    }
    return actor;
  }

  async function handleChatEvent(req, res) {
    try {
      const body = await readJsonBody(req);
      const actor = await requireActor(req, body);
      const eventType = String(body.event_type || '').trim();
      if (!VALID_EVENT_TYPES.has(eventType)) {
        return sendJson(res, 400, { error: 'Invalid event_type' });
      }
      const jobId = body.job_id ? String(body.job_id) : null;
      if (jobId) {
        const job = getChatJob(jobId);
        if (job) flywheel.persistJobCandidates(job);
      }
      const id = flywheel.recordEvent({
        job_id: jobId,
        lid: actor.lid,
        user_id: actor.user_id || null,
        trust_tier: actor.trust_tier || getTrustTier(actor.lid),
        session_chain_id: body.session_chain_id ? String(body.session_chain_id).slice(0, 80) : null,
        prev_job_id: body.prev_job_id ? String(body.prev_job_id).slice(0, 80) : null,
        event_type: eventType,
        candidate_idx: Number.isInteger(body.candidate_idx) ? body.candidate_idx : null,
        rank: Number.isInteger(body.rank) ? body.rank : null,
        shown_view: body.shown_view === 'grid' || body.shown_view === 'top4' ? body.shown_view : null,
        payload: body.payload ?? null,
      });
      return sendJson(res, 200, { ok: true, id });
    } catch (e) {
      return sendJson(res, e.status || 400, { error: e.message || 'Bad request' });
    }
  }

  async function handleChatEdit(req, res) {
    try {
      const body = await readJsonBody(req, 512 * 1024);
      const actor = await requireActor(req, body);
      const trustTier = actor.trust_tier || getTrustTier(actor.lid);
      if (trustTier === 'excluded') {
        return sendJson(res, 403, { error: 'Account is excluded from saving edits.' });
      }
      if (!checkEditRateLimit(actor.lid || 'anon')) {
        return sendJson(res, 429, { error: 'Too many saves right now (max 20/min). Please wait a moment.' });
      }
      const jobId = body.job_id ? String(body.job_id) : null;
      const job = jobId ? getChatJob(jobId) : null;
      if (job) flywheel.persistJobCandidates(job);

      const saved = flywheel.recordEdit({
        job_id: jobId,
        parent_share_id: body.parent_share_id ? String(body.parent_share_id) : null,
        candidate_idx: Number.isInteger(body.candidate_idx) ? body.candidate_idx : null,
        lid: actor.lid,
        user_id: actor.user_id || null,
        trust_tier: trustTier,
        prompt: body.prompt || job?.prompt || '',
        cond: body.cond || job?.cond || null,
        original_indices: body.original_indices,
        edited_indices: body.edited_indices,
        ops_count: body.ops_count,
        tool_counts: body.tool_counts,
        duration_ms: body.duration_ms,
        action: body.action || 'save',
        opt_in_train: body.opt_in_train !== false,
      });
      return sendJson(res, 200, { ok: true, ...saved });
    } catch (e) {
      return sendJson(res, e.status || 400, { error: e.message || 'Could not save edit' });
    }
  }

  function handleGetEditPng(req, res, editId, query = {}) {
    const rec = flywheel.getEditRecord(editId);
    if (!rec || !fs.existsSync(rec.abs_png_path)) {
      res.writeHead(404);
      return res.end('Not found');
    }
    const rawBuf = fs.readFileSync(rec.abs_png_path);
    const size = Number(query.size || 16);
    const bg = query.bg || 'transparent';
    let outBuf = rawBuf;
    if ((size && size !== 16 && [16, 32, 64, 128, 256, 512, 1024].includes(size)) || (bg && bg !== 'transparent')) {
      const { width, height, rgba } = decodePng(rawBuf);
      outBuf = upscaleNearestPng(rgba, width, height, size || 16, bg);
    }
    res.writeHead(200, {
      'Content-Type': 'image/png',
      'Content-Length': outBuf.length,
      'Cache-Control': 'public, max-age=31536000, immutable',
    });
    res.end(outBuf);
  }

  async function handleAdminFlywheel(req, res) {
    if (!isAdmin(req)) return sendJson(res, 401, { error: 'Admin login required' });
    const chatRows = readChatJsonl();
    const diag = flywheel.getDiagnostics({ chatRows, trustLookup: getTrustTier });
    return sendJson(res, 200, diag);
  }

  function handleAdminCandThumb(req, res, urlObj) {
    if (!isAdmin(req)) { res.writeHead(401); return res.end('Unauthorized'); }
    const jobId = urlObj.searchParams.get('job') || '';
    const slot = Number(urlObj.searchParams.get('slot') || 0);
    const png = flywheel.renderCandidateThumbPng(jobId, slot, 64);
    if (!png) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, {
      'Content-Type': 'image/png',
      'Content-Length': png.length,
      'Cache-Control': 'private, max-age=3600',
    });
    res.end(png);
  }

  async function handleAdminQueueAb(req, res) {
    if (!isAdmin(req)) return sendJson(res, 401, { error: 'Admin login required' });
    try {
      const body = await readJsonBody(req);
      if (Array.isArray(body.prompts)) {
        const results = body.prompts.map((p) => flywheel.queuePromptForAb({ prompt: p, queuedBy: 'admin' }));
        return sendJson(res, 200, { ok: true, results });
      }
      const r = flywheel.queuePromptForAb({
        prompt: body.prompt,
        queuedBy: 'admin',
        noneRate: body.none_rate ?? null,
        runsCount: body.runs_count ?? null,
      });
      return sendJson(res, 200, r);
    } catch (e) {
      return sendJson(res, e.status || 400, { error: e.message || 'Could not queue prompt' });
    }
  }

  function handleAdminExportFlywheel(req, res, urlObj) {
    if (!isAdmin(req)) return sendJson(res, 401, { error: 'Admin login required' });
    const fmt = urlObj.searchParams.get('fmt') === 'csv' ? 'csv' : 'jsonl';
    const who = urlObj.searchParams.get('who') || 'trusted';
    const out = flywheel.exportFlywheelTuples({ who, format: fmt });
    const ext = fmt === 'csv' ? 'csv' : 'jsonl';
    res.writeHead(200, {
      'Content-Type': out.contentType,
      'Content-Disposition': `attachment; filename="pixie-flywheel-v2-${who}.${ext}"`,
      'Content-Length': Buffer.byteLength(out.body),
    });
    res.end(out.body);
  }

  // ---- Hooks to drop into existing server.mjs handlers ----

  function onChatVote({ actor, body, job, voteRecord }) {
    if (job) flywheel.persistJobCandidates(job);
    const choice = body.choice;
    const isNone = choice === 'none' || voteRecord?.none_good;
    const shownView = body.shown_view || body.mode || 'top4';
    const rank = Number.isInteger(body.rank)
      ? body.rank
      : (typeof choice === 'number' && choice >= 0 ? choice + 1 : null);
    const top4Max = job?.candidates?.length
      ? Math.max(...job.candidates.slice(0, 4).map((c) => Number(c.score)).filter(Number.isFinite))
      : null;

    flywheel.recordEvent({
      ts: voteRecord?.ts || new Date().toISOString(),
      job_id: body.job_id || job?.job_id || job?.id || null,
      lid: actor?.lid || voteRecord?.lid || null,
      user_id: actor?.user_id || null,
      trust_tier: actor?.trust_tier || getTrustTier(actor?.lid || voteRecord?.lid),
      session_chain_id: body.session_chain_id || null,
      prev_job_id: body.prev_job_id || null,
      event_type: 'vote',
      candidate_idx: Number.isInteger(body.candidate_idx) ? body.candidate_idx : null,
      rank,
      shown_view: shownView === 'grid' ? 'grid' : 'top4',
      payload: {
        prompt: job?.prompt || voteRecord?.prompt || '',
        name: actor?.name || voteRecord?.name || '',
        choice,
        none_good: Boolean(isNone),
        response_ms: Number(body.response_ms) || 0,
        mode: body.mode || shownView,
        top4_max_reward: Number.isFinite(top4Max) ? top4Max : null,
      },
    });
  }

  function onChatDownload({ actor, body, job }) {
    if (job) flywheel.persistJobCandidates(job);
    flywheel.recordEvent({
      job_id: body.job_id || job?.job_id || job?.id || null,
      lid: actor?.lid || null,
      user_id: actor?.user_id || null,
      trust_tier: actor?.trust_tier || getTrustTier(actor?.lid),
      event_type: 'download',
      candidate_idx: Number.isInteger(body.candidate_idx)
        ? body.candidate_idx
        : (Number.isInteger(body.index) ? body.index : null),
      payload: {
        size: Number(body.size || 512),
        bg: body.bg || 'transparent',
        edit_id: body.edit_id || null,
      },
    });
  }

  function onShareCreate({ actor, body, job, shareRecord }) {
    if (job) flywheel.persistJobCandidates(job);
    // Do not mutate edits.action (a prior save must keep weight 2.5).
    // Edit→Share is counted from share events that carry edit_id, including job-less shares.
    flywheel.recordEvent({
      job_id: body.job_id || job?.job_id || job?.id || null,
      lid: actor?.lid || null,
      user_id: actor?.user_id || null,
      trust_tier: actor?.trust_tier || getTrustTier(actor?.lid),
      event_type: 'share',
      candidate_idx: Number.isInteger(body.candidate_idx)
        ? body.candidate_idx
        : (Number.isInteger(body.index) ? body.index : null),
      payload: {
        share_id: shareRecord?.id || null,
        edit_id: body.edit_id || null,
        refined: Boolean(body.edit_id),
      },
    });
  }

  function renderEditedShareAssets(editId, prompt) {
    const rec = flywheel.getEditRecord(editId);
    if (!rec || !fs.existsSync(rec.abs_png_path)) return null;
    const rawBuf = fs.readFileSync(rec.abs_png_path);
    const { width, height, rgba } = decodePng(rawBuf);
    return {
      editRecord: rec,
      png16: rawBuf,
      png512: upscaleNearestPng(rgba, width, height, 512, 'surface'),
      png512t: upscaleNearestPng(rgba, width, height, 512, 'transparent'),
      ogPng: renderOgPng(rgba, prompt || rec.prompt || ''),
    };
  }

  return {
    handleChatEvent,
    handleChatEdit,
    handleGetEditPng,
    handleAdminFlywheel,
    handleAdminCandThumb,
    handleAdminQueueAb,
    handleAdminExportFlywheel,
    onChatVote,
    onChatDownload,
    onShareCreate,
    renderEditedShareAssets,
  };
}
