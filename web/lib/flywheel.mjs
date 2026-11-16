// Training Data Flywheel v2 persistence, candidate archiving, diagnostics & export (Node 22+ zero-dependency).
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { encodePng, upscaleNearestPng } from './png.mjs';
import { indicesToRgba, pngToIndices } from './palette.mjs';

const CAND_MAGIC = Buffer.from('PXCD', 'ascii');
export const NONE_REASON_KEYS = ['wrong_subject', 'messy_pixels', 'cut_off', 'wrong_style'];
export const NONE_REASON_LABELS = {
  wrong_subject: 'Wrong subject',
  messy_pixels: 'Messy pixels',
  cut_off: 'Cut off',
  wrong_style: 'Wrong colour/style',
  untagged: 'Untagged',
};
export const VALID_EVENT_TYPES = new Set([
  'none_reason',
  'grid_toggle',
  'edit_open',
  'anim_open',
  'remix_start',
  'tweak_prompt',
  'retry_submit',
  'vote',
  'download',
  'share',
  'edit_save',
]);

function safeId(id) {
  return String(id || '').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 120);
}

export function normalizePrompt(p) {
  return String(p || '').replace(/\s+/g, ' ').trim().toLowerCase();
}

function loadHoldoutSet(dataDir, extraPrompts = []) {
  const set = new Set();
  const add = (p) => { const n = normalizePrompt(p); if (n) set.add(n); };
  for (const p of extraPrompts) add(p);
  for (const name of ['eval50.json', 'eval-50.json', 'blind50.json', 'blind-50.json', 'holdout_prompts.json']) {
    const fp = path.join(dataDir, name);
    if (!fs.existsSync(fp)) continue;
    try {
      const parsed = JSON.parse(fs.readFileSync(fp, 'utf8'));
      const arr = Array.isArray(parsed) ? parsed : (parsed.prompts || []);
      for (const item of arr) {
        add(typeof item === 'string' ? item : item?.prompt);
      }
    } catch {}
  }
  return set;
}

function percentile(sortedAsc, p) {
  if (!sortedAsc.length) return null;
  if (sortedAsc.length === 1) return sortedAsc[0];
  const idx = (sortedAsc.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedAsc[lo];
  return sortedAsc[lo] + (sortedAsc[hi] - sortedAsc[lo]) * (idx - lo);
}

export function createFlywheel({ dataDir, palette, holdoutPrompts = [] }) {
  fs.mkdirSync(dataDir, { recursive: true });
  const candsDir = path.join(dataDir, 'chat-cands');
  const editsDir = path.join(dataDir, 'edits');
  fs.mkdirSync(candsDir, { recursive: true });
  fs.mkdirSync(editsDir, { recursive: true });
  const eventsLogPath = path.join(dataDir, 'chat-events.jsonl');
  const abQueueJsonPath = path.join(dataDir, 'ab-flywheel-queue.json');

  const db = new DatabaseSync(path.join(dataDir, 'flywheel.sqlite'));
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA busy_timeout = 5000;

    CREATE TABLE IF NOT EXISTS chat_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL,
      job_id TEXT,
      lid TEXT,
      user_id TEXT,
      trust_tier TEXT NOT NULL,
      session_chain_id TEXT,
      prev_job_id TEXT,
      event_type TEXT NOT NULL,
      candidate_idx INTEGER,
      rank INTEGER,
      shown_view TEXT,
      payload_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_chat_events_job ON chat_events(job_id);
    CREATE INDEX IF NOT EXISTS idx_chat_events_type_ts ON chat_events(event_type, ts);
    CREATE INDEX IF NOT EXISTS idx_chat_events_chain ON chat_events(session_chain_id);

    CREATE TABLE IF NOT EXISTS job_candidates (
      job_id TEXT PRIMARY KEY,
      ts TEXT NOT NULL,
      prompt TEXT NOT NULL,
      cond_json TEXT,
      model_key TEXT,
      scorer TEXT,
      n_cands INTEGER NOT NULL,
      top4_max_reward REAL,
      top4_indices_json TEXT,
      rewards_json TEXT,
      cands_path TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_job_cands_ts ON job_candidates(ts);

    CREATE TABLE IF NOT EXISTS edits (
      edit_id TEXT PRIMARY KEY,
      ts TEXT NOT NULL,
      job_id TEXT,
      parent_share_id TEXT,
      candidate_idx INTEGER,
      lid TEXT,
      user_id TEXT,
      trust_tier TEXT NOT NULL,
      prompt TEXT,
      cond_json TEXT,
      original_indices_blob BLOB NOT NULL,
      edited_indices_blob BLOB NOT NULL,
      diff_pixels INTEGER NOT NULL,
      diff_mask_json TEXT NOT NULL,
      opaque_px INTEGER NOT NULL,
      ops_count INTEGER NOT NULL,
      tool_counts_json TEXT NOT NULL,
      duration_ms INTEGER NOT NULL,
      action TEXT NOT NULL,
      opt_in_train INTEGER NOT NULL,
      png_path TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_edits_job ON edits(job_id);
    CREATE INDEX IF NOT EXISTS idx_edits_ts ON edits(ts);

    CREATE TABLE IF NOT EXISTS ab_queue (
      prompt TEXT PRIMARY KEY,
      queued_ts TEXT NOT NULL,
      queued_by TEXT,
      none_rate REAL,
      runs_count INTEGER,
      status TEXT NOT NULL DEFAULT 'queued'
    );
  `);

  const stmtInsertEvent = db.prepare(`
    INSERT INTO chat_events (ts, job_id, lid, user_id, trust_tier, session_chain_id, prev_job_id, event_type, candidate_idx, rank, shown_view, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const stmtHasJobCands = db.prepare('SELECT job_id, cands_path FROM job_candidates WHERE job_id = ?');
  const stmtInsertJobCands = db.prepare(`
    INSERT OR IGNORE INTO job_candidates (job_id, ts, prompt, cond_json, model_key, scorer, n_cands, top4_max_reward, top4_indices_json, rewards_json, cands_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const stmtInsertEdit = db.prepare(`
    INSERT INTO edits (edit_id, ts, job_id, parent_share_id, candidate_idx, lid, user_id, trust_tier, prompt, cond_json,
      original_indices_blob, edited_indices_blob, diff_pixels, diff_mask_json, opaque_px, ops_count, tool_counts_json, duration_ms, action, opt_in_train, png_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const stmtGetEdit = db.prepare('SELECT * FROM edits WHERE edit_id = ?');

  const holdouts = loadHoldoutSet(dataDir, holdoutPrompts);
  function isHoldoutPrompt(prompt) {
    return holdouts.has(normalizePrompt(prompt));
  }

  function appendEventLog(row) {
    try {
      fs.appendFileSync(eventsLogPath, JSON.stringify(row) + '\n', 'utf8');
    } catch {}
  }

  function recordEvent({
    ts = new Date().toISOString(),
    job_id = null,
    lid = null,
    user_id = null,
    trust_tier = 'anon',
    session_chain_id = null,
    prev_job_id = null,
    event_type,
    candidate_idx = null,
    rank = null,
    shown_view = null,
    payload = null,
  }) {
    if (!VALID_EVENT_TYPES.has(event_type)) {
      throw new Error('Invalid event_type: ' + event_type);
    }
    const payloadJson = payload != null ? JSON.stringify(payload) : null;
    const candIdx = Number.isInteger(candidate_idx) ? candidate_idx : null;
    const rankNum = Number.isInteger(rank) ? rank : null;
    const info = stmtInsertEvent.run(
      ts,
      job_id || null,
      lid || null,
      user_id || null,
      trust_tier || 'anon',
      session_chain_id || null,
      prev_job_id || null,
      event_type,
      candIdx,
      rankNum,
      shown_view || null,
      payloadJson
    );
    appendEventLog({
      id: Number(info.lastInsertRowid),
      ts,
      job_id: job_id || null,
      lid: lid || null,
      user_id: user_id || null,
      trust_tier: trust_tier || 'anon',
      session_chain_id: session_chain_id || null,
      prev_job_id: prev_job_id || null,
      event_type,
      candidate_idx: candIdx,
      rank: rankNum,
      shown_view: shown_view || null,
      payload: payload ?? null,
    });
    return Number(info.lastInsertRowid);
  }

  /**
   * Persist all kept candidates (typically 20) as compact gzipped uint16[256] grids (~2.5 KB/job).
   */
  function persistJobCandidates(job) {
    if (!job) return null;
    const jobId = job.job_id || job.id;
    if (!jobId) return null;
    const existing = stmtHasJobCands.get(jobId);
    if (existing) return existing;

    const rawCands = Array.isArray(job.candidates) && job.candidates.length
      ? job.candidates
      : (Array.isArray(job.images) ? job.images.map((im, i) => ({
          idx: im.idx ?? im.index ?? i,
          rank: im.rank ?? (i + 1),
          score: im.score ?? null,
          img: im.img,
          indices: im.indices,
          seed: im.seed ?? null,
        })) : []);
    if (!rawCands.length) return null;

    const sorted = rawCands.slice().sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999));
    const n = sorted.length;
    const grids = new Uint16Array(n * 256);
    const metaCands = [];

    for (let k = 0; k < n; k++) {
      const c = sorted[k];
      let arr = null;
      if (Array.isArray(c.indices) && c.indices.length === 256) {
        arr = Uint16Array.from(c.indices, (v) => Math.max(0, Math.min(4096, v | 0)));
      } else if (c.indices instanceof Uint16Array && c.indices.length === 256) {
        arr = c.indices;
      } else if (c.img || c.png) {
        try {
          arr = pngToIndices(palette, c.img || c.png);
        } catch {
          arr = new Uint16Array(256);
        }
      } else {
        arr = new Uint16Array(256);
      }
      grids.set(arr, k * 256);
      metaCands.push({
        idx: c.idx ?? k,
        rank: c.rank ?? (k + 1),
        score: c.score != null && Number.isFinite(Number(c.score)) ? Number(c.score) : null,
        seed: c.seed ?? null,
        fill: c.fill ?? null,
        ncolors: c.ncolors ?? null,
      });
    }

    const top4 = metaCands.slice(0, 4);
    const top4Scores = top4.map((c) => c.score).filter((x) => x != null && Number.isFinite(x));
    const top4MaxReward = top4Scores.length ? Math.max(...top4Scores) : null;
    const ts = job.ts || new Date().toISOString();
    const meta = {
      job_id: jobId,
      ts,
      prompt: job.prompt || '',
      cond: job.cond || null,
      model: job.model || null,
      scorer: job.scorer || null,
      sampling: job.sampling || null,
      candidates: metaCands,
    };
    const metaBuf = Buffer.from(JSON.stringify(meta), 'utf8');
    const header = Buffer.allocUnsafe(8);
    CAND_MAGIC.copy(header, 0);
    header.writeUInt32LE(metaBuf.length, 4);
    const gridBuf = Buffer.allocUnsafe(grids.byteLength);
    for (let i = 0; i < grids.length; i++) {
      gridBuf.writeUInt16LE(grids[i], i * 2);
    }
    const packed = Buffer.concat([header, metaBuf, gridBuf]);
    const gz = zlib.gzipSync(packed, { level: 9 });

    const relPath = path.join('chat-cands', `${safeId(jobId)}.cands.gz`);
    const absPath = path.join(dataDir, relPath);
    const tmpPath = absPath + '.tmp-' + process.pid;
    fs.writeFileSync(tmpPath, gz);
    fs.renameSync(tmpPath, absPath);

    stmtInsertJobCands.run(
      jobId,
      ts,
      job.prompt || '',
      job.cond ? JSON.stringify(job.cond) : null,
      job.model?.key || (typeof job.model === 'string' ? job.model : null),
      job.scorer || null,
      n,
      top4MaxReward,
      JSON.stringify(top4.map((c) => c.idx)),
      JSON.stringify(metaCands.map((c) => c.score)),
      relPath
    );

    return { job_id: jobId, cands_path: relPath, bytes: gz.length, n_cands: n };
  }

  function readJobCandidates(jobId) {
    const row = stmtHasJobCands.get(jobId);
    if (!row) return null;
    const absPath = path.join(dataDir, row.cands_path);
    if (!fs.existsSync(absPath)) return null;
    const raw = zlib.gunzipSync(fs.readFileSync(absPath));
    if (raw.length < 8 || !raw.subarray(0, 4).equals(CAND_MAGIC)) return null;
    const metaLen = raw.readUInt32LE(4);
    const meta = JSON.parse(raw.subarray(8, 8 + metaLen).toString('utf8'));
    const dataBuf = raw.subarray(8 + metaLen);
    const n = meta.candidates.length;
    const grids = [];
    for (let k = 0; k < n; k++) {
      const g = new Uint16Array(256);
      const base = k * 512;
      for (let i = 0; i < 256; i++) {
        g[i] = dataBuf.readUInt16LE(base + i * 2);
      }
      grids.push(g);
    }
    return { meta, grids };
  }

  function renderCandidateThumbPng(jobId, candSlotOrIdx = 0, size = 64) {
    const parsed = readJobCandidates(jobId);
    if (!parsed || !parsed.grids.length) return null;
    let slot = parsed.meta.candidates.findIndex((c) => c.idx === candSlotOrIdx);
    if (slot < 0) slot = Math.max(0, Math.min(parsed.grids.length - 1, candSlotOrIdx | 0));
    const rgba16 = indicesToRgba(palette, parsed.grids[slot]);
    return upscaleNearestPng(rgba16, 16, 16, size, 'transparent');
  }

  function validateIndices256(arr) {
    if (!Array.isArray(arr) || arr.length !== 256) return null;
    const out = new Uint16Array(256);
    for (let i = 0; i < 256; i++) {
      const v = Number(arr[i]);
      if (!Number.isInteger(v) || v < 0 || v > 4096) return null;
      out[i] = v;
    }
    return out;
  }

  function indicesToBlob(u16) {
    const buf = Buffer.allocUnsafe(512);
    for (let i = 0; i < 256; i++) buf.writeUInt16LE(u16[i], i * 2);
    return buf;
  }

  function blobToArray(buf) {
    const out = new Array(256);
    for (let i = 0; i < 256; i++) out[i] = buf.readUInt16LE(i * 2);
    return out;
  }

  function recordEdit({
    job_id = null,
    parent_share_id = null,
    candidate_idx = null,
    lid = null,
    user_id = null,
    trust_tier = 'anon',
    prompt = '',
    cond = null,
    original_indices,
    edited_indices,
    ops_count = 0,
    tool_counts = {},
    duration_ms = 0,
    action = 'save',
    opt_in_train = true,
  }) {
    const orig = validateIndices256(original_indices);
    const edited = validateIndices256(edited_indices);
    if (!orig || !edited) {
      const err = new Error('original_indices and edited_indices must be integer[256] in 0..4096');
      err.status = 400;
      throw err;
    }

    let opaquePx = 0;
    const diffMask = [];
    for (let i = 0; i < 256; i++) {
      if (edited[i] > 0) opaquePx++;
      if (orig[i] !== edited[i]) {
        diffMask.push({ i, from: orig[i], to: edited[i] });
      }
    }
    if (opaquePx < 4) {
      const err = new Error('Edited sprite needs at least 4 non-transparent pixels.');
      err.status = 400;
      throw err;
    }

    const editId = crypto.randomBytes(9).toString('base64url');
    const ts = new Date().toISOString();
    const rgba = indicesToRgba(palette, edited);
    const pngBuf = encodePng(16, 16, rgba);
    const relPng = path.join('edits', `${editId}.png`);
    const absPng = path.join(dataDir, relPng);
    fs.writeFileSync(absPng, pngBuf);

    const safeAction = ['save', 'download', 'share'].includes(action) ? action : 'save';
    const candIdx = Number.isInteger(candidate_idx) ? candidate_idx : null;

    stmtInsertEdit.run(
      editId,
      ts,
      job_id || null,
      parent_share_id || null,
      candIdx,
      lid || null,
      user_id || null,
      trust_tier || 'anon',
      prompt || '',
      cond ? JSON.stringify(cond) : null,
      indicesToBlob(orig),
      indicesToBlob(edited),
      diffMask.length,
      JSON.stringify(diffMask),
      opaquePx,
      Math.max(0, Number(ops_count) | 0),
      JSON.stringify(tool_counts || {}),
      Math.max(0, Number(duration_ms) | 0),
      safeAction,
      opt_in_train ? 1 : 0,
      relPng
    );

    recordEvent({
      ts,
      job_id,
      lid,
      user_id,
      trust_tier,
      event_type: 'edit_save',
      candidate_idx: candIdx,
      payload: {
        edit_id: editId,
        parent_share_id: parent_share_id || null,
        diff_pixels: diffMask.length,
        opaque_px: opaquePx,
        ops_count: Math.max(0, Number(ops_count) | 0),
        action: safeAction,
        opt_in_train: Boolean(opt_in_train),
      },
    });

    return {
      edit_id: editId,
      diff_pixels: diffMask.length,
      opaque_px: opaquePx,
      png_url: `/api/chat/edit/${editId}.png`,
    };
  }

  function markEditShared(editId) {
    if (!editId || typeof editId !== 'string') return false;
    const rec = stmtGetEdit.get(editId);
    if (!rec) return false;
    if (rec.action === 'share') return true;
    db.prepare("UPDATE edits SET action = 'share' WHERE edit_id = ?").run(editId);
    return true;
  }

  function getEditRecord(editId) {
    if (!editId || typeof editId !== 'string') return null;
    const row = stmtGetEdit.get(editId);
    if (!row) return null;
    return {
      ...row,
      abs_png_path: path.join(dataDir, row.png_path),
      original_indices: blobToArray(Buffer.from(row.original_indices_blob)),
      edited_indices: blobToArray(Buffer.from(row.edited_indices_blob)),
      diff_mask: JSON.parse(row.diff_mask_json || '[]'),
      tool_counts: JSON.parse(row.tool_counts_json || '{}'),
      cond: row.cond_json ? JSON.parse(row.cond_json) : null,
    };
  }

  function queuePromptForAb({ prompt, queuedBy = 'admin', noneRate = null, runsCount = null }) {
    const clean = String(prompt || '').replace(/\s+/g, ' ').trim();
    if (!clean) {
      const e = new Error('Prompt is required');
      e.status = 400;
      throw e;
    }
    if (isHoldoutPrompt(clean)) {
      const e = new Error('Refused: prompt belongs to eval-50 / blind-50 holdout set');
      e.status = 400;
      throw e;
    }
    const ts = new Date().toISOString();
    db.prepare(`
      INSERT INTO ab_queue (prompt, queued_ts, queued_by, none_rate, runs_count, status)
      VALUES (?, ?, ?, ?, ?, 'queued')
      ON CONFLICT(prompt) DO UPDATE SET
        queued_ts = excluded.queued_ts,
        queued_by = excluded.queued_by,
        none_rate = COALESCE(excluded.none_rate, ab_queue.none_rate),
        runs_count = COALESCE(excluded.runs_count, ab_queue.runs_count),
        status = 'queued'
    `).run(clean, ts, queuedBy, noneRate, runsCount);

    const allQueued = db.prepare('SELECT prompt, queued_ts, queued_by, none_rate, runs_count, status FROM ab_queue ORDER BY queued_ts DESC').all();
    fs.writeFileSync(abQueueJsonPath, JSON.stringify({ updated_at: ts, prompts: allQueued }, null, 2), 'utf8');
    return { ok: true, prompt: clean, status: 'queued', queued_ts: ts };
  }

  function getDiagnostics({ chatRows = [], trustLookup = null } = {}) {
    const totalEvents = db.prepare('SELECT COUNT(*) AS n FROM chat_events').get().n;
    const archivedCands = db.prepare('SELECT COUNT(*) AS n FROM job_candidates').get().n;

    // Load indexed candidate stats keyed by job_id
    const candMap = new Map();
    for (const r of db.prepare('SELECT job_id, prompt, top4_max_reward, rewards_json, cond_json FROM job_candidates').all()) {
      candMap.set(r.job_id, r);
    }

    // Load latest none_reason per job_id + event summaries
    const noneReasonByJob = new Map();
    const voteEventsByJob = new Map();
    const dlShareByJob = new Map();
    let retryChainsCount = 0;
    for (const ev of db.prepare('SELECT id, ts, job_id, lid, trust_tier, event_type, candidate_idx, rank, shown_view, payload_json, prev_job_id FROM chat_events ORDER BY id ASC').all()) {
      let payload = null;
      if (ev.payload_json) { try { payload = JSON.parse(ev.payload_json); } catch {} }
      if (ev.prev_job_id && (ev.event_type === 'retry_submit' || ev.event_type === 'vote')) {
        retryChainsCount++;
      }
      if (ev.event_type === 'none_reason' && ev.job_id && payload?.reason) {
        noneReasonByJob.set(ev.job_id, payload.reason);
      } else if (ev.event_type === 'vote' && ev.job_id) {
        voteEventsByJob.set(ev.job_id, { ...ev, payload });
      } else if ((ev.event_type === 'download' || ev.event_type === 'share') && ev.job_id) {
        const prev = dlShareByJob.get(ev.job_id) || { download: false, share: false };
        if (ev.event_type === 'download') prev.download = true;
        if (ev.event_type === 'share') prev.share = true;
        dlShareByJob.set(ev.job_id, prev);
      }
    }

    // Merge chatRows (from chat.jsonl) with vote events so neither old nor new rows are missed or duplicated
    const jobsById = new Map();
    for (const c of chatRows) {
      if (!c) continue;
      const jid = c.job_id || `${c.ts || ''}|${c.lid || ''}|${c.prompt || ''}`;
      const candRow = c.job_id ? candMap.get(c.job_id) : null;
      const shownScores = Array.isArray(c.shown)
        ? c.shown.slice(0, 4).map((x) => Number(x?.score)).filter(Number.isFinite)
        : [];
      const top4Max = candRow?.top4_max_reward != null
        ? Number(candRow.top4_max_reward)
        : (shownScores.length ? Math.max(...shownScores) : null);

      jobsById.set(jid, {
        job_id: c.job_id || null,
        ts: c.ts || c.at || '',
        prompt: c.prompt || '',
        lid: c.lid || '',
        name: c.name || c.lid || '',
        trust_tier: trustLookup ? trustLookup(c.lid) : (c.trust_tier || 'auto'),
        none_good: Boolean(c.none_good),
        chosen: c.chosen ?? null,
        candidate_idx: c.candidate_idx ?? null,
        rank: c.chosen_rank ?? c.rank ?? (Number.isInteger(c.chosen) && c.chosen >= 0 ? c.chosen + 1 : null),
        shown_view: c.shown_view || c.mode || 'top4',
        top4_max_reward: top4Max,
        chosen_score: c.chosen_score ?? null,
        thumb: c.thumb || null,
        has_cands: Boolean(candRow),
      });
    }

    for (const [jid, ev] of voteEventsByJob.entries()) {
      const p = ev.payload || {};
      const candRow = candMap.get(jid);
      const prev = jobsById.get(jid);
      let top4Max = candRow?.top4_max_reward != null ? Number(candRow.top4_max_reward) : (p.top4_max_reward ?? prev?.top4_max_reward ?? null);
      if (top4Max != null) top4Max = Number(top4Max);
      let rewards = null;
      if (candRow?.rewards_json) { try { rewards = JSON.parse(candRow.rewards_json); } catch {} }
      const rank = ev.rank ?? prev?.rank ?? (Number.isInteger(p.choice) && p.choice >= 0 ? p.choice + 1 : null);
      const chosenScore = (rank && rewards && rewards[rank - 1] != null) ? Number(rewards[rank - 1]) : (prev?.chosen_score ?? null);
      jobsById.set(jid, {
        job_id: jid,
        ts: prev?.ts || ev.ts,
        prompt: prev?.prompt || p.prompt || candRow?.prompt || '',
        lid: prev?.lid || ev.lid || '',
        name: prev?.name || p.name || ev.lid || '',
        trust_tier: ev.trust_tier || prev?.trust_tier || 'anon',
        none_good: p.none_good != null ? Boolean(p.none_good) : Boolean(prev?.none_good),
        chosen: p.choice !== undefined ? p.choice : (prev?.chosen ?? null),
        candidate_idx: ev.candidate_idx ?? prev?.candidate_idx ?? null,
        rank,
        shown_view: ev.shown_view || prev?.shown_view || p.mode || 'top4',
        top4_max_reward: top4Max,
        chosen_score: chosenScore,
        top1_score: (rewards && rewards[0] != null) ? Number(rewards[0]) : top4Max,
        thumb: prev?.thumb || (candRow ? `/api/admin/flywheel/cand_thumb?job=${encodeURIComponent(jid)}&slot=0` : null),
        has_cands: Boolean(candRow),
      });
    }

    const allJobs = [...jobsById.values()];
    const noneJobs = allJobs.filter((j) => j.none_good);

    // 1. "None" reason breakdown
    const reasonCounts = { wrong_subject: 0, messy_pixels: 0, cut_off: 0, wrong_style: 0, untagged: 0 };
    for (const j of noneJobs) {
      const r = j.job_id ? noneReasonByJob.get(j.job_id) : null;
      if (r && reasonCounts[r] !== undefined) reasonCounts[r]++;
      else reasonCounts.untagged++;
    }
    const noneTotal = noneJobs.length;
    const noneTaggedTotal = noneTotal - reasonCounts.untagged;
    const none_reasons = [...NONE_REASON_KEYS, 'untagged'].map((k) => ({
      reason: k,
      label: NONE_REASON_LABELS[k] || k,
      count: reasonCounts[k],
      pct: noneTotal ? reasonCounts[k] / noneTotal : 0,
    }));

    // 2. High-Reward False Positives (>= p90 among "none" picks with reward-v7 scores)
    const noneRewards = noneJobs
      .map((j) => j.top4_max_reward)
      .filter((x) => x != null && Number.isFinite(x))
      .sort((a, b) => a - b);
    const p90 = percentile(noneRewards, 0.9);
    const fpItems = p90 == null
      ? []
      : noneJobs
          .filter((j) => j.top4_max_reward != null && j.top4_max_reward >= p90)
          .sort((a, b) => b.top4_max_reward - a.top4_max_reward)
          .slice(0, 40)
          .map((j) => ({
            job_id: j.job_id,
            prompt: j.prompt,
            ts: j.ts,
            lid: j.lid,
            name: j.name,
            trust_tier: j.trust_tier,
            top4_max_reward: j.top4_max_reward,
            none_reason: (j.job_id && noneReasonByJob.get(j.job_id)) || null,
            thumb_url: j.thumb || (j.has_cands && j.job_id ? `/api/admin/flywheel/cand_thumb?job=${encodeURIComponent(j.job_id)}&slot=0` : null),
          }));

    // 3. Rank-vs-Pick Discrepancy (All 20 picks where rank >= 5 beat top 4)
    const gridPicks = allJobs.filter((j) => !j.none_good && j.shown_view === 'grid' && Number.isInteger(j.rank));
    const rank1to4 = gridPicks.filter((j) => j.rank >= 1 && j.rank <= 4).length;
    const rank5to20List = gridPicks.filter((j) => j.rank >= 5);
    const buckets = [
      { bucket: 'Rank 1–4', count: rank1to4 },
      { bucket: 'Rank 5–8', count: gridPicks.filter((j) => j.rank >= 5 && j.rank <= 8).length },
      { bucket: 'Rank 9–12', count: gridPicks.filter((j) => j.rank >= 9 && j.rank <= 12).length },
      { bucket: 'Rank 13–16', count: gridPicks.filter((j) => j.rank >= 13 && j.rank <= 16).length },
      { bucket: 'Rank 17–20', count: gridPicks.filter((j) => j.rank >= 17).length },
    ];
    const recentDiscrepant = rank5to20List
      .slice()
      .sort((a, b) => String(b.ts).localeCompare(String(a.ts)))
      .slice(0, 30)
      .map((j) => ({
        job_id: j.job_id,
        prompt: j.prompt,
        ts: j.ts,
        rank: j.rank,
        candidate_idx: j.candidate_idx,
        chosen_score: j.chosen_score,
        top1_score: j.top1_score ?? j.top4_max_reward,
        name: j.name,
        thumb_url: j.has_cands && j.job_id
          ? `/api/admin/flywheel/cand_thumb?job=${encodeURIComponent(j.job_id)}&slot=${encodeURIComponent(j.candidate_idx ?? (j.rank - 1))}`
          : j.thumb,
      }));

    // 4. Top Failing Prompts (>= 5 runs, highest "none" rate, excluding eval-50 / blind-50)
    const queuedPrompts = new Set(
      db.prepare("SELECT prompt FROM ab_queue WHERE status = 'queued'").all().map((r) => normalizePrompt(r.prompt))
    );
    const byPrompt = new Map();
    for (const j of allJobs) {
      const norm = normalizePrompt(j.prompt);
      if (!norm || isHoldoutPrompt(norm)) continue;
      if (j.trust_tier === 'excluded') continue;
      let rec = byPrompt.get(norm);
      if (!rec) {
        rec = { prompt: j.prompt.trim(), total_runs: 0, none_count: 0, reasons: {}, last_ts: '' };
        byPrompt.set(norm, rec);
      }
      rec.total_runs++;
      if (j.none_good) {
        rec.none_count++;
        const r = (j.job_id && noneReasonByJob.get(j.job_id)) || 'untagged';
        rec.reasons[r] = (rec.reasons[r] || 0) + 1;
      }
      if (String(j.ts) > rec.last_ts) rec.last_ts = String(j.ts);
    }
    const failing_prompts = [...byPrompt.entries()]
      .filter(([, r]) => r.total_runs >= 5 && r.none_count > 0)
      .map(([norm, r]) => {
        const topReason = Object.entries(r.reasons)
          .filter(([k]) => k !== 'untagged')
          .sort((a, b) => b[1] - a[1])[0]?.[0] || null;
        return {
          prompt: r.prompt,
          total_runs: r.total_runs,
          none_count: r.none_count,
          none_rate: r.none_count / r.total_runs,
          top_reason: topReason,
          top_reason_label: topReason ? (NONE_REASON_LABELS[topReason] || topReason) : '—',
          last_ts: r.last_ts,
          queued_ab: queuedPrompts.has(norm),
        };
      })
      .sort((a, b) => (b.none_rate - a.none_rate) || (b.none_count - a.none_count))
      .slice(0, 40);

    // 5. Edits summary & recent edits
    const allEdits = db.prepare(`
      SELECT edit_id, ts, job_id, parent_share_id, candidate_idx, lid, trust_tier, prompt,
             diff_pixels, opaque_px, ops_count, action, opt_in_train
      FROM edits ORDER BY ts DESC
    `).all();
    const goldEdits = allEdits.filter((e) =>
      e.diff_pixels >= 1 && e.diff_pixels <= 24 && e.opaque_px >= 12 && e.opt_in_train === 1 && e.trust_tier !== 'excluded'
    );
    function countEditToShare(edits) {
      const ids = new Set();
      for (const e of edits) {
        if (e.action === 'share' && e.edit_id) ids.add(e.edit_id);
      }
      // Share events with an edit_id count even when job_id is missing (remix / job-less save & share).
      for (const ev of db.prepare("SELECT payload_json FROM chat_events WHERE event_type = 'share'").all()) {
        try {
          const payload = ev.payload_json ? JSON.parse(ev.payload_json) : null;
          if (payload && payload.edit_id) ids.add(String(payload.edit_id));
        } catch { /* torn payload */ }
      }
      return ids.size;
    }

    const recent_edits = allEdits.slice(0, 30).map((e) => ({
      ...e,
      opt_in_train: Boolean(e.opt_in_train),
      png_url: `/api/chat/edit/${e.edit_id}.png`,
    }));

    return {
      summary: {
        total_events: totalEvents,
        total_candidates_archived: archivedCands,
        total_edits: allEdits.length,
        gold_polish_edits: goldEdits.length,
        none_total: noneTotal,
        none_tagged_total: noneTaggedTotal,
        edit_to_share_count: countEditToShare(allEdits),
        ab_queued_prompts: queuedPrompts.size,
        retry_chains: retryChainsCount,
        p90_threshold: p90,
        rank_discrepancy_rate: gridPicks.length ? rank5to20List.length / gridPicks.length : 0,
      },
      none_reasons,
      false_positives: {
        p90_threshold: p90,
        total_none_with_scores: noneRewards.length,
        items: fpItems,
      },
      rank_discrepancy: {
        total_grid_picks: gridPicks.length,
        rank_1_4_picks: rank1to4,
        rank_5_20_picks: rank5to20List.length,
        discrepancy_rate: gridPicks.length ? rank5to20List.length / gridPicks.length : 0,
        rank_histogram: buckets,
        recent: recentDiscrepant,
      },
      failing_prompts,
      recent_edits,
    };
  }

  /**
   * Export confidence-weighted Flywheel v2 preference tuples:
   * { prompt, cond, chosen_grid, rejected_grids[], weight, source, meta }
   */
  function exportFlywheelTuples({ who = 'trusted', format = 'jsonl' } = {}) {
    const tuples = [];
    const allowTier = (tier) => {
      if (who === 'all') return true;
      if (who === 'owner') return tier === 'owner';
      return tier !== 'excluded';
    };

    // A) Edit diffs (gold polish pairs weight 3.0; heavy redraws 1.5; vandalism/blanking 0.0)
    const editRows = db.prepare('SELECT * FROM edits ORDER BY ts ASC').all();
    for (const e of editRows) {
      if (!e.opt_in_train || isHoldoutPrompt(e.prompt)) continue;
      if (!allowTier(e.trust_tier)) continue;
      let weight = 0.0;
      if (e.trust_tier === 'excluded' || e.diff_pixels > 160 || e.opaque_px < 12) {
        weight = 0.0;
      } else if (e.diff_pixels >= 1 && e.diff_pixels <= 24) {
        weight = (e.action === 'download' || e.action === 'share') ? 3.0 : 2.5;
      } else if (e.diff_pixels >= 25 && e.diff_pixels <= 96) {
        weight = 1.5;
      }
      if (weight <= 0 && who !== 'all') continue;
      tuples.push({
        prompt: e.prompt || '',
        cond: e.cond_json ? JSON.parse(e.cond_json) : null,
        chosen_grid: blobToArray(Buffer.from(e.edited_indices_blob)),
        rejected_grids: [blobToArray(Buffer.from(e.original_indices_blob))],
        weight,
        source: 'edit_diff',
        meta: {
          edit_id: e.edit_id,
          job_id: e.job_id,
          diff_pixels: e.diff_pixels,
          action: e.action,
          trust_tier: e.trust_tier,
          ts: e.ts,
        },
      });
    }

    // Preload download/share & none_reason lookup per job
    const dlShareJobs = new Set();
    const noneReasonByJob = new Map();
    const voteEvents = [];
    for (const ev of db.prepare('SELECT * FROM chat_events ORDER BY id ASC').all()) {
      let payload = null;
      if (ev.payload_json) { try { payload = JSON.parse(ev.payload_json); } catch {} }
      if ((ev.event_type === 'download' || ev.event_type === 'share') && ev.job_id) {
        dlShareJobs.add(ev.job_id);
      } else if (ev.event_type === 'none_reason' && ev.job_id && payload?.reason) {
        noneReasonByJob.set(ev.job_id, payload.reason);
      } else if (ev.event_type === 'vote' && ev.job_id) {
        voteEvents.push({ ...ev, payload });
      }
    }

    // B) Vote preference pairs & high-reward none negatives backed by archived 20-candidate grids
    for (const ev of voteEvents) {
      const p = ev.payload || {};
      const prompt = p.prompt || '';
      if (isHoldoutPrompt(prompt)) continue;
      if (!allowTier(ev.trust_tier)) continue;
      const cands = readJobCandidates(ev.job_id);
      if (!cands) continue;

      const { meta, grids } = cands;
      const top4Slots = [0, 1, 2, 3].filter((i) => i < grids.length);
      if (p.none_good) {
        const weight = ev.trust_tier === 'excluded' ? 0.0 : (ev.trust_tier === 'anon' ? 0.2 : 1.0);
        if (weight <= 0 && who !== 'all') continue;
        tuples.push({
          prompt: meta.prompt || prompt,
          cond: meta.cond || null,
          chosen_grid: null,
          rejected_grids: top4Slots.map((i) => Array.from(grids[i])),
          weight,
          source: 'none_negative',
          meta: {
            job_id: ev.job_id,
            none_reason: noneReasonByJob.get(ev.job_id) || null,
            top4_max_reward: meta.candidates?.[0]?.score ?? null,
            trust_tier: ev.trust_tier,
            ts: ev.ts,
          },
        });
        continue;
      }

      // Find picked grid slot
      let pickedSlot = -1;
      if (ev.candidate_idx != null) {
        pickedSlot = meta.candidates.findIndex((c) => c.idx === ev.candidate_idx);
      }
      if (pickedSlot < 0 && Number.isInteger(ev.rank) && ev.rank >= 1 && ev.rank <= grids.length) {
        pickedSlot = ev.rank - 1;
      }
      if (pickedSlot < 0 && Number.isInteger(p.choice) && p.choice >= 0 && p.choice < grids.length) {
        pickedSlot = p.choice;
      }
      if (pickedSlot < 0) continue;

      const pickedRank = meta.candidates[pickedSlot]?.rank ?? (pickedSlot + 1);
      const isDlOrShare = dlShareJobs.has(ev.job_id);
      const isGridBeatTop4 = ev.shown_view === 'grid' && pickedRank >= 5;

      let weight = 1.0;
      let source = 'chat_pick';
      if (ev.trust_tier === 'excluded') {
        weight = 0.0;
      } else if (isDlOrShare) {
        weight = 2.0;
        source = 'chat_download_share';
      } else if (isGridBeatTop4) {
        weight = 1.8;
        source = 'chat_grid_discrepancy';
      } else if (ev.trust_tier === 'anon') {
        weight = 0.2;
      }
      if (weight <= 0 && who !== 'all') continue;

      const rejectedSlots = top4Slots.filter((i) => i !== pickedSlot);
      tuples.push({
        prompt: meta.prompt || prompt,
        cond: meta.cond || null,
        chosen_grid: Array.from(grids[pickedSlot]),
        rejected_grids: rejectedSlots.map((i) => Array.from(grids[i])),
        weight,
        source,
        meta: {
          job_id: ev.job_id,
          picked_rank: pickedRank,
          shown_view: ev.shown_view || 'top4',
          downloaded_or_shared: isDlOrShare,
          trust_tier: ev.trust_tier,
          ts: ev.ts,
        },
      });
    }

    if (format === 'csv') {
      const escCsv = (v) => {
        const s = String(v ?? '');
        return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      };
      const lines = ['source,weight,prompt,trust_tier,rejected_count,job_id,ts'];
      for (const t of tuples) {
        lines.push([
          t.source,
          t.weight,
          t.prompt,
          t.meta?.trust_tier || '',
          t.rejected_grids.length,
          t.meta?.job_id || '',
          t.meta?.ts || '',
        ].map(escCsv).join(','));
      }
      return { contentType: 'text/csv; charset=utf-8', body: lines.join('\n') + '\n' };
    }

    return {
      contentType: 'application/x-ndjson; charset=utf-8',
      body: tuples.map((t) => JSON.stringify(t)).join('\n') + (tuples.length ? '\n' : ''),
      count: tuples.length,
    };
  }

  return {
    db,
    isHoldoutPrompt,
    recordEvent,
    persistJobCandidates,
    readJobCandidates,
    renderCandidateThumbPng,
    recordEdit,
    markEditShared,
    getEditRecord,
    queuePromptForAb,
    getDiagnostics,
    exportFlywheelTuples,
    close: () => db.close(),
  };
}
