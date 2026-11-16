// Admin Comparison Grid runner, sequential chat-queue interleaver, persistence & HTTP handlers (Node 22+ zero-dependency).
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { renderGridPng } from './png.mjs';
import { sendJson, readJsonBody } from './handlers.mjs';

function safeId(id) {
  return String(id || '').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 120);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Construct the prompt for a comparison grid cell.
 * 'plain' (case-insensitive) or empty phrase produces the subject alone;
 * otherwise '<subject> <phrase>'.
 */
export function buildCellPrompt(subject, phrase) {
  const subj = String(subject ?? '').replace(/\s+/g, ' ').trim();
  const phr = String(phrase ?? '').replace(/\s+/g, ' ').trim();
  if (!phr || phr.toLowerCase() === 'plain') return subj;
  return `${subj} ${phr}`;
}

function parseLineList(val) {
  const arr = Array.isArray(val)
    ? val
    : (typeof val === 'string' ? val.split(/\r?\n/) : []);
  return arr
    .map((s) => String(s ?? '').replace(/\s+/g, ' ').trim())
    .filter(Boolean);
}

/**
 * Validate and normalize POST /api/admin/grid input.
 */
export function normalizeGridInput(body) {
  if (!body || typeof body !== 'object') {
    const e = new Error('JSON body required');
    e.status = 400;
    throw e;
  }
  const title = String(body.title ?? '').replace(/\s+/g, ' ').trim().slice(0, 120) || 'Comparison Grid';
  const subjects = parseLineList(body.subjects ?? body.rows);
  const phrases = parseLineList(body.phrases ?? body.variants ?? body.columns);

  if (subjects.length < 1) {
    const e = new Error('At least 1 subject is required');
    e.status = 400;
    throw e;
  }
  if (subjects.length > 50) {
    const e = new Error('Maximum 50 subjects per grid');
    e.status = 400;
    throw e;
  }
  if (phrases.length < 1) {
    const e = new Error('At least 1 variant phrase is required');
    e.status = 400;
    throw e;
  }
  if (phrases.length > 30) {
    const e = new Error('Maximum 30 variant phrases per grid');
    e.status = 400;
    throw e;
  }
  const totalCells = subjects.length * phrases.length;
  if (totalCells > 250) {
    const e = new Error(`Grid too large (${totalCells} cells); maximum is 250 cells`);
    e.status = 400;
    throw e;
  }

  let samples = 3;
  const rawSamples = body.samples ?? body.n_samples;
  if (rawSamples !== undefined && rawSamples !== null && rawSamples !== '') {
    const n = Number(rawSamples);
    if (!Number.isInteger(n) || n < 1 || n > 8) {
      const e = new Error('samples must be an integer between 1 and 8');
      e.status = 400;
      throw e;
    }
    samples = n;
  }

  let seed = null;
  if (body.seed !== undefined && body.seed !== null && String(body.seed).trim() !== '') {
    const n = Number(body.seed);
    if (!Number.isInteger(n) || n < 0 || n > 2147483647) {
      const e = new Error('seed must be an integer between 0 and 2147483647');
      e.status = 400;
      throw e;
    }
    seed = n;
  }

  const palette = (body.palette !== undefined && body.palette !== null && body.palette !== '')
    ? body.palette
    : 'auto';
  const rawStyle = (body.style && typeof body.style === 'object') ? body.style : {};
  const style = {
    fill: String(rawStyle.fill || 'any'),
    outline: String(rawStyle.outline || 'any'),
    colours: String(rawStyle.colours || 'any'),
  };

  return {
    title,
    subjects,
    phrases,
    samples,
    seed,
    cond: { palette, style },
  };
}

function loadPhraseCountsFromDir(dataDir) {
  const map = new Map();
  if (!dataDir) return map;
  for (const name of ['train-phrase-counts.json', 'phrase-counts.json', 'train_counts.json']) {
    const fp = path.join(dataDir, name);
    if (!fs.existsSync(fp)) continue;
    try {
      const parsed = JSON.parse(fs.readFileSync(fp, 'utf8'));
      const obj = (parsed && typeof parsed === 'object' && parsed.counts && typeof parsed.counts === 'object')
        ? parsed.counts
        : parsed;
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
        for (const [k, v] of Object.entries(obj)) {
          if (v !== null && v !== undefined && Number.isFinite(Number(v))) {
            map.set(String(k).replace(/\s+/g, ' ').trim().toLowerCase(), Number(v));
          }
        }
      }
      break;
    } catch {}
  }
  return map;
}

function extractCellImages(job, samples) {
  if (!job) return [];
  const raw = Array.isArray(job.candidates) && job.candidates.length
    ? job.candidates.slice().sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999))
    : (Array.isArray(job.images) ? job.images : []);
  return raw.slice(0, samples).map((c, idx) => ({
    idx: c?.idx ?? c?.index ?? idx,
    rank: c?.rank ?? (idx + 1),
    score: c?.score != null && Number.isFinite(Number(c.score)) ? Number(c.score) : null,
    seed: c?.seed ?? null,
    img: typeof c === 'string' ? c : (c?.img || c?.dataUrl || c?.src || c?.url || ''),
  })).filter((im) => Boolean(im.img));
}

function tryResolveLocalImageUrl(dataDir, src) {
  if (!dataDir || typeof src !== 'string') return null;
  const clean = src.split('?')[0];
  const mEdit = /^\/api\/chat\/edit\/([a-zA-Z0-9._-]+)\.png$/.exec(clean);
  if (mEdit) {
    const abs = path.join(dataDir, 'edits', `${safeId(mEdit[1])}.png`);
    if (fs.existsSync(abs)) return fs.readFileSync(abs);
  }
  return null;
}

/**
 * Create the Comparison Grid manager & sequential interleaved worker loop.
 *
 * Options:
 * - dataDir: persistence root (stores grids under <dataDir>/grids/<grid_id>.json)
 * - submitChatJob({ prompt, palette, style, seed, n_samples, grid_id, row, col }): returns { job_id }
 * - getChatJob(jobId): returns { job_id, status, candidates?, images?, error? } | null
 * - getQueueStats(): returns { waiting, running } for live public chat queue interleaving
 * - getTrainCount(phrase): optional (phrase) => number | null | Promise<number | null>
 * - pollIntervalMs, interleaveDelayMs, cellTimeoutMs: timing controls
 */
export function createGridManager({
  dataDir,
  submitChatJob,
  getChatJob,
  getQueueStats = () => ({ waiting: 0, running: 0 }),
  getTrainCount = null,
  pollIntervalMs = 150,
  interleaveDelayMs = 40,
  cellTimeoutMs = 120_000,
} = {}) {
  if (typeof submitChatJob !== 'function') {
    throw new Error('createGridManager requires submitChatJob');
  }
  if (typeof getChatJob !== 'function') {
    throw new Error('createGridManager requires getChatJob');
  }

  const gridsDir = path.join(dataDir, 'grids');
  fs.mkdirSync(gridsDir, { recursive: true });
  const filePhraseCounts = loadPhraseCountsFromDir(dataDir);

  const grids = new Map();
  const pendingQueue = [];
  let runningLoop = null;
  let inFlightCount = 0;
  let closed = false;

  // Load persisted grids from disk on startup
  try {
    for (const file of fs.readdirSync(gridsDir)) {
      if (!file.endsWith('.json')) continue;
      try {
        const g = JSON.parse(fs.readFileSync(path.join(gridsDir, file), 'utf8'));
        if (!g || !g.grid_id) continue;
        // Any previously mid-flight grid from an earlier process cannot resume its old RAM job
        if (g.status === 'queued' || g.status === 'running') {
          let done = 0;
          let failed = 0;
          for (const row of g.cells || []) {
            for (const cell of row || []) {
              if (cell.status === 'done') done++;
              else if (cell.status === 'error') failed++;
              else {
                cell.status = 'error';
                cell.error = 'Interrupted by server restart';
                failed++;
              }
            }
          }
          g.completed_cells = done;
          g.failed_cells = failed;
          g.status = failed === g.total_cells ? 'error' : 'done';
          g.active_cell = null;
        }
        grids.set(g.grid_id, g);
      } catch {}
    }
  } catch {}

  function persistGrid(grid) {
    if (!grid || !grid.grid_id) return;
    const absPath = path.join(gridsDir, `${safeId(grid.grid_id)}.json`);
    const tmpPath = `${absPath}.tmp-${process.pid}-${Date.now()}`;
    fs.writeFileSync(tmpPath, JSON.stringify(grid), 'utf8');
    fs.renameSync(tmpPath, absPath);
  }

  async function resolveTrainCount(phrase) {
    const norm = String(phrase ?? '').replace(/\s+/g, ' ').trim();
    if (typeof getTrainCount === 'function') {
      try {
        const v = await getTrainCount(norm);
        if (v !== null && v !== undefined && Number.isFinite(Number(v))) {
          return Number(v);
        }
      } catch {}
    }
    const lower = norm.toLowerCase();
    if (filePhraseCounts.has(lower)) return filePhraseCounts.get(lower);
    return null;
  }

  function recomputeProgress(grid) {
    let completed = 0;
    let failed = 0;
    for (const row of grid.cells) {
      for (const cell of row) {
        if (cell.status === 'done') completed++;
        else if (cell.status === 'error') failed++;
      }
    }
    grid.completed_cells = completed;
    grid.failed_cells = failed;
    if (completed + failed >= grid.total_cells) {
      grid.status = failed === grid.total_cells ? 'error' : 'done';
      grid.active_cell = null;
      grid.finished_at = grid.finished_at || new Date().toISOString();
    }
  }

  async function waitForLiveQueueIdle() {
    while (!closed) {
      let stats = null;
      try {
        stats = await getQueueStats();
      } catch {
        stats = null;
      }
      const waiting = Number(stats?.waiting) || 0;
      const running = Number(stats?.running) || 0;
      if (waiting <= 0 && running <= 0) return;
      await sleep(pollIntervalMs);
    }
  }

  async function runSingleCell(grid, cell) {
    cell.status = 'running';
    if (grid.status === 'queued') grid.status = 'running';
    grid.active_cell = { row: cell.row, col: cell.col, prompt: cell.prompt };
    grid.updated_at = new Date().toISOString();

    let sub;
    try {
      sub = await submitChatJob({
        prompt: cell.prompt,
        palette: grid.cond.palette,
        style: grid.cond.style,
        seed: grid.seed,
        n_samples: grid.samples,
        source: 'admin_grid',
        grid_id: grid.grid_id,
        row: cell.row,
        col: cell.col,
      });
    } catch (err) {
      cell.status = 'error';
      cell.error = err?.message || 'Failed to queue job';
      recomputeProgress(grid);
      grid.updated_at = new Date().toISOString();
      persistGrid(grid);
      return;
    }

    if (!sub || !sub.job_id || sub.status === 'error') {
      cell.status = 'error';
      cell.error = sub?.error || 'Failed to queue job';
      recomputeProgress(grid);
      grid.updated_at = new Date().toISOString();
      persistGrid(grid);
      return;
    }

    cell.job_id = sub.job_id;
    const deadline = Date.now() + cellTimeoutMs;
    let missingCount = 0;

    while (!closed) {
      let job = null;
      try {
        job = await getChatJob(cell.job_id);
      } catch {
        job = null;
      }

      if (!job) {
        missingCount++;
        if (missingCount > 30 || Date.now() > deadline) {
          cell.status = 'error';
          cell.error = 'Chat job disappeared from queue';
          break;
        }
      } else if (job.status === 'done') {
        cell.images = extractCellImages(job, grid.samples);
        cell.sampling = job.sampling || null;
        cell.status = cell.images.length ? 'done' : 'error';
        if (!cell.images.length) cell.error = 'Job completed without images';
        break;
      } else if (job.status === 'error') {
        cell.status = 'error';
        cell.error = job.error || 'Generation failed';
        break;
      } else if (Date.now() > deadline) {
        cell.status = 'error';
        cell.error = 'Cell generation timed out';
        break;
      }

      await sleep(pollIntervalMs);
    }

    if (closed && cell.status === 'running') {
      cell.status = 'error';
      cell.error = 'Server shutting down';
    }

    recomputeProgress(grid);
    grid.updated_at = new Date().toISOString();
    persistGrid(grid);
  }

  async function runSequentialLoop() {
    let completedAny = false;
    while (!closed && pendingQueue.length > 0) {
      await waitForLiveQueueIdle();
      if (closed) break;

      // Brief yield between consecutive grid cells so waiting live chat jobs take priority
      if (completedAny && interleaveDelayMs > 0) {
        await sleep(interleaveDelayMs);
        if (closed) break;
        await waitForLiveQueueIdle();
        if (closed) break;
      }

      const task = pendingQueue.shift();
      if (!task) continue;
      const grid = grids.get(task.gridId);
      const cell = grid?.cells?.[task.row]?.[task.col];
      if (!grid || !cell) continue;

      inFlightCount++;
      try {
        await runSingleCell(grid, cell);
      } finally {
        inFlightCount--;
      }
      completedAny = true;
    }
  }

  function ensureRunner() {
    if (closed || runningLoop || pendingQueue.length === 0) return;
    runningLoop = runSequentialLoop().finally(() => {
      runningLoop = null;
      if (!closed && pendingQueue.length > 0) ensureRunner();
    });
  }

  async function createGrid(input) {
    const cfg = normalizeGridInput(input);
    const gridId = 'grid-' + crypto.randomBytes(5).toString('hex');
    const now = new Date().toISOString();

    const columns = await Promise.all(cfg.phrases.map(async (phrase, index) => ({
      index,
      phrase,
      train_count: await resolveTrainCount(phrase),
    })));

    const cells = cfg.subjects.map((subject, r) =>
      columns.map((col, c) => ({
        row: r,
        col: c,
        subject,
        phrase: col.phrase,
        prompt: buildCellPrompt(subject, col.phrase),
        status: 'queued',
        job_id: null,
        seed: cfg.seed,
        error: null,
        images: [],
      }))
    );

    const totalCells = cfg.subjects.length * columns.length;
    const grid = {
      ok: true,
      grid_id: gridId,
      title: cfg.title,
      status: 'queued',
      created_at: now,
      updated_at: now,
      finished_at: null,
      total_cells: totalCells,
      total_jobs: totalCells,
      completed_cells: 0,
      failed_cells: 0,
      active_cell: null,
      samples: cfg.samples,
      seed: cfg.seed,
      cond: cfg.cond,
      rows: cfg.subjects,
      columns,
      cells,
    };

    grids.set(gridId, grid);
    persistGrid(grid);

    for (let r = 0; r < cfg.subjects.length; r++) {
      for (let c = 0; c < columns.length; c++) {
        pendingQueue.push({ gridId, row: r, col: c });
      }
    }
    ensureRunner();

    return getGrid(gridId);
  }

  function getGrid(gridId) {
    if (!gridId) return null;
    let g = grids.get(String(gridId));
    if (!g) {
      const fp = path.join(gridsDir, `${safeId(gridId)}.json`);
      if (fs.existsSync(fp)) {
        try {
          g = JSON.parse(fs.readFileSync(fp, 'utf8'));
          if (g && g.grid_id) grids.set(g.grid_id, g);
        } catch {
          return null;
        }
      }
    }
    return g || null;
  }

  function listGrids(limit = 30) {
    return [...grids.values()]
      .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))
      .slice(0, limit)
      .map((g) => ({
        grid_id: g.grid_id,
        title: g.title,
        status: g.status,
        created_at: g.created_at,
        finished_at: g.finished_at || null,
        rows_count: (g.rows || []).length,
        cols_count: (g.columns || []).length,
        total_cells: g.total_cells,
        completed_cells: g.completed_cells,
        failed_cells: g.failed_cells,
        samples: g.samples,
        seed: g.seed,
        cond: g.cond,
      }));
  }

  function renderGridExportPng(gridId, opts = {}) {
    const g = getGrid(gridId);
    if (!g) return null;
    // Clone shallow cell image wrappers when resolving relative image paths so persisted grid state stays unchanged
    const renderGrid = {
      ...g,
      cells: Array.isArray(g.cells)
        ? g.cells.map((row) => Array.isArray(row)
          ? row.map((cell) => {
            if (!cell || !Array.isArray(cell.images) || !cell.images.some((im) => typeof im?.img === 'string' && im.img.startsWith('/'))) {
              return cell;
            }
            return {
              ...cell,
              images: cell.images.map((im) => {
                if (typeof im?.img === 'string' && im.img.startsWith('/')) {
                  const localBuf = tryResolveLocalImageUrl(dataDir, im.img);
                  if (localBuf) return { ...im, img: `data:image/png;base64,${localBuf.toString('base64')}` };
                }
                return im;
              }),
            };
          })
          : row)
        : g.cells,
    };
    return {
      grid: g,
      png: renderGridPng(renderGrid, opts),
    };
  }

  async function close() {
    closed = true;
    pendingQueue.length = 0;
    if (runningLoop) {
      try {
        await runningLoop;
      } catch {}
    }
  }

  return {
    createGrid,
    getGrid,
    listGrids,
    resolveTrainCount,
    renderGridExportPng,
    getStats: () => ({
      in_flight: inFlightCount,
      pending_cells: pendingQueue.length,
      total_grids: grids.size,
    }),
    close,
  };
}

/**
 * Create HTTP route handlers for /api/admin/grid*.
 */
export function createGridHandlers({ gridManager, isAdmin }) {
  if (!gridManager) throw new Error('createGridHandlers requires gridManager');
  if (typeof isAdmin !== 'function') throw new Error('createGridHandlers requires isAdmin');

  function requireAdmin(req, res) {
    if (!isAdmin(req)) {
      sendJson(res, 401, { error: 'Admin login required' });
      return false;
    }
    return true;
  }

  async function handleCreateGrid(req, res) {
    if (!requireAdmin(req, res)) return;
    try {
      const body = await readJsonBody(req, 512 * 1024);
      const grid = await gridManager.createGrid(body);
      return sendJson(res, 200, grid);
    } catch (e) {
      return sendJson(res, e.status || 400, { error: e.message || 'Could not create comparison grid' });
    }
  }

  async function handleListGrids(req, res, urlObj = null) {
    if (!requireAdmin(req, res)) return;
    const out = { ok: true, grids: gridManager.listGrids() };
    const phrasesRaw = urlObj?.searchParams?.get('phrases');
    if (phrasesRaw) {
      const phrases = phrasesRaw.split('\n').map((s) => s.trim()).filter(Boolean);
      const train_counts = {};
      for (const p of phrases) {
        train_counts[p] = await gridManager.resolveTrainCount(p);
      }
      out.train_counts = train_counts;
    }
    return sendJson(res, 200, out);
  }

  function handleGetGrid(req, res, gridId) {
    if (!requireAdmin(req, res)) return;
    const grid = gridManager.getGrid(gridId);
    if (!grid) return sendJson(res, 404, { error: 'Grid not found' });
    return sendJson(res, 200, grid);
  }

  function handleGetGridPng(req, res, gridId, urlObj = null) {
    if (!requireAdmin(req, res)) return;
    const widthParam = urlObj?.searchParams?.get('width');
    const bgParam = urlObj?.searchParams?.get('bg');
    const rendered = gridManager.renderGridExportPng(gridId, {
      targetWidth: widthParam ? Number(widthParam) : 1024,
      bg: bgParam || undefined,
    });
    if (!rendered) {
      return sendJson(res, 404, { error: 'Grid not found' });
    }
    const slug = String(rendered.grid.title || gridId)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 60) || safeId(gridId);
    res.writeHead(200, {
      'Content-Type': 'image/png',
      'Content-Length': rendered.png.length,
      'Content-Disposition': `attachment; filename="${slug}-${safeId(gridId)}.png"`,
      'Cache-Control': 'no-store',
    });
    res.end(rendered.png);
  }

  return {
    handleCreateGrid,
    handleListGrids,
    handleGetGrid,
    handleGetGridPng,
  };
}
