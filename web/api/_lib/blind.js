// Blind-50 results live in DATA_DIR/blind50/, never in labels.jsonl or chat.jsonl.
import fs from 'node:fs';
import path from 'node:path';

const DIR = path.join(process.env.DATA_DIR || path.join(process.cwd(), 'data'), 'blind50');
const MANIFEST = path.join(DIR, 'manifest.json');
const MAPPING = path.join(DIR, 'mapping.json');
const RESULTS = path.join(DIR, 'results.jsonl');

function loadManifest() {
  const m = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
  if (!m || !Array.isArray(m.pairs) || m.pairs.length !== 50) throw Object.assign(new Error('blind set not ready'), { status: 503 });
  return m;
}
function loadMapping() {
  return JSON.parse(fs.readFileSync(MAPPING, 'utf8'));
}
function allResults() {
  if (!fs.existsSync(RESULTS)) return [];
  const rows = [];
  for (const line of fs.readFileSync(RESULTS, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); } catch { /* skip torn last line */ }
  }
  return rows;
}
function writeResults(rows) {
  const tmp = RESULTS + '.tmp';
  fs.writeFileSync(tmp, rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''), { mode: 0o600 });
  fs.renameSync(tmp, RESULTS);
  try { fs.chmodSync(RESULTS, 0o600); } catch {}
}
function cleanRater(raw) {
  if (typeof raw !== 'string') return null;
  const n = raw.normalize('NFKC').replace(/[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e]/g, '').replace(/\s+/g, ' ').trim();
  if (n.length < 2 || n.length > 30) return null;
  if (!/^[\p{L}\p{N} ._'\-]+$/u.test(n)) return null;
  return n;
}
function forRater(rows, rater) {
  return rows.filter((r) => r.rater === rater && r.v === 1);
}
function doneSet(rows, rater) {
  const s = new Set();
  for (const r of forRater(rows, rater)) if (typeof r.i === 'number') s.add(r.i);
  return s;
}
function stateFor(rater) {
  const man = loadManifest();
  const map = loadMapping();
  const rows = allResults();
  const done = doneSet(rows, rater);
  const n = man.pairs.length;
  // first unanswered pair, in order
  let next = man.pairs.find((p) => !done.has(p.i));
  if (!next) {
    return { n, done: done.size, finished: true };
  }
  return { n, done: done.size, finished: false, i: next.i, prompt: next.prompt, left: next.left, right: next.right };
}
function json(res, code, obj) {
  res.statusCode = code;
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(obj));
}

export function handleNext(req, res) {
  const u = new URL(req.url, 'http://x');
  const rater = cleanRater(u.searchParams.get('rater') || (req.body && req.body.rater));
  if (!rater) return json(res, 400, { error: 'name required' });
  try { return json(res, 200, stateFor(rater)); }
  catch (e) { return json(res, e.status || 500, { error: e.message || 'error' }); }
}

const ACTIONS = new Set(['left', 'right', 'both_good', 'both_bad', 'skip']);

export function handleSubmit(req, res) {
  const rater = cleanRater(req.body && req.body.rater);
  const action = req.body && req.body.action;
  const i = Number(req.body && req.body.i);
  if (!rater) return json(res, 400, { error: 'name required' });
  if (!ACTIONS.has(action)) return json(res, 400, { error: 'bad action' });
  try {
    const man = loadManifest();
    const map = loadMapping();
    const pair = man.pairs.find((p) => p.i === i);
    const hidden = (map.pairs || []).find((p) => p.i === i);
    if (!pair || !hidden) return json(res, 400, { error: 'bad pair' });
    const rows = allResults();
    if (doneSet(rows, rater).has(i)) return json(res, 200, stateFor(rater)); // idempotent
    const pick_model = action === 'left' ? hidden.left : action === 'right' ? hidden.right : null;
    const rec = {
      v: 1,
      ts: new Date().toISOString(),
      rater,
      i,
      prompt: pair.prompt,
      action,
      left_model: hidden.left,
      right_model: hidden.right,
      pick_model,
    };
    fs.mkdirSync(DIR, { recursive: true, mode: 0o750 });
    fs.appendFileSync(RESULTS, JSON.stringify(rec) + '\n', { mode: 0o600 });
    return json(res, 200, stateFor(rater));
  } catch (e) { return json(res, e.status || 500, { error: e.message || 'error' }); }
}

export function handleUndo(req, res) {
  const rater = cleanRater(req.body && req.body.rater);
  if (!rater) return json(res, 400, { error: 'name required' });
  try {
    const rows = allResults();
    let last = -1;
    for (let k = rows.length - 1; k >= 0; k--) if (rows[k].v === 1 && rows[k].rater === rater) { last = k; break; }
    if (last < 0) return json(res, 200, stateFor(rater));
    rows.splice(last, 1);
    writeResults(rows);
    return json(res, 200, stateFor(rater));
  } catch (e) { return json(res, e.status || 500, { error: e.message || 'error' }); }
}
