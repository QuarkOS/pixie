// Label storage: append-only JSONL on the container disk (single Node process).
// Each line = one submitted block {v, lid, name, device, at, labels:[...]}. Purge rewrites the file atomically.
import fs from 'node:fs';
import path from 'node:path';
import { COUNTS, INDEX, GRID_ENABLED } from './core.js';
import { AB_INDEX, AB_COUNTS } from './ab.js';

const DIR = process.env.DATA_DIR || path.join(process.cwd(), 'data');
const FILE = path.join(DIR, 'labels.jsonl');
fs.mkdirSync(DIR, { recursive: true });
let rows = null;
// lid -> Set(pair index) of A/B pairs this labeller already has stored: server-side dedupe for resubmits from localStorage / beacons
const AB_SEEN = new Map();
function markSeen(lid, pairId) { const j = AB_INDEX.get(pairId); if (j === undefined) return; let s = AB_SEEN.get(lid); if (!s) AB_SEEN.set(lid, (s = new Set())); s.add(j); }
export function abSeen(lid) { load(); return AB_SEEN.get(lid) || new Set(); }
// lid -> Set(round index) of grid rounds already stored for this labeller (dedupe + never serve a round twice)
const GRID_SEEN = new Map();
function markGrid(lid, rid) { const i = INDEX.get(rid); if (i === undefined) return; let s = GRID_SEEN.get(lid); if (!s) GRID_SEEN.set(lid, (s = new Set())); s.add(i); }
export function gridSeen(lid) { load(); return GRID_SEEN.get(lid) || new Set(); }
// Grid mode: everyone only if GRID_ENABLED=1, otherwise only labeller ids the admin marked owner or trusted (excluded/auto stay A/B-only).
export const GRID_STATUSES = new Set(['owner', 'trusted']);
export function gridAllowed(lid) { const st = typeof lid === 'string' ? overrideMap()[lid] : undefined; return st !== 'excluded' && (GRID_ENABLED || GRID_STATUSES.has(st)); }

function load() {
  if (rows) return rows;
  rows = [];
  if (fs.existsSync(FILE)) {
    for (const line of fs.readFileSync(FILE, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try { addBatch(JSON.parse(line)); } catch { /* skip a torn last line */ }
    }
  }
  return rows;
}
function addBatch(b) {
  for (const r of b.labels) {
    rows.push({ ...r, lid: b.lid, name: b.name, device: b.device });
    if (r.type === 'ab') { markSeen(b.lid, r.pair_id); const j = AB_INDEX.get(r.pair_id); if (j !== undefined) AB_COUNTS.set(j, (AB_COUNTS.get(j) || 0) + 1); continue; }
    markGrid(b.lid, r.round_id);
    const i = INDEX.get(r.round_id);
    if (i !== undefined) COUNTS.set(i, (COUNTS.get(i) || 0) + 1);
  }
}

export async function putBatch(lid, batch) {
  load();
  fs.appendFileSync(FILE, JSON.stringify(batch) + '\n', { mode: 0o600 });
  // COUNTS for these rounds are bumped by the caller (submit.js); only add rows here.
  for (const r of batch.labels) { rows.push({ ...r, lid: batch.lid, name: batch.name, device: batch.device }); if (r.type === 'ab') markSeen(batch.lid, r.pair_id); else markGrid(batch.lid, r.round_id); }
}

export async function ensureCounts() { load(); }

export async function compact({ dropLids = null } = {}) {
  load();
  if (dropLids && dropLids.size) {
    const keep = [];
    const lines = fs.existsSync(FILE) ? fs.readFileSync(FILE, 'utf8').split('\n') : [];
    for (const line of lines) {
      if (!line.trim()) continue;
      try { const b = JSON.parse(line); if (!dropLids.has(b.lid)) keep.push(line); } catch { /* drop torn line */ }
    }
    const tmp = FILE + '.tmp';
    fs.writeFileSync(tmp, keep.length ? keep.join('\n') + '\n' : '', { mode: 0o600 });
    fs.renameSync(tmp, FILE);
    rows = null; COUNTS.clear(); AB_COUNTS.clear(); AB_SEEN.clear(); GRID_SEEN.clear(); load();
  }
  return { rows, merged: 0 };
}

// ---- manual per-labeller overrides (admin): { lid: { status, name, at } } ----
const OV_FILE = path.join(DIR, 'overrides.json');
export function getOverrides() {
  try { return JSON.parse(fs.readFileSync(OV_FILE, 'utf8')); } catch { return {}; }
}
export function overrideMap() {
  const o = {};
  for (const [lid, v] of Object.entries(getOverrides())) if (v && v.status && v.status !== 'auto') o[lid] = v.status;
  return o;
}
export function setOverride(lid, status, name) {
  const all = getOverrides();
  if (status === 'auto') delete all[lid];
  else all[lid] = { status, name: name || all[lid]?.name || null, at: new Date().toISOString() };
  const tmp = OV_FILE + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(all, null, 1) + '\n', { mode: 0o600 });
  fs.renameSync(tmp, OV_FILE);
  return all;
}
