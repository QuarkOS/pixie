// Public sprite share records. Stored under DATA_DIR/shares/ (jsonl + files).
// Does not read or write labels.jsonl or chat.jsonl.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { decodePng, encodePng, native16FromDataUri, renderOgPng, renderSpritePng, parseBg } from './png.js';

const DIR = process.env.DATA_DIR || path.join(process.cwd(), 'data');
const ROOT = path.join(DIR, 'shares');
const FILE = path.join(ROOT, 'shares.jsonl');
export const THUMBS_DIR = path.join(DIR, 'chat-thumbs');

function ensure() {
  fs.mkdirSync(ROOT, { recursive: true, mode: 0o700 });
  fs.mkdirSync(THUMBS_DIR, { recursive: true, mode: 0o700 });
  try {
    if (!fs.existsSync(FILE)) fs.writeFileSync(FILE, '', { mode: 0o600 });
  } catch (e) {
    console.error('share ensure', e && e.message);
  }
}

let cache = null;
function load() {
  if (cache) return cache;
  const next = new Map();
  ensure();
  try {
    if (fs.existsSync(FILE)) {
      for (const line of fs.readFileSync(FILE, 'utf8').split('\n')) {
        if (!line.trim()) continue;
        try { const r = JSON.parse(line); if (r && r.id) next.set(r.id, r); } catch { /* torn line */ }
      }
    }
  } catch (e) {
    console.error('share load', e && e.message);
  }
  cache = next;
  return cache;
}

export function getShare(id) {
  if (typeof id !== 'string' || !/^[A-Za-z0-9_-]{10,24}$/.test(id)) return null;
  return load().get(id) || null;
}

export function native16Path(id) { return path.join(ROOT, id, 'sprite16.png'); }
export function ogPath(id) { return path.join(ROOT, id, 'og.png'); }

export function readNative16Rgba(id) {
  const p = native16Path(id);
  if (!fs.existsSync(p)) return null;
  const im = decodePng(fs.readFileSync(p));
  if (im.width !== 16 || im.height !== 16) throw new Error('stored sprite is not 16×16');
  return im.rgba;
}

export function createShare({ prompt, png, index, job_id, lid, name, model, cond, seed, score, edit_id, refined, png16, ogPng, candidate_idx }) {
  ensure();
  const id = crypto.randomBytes(10).toString('base64url');
  const dir = path.join(ROOT, id);
  fs.mkdirSync(dir, { mode: 0o700 });
  if (png16) {
    fs.writeFileSync(native16Path(id), png16, { mode: 0o600 });
  } else {
    const native = native16FromDataUri(png);
    fs.writeFileSync(native16Path(id), encodePng(native, 16, 16), { mode: 0o600 });
  }
  if (ogPng) {
    fs.writeFileSync(ogPath(id), ogPng, { mode: 0o600 });
  } else {
    const rgba = readNative16Rgba(id);
    fs.writeFileSync(ogPath(id), renderOgPng(rgba, prompt), { mode: 0o600 });
  }
  const rec = {
    v: 1, id, prompt: String(prompt || '').slice(0, 80),
    job_id: job_id || null, index: Number.isInteger(index) ? index : null,
    candidate_idx: Number.isInteger(candidate_idx) ? candidate_idx : null,
    lid: lid || null, name: name || null,
    model: model || null, cond: cond || null,
    seed: seed ?? null, score: score ?? null,
    edit_id: edit_id || null, refined: Boolean(refined || edit_id),
    ts: new Date().toISOString(),
  };
  fs.appendFileSync(FILE, JSON.stringify(rec) + '\n', { mode: 0o600 });
  load().set(id, rec);
  return rec;
}

export function spritePngFor(id, size, bg) {
  const rgba = readNative16Rgba(id);
  if (!rgba) return null;
  return renderSpritePng(rgba, size || 512, parseBg(bg));
}

export function ogPngFor(id) {
  const p = ogPath(id);
  if (fs.existsSync(p)) return fs.readFileSync(p);
  const rgba = readNative16Rgba(id);
  if (!rgba) return null;
  const rec = getShare(id);
  return renderOgPng(rgba, rec && rec.prompt);
}

// Persist a 64×64 preview of a chat pick so the admin table can show a thumbnail.
// Existing votes are not rewritten. Files are named by job id.
export function saveChatThumb(jobId, pngUri) {
  if (!jobId || !pngUri) return null;
  ensure();
  const native = native16FromDataUri(pngUri);
  const buf = renderSpritePng(native, 64, null);
  const fp = path.join(THUMBS_DIR, jobId + '.png');
  fs.writeFileSync(fp, buf, { mode: 0o600 });
  return '/api/admin/chat_thumb?job=' + encodeURIComponent(jobId);
}

export function readChatThumb(jobId) {
  if (typeof jobId !== 'string' || !/^[A-Za-z0-9_-]{8,32}$/.test(jobId)) return null;
  const fp = path.join(THUMBS_DIR, jobId + '.png');
  if (!fs.existsSync(fp)) return null;
  return fs.readFileSync(fp);
}

export const PUBLIC_PATH = (id) => '/s/' + id;
