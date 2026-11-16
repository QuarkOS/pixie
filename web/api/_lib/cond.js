// Palette and style conditioning for the s7 v3 model (n_pal=3, style=true).
// Tensor layout matches v3_model.default_cond / train_v3.py make_plan:
//   cond[0:3] = palette ids (1..4096) or NO_PALETTE=4097, slots may be shuffled by the trainer
//               but the client can send them in any order (we do not reshuffle).
//   cond[3:6] = style tags [fill, outline, colours]; 0 = ANY in each channel.
// Invalid or missing input becomes the default (all NO_PALETTE + ANY), which is what
// plain chat already uses.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const NO_PALETTE = 4097;
export const N_PAL = 3;
export const STYLE_SIZES = [4, 3, 4]; // fill, outline, colours
export const FILL = { any: 0, small: 1, medium: 2, full: 3 };
export const OUTLINE = { any: 0, dark: 1, soft: 2 };
export const COLOURS = { any: 0, low: 1, mid: 2, high: 3 };

const NAMED_FILE = path.join(path.dirname(fileURLToPath(import.meta.url)), 'named_palettes.json');
const NAMED = JSON.parse(fs.readFileSync(NAMED_FILE, 'utf8'));

const FILL_LABEL = { any: 'Any fill', small: 'Small (under 80 pixels)', medium: 'Medium (80–180 pixels)', full: 'Full (over 180 pixels)' };
const OUTLINE_LABEL = { any: 'Any outline', dark: 'Dark outline', soft: 'Soft / light outline' };
const COLOURS_LABEL = { any: 'Any colour count', low: 'Few colours (≤12)', mid: 'Medium (13–24)', high: 'Many colours (>24)' };
const PALETTE_LABEL = {
  auto: 'Auto (no palette hint)',
  ember: 'Ember', gold: 'Gold', forest: 'Forest', ocean: 'Ocean', royal: 'Royal',
  rose: 'Rose', poison: 'Poison', ice: 'Ice', earth: 'Earth', mono: 'Mono',
  gameboy: 'Game Boy', sunset: 'Sunset', candy: 'Candy', blood: 'Blood', copper: 'Copper',
};

export const PALETTE_IDS = Object.fromEntries(Object.entries(NAMED).map(([k, v]) => [k, v.ids]));

export function publicCondOptions() {
  const palettes = [{ id: 'auto', label: PALETTE_LABEL.auto, ids: null, swatches: [] }];
  for (const [id, v] of Object.entries(NAMED)) {
    palettes.push({
      id, label: PALETTE_LABEL[id] || id,
      ids: v.ids,
      swatches: v.colors.map((c) => ({ hex: rgbHex(c.palette_rgb), id: c.id })),
    });
  }
  return {
    model: { n_pal: N_PAL, style: true, no_palette: NO_PALETTE, text_len: 18 },
    palettes,
    style: {
      fill: Object.keys(FILL).map((id) => ({ id, value: FILL[id], label: FILL_LABEL[id] })),
      outline: Object.keys(OUTLINE).map((id) => ({ id, value: OUTLINE[id], label: OUTLINE_LABEL[id] })),
      colours: Object.keys(COLOURS).map((id) => ({ id, value: COLOURS[id], label: COLOURS_LABEL[id] })),
    },
    note: 'Palette slots are the s7 model’s top-colour hints (1–3 palette ids, or auto). Style tags are fill, outline and colour-count; 0/any is the training default and what live chat already uses.',
  };
}

function rgbHex(rgb) { return '#' + rgb.map((n) => n.toString(16).padStart(2, '0')).join(''); }

function isPid(n) { return Number.isInteger(n) && n >= 1 && n <= 4096; }
function padPalette(ids) {
  const out = [NO_PALETTE, NO_PALETTE, NO_PALETTE];
  for (let i = 0; i < Math.min(3, ids.length); i++) out[i] = ids[i];
  return out;
}

function parsePalette(raw) {
  if (raw == null || raw === '' || raw === 'auto' || raw === 'none' || raw === 'default') return { ids: [NO_PALETTE, NO_PALETTE, NO_PALETTE], key: 'auto' };
  if (typeof raw === 'string' && NAMED[raw]) return { ids: padPalette(NAMED[raw].ids), key: raw };
  if (Array.isArray(raw)) {
    if (raw.length > 3) throw bad('palette: at most 3 colour ids');
    const ids = raw.map((x) => {
      if (typeof x === 'string' && x.startsWith('#')) throw bad('palette: send named ids or integer palette indexes, not hex (map hex on the client via /api/chat/info swatches, or use a named palette)');
      const n = Number(x);
      if (n === NO_PALETTE) return NO_PALETTE;
      if (!isPid(n)) throw bad('palette: each id must be 1–4096 (or 4097 for empty)');
      return n;
    });
    return { ids: padPalette(ids), key: 'custom' };
  }
  if (raw && typeof raw === 'object' && Array.isArray(raw.ids)) return parsePalette(raw.ids);
  if (raw && typeof raw === 'object' && typeof raw.id === 'string') return parsePalette(raw.id);
  throw bad('palette: use "auto", a named palette, or an array of up to 3 palette ids');
}

function tag(map, raw, name) {
  if (raw == null || raw === '' || raw === 'auto' || raw === 'any') return 0;
  if (typeof raw === 'string' && raw in map) return map[raw];
  const n = Number(raw);
  const max = Math.max(...Object.values(map));
  if (Number.isInteger(n) && n >= 0 && n <= max) return n;
  throw bad(name + ': expected ' + Object.keys(map).join('|') + ' or 0–' + max);
}

function parseStyle(raw) {
  if (raw == null || raw === '' || raw === 'auto' || raw === 'any' || raw === 'default') return { tags: [0, 0, 0], key: 'any' };
  if (Array.isArray(raw)) {
    if (raw.length !== 3) throw bad('style: expected [fill, outline, colours]');
    return { tags: [tag(FILL, raw[0], 'fill'), tag(OUTLINE, raw[1], 'outline'), tag(COLOURS, raw[2], 'colours')], key: 'custom' };
  }
  if (typeof raw !== 'object') throw bad('style: object or [fill, outline, colours]');
  return {
    tags: [
      tag(FILL, raw.fill, 'fill'),
      tag(OUTLINE, raw.outline, 'outline'),
      tag(COLOURS, raw.colours ?? raw.colors ?? raw.colour, 'colours'),
    ],
    key: 'custom',
  };
}

function bad(msg) { const e = new Error(msg); e.status = 400; return e; }

export function parseCond(body) {
  // Accept top-level {palette, style} or {cond:{palette, style}} or {cond:{ids, tags}}.
  const src = (body && body.cond && typeof body.cond === 'object') ? body.cond : (body || {});
  const pal = parsePalette(src.palette ?? src.pal);
  const sty = parseStyle(src.style);
  const tensor = pal.ids.concat(sty.tags); // length 6
  const isDefault = pal.ids.every((x) => x === NO_PALETTE) && sty.tags.every((x) => x === 0);
  return {
    tensor, // [pal0, pal1, pal2, fill, outline, colours]
    palette: pal.ids,
    style: { fill: sty.tags[0], outline: sty.tags[1], colours: sty.tags[2] },
    palette_key: pal.key,
    style_key: sty.key,
    default: isDefault,
  };
}

export function isDefaultCond(c) { return !c || c.default; }

// Payload the worker understands. Omitted entirely when default so old workers stay identical.
export function workerCond(c) {
  if (!c || c.default) return undefined;
  return { palette: c.palette, style: [c.style.fill, c.style.outline, c.style.colours] };
}

export function publicCond(c) {
  if (!c) return { palette: 'auto', style: { fill: 'any', outline: 'any', colours: 'any' } };
  return {
    palette: c.palette_key, palette_ids: c.default ? null : c.palette,
    style: { fill: inv(FILL, c.style.fill), outline: inv(OUTLINE, c.style.outline), colours: inv(COLOURS, c.style.colours) },
  };
}
function inv(map, v) { return Object.keys(map).find((k) => map[k] === v) || 'any'; }
