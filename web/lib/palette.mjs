// Zero-dependency 4097-colour discrete palette loader & Oklab nearest matcher (Node 22+).
import fs from 'node:fs';
import { decodePng } from './png.mjs';

function srgbToLinear(c) {
  const u = c / 255;
  return u <= 0.04045 ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4);
}

export function rgbToOklab(r, g, b) {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  ];
}

function buildFallbackEntries() {
  const entries = [null];
  for (let r = 0; r < 16; r++) {
    for (let g = 0; g < 16; g++) {
      for (let b = 0; b < 16; b++) {
        entries.push([r * 17, g * 17, b * 17]);
      }
    }
  }
  return entries;
}

export function loadPalette(paletteJsonPath) {
  let raw;
  if (paletteJsonPath && fs.existsSync(paletteJsonPath)) {
    raw = JSON.parse(fs.readFileSync(paletteJsonPath, 'utf8'));
  } else {
    raw = buildFallbackEntries();
  }
  const entries = new Array(4097);
  entries[0] = null;
  const exactMap = new Map();
  const oklab = new Float64Array(4097 * 3);

  for (let i = 1; i <= 4096; i++) {
    const e = raw[i];
    const rgb = Array.isArray(e) && e.length >= 3
      ? [e[0] & 255, e[1] & 255, e[2] & 255]
      : [0, 0, 0];
    entries[i] = rgb;
    const key = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
    if (!exactMap.has(key)) exactMap.set(key, i);
    const [L, a, b] = rgbToOklab(rgb[0], rgb[1], rgb[2]);
    oklab[i * 3] = L;
    oklab[i * 3 + 1] = a;
    oklab[i * 3 + 2] = b;
  }

  return { entries, exactMap, oklab };
}

export function nearestPaletteIndex(palette, r, g, b, a = 255) {
  if (a < 128) return 0;
  const key = ((r & 255) << 16) | ((g & 255) << 8) | (b & 255);
  const exact = palette.exactMap.get(key);
  if (exact !== undefined) return exact;

  const [L, oa, ob] = rgbToOklab(r, g, b);
  const lab = palette.oklab;
  let best = 1;
  let bestDist = Infinity;
  for (let i = 1; i <= 4096; i++) {
    const dL = lab[i * 3] - L;
    const da = lab[i * 3 + 1] - oa;
    const db = lab[i * 3 + 2] - ob;
    const d = dL * dL + da * da + db * db;
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

export function indicesToRgba(palette, indices) {
  const rgba = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const idx = indices[i] | 0;
    const o = i * 4;
    if (idx <= 0 || idx > 4096 || !palette.entries[idx]) {
      rgba[o] = 0; rgba[o + 1] = 0; rgba[o + 2] = 0; rgba[o + 3] = 0;
    } else {
      const rgb = palette.entries[idx];
      rgba[o] = rgb[0];
      rgba[o + 1] = rgb[1];
      rgba[o + 2] = rgb[2];
      rgba[o + 3] = 255;
    }
  }
  return rgba;
}

export function rgbaToIndices(palette, rgba, width = 16, height = 16) {
  const out = new Uint16Array(256);
  for (let y = 0; y < 16; y++) {
    const sy = height === 16 ? y : Math.min(height - 1, Math.floor((y * height) / 16));
    for (let x = 0; x < 16; x++) {
      const sx = width === 16 ? x : Math.min(width - 1, Math.floor((x * width) / 16));
      const o = (sy * width + sx) * 4;
      out[y * 16 + x] = nearestPaletteIndex(palette, rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]);
    }
  }
  return out;
}

export function pngToIndices(palette, pngInput) {
  let buf = pngInput;
  if (typeof pngInput === 'string') {
    const m = /^data:image\/png;base64,(.+)$/i.exec(pngInput.trim());
    buf = Buffer.from(m ? m[1] : pngInput, 'base64');
  }
  const { width, height, rgba } = decodePng(buf);
  return rgbaToIndices(palette, rgba, width, height);
}
