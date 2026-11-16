// Zero-dependency PNG decode/encode (node:zlib only). Used for sprite downloads,
// 16→512 nearest upscales, a transparent vs flattened background, and OG images.
import zlib from 'node:zlib';

const SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return (c ^ 0xffffffff) >>> 0;
}
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32BE(n >>> 0); return b; }
function chunk(type, data) {
  const t = Buffer.from(type, 'ascii');
  const crc = u32(crc32(Buffer.concat([t, data])));
  return Buffer.concat([u32(data.length), t, data, crc]);
}

function paeth(a, b, c) {
  const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function unfilter(data, w, h, bpp) {
  const stride = w * bpp, out = Buffer.alloc(h * stride);
  let i = 0, o = 0;
  for (let y = 0; y < h; y++) {
    const ft = data[i++];
    for (let x = 0; x < stride; x++) {
      const raw = data[i++];
      const a = x >= bpp ? out[o + x - bpp] : 0;
      const b = y ? out[o + x - stride] : 0;
      const c = y && x >= bpp ? out[o + x - stride - bpp] : 0;
      let v;
      if (ft === 0) v = raw;
      else if (ft === 1) v = raw + a;
      else if (ft === 2) v = raw + b;
      else if (ft === 3) v = raw + ((a + b) >> 1);
      else if (ft === 4) v = raw + paeth(a, b, c);
      else throw new Error('unsupported png filter ' + ft);
      out[o + x] = v & 255;
    }
    o += stride;
  }
  if (i !== data.length) throw new Error('png idat length mismatch');
  return out;
}

export function decodePng(buf) {
  if (!Buffer.isBuffer(buf)) buf = Buffer.from(buf);
  if (buf.length < 8 || !buf.subarray(0, 8).equals(SIG)) throw new Error('not a png');
  let p = 8, w = 0, h = 0, depth = 0, ctype = 0, plte = null, trns = null;
  const idats = [];
  while (p + 8 <= buf.length) {
    const len = buf.readUInt32BE(p); p += 4;
    const type = buf.subarray(p, p + 4).toString('ascii'); p += 4;
    const data = buf.subarray(p, p + len); p += len;
    const crc = buf.readUInt32BE(p); p += 4;
    const got = crc32(Buffer.concat([Buffer.from(type, 'ascii'), data]));
    if (got !== crc) throw new Error('png crc ' + type);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      depth = data[8]; ctype = data[9];
      if (data[10] || data[11] || data[12]) throw new Error('unsupported png compression/filter/interlace');
      if (depth !== 8) throw new Error('only 8-bit png supported');
    } else if (type === 'PLTE') plte = data;
    else if (type === 'tRNS') trns = data;
    else if (type === 'IDAT') idats.push(data);
    else if (type === 'IEND') break;
  }
  if (!w || !h) throw new Error('missing ihdr');
  const infl = zlib.inflateSync(Buffer.concat(idats));
  let rgba = Buffer.alloc(w * h * 4);
  if (ctype === 6) {
    const raw = unfilter(infl, w, h, 4);
    raw.copy(rgba);
  } else if (ctype === 2) {
    const raw = unfilter(infl, w, h, 3);
    for (let i = 0, j = 0; i < raw.length; i += 3, j += 4) {
      rgba[j] = raw[i]; rgba[j + 1] = raw[i + 1]; rgba[j + 2] = raw[i + 2]; rgba[j + 3] = 255;
    }
    if (trns && trns.length >= 6) {
      const tr = trns.readUInt16BE(0), tg = trns.readUInt16BE(2), tb = trns.readUInt16BE(4);
      for (let j = 0; j < rgba.length; j += 4)
        if (rgba[j] === tr && rgba[j + 1] === tg && rgba[j + 2] === tb) rgba[j + 3] = 0;
    }
  } else if (ctype === 3) {
    if (!plte) throw new Error('indexed png missing plte');
    const raw = unfilter(infl, w, h, 1);
    for (let i = 0, j = 0; i < raw.length; i++, j += 4) {
      const idx = raw[i], o = idx * 3;
      rgba[j] = plte[o]; rgba[j + 1] = plte[o + 1]; rgba[j + 2] = plte[o + 2];
      rgba[j + 3] = (trns && idx < trns.length) ? trns[idx] : 255;
    }
  } else if (ctype === 0) {
    const raw = unfilter(infl, w, h, 1);
    const tk = (trns && trns.length >= 2) ? trns.readUInt16BE(0) : -1;
    for (let i = 0, j = 0; i < raw.length; i++, j += 4) {
      const g = raw[i]; rgba[j] = rgba[j + 1] = rgba[j + 2] = g; rgba[j + 3] = g === tk ? 0 : 255;
    }
  } else if (ctype === 4) {
    const raw = unfilter(infl, w, h, 2);
    for (let i = 0, j = 0; i < raw.length; i += 2, j += 4) {
      const g = raw[i]; rgba[j] = rgba[j + 1] = rgba[j + 2] = g; rgba[j + 3] = raw[i + 1];
    }
  } else throw new Error('unsupported png color type ' + ctype);
  return { width: w, height: h, rgba };
}

export function encodePng(rgba, width, height) {
  const stride = width * 4, raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6; // 8-bit RGBA
  const idat = zlib.deflateSync(raw, { level: 9 });
  return Buffer.concat([SIG, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}

export function fromDataUri(uri) {
  if (typeof uri !== 'string') throw new Error('expected data uri');
  const m = uri.match(/^data:image\/png;base64,([A-Za-z0-9+/]+=*)$/);
  if (!m) throw new Error('expected png data uri');
  return decodePng(Buffer.from(m[1], 'base64'));
}

export function scaleNearest(src, sw, sh, dw, dh) {
  const out = Buffer.alloc(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor(y * sh / dh));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor(x * sw / dw));
      src.copy(out, (y * dw + x) * 4, (sy * sw + sx) * 4, (sy * sw + sx) * 4 + 4);
    }
  }
  return out;
}

// Exact 16×16 recovery from a nearest-neighbour upscale (256, 512, …) or a native 16×16.
export function toNative16(rgba, w, h) {
  if (w === 16 && h === 16) return Buffer.from(rgba);
  if (w % 16 || h % 16 || w !== h) throw new Error('sprite is not a square multiple of 16');
  return scaleNearest(rgba, w, h, 16, 16);
}

export function compositeOn(rgba, w, h, rgb = [0, 0, 0]) {
  const out = Buffer.alloc(w * h * 4);
  const [cr, cg, cb] = rgb;
  for (let i = 0; i < rgba.length; i += 4) {
    const a = rgba[i + 3] / 255, ia = 1 - a;
    out[i] = Math.round(rgba[i] * a + cr * ia);
    out[i + 1] = Math.round(rgba[i + 1] * a + cg * ia);
    out[i + 2] = Math.round(rgba[i + 2] * a + cb * ia);
    out[i + 3] = 255;
  }
  return out;
}

export function parseBg(s) {
  if (!s || s === 'transparent' || s === 'none') return null;
  if (s === 'black' || s === 'opaque') return [0, 0, 0];
  if (s === 'surface' || s === 'dark') return [17, 17, 17];
  if (s === 'white') return [255, 255, 255];
  const m = String(s).match(/^#?([0-9a-fA-F]{6})$/);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function renderSpritePng(native16, size, bg) {
  let rgba = native16, w = 16, h = 16;
  const n = Number(size);
  if (n && n !== 16) {
    if (![32, 64, 128, 256, 512, 1024].includes(n)) throw new Error('size must be 16, 32, 64, 128, 256, 512 or 1024');
    rgba = scaleNearest(native16, 16, 16, n, n); w = h = n;
  }
  if (bg) rgba = compositeOn(rgba, w, h, bg);
  return encodePng(rgba, w, h);
}

// Tiny 5×7 bitmap font for OG captions (ASCII 32–126). Bits are rows, MSB left, 5 cols used.
const FONT = {
  32: [0, 0, 0, 0, 0, 0, 0],
  33: [4, 4, 4, 4, 0, 4, 0],
  39: [4, 4, 0, 0, 0, 0, 0],
  44: [0, 0, 0, 0, 4, 4, 8],
  45: [0, 0, 0, 14, 0, 0, 0],
  46: [0, 0, 0, 0, 0, 4, 0],
  48: [14, 17, 19, 21, 25, 17, 14],
  49: [4, 12, 4, 4, 4, 4, 14],
  50: [14, 17, 1, 2, 4, 8, 31],
  51: [14, 17, 1, 6, 1, 17, 14],
  52: [2, 6, 10, 18, 31, 2, 2],
  53: [31, 16, 30, 1, 1, 17, 14],
  54: [6, 8, 16, 30, 17, 17, 14],
  55: [31, 1, 2, 4, 4, 4, 4],
  56: [14, 17, 17, 14, 17, 17, 14],
  57: [14, 17, 17, 15, 1, 2, 12],
  58: [0, 4, 0, 0, 0, 4, 0],
  63: [14, 17, 1, 2, 4, 0, 4],
  65: [14, 17, 17, 31, 17, 17, 17],
  66: [30, 17, 17, 30, 17, 17, 30],
  67: [14, 17, 16, 16, 16, 17, 14],
  68: [30, 17, 17, 17, 17, 17, 30],
  69: [31, 16, 16, 30, 16, 16, 31],
  70: [31, 16, 16, 30, 16, 16, 16],
  71: [14, 17, 16, 19, 17, 17, 15],
  72: [17, 17, 17, 31, 17, 17, 17],
  73: [14, 4, 4, 4, 4, 4, 14],
  74: [1, 1, 1, 1, 17, 17, 14],
  75: [17, 18, 20, 24, 20, 18, 17],
  76: [16, 16, 16, 16, 16, 16, 31],
  77: [17, 27, 21, 21, 17, 17, 17],
  78: [17, 25, 21, 19, 17, 17, 17],
  79: [14, 17, 17, 17, 17, 17, 14],
  80: [30, 17, 17, 30, 16, 16, 16],
  81: [14, 17, 17, 17, 21, 18, 13],
  82: [30, 17, 17, 30, 20, 18, 17],
  83: [14, 17, 16, 14, 1, 17, 14],
  84: [31, 4, 4, 4, 4, 4, 4],
  85: [17, 17, 17, 17, 17, 17, 14],
  86: [17, 17, 17, 17, 17, 10, 4],
  87: [17, 17, 17, 21, 21, 21, 10],
  88: [17, 17, 10, 4, 10, 17, 17],
  89: [17, 17, 10, 4, 4, 4, 4],
  90: [31, 1, 2, 4, 8, 16, 31],
  95: [0, 0, 0, 0, 0, 0, 31],
};
for (let c = 97; c <= 122; c++) FONT[c] = FONT[c - 32];

function glyph(ch) { return FONT[ch.charCodeAt(0)] || FONT[63]; }

function drawText(rgba, w, h, text, x, y, scale, color) {
  const [r, g, b, a] = color;
  let cx = x;
  for (const ch of text) {
    const rows = glyph(ch);
    for (let row = 0; row < 7; row++) {
      for (let col = 0; col < 5; col++) {
        if (!(rows[row] & (16 >> col))) continue;
        for (let dy = 0; dy < scale; dy++) for (let dx = 0; dx < scale; dx++) {
          const px = cx + col * scale + dx, py = y + row * scale + dy;
          if (px < 0 || py < 0 || px >= w || py >= h) continue;
          const i = (py * w + px) * 4;
          rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = a;
        }
      }
    }
    cx += 6 * scale;
    if (cx > w - 6 * scale) break;
  }
}

export const OG_W = 1200, OG_H = 630;

export function renderOgPng(native16, prompt) {
  const W = OG_W, H = OG_H;
  const rgba = Buffer.alloc(W * H * 4, 0);
  // fill #000 already; sprite 512 centered, slightly above mid so caption fits
  const spr = scaleNearest(native16, 16, 16, 512, 512);
  const sx = Math.floor((W - 512) / 2), sy = 48;
  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      const i = (y * 512 + x) * 4, j = ((sy + y) * W + (sx + x)) * 4;
      const a = spr[i + 3] / 255, ia = 1 - a;
      // composite onto #111 so transparent pixels are visible on a dark card
      rgba[j] = Math.round(spr[i] * a + 17 * ia);
      rgba[j + 1] = Math.round(spr[i + 1] * a + 17 * ia);
      rgba[j + 2] = Math.round(spr[i + 2] * a + 17 * ia);
      rgba[j + 3] = 255;
    }
  }
  // fill the rest opaque black
  for (let i = 0; i < rgba.length; i += 4) if (rgba[i + 3] === 0) { rgba[i] = 0; rgba[i + 1] = 0; rgba[i + 2] = 0; rgba[i + 3] = 255; }
  drawText(rgba, W, H, 'PIXIE', 40, 24, 4, [182, 220, 60, 255]);
  const cap = String(prompt || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  const scale = 4;
  const tw = cap.length * 6 * scale;
  const tx = Math.max(40, Math.floor((W - tw) / 2));
  drawText(rgba, W, H, cap, tx, H - 24 - 7 * scale, scale, [255, 255, 255, 255]);
  return encodePng(rgba, W, H);
}

export function native16FromDataUri(uri) {
  const im = fromDataUri(uri);
  return toNative16(im.rgba, im.width, im.height);
}
