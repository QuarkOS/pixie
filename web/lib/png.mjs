// Zero-dependency PNG encoder, decoder, nearest-neighbour scaler, Inter text renderer, and grid/OG sheet renderers (Node 22+).
import zlib from 'node:zlib';

const PNG_SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const typeBuf = Buffer.from(type, 'ascii');
  const len = Buffer.allocUnsafe(4);
  len.writeUInt32BE(data.length, 0);
  const crcInput = Buffer.concat([typeBuf, data]);
  const crc = Buffer.allocUnsafe(4);
  crc.writeUInt32BE(crc32(crcInput), 0);
  return Buffer.concat([len, crcInput, crc]);
}

/**
 * Encode RGBA pixels (Uint8Array of length width * height * 4) to an RGBA PNG Buffer.
 */
export function encodePng(width, height, rgba) {
  const w = Number(width) | 0;
  const h = Number(height) | 0;
  if (w <= 0 || h <= 0) throw new Error('Invalid PNG dimensions');
  if (rgba.length !== w * h * 4) throw new Error('RGBA buffer length mismatch');

  const ihdr = Buffer.allocUnsafe(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const stride = w * 4;
  const raw = Buffer.allocUnsafe((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    const dst = y * (stride + 1);
    raw[dst] = 0; // filter None
    Buffer.from(rgba.buffer, rgba.byteOffset + y * stride, stride).copy(raw, dst + 1);
  }
  const idat = zlib.deflateSync(raw, { level: 6 });

  return Buffer.concat([
    PNG_SIG,
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function unfilterScanlines(inflated, srcOffset, w, h, rowBytes, bpp) {
  const scan = new Uint8Array(h * rowBytes);
  for (let y = 0; y < h; y++) {
    const lineOff = srcOffset + y * (rowBytes + 1);
    const f = inflated[lineOff];
    const rowOff = y * rowBytes;
    const prevOff = (y - 1) * rowBytes;
    for (let i = 0; i < rowBytes; i++) {
      const x = inflated[lineOff + 1 + i];
      const a = i >= bpp ? scan[rowOff + i - bpp] : 0;
      const b = y > 0 ? scan[prevOff + i] : 0;
      const c = (y > 0 && i >= bpp) ? scan[prevOff + i - bpp] : 0;
      let v = x;
      if (f === 1) v = (x + a) & 0xff;
      else if (f === 2) v = (x + b) & 0xff;
      else if (f === 3) v = (x + ((a + b) >> 1)) & 0xff;
      else if (f === 4) v = (x + paeth(a, b, c)) & 0xff;
      else if (f !== 0) throw new Error(`Unknown PNG filter=${f}`);
      scan[rowOff + i] = v;
    }
  }
  return scan;
}

function decodePassToRgba(scan, pw, ph, rowBytes, bitDepth, colorType, plte, trns, outRgba, fullW, x0, xStep, y0, yStep) {
  const trnsGray = (colorType === 0 && trns && trns.length >= 2) ? trns.readUInt16BE(0) : -1;
  const trnsR = (colorType === 2 && trns && trns.length >= 6) ? trns.readUInt16BE(0) : -1;
  const trnsG = (colorType === 2 && trns && trns.length >= 6) ? trns.readUInt16BE(2) : -1;
  const trnsB = (colorType === 2 && trns && trns.length >= 6) ? trns.readUInt16BE(4) : -1;
  const grayScale = bitDepth < 8 ? 255 / ((1 << bitDepth) - 1) : 1;

  for (let py = 0; py < ph; py++) {
    const dstY = y0 + py * yStep;
    const rowStart = py * rowBytes;
    for (let px = 0; px < pw; px++) {
      const dstX = x0 + px * xStep;
      const o = (dstY * fullW + dstX) * 4;

      if (colorType === 6) {
        const s = rowStart + px * (bitDepth === 16 ? 8 : 4);
        const step = bitDepth === 16 ? 2 : 1;
        outRgba[o] = scan[s];
        outRgba[o + 1] = scan[s + step];
        outRgba[o + 2] = scan[s + step * 2];
        outRgba[o + 3] = scan[s + step * 3];
      } else if (colorType === 2) {
        if (bitDepth === 16) {
          const s = rowStart + px * 6;
          const r16 = (scan[s] << 8) | scan[s + 1];
          const g16 = (scan[s + 2] << 8) | scan[s + 3];
          const b16 = (scan[s + 4] << 8) | scan[s + 5];
          outRgba[o] = scan[s];
          outRgba[o + 1] = scan[s + 2];
          outRgba[o + 2] = scan[s + 4];
          outRgba[o + 3] = (r16 === trnsR && g16 === trnsG && b16 === trnsB) ? 0 : 255;
        } else {
          const s = rowStart + px * 3;
          outRgba[o] = scan[s];
          outRgba[o + 1] = scan[s + 1];
          outRgba[o + 2] = scan[s + 2];
          outRgba[o + 3] = (scan[s] === trnsR && scan[s + 1] === trnsG && scan[s + 2] === trnsB) ? 0 : 255;
        }
      } else if (colorType === 3) {
        let idx = 0;
        if (bitDepth === 8) {
          idx = scan[rowStart + px];
        } else {
          const perByte = 8 / bitDepth;
          const b = scan[rowStart + Math.floor(px / perByte)];
          const shift = 8 - bitDepth * ((px % perByte) + 1);
          idx = (b >> shift) & ((1 << bitDepth) - 1);
        }
        outRgba[o] = plte ? plte[idx * 3] : idx;
        outRgba[o + 1] = plte ? plte[idx * 3 + 1] : idx;
        outRgba[o + 2] = plte ? plte[idx * 3 + 2] : idx;
        outRgba[o + 3] = (trns && idx < trns.length) ? trns[idx] : 255;
      } else if (colorType === 4) {
        const s = rowStart + px * (bitDepth === 16 ? 4 : 2);
        const v = scan[s];
        const a = scan[s + (bitDepth === 16 ? 2 : 1)];
        outRgba[o] = outRgba[o + 1] = outRgba[o + 2] = v;
        outRgba[o + 3] = a;
      } else if (colorType === 0) {
        let sample = 0;
        let v = 0;
        if (bitDepth === 16) {
          const s = rowStart + px * 2;
          sample = (scan[s] << 8) | scan[s + 1];
          v = scan[s];
        } else if (bitDepth === 8) {
          sample = scan[rowStart + px];
          v = sample;
        } else {
          const perByte = 8 / bitDepth;
          const b = scan[rowStart + Math.floor(px / perByte)];
          const shift = 8 - bitDepth * ((px % perByte) + 1);
          sample = (b >> shift) & ((1 << bitDepth) - 1);
          v = Math.round(sample * grayScale);
        }
        outRgba[o] = outRgba[o + 1] = outRgba[o + 2] = v;
        outRgba[o + 3] = (sample === trnsGray) ? 0 : 255;
      }
    }
  }
}

/**
 * Decode a PNG Buffer (color types 0, 2, 3, 4, 6; bit depths 1, 2, 4, 8, 16; non-interlaced or Adam7) to { width, height, rgba }.
 */
export function decodePng(buf) {
  if (!Buffer.isBuffer(buf)) buf = Buffer.from(buf);
  if (buf.length < 24 || !buf.subarray(0, 8).equals(PNG_SIG)) {
    throw new Error('Invalid PNG signature');
  }
  let pos = 8;
  let width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  let plte = null;
  let trns = null;
  const idatParts = [];

  while (pos + 12 <= buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const dataStart = pos + 8;
    const dataEnd = dataStart + len;
    if (dataEnd + 4 > buf.length) break;
    const data = buf.subarray(dataStart, dataEnd);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      interlace = data[12];
    } else if (type === 'PLTE') {
      plte = data;
    } else if (type === 'tRNS') {
      trns = data;
    } else if (type === 'IDAT') {
      idatParts.push(data);
    } else if (type === 'IEND') {
      break;
    }
    pos = dataEnd + 4;
  }

  if (!width || !height) throw new Error('Missing IHDR');
  if (![1, 2, 4, 8, 16].includes(bitDepth) || (interlace !== 0 && interlace !== 1)) {
    throw new Error(`Unsupported PNG format (depth=${bitDepth}, interlace=${interlace})`);
  }

  const chMap = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
  const channels = chMap[colorType];
  if (!channels) throw new Error(`Unsupported PNG colorType=${colorType}`);

  const inflated = zlib.inflateSync(Buffer.concat(idatParts));
  const bpp = Math.max(1, (channels * bitDepth) >> 3);
  const rgba = new Uint8Array(width * height * 4);

  if (interlace === 0) {
    const rowBytes = Math.ceil((width * channels * bitDepth) / 8);
    if (inflated.length < height * (rowBytes + 1)) {
      throw new Error('Truncated PNG IDAT payload');
    }
    const scan = unfilterScanlines(inflated, 0, width, height, rowBytes, bpp);
    decodePassToRgba(scan, width, height, rowBytes, bitDepth, colorType, plte, trns, rgba, width, 0, 1, 0, 1);
  } else {
    const x0s = [0, 4, 0, 2, 0, 1, 0];
    const xSs = [8, 8, 4, 4, 2, 2, 1];
    const y0s = [0, 0, 4, 0, 2, 0, 1];
    const ySs = [8, 8, 8, 4, 4, 2, 2];
    let offset = 0;
    for (let p = 0; p < 7; p++) {
      const pw = Math.ceil(Math.max(0, width - x0s[p]) / xSs[p]);
      const ph = Math.ceil(Math.max(0, height - y0s[p]) / ySs[p]);
      if (pw <= 0 || ph <= 0) continue;
      const rowBytes = Math.ceil((pw * channels * bitDepth) / 8);
      const passLen = ph * (rowBytes + 1);
      if (offset + passLen > inflated.length) {
        throw new Error('Truncated interlaced PNG IDAT payload');
      }
      const scan = unfilterScanlines(inflated, offset, pw, ph, rowBytes, bpp);
      decodePassToRgba(scan, pw, ph, rowBytes, bitDepth, colorType, plte, trns, rgba, width, x0s[p], xSs[p], y0s[p], ySs[p]);
      offset += passLen;
    }
  }

  return { width, height, rgba };
}

export function parseBgColor(bg) {
  if (!bg || bg === 'transparent') return null;
  if (bg === 'black') return [0, 0, 0, 255];
  if (bg === 'white') return [255, 255, 255, 255];
  if (bg === 'surface') return [17, 17, 17, 255];
  const m = /^#([0-9a-f]{6})$/i.exec(String(bg).trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 255];
}

/**
 * Nearest-neighbour upscale (with optional flat background composite) returning RGBA PNG Buffer.
 */
export function upscaleNearestPng(srcRgba, srcW, srcH, targetW, bgSpec = 'transparent') {
  const tw = Number(targetW) | 0;
  const th = Math.round((tw * srcH) / srcW);
  const bg = Array.isArray(bgSpec) ? bgSpec : parseBgColor(bgSpec);
  const out = new Uint8Array(tw * th * 4);

  for (let y = 0; y < th; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y * srcH) / th));
    for (let x = 0; x < tw; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x * srcW) / tw));
      const si = (sy * srcW + sx) * 4;
      const di = (y * tw + x) * 4;
      const a = srcRgba[si + 3];
      if (!bg) {
        out[di] = srcRgba[si];
        out[di + 1] = srcRgba[si + 1];
        out[di + 2] = srcRgba[si + 2];
        out[di + 3] = a;
      } else if (a === 255) {
        out[di] = srcRgba[si];
        out[di + 1] = srcRgba[si + 1];
        out[di + 2] = srcRgba[si + 2];
        out[di + 3] = 255;
      } else if (a === 0) {
        out[di] = bg[0];
        out[di + 1] = bg[1];
        out[di + 2] = bg[2];
        out[di + 3] = 255;
      } else {
        const inv = 255 - a;
        out[di] = Math.round((srcRgba[si] * a + bg[0] * inv) / 255);
        out[di + 1] = Math.round((srcRgba[si + 1] * a + bg[1] * inv) / 255);
        out[di + 2] = Math.round((srcRgba[si + 2] * a + bg[2] * inv) / 255);
        out[di + 3] = 255;
      }
    }
  }
  return encodePng(tw, th, out);
}

// Crisp 5x7 pixel bitmap font preserved for OG card renderer & public helper compatibility
export const FONT5X7 = {
  A: [0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
  B: [0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e],
  C: [0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e],
  D: [0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e],
  E: [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f],
  F: [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
  G: [0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f],
  H: [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
  I: [0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e],
  J: [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0c],
  K: [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
  L: [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f],
  M: [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
  N: [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
  O: [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
  P: [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
  Q: [0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d],
  R: [0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11],
  S: [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
  T: [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
  U: [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
  V: [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04],
  W: [0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11],
  X: [0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11],
  Y: [0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04],
  Z: [0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f],
  a: [0x00, 0x00, 0x0e, 0x01, 0x0f, 0x11, 0x0f],
  b: [0x10, 0x10, 0x1e, 0x11, 0x11, 0x11, 0x1e],
  c: [0x00, 0x00, 0x0e, 0x11, 0x10, 0x11, 0x0e],
  d: [0x01, 0x01, 0x0f, 0x11, 0x11, 0x11, 0x0f],
  e: [0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0e],
  f: [0x06, 0x09, 0x08, 0x1e, 0x08, 0x08, 0x08],
  g: [0x00, 0x0f, 0x11, 0x0f, 0x01, 0x11, 0x0e],
  h: [0x10, 0x10, 0x1e, 0x11, 0x11, 0x11, 0x11],
  i: [0x04, 0x00, 0x0c, 0x04, 0x04, 0x04, 0x0e],
  j: [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0c],
  k: [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
  l: [0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e],
  m: [0x00, 0x00, 0x1a, 0x15, 0x15, 0x11, 0x11],
  n: [0x00, 0x00, 0x1e, 0x11, 0x11, 0x11, 0x11],
  o: [0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e],
  p: [0x00, 0x1e, 0x11, 0x1e, 0x10, 0x10, 0x10],
  q: [0x00, 0x0f, 0x11, 0x0f, 0x01, 0x01, 0x01],
  r: [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
  s: [0x00, 0x00, 0x0f, 0x10, 0x0e, 0x01, 0x1e],
  t: [0x08, 0x08, 0x1e, 0x08, 0x08, 0x09, 0x06],
  u: [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d],
  v: [0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04],
  w: [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a],
  x: [0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11],
  y: [0x00, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e],
  z: [0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f],
  '0': [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
  '1': [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
  '2': [0x0e, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1f],
  '3': [0x1f, 0x02, 0x04, 0x06, 0x01, 0x11, 0x0e],
  '4': [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
  '5': [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
  '6': [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
  '7': [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
  '8': [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
  '9': [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
  ' ': [0, 0, 0, 0, 0, 0, 0],
  '-': [0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00],
  '_': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f],
  '+': [0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00],
  ':': [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
  ';': [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x08],
  '.': [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c],
  ',': [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08],
  '!': [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04],
  '?': [0x0e, 0x11, 0x01, 0x06, 0x04, 0x00, 0x04],
  "'": [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00],
  '"': [0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00],
  '/': [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
  '\\': [0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01],
  '|': [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
  '(': [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
  ')': [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
  '[': [0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e],
  ']': [0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e],
  '&': [0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0d],
  '#': [0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a],
  '%': [0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13],
  '=': [0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00],
  '*': [0x00, 0x15, 0x0e, 0x1f, 0x0e, 0x15, 0x00],
};

export function measureBitmapText(text, scale = 1) {
  const s = String(text ?? '');
  if (!s.length) return 0;
  return s.length * 6 * scale - scale;
}

export function drawBitmapText(rgba, imgW, imgH, text, x0, y0, scale, color, opts = {}) {
  const raw = String(text || '');
  const str = opts.uppercase ? raw.toUpperCase() : raw;
  const maxWidth = opts.maxWidth || imgW;
  let cx = x0 | 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (cx + 5 * scale > x0 + maxWidth) break;
    const glyph = FONT5X7[ch] || FONT5X7[ch.toUpperCase()] || FONT5X7[' '];
    for (let row = 0; row < 7; row++) {
      const bits = glyph[row];
      for (let col = 0; col < 5; col++) {
        if (!((bits >> (4 - col)) & 1)) continue;
        for (let dy = 0; dy < scale; dy++) {
          const y = y0 + row * scale + dy;
          if (y < 0 || y >= imgH) continue;
          for (let dx = 0; dx < scale; dx++) {
            const x = cx + col * scale + dx;
            if (x < 0 || x >= imgW) continue;
            const o = (y * imgW + x) * 4;
            rgba[o] = color[0];
            rgba[o + 1] = color[1];
            rgba[o + 2] = color[2];
            rgba[o + 3] = 255;
          }
        }
      }
    }
    cx += 6 * scale;
  }
}

export function wrapBitmapText(text, maxWidthPx, scale = 1, maxLines = 3) {
  const words = String(text ?? '').trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [''];
  const maxChars = Math.max(1, Math.floor((maxWidthPx + scale) / (6 * scale)));
  const lines = [];
  let cur = '';
  for (const w of words) {
    if (!cur) {
      if (w.length <= maxChars) {
        cur = w;
      } else {
        let rest = w;
        while (rest.length > maxChars && lines.length < maxLines - 1) {
          lines.push(rest.slice(0, maxChars));
          rest = rest.slice(maxChars);
        }
        cur = rest;
      }
    } else if ((cur.length + 1 + w.length) <= maxChars) {
      cur += ' ' + w;
    } else {
      lines.push(cur);
      if (lines.length >= maxLines) {
        cur = '';
        break;
      }
      if (w.length <= maxChars) {
        cur = w;
      } else {
        let rest = w;
        while (rest.length > maxChars && lines.length < maxLines - 1) {
          lines.push(rest.slice(0, maxChars));
          rest = rest.slice(maxChars);
        }
        cur = rest;
      }
    }
    if (lines.length >= maxLines) break;
  }
  if (cur && lines.length < maxLines) lines.push(cur);
  if (lines.length > maxLines) lines.length = maxLines;
  const last = lines[lines.length - 1];
  if (last && last.length > maxChars) {
    lines[lines.length - 1] = maxChars > 2 ? last.slice(0, maxChars - 1) + '.' : last.slice(0, maxChars);
  }
  return lines;
}

// Embedded gzipped Inter glyph atlas (built directly from workspace fonts/inter-latin.woff2 + inter-latin-ext.woff2)
const INTER_ATLAS_GZ_B64 = 'H4sIABc49moC/+zd248lR54f9n42MP+BXkTPsldkVTEuv7iRwx0WyWUPu3umushmXzg1bLGb1ewZsi/sJptd5EyvsYYEG5ZhQNgXQWvAWgEC/GLYehAg7L74srZheQHNSgsbXgEy5DW0a1m70t7GgxmznBG/7zczMs85bHI4I8CAOBxmVv4qMyMiIyPjZMXnxMeP3Xv/6N3De489/fFjV22ui3vf/ujwsadt3nrsw8Nvv33j/ceeTsZsPfbmvWuHt97XwFuH+CFsPfbut28dfg2/6PzWY2+/e3TnRjueDIf7pi2yZba2rWxZt2XDlvnW1mNSasBpwG+lreE/NtdIMDWSLXexW8N/XbI1ZFsozo4mLtSQaycKOF4N+a1oWsjXUCoMtQOm2A4o7YBpdsBi216hhmLq9hr+a0qLxTW7WedzjaV1MbE15msiYx5C+j8bSttcE1iGImhnkpo8Rlr6QkEob8UtG4PUSMBptjWEf5NxNajp0xyHlobh6LmVoa/pc8W3Qw4nijV9zqSaL7+4Wr7FfCsPXy9YyWMi45YLtl3Jdr1MwQFjPZcLviZSLEKmC0Ufa6hdr2jHnOXhGrcCFI8rWSNDGbV/XW4Zk1oepV1/txVqGopub6Xh29VPw6H8EPAtEHWH4Vg10UNFKqGdvl2loWC2NdWlFoQrqe4T8rjPcOZWA721da/QykCzY7bqZR0isRZCNF2ibUv0sJfzLWanWKmhtOW9tIib79X+9dKKNXoUEK5FrS1bPvh6DaPgGqJu5lqwPuYWq0Xh7HgjtMvoS0411upFsbObRGwsNdbqhfWzay++3Vyxlogzmfu1fyW4ls52I6cwu08k5nrM1EollnkstyuTbDtfnJ+vtHqY3DwP7d8wVO4aq+WSmYXQIt63s7WbJY6pNDXzQUytiamViknzIwZpsVYq0Xf7DbFkfY21UpmOGVssS4u1UnFmdsyhyalXIZUWc8xdvaO3omsllk3XQoxNlbS6klupOJnvF3Ktfdl1d/W4Xyot5rsYW51YQi3p3MqluNl+Q/ZaTGuL7fIwtI0u1xsxxzU1KUm7CllbkTEtpcWini93NWncL5d6jXJZOd9w85tWc4tZUzuzC/V8xa42MllMzXpxa27lHEzNQvGrDVDWVJZWKsKrV4ZgTq1pLdqYtKZVG5+tnPVwraaYgFQM9+MQcS2SuqeFwf2ftZ6U1qpmN3v+5BJarHR3iEELUKypd6s1Zs2OxbUTWtNKJLn5nj67FtQyCV2ZDP8Jxreg7w5reNjoSgvKysNyKJyUdc9aMimjNFv7V4qLLRQ1tD1eoDzEisbS2kepbXemNXk6KG7oIf2iJdCqC25N026/Wrjetlxa06VVM1KjkjSqBZRmBTRkQUvIujWFMKRUi8j6tdEcWqpsKyPLI2e93MbaFgxdu82aMESLRlmDtvnca10cZzSaVjJU//VWE9XqUfLz7Fppj1prW1FJmBeV1SZuOMG8fmLfpJfA2ZUj11Rlq1F9VIcuR/V3Smz10Pnp0VvrdetqOINgKymjXZtcH7/1vnY4a1i3p9M64+L0aMw1RbUbI63Nsu1h23ejnLTHm21PWnSkDDpSY8yteRYNaWi3qNWnbZZZp3NIRGt4rT5vre7r2NdyuT3EbdQrLvMje2PaDdWeuV2/qT1CbNCMpG5PfcbXjrCzetaM2uBQPvWoHinijYFeXGj/outg9bnrZSzZ2jxHH1pMeyOGV9O1YNIWJ3UdkrSlfRyfre5Yi8hMhT6kRLdL12zro8d20VY4Lo4Nd81EKXq2do29G5M5nFIMzpbWdKiGaOvD2vbo5Z5RMzh0sTXWmlQfZzHXWlurj95ZA27Rx7HZdp87WsG0B7D21Kw+fJObdRKs6GPGtsevdhlz62ZaiV4j0p1xOGFtNLUHb9uztzCdsaVzeJy3UCsZYZmWFkte85DWXEDBvarPXhlrVNBMDA+NFm21JozRqB0M7QrY9vx1kfdjrYo16luH2eoTeNau1Wgo7byl77HpjsN/Q9K2qT2GV6Pa3bZF1kWj07pcwtqoSCveEtdGcd+W1EW1qR3+G4vTVLWyKtL1ousnqKFatmhZ6Q/XFkrvMKeP5jSe17bzpthaW4dn8zKaW+fd6cN5Gc2mfUR0+nReibr2Uda1x3O2+vhI2qEZal9xLRimrvEYk6JJitjRtWDRYGiNrWsP6OmoGR8XpbRgRuPHT5r6iM45a4pK1wvWj/HDf4tpn+Vde0SzH5x4BYpvzztn7dpoaDera4/o1WjWkmiP6GV0uC7tEe3aI3o16lub7PQZvfyIaULUVMU1ffehABFNXQ8Wry2GqH64djaviw6Pf01zWRvVjy+uPaNXo8NN1KJ2JTp8mB6qTcuvPqOnT4TtGTQcsfUVnVv9dDA8ioestDTrQzqm2SfNYR+kKqyNStZoXBuN2DetjSbsm9dGC/YtXTRsaZdm+ECt+3ptq1zX27FD1GmV9XalV127HvqmyPm+KddnfX3LEZLu69dGk97Z+kJmJVq0XulLmWV0eGjpvitd5CFR4jS77bXM2LVGzLt26T16yNtu6swPbbxe+fZeZgpmDYb2rHD6bib5RYKitILQ1zOpzN+0DY+SduvqG5qUFvuW2DKjb2mW0WDbpwgnsjbqtC3StzUrUQm6b1wbjU73XdcvcAFVSl83Yl/tPbshqp89nb5znOV3OHI02iwHszZqtYkMdm3Ua6Ogbx9XosHpvn7lKgw3doxF95WVT3n1LVnWHOlryOW+Q3dN3wjoh5zheb+tPTZbX30l1ypcRguaM6JDI12jvlWcrO1c5rvF9jYt+YhYWhMrGtPPEiEiWFpQHIItQcHPg61nlnHX6mHby5T6Yje0fmlGU+KTdqJS2zHoS4NQmBHTPg3UsBsetlaD7UmAV4naXtfjJs1K0O7FYt8SvQZDt6/nvllbmeEDuPF4S7Xdd7Td8KE/fOt7W4+9c3j3Vn3NXZvQbdEXnrXitTdfdUPtNG8HfTdVm+32sqkFhgKO+g6p/lh7KtvDz3/xicee9n7rsYPHHq87mLq2NawNHehhbWdca4fLdW23bmt7nK6HSnXtmh7s4LG3uHLIlbe5cpsr7+nKl3HmL+9qir58SZc/h+2PHzyGFD1+kisMna8bhmM8fgHJevwiyuHxyzV5deXggLH73P1Drhxh5SQzfZJ5Psksn2SOTzLDJ5nfk8juSeT2JDJ7Enk9iayeRE5//mr7o8XP39DFdx57esjmz7+jP72ri5u6uKWLO7q42xZP/EXN9RNfxvLnsHwcy5p2X4aVHa4YRCyWDkuPpWAZsIxYJiwzlgXLr2L5nFauJ3b1zy5PPI/tL2D5Ipa/iOVLWJ7C8mtYvozlaSzPYHkWy69j+Q0s97A8h+U+lq9g+SqW55Gu1/DzUD2GbtVjT1zEz0MVG9qkx564jOXr2H6lXuK64bu64UmnV+7JepcMdeDJXdSTJ2sdsPU3ahWsx37yArfUKjjUjidrPbY1chk16EmtivV3Huhxt1rddkNs6yRXjJ5qy6ICbnk93lZEIOHnjJ+L3i9bX607DFdl6zmcZesFrpziyh5X9rlyHjnaaqkcKs028ryNPG//Qr3Yw3G3mfltZn6bmd9m5reR+W1mfpuZ3x4zv43M74yZ32Hmd5D5HWZ+B5nfQeZ3kPkdZH4Hmd9h5neY+R1mfoeZ32Hmd5j5HWZ+h5l/agvtx1M7XNnVMz51GhkxaHzMlibB7GCJltRc0B3MJRzBXMZZTCuCGrqCBBj9M6bFIe0V/fH91jI4bPVcbukJ/A6Wl3QpiMsWziM7XHkDKwG/ErY0AWFHlxHbY90+ZD/uYMkEpn8XLWx6nFuwS2qtTY3scOVpnC09w5Vadr4+lU7jAqd6YKm//CaPVxtRW59Ub3HlkCtvc+UmV25x5TZX7nDlPa7c5co9XICMJGcUYEYBZhRgQbzgihZc0YIrWXglC69k4ZUsLKiiV/Lp8WnzDOvWM3w2PTPGvrKNW+rZX2hHHtLw7C5S/ex5HPHZC2iMnr2I3Z69xN+5jOJ8th2y/dJ9pPHZBzjvs0fY8gv6S8PaV5HRr+7iiM9t4WTP7XDlKez1nKZ2uIWfu4JNu3rDDr+1+2W97XZPcsMT2uLuPomz725zpfVM6u8+i9TvfhVn391t9Xz3BZxo9xRX9riyz5VXccLzaBh2X9Nrt3uBKbiITO5e5hbNd12r94EbLtfuNb1su29heYjl21jexvI9LN/XmrB7H8W8+yFXjrjyEBl9HoX7PDslL2DDC7t6kBfQsXmB5fkifuFFXoUXeRVe5FV4cRcF8OJ5vTNfvMANaGlfZKV48TIj45V7kWf6RZzpJS7ZzL3EpuKlXRTsS6eRsZeuMPSmPu1fuqb5eOktLA+xfBvLm3pJXrqF5W1sv4Of38PPd/HzBzwBO2YvsT/20kdYOYUEn9rV7J9CMZy6oMc4hTI9dVmPfarlfQh8DTt+TZvUr2mT+jK2vqxbX9atp7H1NBvQ0yyV00/hWp7e1cOevoLSPoMKf+ZJPALPbHPlWex8hvX6DOv1GdbrM6zXZ67hQpx5iyuHXHmbK7e58h5XWHJneNef+RD31Bne9WceIj1n9a4dEn/2JFeQ+LNP4rF3dpsrFjXw7AtaoGdPYbmH5T6WvBHPXuB52PU+exnP4rN6A9ZYvQF93e19LcezvKHO8j46+xBJ+Dquxtf1Gn1dr9E3sPUbuvUbunUPW/d4B+3xDtrjHbTHO2gPVWePd9Ae7qA93kF7vIP2xjtoj3fQOX5COIdTnntSc3KOt9K5bWxg5TnXbqkhlef49Dv3ptbWcw/1V/dxrH0mf5/J32fy95n8fSR/n8nfR/L3mfx9Jn9/TP4+k/8KzvQK7ptXcL+8wvvlVfzCq/gU9ipurPPM93nUmfNP4nTnebec3+YW5vz8U9xiuEVwmc9HbklonM8/jTSeZ7fhPO+g82yUzr+gmT9/CsvTDOxhwz6WV3DDnn8Tj4Pz11AHz7/FlUOuvM2Vm9zrFlduM3SHW97jlrvcco+nqJXa1bN/wC33cTue5315/gESeMQNH+FSn3+IwnoNl+A1FuxrLM/X2Ay9hmboNTZDFx7nyhP6EefCk/jVC1uodBe2uWWHW57CI/mC0Sp0QXCiCxEbnkYqLzzDlWe5wo/BF9i8XWDzdoH1/ALbuQts5y5caPfrhVrd6vIKD/ImXk9cuMbjv8WVQ668zZXbXHmPK/dQiBceIpMXWSAXUZoXn0QSLrJALm5zCwvk4lN6cS6yHC4+zaM8wxXm/+IurvHFF7SsLp7Ckh8QLu5hwz6WF9sbl4vM6sVrKP6Lb3HlkCtvc+UmU3KLK7cZusMt73HLXW65x5WHyOIldNQuPaGfdC+xNC6xEC4Z3H6XvNatS8INkSsZEfbgLr3AlVNc2ePKPlfQY7t0nkd5E5foEvoRl9CPuIR+xCX0Iy6h33AJ/YVLzMzlx1HCl1HZLz/JDVt4eFze5pYdbmHLc5l1+jLr9OVncf0vs05fZp2+zDp9+TSK9DLr9GXW6ctv4ra/XDPk65a3uHLIlbe58m0tw8s3URyXb3HlNn/lDre8xy13ueUeM/EBtzxEJl5HPX+dPe/X2fN+/QWunOLKHlf2ucLe6zf13dQ39d3UN/Xd1Df13dQ39d3UN/Xd1Df13dQ39d3UN/Xd1MHB+Nng4AB17uDg5LjpCXTADw6YzIOD7XGNHxEODr6C7B0csK4dHPDjwsHBcyj5gwP95HBw8MK45dS4tjeu7Y9rrzJJ58cT4APEwcGF8ewXcYceHFweU67P0hZ+Y8wEPkYcHLzFlUOuvM2V21x5jyv4MHFwcH/MGz9OHBwcjdt4Sd7go/cNvth7gy/23hC0e2/gTdwbu0jbGy9gwyks2Sa/sYcN+1jyXdeVJ7hicNArFh/br3huEdy7VwK3RK5krhSuPIcyvvL8Y0/X9F95gRtexIZfxPIlLE/xF76GDS9jeQbLs1h+HctvYLnHHc9hwz43vIIN/DR25TVG+Inxyvj58MobLIKr2OsGlt9u9ezKcDvUMrzyDja/i+VNLG9heQfLu1je5wmOuPJdjbzZbpchWW+exJIv8N88j7S8eQE14s2LXOHH/je1UtZN9/Vqvnmky6s87lUc9+pf0pp/1aIyXeVbvKt8eXeVLyiuXkRVv3pJa+rVesLauFwdX11c/RaOhxNf/RA/46XeVSbkI/z8PY1fY8KuIWHXkLBrTNg1JuwaE3aNCbvGhF1Dwq4xYdfGhF1Dwq4hYdeQsGtI2DUk7BoSdg0Je+svtQbsrW/pQj9evKUfL976XvvpkGk/RNoPkfZDpv2QaT9k2g+Z9kOm/RBpP2TaD8e0HyLth0j7IdJ+iLQfIu2HSPsh0n4dn0Gub6GSX8dnkOs73PAUfpN/xbjOlF5nd+X6FfzKm9gXf9W4jr9qXMdfNa5fR2t6HX/euI4/b1zHnzeu39NUXn8fVfn6h63jcx2fdt4+j0p7g+V5A+V5g5X/Biv/DVb+G6z8N1j5b4yV/waK6waK550n0X6/s80VtpPvfAWfxd5he/AOsvkOsvkOsvkOcvcOcvcOcvfOQxzzXa0w72qFeVcrzLtaYd7VCnOTGbyJDN5kBm8ygzeZwZvM4E1m8OaYwZvI4E1k8BaPewvHvcXj3uJxb/G4t3jcWzzurfG4t3DcWzjubR73No57GxX8Niv4bVab26zgt1nBb7OC30YFv80Kfnus4LdRwW/jxLdRwW+jgt9mQlDBb6OC32HC7iBhd5CwO0zYHSbsDhN2hwm7w4TdQcLuMGF3xoTdQcLuIGF3kLA7SNgdJOwOEnYHCXuPdfku+zN3t3DIu+zX3N3hFny8uMvezd1drjzf6svdF3Xxi7p4SRdf08XLujiNenv3jP58Vhdf18U3dHFOF6/oglX97utMEd4/3NXO3l10Y+6iF3MXnZi719t9exddmbvaJ7yrfcK72ie8q33Cu+jl3EUn5+77uut9jT7QxZEuvqsL9m/unccnqXv869G9iwzxr7f3xr8avf+kXon3t7H8il7U9/me/f3rWsrv329JeF8bn/ePdPFQ9/qAl+sDXqUPnuIKr84HvLYfXEYBfsBXoR8w7fefxC/f5+uf+9tc4V+O7vNT+/2vcK9nGdrF7XqfrfB9ZuM+/8Zwn5+R71/TtN9/C8tDLK+3C3L/bfx4G8v3sNS/SNx/iHN+yO7kh1so3A93uMLkfMha9iHL4EPWnQ+vad358C0sD7F8G8vbWL6nywd4Nj3AI+kBX+o8wIusB2iAH6ABfoAG+IFm6wHa4Qdohx+gHX6gLe0DPFSOeCGOeCGOeCGOeCGOeCGOeCGOeCGOmPMjXogjXogjXogjXogjXIgjXIgjXIgjTfERLsQRLsQRLsSRXogjXoiPeIaPUAIfoQQ+Qgl8hKx/hKx/hKx/rLftx3o/fqz348d6P36s9+PH+hntY/2M9rF+RvtYP6N9F58Uv6uPrO9qQT7En3sf4s+9D9mgPuRfeR+yZX3I+/QhXj0+5F95H/KvvA/H+/Vha0G/973a9Qwd7QybaGfoaKfMaWf9HL+knRiCpyOx3JbLQIFtqJXxCOrwvBz9BDyDRawg1sZZ6TCrEBYHTWUynrMz1j3hNdsgKynTUV3TgWFkniHOj1psGySnI6wAgZicogPGFXqu7Ojb8H2VnitB5SdtkGFI43B5V0IbpOs7vqaoYQoJBrXr0LY0RF1RrKQDCwOGIVEKDtFmG5R7htSNTa+7qllS7ynCAXOipywhj+Az2tl4OW9sG7fWRhZmix3rKDA/FGcm+cQIdKeDFYdQG6faxhSOISWSxrdBuTqgECrBtuH+3ui4MR1NqJyhtOHuQ6SZozaSMBsVn23EvgltBKIOIrStMsR6qBrJQvOpu9RE1wQoBG5jB4vXQnRNG9SYVpE87pSHnWoCUht92cynjs6rgVoSJjXdouZTcRITnds+Kj41O6mNgfOmNPQChTLLqF4KFShd6dQrMdxBZfSewpgOQvPWtpFzyk8SK776C2+9utQIQNzfMr6CAJJPi0Gn49W3sZHa2Guj0aZaHdGv6FNif894m0MY0afYeaw086D4JJv5GZ1pN1ty84zoGZ1t48ybPokc2CnktXF0n/0NXI/pmxtMYV0unLQihz+Z58LpOGCFn24scpwxpkz5afPyqICtrWxy6K5HTY96v2xW2oxhz6EOJOJP8vZxz6F40qg/Q17s6Rs3Vf4Z0qwp8l7a6FQlKNEs9gw6KFMLKM6viNexqcpQllXLayVXAjo2uDoE2/vSKkFeHYjbZHMbapt7MjwFnbpMs67Gim+tS0HrYrvWRbShaENElze3KCVoI0QXjZLEVnKlt3z1thturKEFt6MDtaIOtDKsIdJKtKkTHVHajudrpFUbdaC+sE0Ykj+E2rBnZaDjYw/XSHJrs9SB6k3D/YIBkFx5XroaVHij1ET8bEdnJgVaIotEm5TgipsUaHDjjr7q8pwnBNo/S4dLEEJrRBWBQgnVJrkdNCKpUWPb4wWqZl2fYWCgfnHU5NKoQGOY3d8hw5aWid+pZK2x5pBhQKV0jXAAO8XY6NQXTTQpTfwzoLxxLaKNadKfYboYNebyZD9L1lBt2330rXMC+KkN+Hg6bWiUfWIcdnu0t0MGNUyqPmVWZ2JUk6XmE+O7W4nVMyblXMpJzBhsBRNzq7wQn74vmFhU1Tq7PGi9+EmxCLinR7FVPueTShFYT5go3mPJKaFsjARDxutT19YdvdiJei53FJsm6qnP0aZWaywoA2zP2K4bNQRCGKlnxiVq3agp5NY8gIYbN0zQM/h585KS8rkoXcNOgOtTVjAWOxje71wkjNKz6y/VQs0GaUrLR43UoN7D+qiVssWmq+6o3xUA6GnH5rC5+CGq5arPWqCgumceYh660k4PjIyvnciiDjJ13ZBKJGtigvNrmOewPXTMs2+XSxcMUI7s7tc0qjKB8rR8/NSB/RWHpkl5zrpDwweBMhFPa8YviKnJT9lMxHP8hgGNZQXs7Qk7b5tzTnkSnoubPufiJ+A59rG01hSj9Lo9X7XbkrRLWawa9PZwzUqOm/wcInrrtidrNqNrTzWkd7w+Vy0TWb/8ZIjljnfOL1sBh1fdOX6qUi3ii9f7CN+t4BZRiWbSnY5VRT8H+BIVvKvu7BqvGkt615e+Xxbw3RL1G0XKaDuXUTEmlNF2rkYVPantXI2KLaPtXI2qCVHbuRpNSFXuNLz2mIfLM3S1y2g75x3f4dINV1Qm24nObf0oUA9sXcoT7VwGpVUTyM5lUJt5wM5lMGY/us6IliXog7OC5jy6zlHCMVb86Drrjq4FswbV5qvrnI6a9OuSTM6j66zuiK1Z693J0Nb40XWyKQwgTeIUS6vrZCdXSz8OUSlmdJ2r0di+esHhqxeW0axKDl+9sIgOPf04us7VqH4shevkJ7LWRMnQBzSj6uSejt9944OYUXWyG6y5rcfVlxIO37ywjObkRtW5Eh2qcR5V52rUaV1zvRQeS3nY2UyqM9rZw0okNFoP1bn4RCASU5hUp+d3FrQOnkhStIuvXlgE9buUCLHmwWBsRzqXQdu+0wmicxnUVykAnb6MX7hS77ngWzcOMszmsYtXuwFBtPiVc0rfHxvKQE+pmFP8/JTUmn5dMKvZ09cty2DR21xfuCyCUb/9SyHnrJc7JCi69kBQyDl2jxnTmuLRyW03KzrkEr13I+Scjpo0KCp4FXLi6TSmJ+gp1XFKV+x1zwiK6aZe8LRn8mlSnMtgDjIhzmWwKBrW1y+LYELTooRzGQRhl5VnviTv3MQ3Zz12SWL8hDdlXreSaGMVzLpgVMga7Lqgfu6A3FwGs9YfvFMUXhPcvKnETm7OPtENVTpbLVu8V1zunF1KHd10psXrtz/Ur8bS/hnlpjBYv96rRlv3B3Jz/PxcHwmS9ZvBIDeXsRA7uelN92F3CAJnalPk8iwYXS83BS/VWv9+CEbbyU0r2j1qHaSsH6EpN3XH9lVzLajpgdy0CR1f/Xf4ROxtJzcX++o3hFFuYl++LBmasNbaU27WV00sfv3SsjJ0STbKzbKAm0K4KXO4GQA2g4LNNhipgc08ek03ek03es1Ar1lfpKvX9IVeM9BrBnrNQK8Z6DUDvWZQr1lPTK/5c/hZnWYG08yqNFvgPHLyOP823ZRmAdJsNHP803RDmhlGM4NoZgjNTKDpCDQdgWYA0Kw5PMkMnkT+TiJ7J5G7k8jcSeTtJLJWgaYo0ITMFJWZojJTVGaKykxRmSkqM2tOq8xsy5/D8nEst/CHZ5WZXmVmi1gsHZYeS8EyYBmxTFhmLAuWX8XyOb0oVWbWoqgys21/AcsXsfxFLF/C8hSWX8PyZSxPY3kGy7NYfh3Lb2C5h+U5LPexfAXLV7E8j3S9hp8v6Jd6VpnZfr40/JxUZrbl69jOcd1VZtYNTWYGyMwEmZkJMy1gZlOXF7jlIjDnJSLOy6g5gJkWMDOMMFMIMwUwMxFmOsDMAJiZADMjYGYiyLQEmZYg0xJkWoJMS5BpATIzPWaAxwzwmAkesxK6bWR5m1neZpa3meVtZHmbWd4ejzpmeRtZ3hmzvMMs7yDLO8zyDrK8gyzvIMs7yPIOs7zDLO8wyzvM8g6zvMMs7yDLO0zcU+CIT4EjPoVh72owgxrMWgsMPKOBZzRoHA2YQzOYBQQzU2BGCkyrAlNUYNYDVoEpM4FZt3out/TwTWAGCMygArPGBYBQAAibv8zKL2s8YP+A/SO2R2QjkmUydcovhfzSKr+suyi/NOSXBvwyQ19m4MtqJBRfeuDLqnAUX1riS0d86YgvHfGlI750xJeO+NIRXzriS0d86YAvi9rLmt6MrGdkPaPoCuIFRVBQBAVXsOAKFlzBwitYWEZFr9XT4yPkGVSnZ/i4eWYMNXhZ750GL0uiu3R0lxbuspFMjqB6ln/mfpbDgtVdtl+6r+lr7DJDXRaiy3ozfhU5/CqF53McqfAcByg8hwFCMJcG5rKQXNYWc0Yu24YntClt5DJDXGaCS0dw6QguwwJcGoJLQ3BpCC7NElxagMsAcNnOy9HSu8TTCi7b2huUlxjisIshDrsY4rCLIQ67GOKwiyEODVwmgMtavg1ctpUjrjzUfD6Pcn2e3YsXsOEFiO2ZtyzKLWv8RZb/iyz/F1H+L3Lsw4uoQqotDbRlpLb01JZm1JYG2rIotqzneYlLtGYvoVVQaulALWuWGrXMkJYCaZkgLROkZYK0TJCWAdIyQFomSMsAaZkgLQOkZQa0zHCWGcwyq7KsKT2FJ0pTlgnKMiyUZaKyDKos645f02azU5Z168u6tVOWdetpNJKnURynqZGasQw0lk6NZa1NZ8g8zlB3NGOZSSwNiaUhsTQklgbEskBYFgDLAl9ZwCsLdGUBrsywlYW0MoBWFsjKmgrISkdZ6VRWNkdJhnWW+uosh2qehRs6Czd0FlzoLLjQWQ63Ocs77Sy6y2c5Ku7seKMprLSAlQGwsp2P98tZ0qiv4wJ8XS9LByvr1m/o1g5W1q17vFf2eK/s4V7Z472yh3tlj/fKHu6VPd4re7xX9sZ7ZQ/3yjl068/hfOcwyu8c7plzGO13DpXlHMdrneND7RxG9u3jCPtM8T5TvI8U7zPF+0jxPlO8jxTvM8X7TPH+mOJ9pPgVnOcV3Buv4J54hffEq/gFSsrzyOJ5VA0FlB6AMtNPevjJTD7pwCcz9GQjh/gajIYn28GeBivEQ/88bozzVGrn4XjPw/GqnHSQkxlyMlNOCuRkE4rXeJK3uHLIlbe5cpN73eLKbYbucMt73HKXW+7xFO+jp9nkZNtCG3Oet915PFnPU8ionLSUk17lZC3y11Cir6EgX2PT8hqaltfYtKibdOommyR8kiZyizRxm1t2uIXXpbnJCDeZwSYj2WQgmwxkkwFsshbCBTZZF9hkXWBVvsC26wLbrnVssh3kTaaN355xgd+ecYHfnnGB355xgd+VcYFfkXGB34yhbFLIJp2yyVqUFzlQ9iLL4yJHzF5keTQ2mcEmM9Wko5p0VJMBarLhxxe0qJqajFSTAWoyQk1GVZOBalKoJh3VpKOadFSTDmoyA01mmkkHM5lJJh3IZIaYzASTRqFkTecl5v8Ss92gZIGTDHCSBUyyQEkGKklHJempJD2VpKeS9NCRhThSgCMTcGQCjkzAkQk4MgFHJuJIQxwZFEc23UiorTgyEEcG4sgAHFkblsusuZdZcxVHeuDIJilZcy+z5jYcmWkjDW2koY0U2Mj6KaTZyLZyyJW3ufJtLblmIwtoZIGMbL9wBxve44a72HCP6f8AG4ipX0c9biyyQEUWoMgME5lBIjNEZAaILOohRT0kIKQohBSFkKIQUhRCikJIGSFke9E2h5C66YkRFj5JDbjNFTtyxa9oIqAg3aggw6ggzYqCNKOCNKOCNKOCNCsKshBBhhFBuhFBChGkpnr8QgEgSE8EGYkgIxFkJIKMRJCRCDISQaYRQToiSBtHBOmIIDMMZCaB9CSQngTSgkAKCaQHgRQQSCGB9CCQAgIpIJDNBj7BFWLoK/wCKiWQlgRSSCAtCaQlgbQkkBYEsqiAzACQRf1jVv6YVT9m4Mei9jErfcwqH7PCx6zuMSt7zFCPRdFjhnksSh4zxaNAPBaAx3odr4zdxgYeW4av6j43dNFxR6/cMat2zIods1rHrNQxq3TMgI7twEdc+W4LqHMMcI4BzrGAOdaG9k3WPGWOjszRj8zRgTkKmKOQOQYwxwDemKEbM3BjTcdVvgS4ysM33JiAG2sTcXV8PXD1focZAzCjADMGIsYAxBiAFzPsYgZdrKe9xtNe42mv4bTXeNpr42mv3e+oYgBVFFDFoERRlCiKEkWZEUUhUQwgigE0MUMmZsDEmrJDpuyQKTtEyg6ZssMxZYf3O4gYABEFEDEAIAYCRAFADASIIzwMgIeZ7jDAGwZ4wwBvGOANA72hgTcM8IYB3jDAGwq8YVFuGMANA7mhIzcM4IYB3LBAG9Y6eIN18AbL5Abr4I2xDt5AWdxAGbzDHuM77Ci+wyZJtaGhNhRowwBtGKANA7RhgDYM0IaB2lBUG4pqQ1FtKDNtKNSGAdowQBsWYMOav5vM303m7ybzd3PM303k7ybyd4uHvYXD3sJhb/Gwt3jYWzzsLR721njYWzjsLRz2Ng97G4e9jZp6GxXkNmvqbdbU2zz8bdTU26ypt8eaevt+JwoDRKFAFAZKwgBJGCAIMwBhhh+sp73D097hae/gtHd42jvjae/c77xggBcUeMFAJ+jgBAuZYAATLFSCAUowQQkWIMGiRlDUCIoaQVEjKGoERY2g0AiKGkFRIyhqBEWNoKgRFDWCQiMoMIJZaaCABkbQwAgaGJUGBtDAqDRQZjRQlAYKaGAEDYxKA4PSQFEaKEoDRWmggAYWyMDah73HP2Q0GZgBA1tk/PtFg4ECGCiAgQkwMMMFZnWBQV1gUBcY4AIFLrCABRaowAwUWGgCHU2gwARmkMACEVgAAgs8YAEHLNSAAg2YgQELLGBjgmwZ7yPl9/HO+z4/9t2HwbsPg3cfBq+XgAIJKJCAohJQIAELIGCGA2xT7RGqfsikfMia9CHz/SHqxwO0+A/Q0D/AC4cHSPMDNHQP0NA9QEP3AA3dAzR0D9DQPUBrfYTiO0LxHaH4jlB8Ryy+IxTfEYrviGk+YvEdISlHKL4jFt8Riu8IxXeE4uv9nsDvCfyeqN8T+L0Cvpeh9wL0XoDeC9B7AXovQO8F1Xuieg9sT5TtibI9UbYnyvZE2Z4o26uNzne17e/ZXgDbS2B7mWrPQu01ksd76CHecj3kHwMf8u9tD8ebqam9oGrPd2rPb1J7vlN7fqH20qraE84e0oa5SvHt67DbEJtxYhEdil0kutHsedtPeSQltGE7oRs7Oh1SFVHox2IbDOkpqU3/oKNrXJmOWfdLZZqY0Yf5MXXIqQ6scYu06MAtBXvz/YIxYZqZcRmzbSRyG0bWRgHp/4JxbYxKG0GWWCSiuyAkmGPLorjCEPJt4J7nzEbb/QQTQ7T5R7/CjvwQ0/nffD+7hMEY3mBCG//r11y7YGIbc6nTABgkxrfE6Jfit6FjGOtqdEBaMNmM8zOuhJTX6QxiEbE6ajsYnTgFg8YsJ1tMNdKGZ7YRYzrtgmK9YI0b52csbVxX+y75GpBxfsZpjzrLn2mjz6SfksToaLgh1k7TrJ7ulLbqlAdhuAyGVi9ZBmqerG2DTmOH0nIbcB2sa0mL3eDC2EY9DZE2fczoCOy0j28iRRFBVzj1QljfLgTmiuqvYM2VNFqigiDMp9IKVscyqtXzub9Xgk0a01HhMptVLFgFMuoHglkcteieBRMwdXdLcKbFlA84N4/pXJfgA3F+RqfTIagfWOaDHq+WTsjTPVND0lG98YRacE4kjlRvmQunLCH1I+bHlEY3ztHIWUt14qYa1KnclBGEsjiqEjZ1BNEs8ljcRPX65mLY0ZtmhlUSTNOP6I7e6o4doJl2dLqj7+593sJe5/nI/YjdcT9x4zyN0FrTfjr+UzXB8mL42OajU6U3VlUdWRt80mBeqVU1mMOk9IJbBEuY5mpc1FXRuS/LmlalDpEn0lve1aKz37XBgMvdXBuvrIrA2LHaDDdpHdhNpFeKGr1cWxbxbZRk0wMpjrWwRVrlVTng0BZUwxVEZwlVotc/5WyNtVtXzYBzbEHqXtAvxqzbLQK22WmiY+6X1CI1LpB9Vxq1EJNCMx2/6McWqx0TExCqz+ufnHVHMKSmBUKc3X5BHbz6PDJyg9lTh0dImHze/LkahmbNjD5vPKrXGKboUipQgKfbwOZQ7/jJ52lS0YgO9b3zeS53JRN04Dt43jzzgfM6rimYEBS5NB6gY/zrXMn1ZBHILkxNNk8W4QRRU1TDxRZS4Q2a50r3qAvZdTLPBcytUjSm0AgiIHFWFt+CRTqY5/rHZzRKhjAEdzpmLeWog6rh8mzhKNu6n03THIwlTx23obJHp/WvOYDEyaXbrCtDTA1NYwDL/bx0Ko/PxVIf5xEwpz1M+85SlA7lJdQD7SyNIbfmURNF0wH77metSVSIDZPXgajh+sQYyyTyODfZuGcK09SLXZ+oZS9LmUBe/0BpwRwmkOfK2LWrsaLTC8aynH6vZbOoImtP1BLHEo11ImHXTbyojwX9UoAhFO3E8bSnUZnOUNbJrZ11MVBb6pO0a4BzF2uX1Y4t6ZD65O2E8br0hRpy3YyLsy5S8uB9edrLIleoC6lMVWgM6V7tATpvgZPoNKj6/PSWZVGTmVAZ9PHpwqympKAz8rTnp3b9ovYXE6pCe3qmOEG8eg/bEeLFMtLimsSoRlIfncb1vdykQ9cB8ebXK0U/Ojw7Zk3H/oeUO4fXIVLtj6Si9+NsluMxOvTH0+Twpmaq1q0Mllz6TpeDXxyaIy1tdXgrUZ/C5PBWoqJ0uYS1UZDZEtdGk84nqk/TEOazZ+eMVPVzQWtvuOa26H2mz9RZ/2PYt1gbJoeHniu0TBg+goTJ4S2Dyo3h8JZBUZehD9ZlEBOmtSdryDPaEzA3sjq8kHvaM8R0ni59so47altaOIViWhdUqA2Hx+hYm0ruHV4os9KP1TdMDi+aLlrnINeOKIjJStR18yuuRr2KRH3MrkSDSkcra6NR/OTw2JTUtqm6ez8xvEWXe3jE6VXV5y36nNA4VV2FSeEtgkOvpUN4y6DN08yKK0H9tMIZy9y8gK1oEULDz7paw32h+cTcx7OefrQRe8qsO9U+6wwPbC0/F9YGlU/qo3cZzDFNAG8ZLAqWFOAtgsPtYCeA5+wIsWs+h/qVR4CHKSKzTiobnX52hr+bur7toE7Tqv7OSd/3HQoAis6vC4pSTn1/sgwGtWf6BmUZjOJGfzf1YlsXN9Z57UZ/N+/+DjEoQ3Rim79Dhzs6zHfY3qBMR40I6jym6u/6h1JNT1E8qv5u3h2PHnNv6osUl2d7emvy5O9Wgsoe1d8tgy53cygug15vBvV3y6BCcPi7/jkfPYiyvhzse+R1gkM38btlHqMmJph1waQzRYa1pUNh59YF9T0T+J0LXe9uuG9FvxYF+q7/1FWDVus63hEud3UudPYO35wR2ufGoelqtYv2zmQE63dp1Wh7hwh7N06GXl9yDbGcJ3u3iInJnb2zrvsMOwRd7uydtfOgzlEIe2eyviRrnfghWPJk7wo6RLHtF3zq6J3uF7d0GkcJnBVReVyeTYEbRb92hPRusW9iMKzsa2tY+1Skd9OtpH3kobNvUk/vBH/6U2Ln9EUO50IU0DohrUukdYrndsY1frcsaF0caZ2Q1glpnZDWCWmdkNYJaZ2A1pHUCUldAqlLIHVCUudI6jxIXSapCyOp8yB1CaQugdQlkLoEUtfcHPN2klk7yZydZMZOIl8nka2TyNVJZOok8nQSWfpCpE5A6gSkTkDqBKSufqvtE5zEoZE6AakTkDoBqROQOgGpE5A6AakTkDoBqROQOlmQugRSJyB1AlInIHUCUicgdQJSJyB1AlInIHUCUicgdQJSJyB1AlInIHUCUicgdQJSl0DqBKQugtQJSJ2A1AlInZDUCUidgNQJSF0EqUsgdYWiLkLUFYC6AFCX6ek8PV1Z4XSWnM6C00VwulrpZpwugtMFcLoITlf/oLWFwVVbGFW1hQFUWxgWtYXxUFtQDFtM1jYyuI0MqqQzkHQJkq4Q0kVAugJHF+DoMhmdJ6MrK4rOUtFZKLoIRVdzOlN0EYouQNFFKLqa0x3kdAc53UFOd5DTHeR0BzndYbKewkDopzAQ+imMf36K3+lp0JAYTIlnMCWeQRNoMLrdIMMGo9wNR7kbDIfv/ZyMfs7hR8/llh62wTlZwDkBnEuAcwlwLgHOyQY4J4BzEXAuAs4VujkHN1fA5gRsznHWQsdZCxPYXCKbc2RzhmzOg80VqjlLNWep5izVnKWas1RzlmrOUs1ZqjlLNWeh5jLUnEDNCdScLNScQM1FqLkINReg5jLUXKKaC1BzZYnmPNBcIprzI5rzRHMGaC4L0ZwFmis0c0Iz52nmLM2cGc2cwMxlmLkEM5dHMxdg5oRmzsPMFZC5AjGXKeYywFwewVwklIuAcglQLgHKJUI5QyhnCeU8hFwBkCvwcQU8rkDFtSkMXwOTIxrcJRrc5ZdH745WcJfTTOxi6MAuhg7sYujALoYO7GLowC6GDuxi1jRVcUIVJ1RxAhWXoOKEKs5BxQlUXISGy9BwAg1XgOEKLFyGhSugcAkULkPCBUo4CwlXCOEyHFyGgxM4OIGDy3BwCQ6utocvcbRJc3AJ/i3Cv0X4twj/FuHfBP5N4N8i/JvAv0X4N4F/S/BvCf4twb8l+DeBf0vwb7Hzb5fx88y9yejeXsaPAG+n8eNptGqnkeHT9COnQYNO07CeQRU9QzZ2hmysSbcE6ZYB3TKcWwZzy1BuGcotQ7llKLcM5Zah3DKUW4Jyy1RuHsotU7lFKreGzTiM/SySe5ZDIM9yCORZjBE8C+ZxFszjLHTHWeiOsxxtd5Y3zll0YM9eHnUd75uznHhnjtyEyE2I3ATITUbk9g38CN22hx/3UOn3UOn3UOn3UOn3UOn3UOn3UOn3WOn3UOn3WOn3UOnPoVN9Dmc6h9Fp51D5z2GU2jnUiXOEMef4uDmHIWn7OMI+0rqPtO4jrftI6z7Suo+07iOt+0zrPtK6z7TuI62v4AyNsgkoWyRlE1A2ErYEwpZI2CwIWyJhsyBsiYTNgrAlEjYBYUskbBGELYOwJRC2BMLWSBjA5HmAyUbYmieDnDwPOXmeuksJWyBhiyRskYQtkrBFEjZHwuZI2CIJmyNhiyRsjoQtkLBZErZAwuZJ2DwIWyJh8yBshYLNQrAJBFuCYEsUbA6CTSjYLAWbhWCLFGyOgs1QsDkKNkPBZiDYAgRbgmALFGyegs1TsHkKtgjBlgHYMv2agV/L4GuZXC2SqzlyNU+u5snVPLmaJ1fz5GoeXK1QqzlqNQutJtBqhVjNAKsVWjUDq5Zg1RKtmqVVs7RqnlYtwKoFWLVAq+Zh1QKsWlhYNUerZmjVDK2aoVUzsGoJVi3RqhlYtUSrZmDVEqxaglUroGoCqpYh1TKgWl5AtQyolpdQzRKqWUI1S6hmCdUsoFomVHOAahFQLQKqRUC1CKgWAdUioFqmU/NwapFOzdOpeTo1T6fm6dSETs3TqXk6NUOnFuHUCphagVJLUGoFSK3QqFkaNU+j5mnUPI2ah1ETGLUMo5Zp1DyMWqZR8zBqmUZNYNQyjZqHURMYtQyjlmHUEoxaglFLMGoJRi1/caMWR5oWR5pmSNMSaVoaaZohTcsjTbMjTfOkaYUirRCkFXq0Qo5WiNAyEZqMCM2MCM2NCC1OCM2MCM0QoQUitECEFojQAhFaIEILRGhxRGh2RGgyIjRLhJaA0BIQWptLj++c3oABnRs0s8mgmU0GTWjQBAatkKA5ELRCgeYg0AoAWoE/K+BnBfosqz5L0GdZ9VlSfZZUnyXos6z6LKk+S6rPkuqzpPosqT5L0GdZ9VmCPsuqzxL1mYM+y9RncdRnkfpMVJ8l1WcJ7MwoO0vKzpKys6TsLCk7S8rOEtlZJDuLys4S2ZmAnQnYWSY7C2RnluzMkp2ZkZ3ZTexMwM44q14CO0tkZ5HszJOdGbCzSHYmIzvzC3YmC3Y2zp0nYGecMy+BnSWys0h25snODNhZJDuTkZ35BTuTBTuTTeys92YCb8ap8BK8WaI3i/Rmnt7MwJtFejMZvZlfeDNZeDNOeCf0Zg7eTOjNHL2ZhzdL9GYe3kzgzQTeTODNBN6sgJsJuJmAm8mCm2XlZgJuJuRmltxMwM0E3CyTmwVyM0tuZsnNzMjN7CZu5sjNHLhZgTarHeV3eO/9ZNrMbdJmPTMTMDMBM8tkZoHMzJKZWTIzMzIzu4mZCZiZgJllMrNAZmbJzCyZmRmZmd3EzATMjFPZJTCzRGYWycw8mZkBM4tkZjIyM79gZrJgZuOEdQJmxonqEphZIjOLZGaezMyAmUUyMxmZmV8wM1kws3E6OgtmlsnMPJhZJjPzYGYRzCyDmWXiMUcS5kDCEixYgAULsGBBLZjAgnEquADvFdR70XRlmq4A01VAuhJJVyDpKhBcEYIrQXClmeCSUXB9gBx/gIx+gI7+B8jfBywalVsOcitBbmXIrQy5lSG3MuRWptxykFsJcitTbgnllofcSpBbiXIrforc+gxkK4NsJZKtQLIVSLaEZMuRbFmQrbSBbCWQrQSyJSBbArLFGdgEZEtAtmRBtjLIVgbZyiBbmWTLgWwlkK1MsiUkWx5kK4FsJZKt+Clk6zNYrQyrlWC1BFZLYLUEVktgtQRWS76g1ZLRaj3EH5we4g9OD9EIPMTfmR6yMXiIe+Ih3t08xPv7h/yDzkPeGz3Qch3QcpuAlvsUoBVWgZYrnH6j/lk+6DRQbQxG9pzzogwB20Zhhm5IsU7ZMUTa6IvQjT4fj+ba+GsdeqFfBsydfKMfOu7CzpOgYxx00IWOgxlD0gZLh25CLp0aJQYdV6Ioa7GT8gQ1WctQGySqJCuMQ19j0PkI2qCiiD10fA8jygSihur40BhyO43vh4Pol04PoTaAUTVW72PcECutILzKEpxLB4fFOrpuxFiun7UlRp36gd/kjRQOyYhqjZRiuZ44DCGfR4rlpmFLLdQGeegIIpNYsHGI+EZmdPiQN9NQ8iHSrpNKLKGrqkkQGadN09HLVWL5GmjjglRiySSxhkAbxSX4QulpXFRNmn7Tv0os4XDcev6gXrAVQWQg14DyKLOa5NhBLG8miDVEWgqiW90n2WnStKlo2kVIbSIBHTbeX7uabB12FMN80qShNsTSyk0Nlusmv4nJtAHGOmBcXH+9k6qWmJdT7cTk2ndY62Bxm7p7IvlWfTBzi59F2ljdZKeveh9PJK2SJLeS7hTaECI1V9jHt0A0ZRRX013ZjqbeTMGVz/3RkjItHRZd+sQpS0ndrBuYjmi4RjrVWp7mt+DhShv/qsxKcp+jbNpwPUVWU/s07JRtm/km25VZkYZQcSOwcrHfybWhitnPI20n34YU5m40HXeSNpldDitzb8UhS3GkVbMCz6GNztPR4YUF0Yo8R/1q9zyvKTWiU4DpyHAxfSS3i66jwuepK42ZFrvaZhQdW9gGeC3vzKJcRDXVvD0ZGjVDTVU8W8nhPiuuDVIrcAPNUqU66q1o41TiXMH6GtHZ2tI0cC21+YeGSJws1fRMqgksvl3a0k3fw50Af4xZs5eEOEmq6Wvm625BKVEb8J0WYzcxrSYk1QRuWoxzkgm+hH+q6UWHucFRuV48DbHSOSrXjUCtRZJd56hs/zQuOU2znHHOtVZnit4iMFT6vf1t2G6qsRQ7Q2Um+TP0LXQiJx3IqIwltcGcyVid/U2Hd8+yPcRinAzVIuYkjoYqZc51VE+m6hWGyuTpAgwhFSwWNWQyVMlgei8YqlnqRWcW0hHdtjdUyQSd+EoHdEdPQ+U0qJMkOc57pcHcYjF2s5vZ2COqhMk8iahmick6x5PzC/3uaiy6CVFZzFPQJkBIprTWA4hqsV9Ru4SpVPC4zcOeySqSBKKaujdDIE2IKoYOMU8ht3xoJGv7ec2cnyrdEFIdpk9CiRyNX0NOp0mAn+J10928t6Oe6vpZQ5W0OhcH8NT0fNCYChbYKTPaqZrngAnUuplyCqDDEGzjbpVO5cJmKQ4R3KT6RHRpev4PIUVxiV0Dq/2ZoQ9idfKXpZwatqvpg0EeH/GpC7Upu4Qd7OFCWkz9lqYZH7XNHCJ6+fWJ2HdOhlCUkU3l3kal4QE2zWBW7DykzZyyqb6xHULakukzcX7bOswSqQ9FG7q6MbTqMqEp0UrYDugsJhSTScjVGUZqRO2coinphjIPIb1r20OxSHehnC0dmZpdKAeGmLvZotFxGGKY8kxv9TKL+fZAB5dKaRYThV2KpUzizdyKI+gUcWVlXqDharqo1b74dcGkczUWWRfM2nCWfl4gzBKXXGk9TkCpxZ7eJD85qWXQIietdEocO6tDLr3L3WRlXcez7qdz/IFI2TTNq1iDOlEniNRKEF7JrQtGRTr6xFwGU0gTkRqntRENZsirMPZDu1hPpGSmoJIvUFlpTXDoFJSJSAmJbhvPPwSznYDUfC7iOrWknXyU5F7+JHGpm6ZsGfStowsdtQwG001SJrmfkjWJTpYDG7XcM6khwyyhfXMhWeJEo2bd5MQ5uWyHNqZjarMAGLUIBqtlZ8u6oHNxglHLoIdust3XjbDGBigS55Yd1hSCzjrn/LJ7nkJUKaMmys58UqpCdjJRK0EQrrguqHclTBR7XNoQVO83kajFjtHYbk6ylWCZ5iTru2TD1cIUsSBRUxe27hxVYoJEWZkfFBPnKYlaBkWvMr5SZhlsfQmQqGVQp3YCiXIzEpViVNSoJMr1JGqI6a3u82zHqEEoNSVRK0HdU0mUDfP0ZD2l2GW/Og0fP8wEouwMJ1WP6ycQtQgOXTs7gahl0KqxxHzwufvkkIanSJo81HJHnTYTHmp2Rw5tS5g81Kx3nQRpKVMGScnScF+FzkMtgzp7XLDrglHn9IOHWgSV2sJD2ThDTcMHc01skMVHp+FuTUUnVgth3Z7Z2DxxKEwUjW5q1snloaGGxyk1lGi09RKoocz0qitl/TYPYqh5yKbOQhlh7Wgx/X4UtDrGzWKxm4WspkUllGtJ0cckJRS+r6DOFZay3nOQUNhPZ+SrwdYXoIRK0k0aOQSVV6mDWu4pOXcOqtuzXqzMj6RgUJwAjrMJp1zfviwZVCKDsusZlCODiiODSiODSiODciODCiODcl+IQTkwKEcGFcGgIhiUI4OyZFBuyaBkZFAODCqCQUUwqAgGFcmgEhlUIoNyZFCBDMp9fgbVOJMDZ3LgTA6cyZEzRXKmCM7kwJkcOJMDZ3LgTA6cyYEzOXAmB87kwJkcOJMDYwpgTA6MyYExOTAmB8bkwJgcGJMDY3JgTA6MyYExOTAmB8bkwJgcGJMDY3JgTA6MyYExBTAmB8YkYEwOjMmDMXkwJkfG5MGY3IIxBTCmuGRMYcGYZMmY3JIxCRlTw0ocCj1nTAmMSTYwpkDGlMCYMhhTBmPKYEwZjCmDMUUyJrdgTAGMyScwprhkTGHBmGTJmNySMQkZU8NKzOmcMSUwJtnAmAIZUwJjymBMGYwpgzFlMKYMxhTJmNyjGJMDY3KfkTElMKa4hjFZHMhh6blcB5cEcMkBLkXApbiASw5wSQCXBHDJPQou2QVccoRLQrgkgEsRcCkSLlnApToeROGSWcIlQ7hkCJcM4ZIhXDKES4ZwyRAuGcIlQ7hkFnDJbYBLArjkHgGXEuBSXAOXnh7b82dQW55h2//MGGpiqd4RTSylcZovsxRLnmLJUSwZiKVafs+O4y9mYimuiCWBWHIUSw5iKUMs5UeLpUCxFCCWAsRShFiKK2LJUCw5iKUMsZQhljLEUqZYSkuxFCiWhGIpjGIpQCw154Tn3S6ed7t43u3iebeL590unne7kBq7HAu/y7HwuxwD38RShFhyFEsWYslBLIWFWHIQSxliKS/EUoZYipvEkoFYypvEkoNYcguxFCmWEsWSh1iKPyOxFCGWIsRShFiKEEsOYikuxBJn6goUSwKx5ECVHIySg1GKMEqRRskujZKBUQo0SoFGKcAoxZ+RUYpLo+SWRimMRinRKCUYpUCj5GmU/MIoBRilAKMUYJQCjVKiUQowSolGKY1GKdAoBRgloVHyNEqeRsnDKDngJAeV5KCSMlRSXqikDJUUN6kkA5WUN6mkCJXkHqGSIlVShEqqDcBMJTmopAyVlBcqKUMlxU0qyUAl5U0qyT1KJTmopAiVFKiSDFRSpEoyUEmRKslAJUWqJA+VFKmSwkIlRaikSJWUoJIiVFKkSkpQSREqKVIlWaokoUoKVEmBKilQJQWqJEuVZKmSAlWSpUoKVEmWKkmokgxVklAlOaokB5UUqZLcUiUZqCQHlRShkiJVkl2qJEOVZKCShCrJQiW1mbZoxS7wa8cvYIK2GUqKS5TkiJIcUZIjSgobUFI7+FqUFIiSLFGSI0pyREmOKMkRJTmiJLdESZYoyQAlOaCkDJRUk3IRA+QuMuMXMRbzIjKsKMkQJRmiJEeUJJtQkluPkoQoyf5MUVIESspASX4DSkpASbJASQkoSYiSDFGSIUoyREmGKMksUZIFSgpASQEoKQAlBaCkAJQUlijJASUJUZIjSnJESY4oyREleaIkR5TkgJLaJFysn5dRPy+jfjaUFIGSMlBSBkpqs21do056iyuHXHmbKz1KSkBJiSjJAiUloiQLlJSIkvwSJTmgJAeUlICSElBSBEqKQEkRKCkCJaUJF4URFwXiIpuJiyJxUVzFRWnERWbERY64KBMXZeKiTFyUiYvyZlxURltkR1sURltUSIs0oXx1dcBXVwd8dXXAV1cHfHV18B5X3qcxuj9m4cPRRx2N29D5fQPPsDf4/ucNvv8ZaREZ7Bts0d7gGw8lQ35JhuySDNnPR4binAxFJUNRyVCck6GoZCgqGYpKhqKSoahkKM7JUJyToUgyZJdkKIxkKJAMeSVDUclQVDJUi6SSoahkKCoZikqGopKhqGQokgwFkqGgZCg+kgwJyZAhGTIgQ/VmfXO0aeuJUAQRiiRCgUTILYlQIBFyIxFyHQ36VBIUQYIiSVAgCXJLEhRIgtxIglxHgXCa9cwngvlEMp9A5uOWzCeQ+biR+biO96xlPXbJeixZjwPriWQ97jOynvwprOfTOI95JOcRch5DzmPAeWq9uDHWi3fYD3qH3Z8534nkOxZ8R8B3BHxHwHcEfEfAd4R8xz6S6AiJjiHRMSA6Na03x7Q+guQISY4hyTEgOfUwt8bDrCc4EQQnkuAEEhy3JDiBBMeNBMd19OZTyU0EuYkkN4Hkxi3JTSC5cSO5cR21mREbA2KTSGwciE0isXGPIjaWxMaC2MSfCrFJJDayIDaRxEaWxCaA2EQQm6jERmbEhpMjJRCbBGITl8TGkNhYEJsIYpMWxCYtiE0isbEgNnFJbDyJjQOxiSA2kcQmLKxMpJURWhmhlfG0MpZWxsDKRFgZgZURWJkIKxN/QisjCyuTFlYm0cpYWJm4tDKeVsbBykRYmUgrExboJX5+9PJd9PxmaiVArcSlWgkLtSJLteKWakWqWrm5Sa2EL6RWemeSsg4YUrViJrWScm5/lte/kvMv9kkjeZpRyJnZ0XR6g7Ay9jsVndwgdJOycBBIMW2c1MqfxmtIR3KEMA0IRSKKa4PjQlyzk29jGEJaF9J5eGr6gp2GHRZpg438NCmKjvwdA200dmDxSA20RMOs+M6spBLaX9BhVqQ3K0Os/c0aZqWNVR3/KF1iG3/u84q2SUWHpqtZMUxfTUZuo9rUrNh+/PkQahPkqFmx/RjzVIrKFDUrphskWEqZZg9yeRpukY1aAJGxiFxLwhDQ84RxZGhowzKHgMqYONujBvSLddWs2NnI+CFmhWYl4Py+nd+206hZGWcPGgrIOJNGs+LyNGh4iLTRRZFTWNg66KAOlx4ibVBWdFOE+/g2FBJmxU41boi0gd46UJdDYJWsZCPNfulIXd/piozB0DEu74hsYhuAGtMKbMgG8wmtkJVsMJlQmd9GNVJaMaySlSHS6vAaspKtzpSzSlay1W8nV7LSjYWv8EMmsmK6G3kItUHJICuuP5pv401WycoQaY0JyEroRpDmIa1hIiuhP1xoo4dBVtIsR1GHk5jlSOtsdUKXbFcG8A2hnCayIv1OagKyX2k5s1UUkBfYbvi/M20YIchKj5GyszrwJq5UFAzPBVnx3SDT7Fz7xmgdnev7gnA6jY6OzfXSR3QIVFktB6dDuYtdaTLqEB4ZycrixnSxtTRKVmbNSXapNSdKVtw0/HoItNakoGkAWbFbFQIJyUr3Vcy+RlodKt3gdVUMQ6TdzToKd+Ye6+iiMpEV2++EMf4Yg1v66uqNTo5g7FK6DGkw0+Q/yc/G1g1B5RTGz+d6aTGn81IoWZndoN7p+Os2+la6h0z2ip8UrEjGs0TL1nsdfKxexaT+gOLK6FXmBxSgGh3Kbzuvkn3QOUF00K2mEG2fD6Wb82cK1QKJOm+Rnc1vo6XvdaZAeJVFeSRNvnoVmbxKxgRW9Cqz0tcnl+W3eE9eJfuiDkGH2s5TX5D6vNQjWQxS303ToH2PGisyaZVpnHoNKc8lVjF9xsTZzqrMd8OI+TbENofZHSbeT1YlRlgVV63KENOr1obYLvfTGcZgVfQpp4MAh5AOwm8PwK5Tk0V5nFqV6Vlq+5Cbj8ktNaRD4+LCF7dQ6Sb7mWaOqKGo7ECfgCZ1ViVLUiES4/il/7We1rG6ArmxxqpkzqOlT0Ebta/UehYC1bOOqgxB0BIz0yBxKwedtRBUxXYCYggVM1EV7REoVcl1WsA1VGXYnkxHVfJEVaZQ6LRMSznGdK9SlQzwDaoy63oFp65BqUo/L1AOXutFexTmNA9psbdH4byRDXAb+iy0ob+bgy8bqEoOkiaqElAHW48u6NfZK1XB5CttOOkQ0emOlKqYjqoMISiWOKOO9UJhVil9GNr+5gqY7waz4qVuQOgQ0/sVVGXWmQhJrz2oipvFsjo9UBXb3+ehaCnq6FQf+3H2OerEmaAqyyAmZdPRqcugmk1QFT9zERlwHFRlGdReEajK8rDaMSJVmfUiIqBWKYsOZ90vKRgw3UOBpiTHrGTA2HVBLR9SlXkw6QyWpCpxmtRkyEhSQgyq4mZUZQjCznS+c4xxdqE42zEgiBl5Uhe0NVhP6XVSHVAV31OVnHQSCFKVMCv1pFM5kKqk3mIMH6E1saAqi2Cy0lGVRVB7O6QqqacqlSPnjqrMg8MHizhRlVlzkfWxRaoy6x7XYd0dVZlNtpOzfnAmVVkExaWOqiyCAVPbmC44JjXqaHzXT0HMGptTLB1V6fveWfvepCq2D2nfG1TFlJ435MKpdMLaoF5IF9cFbSOcoCpm1luEpQZVWcS0FYZUWR5UH+uUKrGTKkM2NPfeznuu7aBBKZfvPOl0UJ24klJlEUwKoCBVlkEVWZAqfpaPHPIIVSTPoMrQxda5ctq7C5nNHDvEtInwebZjaLeWMXpjtbcXY/e2vmQqmPSZTGWW0mKs6ZhK35ceQrlnKnG+n37sJ1NZBFW3k6ksg3ojS1h5S1WMPvzIVOYxVH9ZeXYXox+pqFTipFSGkAo3Kcuk1EMm383aswxm083asxJM3aw982LDDFR4C9cBluFjw/AZKU1EpfuA5WpM75rli7gWs/q1CStApcZcyhuASotyz9aS+0mhFKt3FIDKIuS7yXqKnb5IZgjpt0hoU1PMLISU9GCsTXlTE6K0K7NVcPqqqL7HGmJOVnyKTg5Xg/oOAD5l7HLUB0qxwZQVnzLuqX1p+pRuz7rz8HNZ+JTxo31tpMqKTQmPsClx1abE0abER9mUL+MQP4flemUSfybKJFKZxE9VJj//nWkunAg8EoFHIvBI3IhHIvBIBB6JwCMReCQCj0TgkQg8EoFHIvBIBB6JwCMFeCQCj0TgkQg8EoFHIvBIBB6JwCMReCQCj0TgkQg8EoFHIvBIBB6JwCMReCQCj0TgkQI8EoFHMvBIBB5JwCMJeCQu8UgEHomPwiOywCMZeCQRj8hmPJKIRxLwSFjiEc6B81NDIxlmJC7NiGwyI7IwIxlmJNGMyGYzkmhGEsxIWJoRTn3zU7Mi+bNSkQgqEkBFQkdFlkSklyERMiRChsRHyJC4QYYEyJC4YUqb+HllSPz/lwyJj5AhETIkQIasEyGfH4IIIYj/NwVB4meGIOnREKRsgCBpowNJcCAJDiTBgaSNDkToQBwdiIwOROhAIlyHfHbXEZeuI8J1yMJ1xEe4jrRwHQmeI4NzJHKOtOAcEZwjfi7OgUFFjXUksI4E1pHAOhJYRwDrCGAdCawjgHUksI7wGVlHBOsIn5V1RLCOCNYRPzfrKGQdhayjLFhHAutIYB0JrCP9zFhH2cw6CllHIOsIn5d1ZLIOIecwI+eQz8E5AjhHBOeI4BzxEZwjLThHAuPIUByJiiNtUBwRiiMtFEdaKo6wVBwJiiM+QnGkheJI0BsZeCMRb6QF3ohAGwFWIy6sRqHV8Eur4f+t1fgUq9FKYz3W8MAa8SfHGplYw6xgDbMOaxBpPApnCHBGAs5IS5yRgDPSCs4wxBmGOMMQZxjiDEOcYYgzzBJnmCXOiIsZY0acUXqc8VlRhusRRj8TjPk3gi44E0wCukhAF+knRRcZ5iKDXGSIi/zTAhdpCS4ywYUAXGR6C4G3yJ+HWwi4RQK3SAtukcAtErlFBLeoZdi4RVs55MrbXPm2ZmItt2i/0HOLtuHTuYWAW8SfnFt88zuzKV3WqYu4WV18qrVItBaJ1iLRWiRai/T5rIWssRaRakJ+6moi/ls1sVZNfLs1Uj9NPDGzEn6zlYgzK/HpNMJsohE10Veno2yYPeURIsJsEhH16NfGo2+aJOUREMJsghD16Ifj0TfNd3IdX9k/BxGhBxFP4Tc3wwg0zdfRNF9H0zwDEQEgIgBEhMU8JwkiIvQiYgYg/GYAEVcARIF/KD91/lCW2MFvxg5xhh1mtsFvtg1xZhs+nTKYTZShXv7b01E2zCbyCMFgNgmGevQ749E3TRrys5cMAskgkAwkDALCICAMP6FdwBj99zFGf61lCJNlUMTwPkbi/5Qxw8b5Qr4gZvi884X8zPFD/cWfin7YOFPIF9QPn3emkIWW+Pg70wweEQgiPgpByAJBZCCIRAQhGxCE7RCE3YQg7KcgCFlFEFb6v9/a3L75sv0hNnkd1lbnQyhWvw5R/whrOgQxRGRCELNh0MUZMyEIi290rn/zHSJpQhBmjAyHc7YNBAmy8h2ZQ6j9lVT/9Gpsfzj9EvcQ1+ykf+YOaV1oMhBtcIj+rzj90lA/TVLQhhdOARmHmOVmIIZA++pizzlF6t922/CMIaTfwaoGov86TFdjaZq3oxvy3RIX9Vx5GnzW/lY+RJJKjPZtl4HJq6nIMk7bMfsm7CHS/hwsdk1kmrRD/y6b26C54vSySvfdurHN5jFE2iBs9Q86UEHP740d5+zIraSljfUbAm2sn0x/3Fb/MATasKg2WqAb3loT5o1Wqzzu06b/2KqmJZM/hMhAqQEVAWY5VG6IhIk/aCS072qs3/A/8Yf5Pq4NWmuDP6chefU0rg3J06GfvEf0yxKL900RxDCNhM86xMFLuwIxrtwKPujcIGllIp0hpHOD5Ol7qnm4qIcr8/unRnSqkdR9Z+UYaUfTMZ9ECThRbpnF15Pb/kR637chnzrUBfeC12+R1jGfvBm1gEqrUoofXH8wMW3kEPCD4T41YtsMNjrkM2KfZoUKxh4rfvC+P5zr8MOkAWqGxLdKkrvRNCgFwXdl2jU7CQZNTLcqz6TDWrJfbTGHJiGP+IH3qZ5Jp4PJ3Vco4w4WHdKS40o1kWym6ToSrngdHVZD7bbX4Z5uVg6ljYrVwZ6zSFCqVFaLIehg77LaIgTbvmm0DfJY3JNBG+DiV3dy01wdWXhhhxsseDPBBzvBhyHgxrk6QuGxfA00jAD3gPl1atMzRJp0K924Mzw0gs8TezCOjVXdRzBzhlndKehofR3NaXy/V9Dxs20sZ0zduPsSog531+EoesCIA2JKByMrd2VIOgRWyUM38c0Q0SH+Sh6kJw8lYK6HNoSz5P54GfIij8fDfRmKzmOhgzdD7Eb4DzGxk3gwsWvUo7F2Eg9TKNUQJu9wKzmOnLzDrwnFMHqHmPk9t/VUrl159Q5KCXGBo9PvtOfX9OpjtI6gKtHrjCs6XHOedC9m4g5GtU5tVVpMx3fb7lvXdbqVISZ6TXT0VO6PqNNFgTt0R6z5CnALrtsrtr2iTm3QRmmmzLHu9faI+oXiwA46MUcdbzU0JUNMJ2lpgzQXu3EakDh/cPkawmwZZtF5iQk0wY69l6Q1bYy46S5BdYqK6UAdepY2hIAqZP6903UvNa5KHUrG5WqhpF+FD+ngWD9rg5kww0BMc3NZM5wsZuzIuMqjdBhCsUzSIdrZKLFkdZi8SocR7w4XJTlEuskXcJWHW7JM0kFLV7R0k1eSs5AOw/bkJulgx0k5chdqLZ1j57BmSnwnHUzXbUyi92NKK50sfHG3SodURgJTQ0ELKU1SeQp5P0qHWauaQvQTdJjun1JDOl2LPvfYbLXqnqIOzlfn4CfnMES02cp9L7CNSkwRs3+EsX/aRsfW5CWTR+eQw+QchojeVTmtXibMdoJ5qvrxfym3zgmYQ7D9mMOUFZnps2/GI0oq3YwcxfMeHiLZqN7RIY6OQ99bB2/YSWdM0iGOyyAYVJF1QVWfQA4rQcycEdcFRadiKmldMGgbq0/EsbtQwyVHUI6y6FcOFy8nTH/RPQeoEQq4O5DDMlh0EK9xa4PFTshhESwGdkK6gdEW3YfhqGFEDuM8FYx183FMO2oLWfR72BU5rAZznpCDt/3HsoJZI/Qx6f2sXIfHZJiMgw/9IP5S9PaDcVgGdZ5IjPJdBpMtk3FYBjNUgawLFv1KfTUOfUNRB8ZqThQ5zHq0NZjDpBy09dZs1tkhDUb565NzJeowKUlZG/U6tYQza6MSu0k53Kx4h6h6J1AH231IGWJROuvQg7w6LFuLHthhenS0rCbMrhHWBTNmEIlrg7mbl2MZ1G8IoHaYB2tyO+7QzaplW1RTpAOtox07H7U/ZnXyTYCHqc9a82kdJthwU88UnzdqMHZTcyyD2nkGeFgGMU+ED2uDWhf8oh/a0hP09lTwIHysaAiIIve7BY1FnbJCucMy1qYqg3dYpiXpAHyx/QypKJukFkbcPKY76usrgIdlkDN6yNogHEVYExzaJTOJh2UQekvS6r053DtmMg+mc8RDzGkFkLKaSYeWK5g1Me2bgDwsYmI78bCMacEF33VqC29YF7Tkgsw/J7WK7HQGTje+eZvvqtOvgD1kbYSlzcg4BLMpnXvApIdtBkINOz/BBynT26wakzLJh2UMc2y05if76RNnjaVubg7GAmIldfhBGWabKLelpnDGj2nOsMA9deYn6gfVS+05E1o0pI4/BDv1UobKrVPwkj8sdh16ob7zD6HDxEPQNvk28ofptU2LDg3PZv9gAB5kFTyEETyEETzYETwIwcPQa1DwIAAPsgk8CMGDIXiwAA+J4MGP4MF+VvAQCB4CwYMleBCAh5rEHjwIwIMAPAjAgxA8CMGDADwIwIMAPAjAgwA8CMCDADwIwIMAPAjAgwA8CMBDBngQgAcBeBCABwF4EIAHAXgQgAcBeBCABwF4EIAHAXgQgAcBeBCABwF4EIAHAXjIAA8C8JAAHgTgIQI8RIAHIXhwAA8C8BAAHmQBHjLBg1+AhwTwEAke/GbwEAgewhI8hMVsGQQPQvAQviB4CAAPQvBgF+AhEzz4BXhIAA+R4MFvBg+B4CEswUNYTJJB8CAED+ELgocE8JAAHoTgwQI8yBrwgHzNoEMGdBBABwF0kEdAB9kAHQTQQTZMgSEboEMmdDCADhnQQQgdHKGD2wAd6vDSxAGeDTq4DOiQAR0KnEMBcyhQDgXIocA4FBCHAuFQABwKfEMBb0jgDfII3iAbeENc8IY88ga75A125A2WvMGCN0TOc1GgGzJ1g6NusNANhbghjbjBATekBW5II27wwA1C3GA34IZE3BCBG9KIG4S4QYAb8iNwQ6FtsLANEbYhwjZE2IZI2xBoGjxNg6Vp8KNp8DQNYWkaHE2Do2lwC9MgNA0GpkEWpiHBNMgG05BgGiJMQ4BpiDANCaYh0jREmIYE0yAwDQLTkJamIdI0uA2m4VFTVSxNQ1yYhvg5TYPANAhMQ1iYBlmaBoFpEJgG2WQaDCxDAWXIpAyJlCEtKEMEZYigDBGUIYIyJFCGBMqQQBkSKEMCZUgLypBIGSwoQyJlSCNliKQMEZQhkzIIKYOAMuTPPUOFA2Uoo2RwlAx+KRkcJYOjZBBIBoFkEEgG2SAZEiRDhGQIkAwRkiFBMkRKhgjJkBaSQRaSIW2SDELJkBaSQTZIhgTJECEZAiRDhGRIkAyRkiFCMiRIBllIBllIhkzJ4JaSwS0kQ1lCBreEDALIkDZBhgDIEAAZAiFDBGQIgAyBkMEQMnhCBiFkEEIGIWQQQgZDyGAIGYSQwRAyCCGDIWTwgAytTD7glvvEDrwxGmQodAwWjiHTMTg4BtnkGAz8QgFfKNALCXqhEC8k4IVCu5A22QVLu2BpFyztgoddiLALkXYhwS5E2IVIuyCwC+UL0YUMulAgFwrggixnlUjLWSXSAi4UuIVCtmDJFizZgp3PFVGgFgrQQoFZKCALBWIhQixEiIUCsBABFgq8QtwwSUSEV4jwChFeIS68QoJXiAuvUMAVErhCAldI4AoJXCGRK5ifkCtEcgULrpDIFfySK/glV3DkCpZcwZIrJHIFD64QwRXigitEcIVIriDkCoFcIZArBHKF8CiuEJZcISy5ggNXSOQKHlxBfmpcQUauICNXCJ+BK5RRK1hqhUitEKkVIrVCpFaI1AppFSmYESn4NUghrCCFMhoFR6NQNhIFIVEQEIVMohBIFBKJgiNRcCAKGUShfrBpRCGTKBgQhQyikEEUMohCBlFIc6KQPo0opM9NFNKcKKQFUTAgColEQUaiICQK7tOIQvr8REFIFGROFBKJggNRKBAKhUBBCBTKJp8gS59QljwhzXjCg0/jCLLkCGWpEdJMIzz4NH0gS31QlvggzfDBg3XYwCyxgVliA7vEBvYR2CB9RmwQF9igwBokWgMHa1BADQqlgVAalA3QIBMaBEID8wWhQSI0cIAGBc6gkBkImUGBMkhUBg7KoAAZFBoDoTEom4iBLIlBWQqDNBMGDz5NFMhSFJQlKEgzUPCgAwRl6Qfs0g/YhR9I8AOJfsDQD5jOD8w8gIcHyEsP4OkB8mcd//+p4/4Txv0XDvs3G4b9p8UcBmk5h4FZDPtPHPbvOOzfbhr2z2H8aTmM33MYv+cwfsdh/AbD+MtnHMVfvuAg/rSYwiAtpzAwi0H8iYP4HQfx202D+DkoP60flC8YlB8wKF8Wg/IzB+X7xaD8hEH5kYPy/ZpB+d/73r9zYvgnfOtbz59o/zz3W58cH3/yvx0Mq8//s+P2z//x8okTf+eT43/0XP5vPzn+b06Ef3r8J985ceLV/+v4X772rT8+/t3Hh1/9reP/5+Ev//D4t+oRfuP4k78xrP+Grh//6rT9//2V4ff/acDvD8f5szsnTrz+h8d/8PV6/N+9tP+/HNfdnvj+dN4T+dd/eHz84986pan70quX/oKu/Sf/avjd/F8PiT3+jV/+4Y9/90fD4p9/t573GXcCp/9JF1969Vt6mi/9zT8bjv/Dv//EiRP/6Y+PP/nzT44/+fUv5f/9+I9unTj4/eN//ZeH7Pxmy9kPf1lTdeI3j//0XV37Kz84/u3H29qtPzr+oxtt7Zf++PiPf6mt/Qd/fvyv7re1X/lBPVz95z/70fGf/MffGv4Jz//eMf751W61/s6ZIarH//f+xbDtx//gOT3V8Q9/fHz8/SdOPP6/Hv/gPzyRf7umKPz69//LL5048avYdfjnL/z3x392r639F7/zJ8c/+ttfauu/c3z8o3+om0/81V/7e398/IO/gt8/cfSD43/8pcN/8P2/M6y/+6fHv/f8/v99/IO/+cTrw4X6n06c+Fs/1pS13P1HfzBcix//owPdMXzr1S/xICe+PVyr39HVx//h8bj+Kz8e1/f/4Pif/L6uf+m/O/7TX/o9Xf+rP/zk7z6v60Nl/GfPY/3vfjKkUNe/8yfH//xv/9p//ofH/+ev/eXh6o1l9e3/+fvDP//4z4//9fe1wPH7s/VfxW8P6//VtPobx//iXr0KZ06c+O0hx9jld45/597/+P2/d6rtXAv7+M///RPP/pNPfvSbv/ZbPz7+/aGqPvFyS/yP/hr2+Gs/asn4G/XKDrfEa//y+I/unbj3R/XmQcH9+G8Ne/z1PxzW/vCva2k9z9twONzrXM//w//H3ff9RHVtj59/4Lx+n+apDzzMQx+akEzSmJBJjJkYQ4ixIcTJh2C8QWOJtSn+iFe9qNQqtlw71nIdK6IVtBaFUlsuA9dSEAsOOpQRCgqUGX6MMMNwZpjhcM6c811r7X3ODIitrT/qvRM9s/dwzj77x/q91l4bbtce4swaqBHOF4QzKX0gL6dD05tw1SMF8HS7r14oiOp9tpp7d8qgxQOL+r0Q3K7dtyJsp6I3bkR17TKW4wBzJXHAxqIYoXDWMIwVhga4KGwI6YE1MMvajzZEeUAJRyADJXO6VkNJYzKhmYxiNs6ks2pJ72UDzOrTk3vZg9CJ3/kxscf8sefZfjy5ZPy4HFl20ywQVTuU1LWFBBuKV186JVhuwergAktOwMslvQ4BTC4XxFbqlfW+pj6OaMpVaFf8fA6H9Wg3FFs07VHFxTAiS+G8PmYThJ0LMJ3QLQQtWO2gHX6XDghibUofzMI1p/cC2Ajix4Qsv2wxltqgakh6CqxpCqcrLVgD4qMn4f/9LMExjQi+bkBfOiaUJoj8AJJ2GqQLVnM0G9474RDEJg36IPbo+kzDg1QKQdkxhtPQPkFgbSlr/mor9LNPWN9FhOJMCkADQEltcJ6RCMSQsHA4FYRdU9DnyCcGCnHohLX2wP8GpDLG/yqNqD793zcrufC+ovHgjmZd97oU9Zxgt2N/4V7kGvIRN0x9aUIuw2+nJJcdkfE7UVomT661VVXZcr+ozGIcpE1YO7l4gCOkG0YbLcDymoB8RDgiA9AK2f6Yqi1ICV2RanEV+CflEiz5V/Uh5/891OpohpvTjdmdA3q980AsdmA9woTx8Qpv/hzXtLi0qC9KVwVEp8m1QpveTO+v0gZEcQAhZ3lj9GlsNIHQ3tHB8Xg3TqAe2i2YE4vAgxNe29ZWqwKlBfwL2j0eINwt2WsCunr9loc9SgTjcX0egQhRDF27Y6PaNj9Sow7RgnDv8QCziRbA5DZBWRyACQX+Kl/t6GggImVQbEZ5wsjqQyc4xe7oyBae6P/GYxuN35BI3AIOeAvXwtqPKFcmh86EiPddSV0hYPDAIgG32ZPot/J6G9Tt49FtjJ1IrRJwoGblAoP2bxQdEOto0mc1qF6+FYhNpAhwwmAgYitAzGHAnCEHY1Fyj+iYXmoZ1hsY6wjlCU4paD8iI+KJPYC6GfVzaiu0Y95fPEckLLM9AwfZ+wABTRJcNAXz05/DiPCIrsk6k2oA8VJnxZtIJAkqgOR7mByE6Dl4S4UO0wfxVm1i47DcVHT9G8FW9UUuEQ8ZFr8oBpMHWOx7J6An2lMgOhCzJmoLJAPI98854qdLuvLPNOknNi/YflSQgNQktXacOku+QTUY3slHEJ8QlxCPEIcE4VtpshClLaCvlzXABwC88CZkFwMi4A5CNZMUQJo5Ysg07FpHvwM/aybCBiMVe0nOAQ5eR1OD5K+4xu04GI6egh/G6IcmepgRJmApIPgBQCLCttEfmhEp+TsbiATT7+weeA3c70rBmsG7gGUCo+8QhcIamD1i4EjBtg3u9gf3PdzzlaIsLirKdWHDL7o+sUFwRbSUqkln90TOPgiemy0TLWsCQTsiiLiRQ1X+5Ow+NmE4SctKwLIOxOTPQAAw+ZjfJUlACRYkqXu1J5aVjKWoULRzwPjnCxls+GEkawKJUuGtzz4FMa2yBj5f5DLpTi4DjIfP3W1F4+NFIDN7XIriQipJnw6pl9jD2kkG8QKuLRXqiF3DsPqZiA3Tpp3jwgeXQ5s5lAJ/76QfCudTLiZ+paBvTGZlsAudMds/wuBXAwjB9nuBODEqvkr/DGxZ+TeHP3QQyCTww9K49h1M7Hph1c9vzpTxsbzN9ZP6BcLBvC7CRnH3Q+Sk/nfLkaNGr+WQzvFwt8i1D23sOM6d5RT+eaGecPn9B/jIV0zwvMU1FUTxq7WFNfxTaq69B9aefdI0XWBiDXzSso4nXXrjDfPGslhXnmBipvaAidk5LQtIeU/RsrK+SbVMWdjqTXGhc2s3jBAWz+qmoV21CfvwqSkaEKg8fi6y7/4+L2My7Vzm2TyExH0wF7FTXwRiMWKDDty22IaB6RbWnN+UobYUk1AKn1MJzutBSIp+wuEnzrQTuD1oX7Fy4qVIp5UDns65H0j4j/P5mm0y18/iGgSKq842G33dFTKmS7lp4fpSsu1di2D9+wMN5VXUCSa4NgFKUuoSKC0a9OWTsKb2PpzMb9SmHaBCthu8VHIWzisVTkmpAMLmzbG2aZIT/l5XJidKC6KxIuwjSHh+VBdRvEDMsQfpl4YyefHApjBizI645FwTQEU15QKsGigqGgApsFyOFEBbvWJpHJnnKLLfZlSKgdds6Y6Mffz2NzNjn4skJydc4rI5gsnVZr6pdFZ+M/kh/yl3UONj7zRvy6v3R6TAnVPmjIF4b4jChIA6ifj0ASI+rnBERyRNXYkiu+CUILZtmC8nCqaj2Q0o8bMbAdRBWCRsh0laqnLumNATqIYCdeUfpACXje4hGccbjwlE1EEchhuniW02IfbVGSOAVkl7IMEEJQpTbWWaGkDORtBUvLyMYqrfKJ9c0hqxnOj3DSlal9V8NvatDctTJc73OlSQffn9MMREKS+DtCaXsXuKrsswbPO9oKgb/QlftJj6AIdgJkNpUyiLiU1cKJLPonChy9eLilpVkH9RMrvALRrngE8y8M/9oqYUGLM3jeJpCkIaV59ZISWACudrSmEJEYZRRYSlAwRTfnzvvfuaDtILav70Qd0AEHqMNOktwnLtgA9H6dsiZMyvUi3wSdrbqejJQ8asCp+rAD5GBYA3WmBUgM8uHjArHjaDrNKZ8RdY8PAmo3KBtxa5UvMlkE/AQ7MHiVNGd7T5H9dl9k1yLi9CJ+GTbzFfJwjLi2qM0VnzMY/A3okUWXjqYxntrv5iQxtdXmSPbRR/u92CNEACuVxeXP2x3y6ugEThxEwaEq9lQCJAiAmJAPImJF42IREtBKtAIuuin0EirwAkMrQFxPMakChcSoH4Bu9RO94r/kEhjc+AxKTLhERtbNeTkGjADhmo/kyFLfXiMgsXq7DlHvjzTa+oPM9FMHRc94spgSzBZV72ezVcjy3pQ+vWjQJRQ1aNH1Dzc3pUw7oHEiVNu0H+WjQ9kMsrgJZzxfwvyECP8tuA6dAKYqUowIEHKhNdut4t8ooCHIYZTnAJtIAKMgKraC3W+8ThsAI6b1EEKYvRA9Bqxu1GBdQSRmppsvC9R9Mz91JLDH3TOO1ZUfJwsRfAn+qEAKBsGXXiag04C1ifiyMrA5oXiFF94iFqR8DXPBLVgzdQZ2rSEzUJVv97XPcCWR2rkFl9w6A+eTCuN5Xxuv2ylvguhUYMXi+c12aB95p14ul1gllHjg98O10HnQm4Oq+j9sMwHCuo45oVXA2jsqQAMzUqE/1IqHkl+GFC77fySji/VVddvCI5QdQKnTYqoNVpvWYFxLmkYlTIkmtWhCtaRgWB12OIoAZxEz9OU7c0o0Wehd3i9A36aBK4RvQwmBQOBJ1gmsTBWkrONLdFg53JbmElJKfJb0HuCawxGW4LtpnBcSNFQgahY7buFTw3bZLjQEW2OhPAGM4BABbf09JcFnprMkyaFV4uiTNyAoStTdK1NHtD2wIjhTO9KNVncolnKb8Gs/1Gk+/TjNk+mkwOZcx2E1CYZbMtf5Qx27JumH1wtg8ltSZTZEOWs0KExNm+owFG8BkA7FhttsXPZZB52awCd0FjI28neUlk0sTVqD631ZjJK9RdVgY0YIINtlmd0nuMlQWyQBiXMdm/YcyzWNLGPEFIG/MM0R6NcEpgdWNeQyBtzCPPETfmCULamGdf//vGPK4FujNYa2YZ9derun7Vmb7P8BG9srIJyf40qrykCrAwPoFwWTrJXFLM6yhVO08C1rRnGNjIQmi9Rq6/rhxTYmOWo8saxw9gDWM2AH1mIQF1qAnximwrgB7oOQB67mUsB61hRB3I0URGFi/eCjhG6hcaurzoYCPnKIJ2aANoa3WGT1Eu85p+RRAX2kLcYkP2Rkk2biShx9D9GNJzvbCO0eYjss7UXugfdAsbgf5gF8i5B++NFuA8IV9Cy00n9lOpAEUmFcOxtNH4oGcN0NTiIDQGQ0UrEjzeLzbo4VpgeDAn2DD8JZQ3oA9sjenVzHSI3UscD+kN8CdvA3eUuVKaB81ZHn10gM8lNBBMQbFCiUW43QkmKoVF4KEpY7QNrJvI4dkk0oDJCtGscxVmmLEAEEviHxAhjm4jiQWN1Yj6fVkgzJCpWrD260vnpnVmqCYAUTQmJnNRmd+Hnx2Srl8zWQXCjeGAwhvHlpgghWZwPVzIJSn0YKoukpAOMaaEHB1wGMANKAiqgVyYOrnEa0PQUbEHOQZ5toDNQ68iBchqCYgbUaF3qczlBYsa3zdmWGvg9mkSBbiDTGdNElGJYz8EIU3X/2zFKemmlpUuz+ZZ8lEy+5tfo17yWWnltgzgDAFueHcEaC4ExqSN3mLfjakGwZO7znFuDLM6ztpP/8d0voLoan2QnGi1G1zj5v5I+bONopvMXc93TT1syWZtoqWwMsvSK3UY0EhEBmGQQyNgE4dUL4di054K+MLdxMAYmoVlVti0Xda01Jq2W3gLZ/5HjLeYFl8gPr3ivllJmt2HeC+XWQuuvoARy2Xb7o7DlbWZ+0UNWeDtQ1KnyK1UBHG2seQdrZ2c+n0bp4EEWYdAvm4C+DwYfmATiqfHN5ur40mXXHMTRQjnTQiWwNLEjycVLfqtTRCbmQ1qwFoY0gbybD1ELiwAFheRfVjdA9FFZGToKlmceUROx1StyFUr+QhhJTog5eHoYkp3i1eh/7NwcaM+tVF8Zq29bEKLNZEJII4MgckE+kBxeRQUCBQnCognNEMR1w9FJCgOiET3/dyhDrOEcrLWaskNAAmF4jzJDCM2jESoSpKhlktCzAm3rmtBX+jexGgXFzmZa7YBLTHQNhB2sbXblQWsg0MQUuyGtB0Kec/5i/+H8SrJj4SdUeyfS9G1RY0JvZ9Q4MEnwsrAg+UBLU8UT0xpsUutWHShUKfOQhFeod4sgp7JZfDiQfIPyWXQMw8FP8hlVZpZ5B5+ZGroy64QP0piY6hL6NpjLFqaZF3rvcBebMk3OgZTos235KSnBMgITIn04Rswge043U1kKA1tgBcnyLggOeHFCTLeT+ZDz4zZgRfPFAk5w7Mtb6I2q8ZgWmqNKVm4ZMRiGP4Ky9nh6f+sYUSbWDQQHenGh2yacVxrBhc0fVHqNmRNvyBu/EzWO6gFU4b+cwWa79NoN2DzPQhwbc637jbnW3eb800eeKMINwD1EbsQWaYRgBAT3ILbbAxk2RFp5sI5KooWkhbksrd7lInt4vszCOBc0CRF4v37USmK0UIZ8TRPL+6a0BaaCXsmI8xojmjgK72pMuEdkCPTnJthi/yjRbExXMNUGfG7aCNbwTWDwAL+mL2HmuEqUboh3hSnDX/0unYSrqW3VVCePKQlVPmanc9sNDwclh6sN4w63NhTFDBMPdO/MFWOAV6fxSwS60W96GqSDHvUAggM97NYEWQIeJC9AiSCueLntpXBFHmOhFMTRbb2pPrLZmz7MbpLhvtMi6vaWvoLKmJlM0DOmVwOkmMbA1TWF+ANHqYjvOg69K/72eDRVCA8f774OyR8xAb0CvGyF6Cfk/A6hrgNXEfiJBwI5RFGwvdIqPVVyETCSRZajYTzYXKmNT33ApnWkmIyreAlxrRoxIxprQ7EpP4afIXTstl9Ji17gbwjkeYdt9K847dYGWOavrvbcM6qYuQhAZBlLoNQXZWNT5U5nmf9FiwXZ1UtfNXKWAaZdIGmDNssrUpoK4DkZH46GnIxUAvagu3bebxztghpsPXvo7pefXgudU1EeacaVAdtIU6WK+bXHdrChAUe9WHnQWPGsNj3msFFeGwynxEE0H03doAgjHat5fet9o0RJ6SWEihbgXixkABcakYLhUZYxz9PtwH1AdPkMtAF1etrzqbXUZ3CorVF0ZVvDBlgI7PF3ZLCqN1VS/NnQaP0MG/ndoAEaBeURgANGsCZVOoM6eceACUUuDEe5W9h1PERyeLXFDL9Cttj2hSp5tSQTlYWLtLuoLc93iUI3dJ4Hoq7Y7mg9lajUA8TAVjSKwLl8iLqAMJWaSCEw0gCa+r0ea1fZFHADXD16p54yDlKaFYmp6onF08GF90xQnLouD8RtHfqvQqb9k49pXXCpMc4FMH0gwYAIzNDN1DZRl0Y9YLjYZTBd4Sk+2+CkiiXATChFgoq0+DJJaZyXkotxTAUkCmTOlfHMK6bFxu1hRl2A/CAnuP0mBWtxmj2PQSMH5Uy+NOIjc/955JUu9rcA95SuM85zZz7fqIiQPHGSubZ3F/W4o0pFulZEElFjLkH2YV7qNG4wZUfNsdkaavxM1s984Izm2FORyRwwWKwZvoQNHLnnLgRlMOLCr8Z6C9TOdE9v8dQJlkI2hUNNG4efzD+N7SH5c0aVjJTT3GTcXWMIkAwpsvDrEkfZqcFnpXfJ2RcNOzRVBF9h47Ba7K+mOBabEGE2VJXPgcKIZsWr65NYnBhcZjb1S2lNaVZlTXleOeL/g/ztX7fbITct6jpof1CzP+Zx3QckZm2WiYzy53xDUt0G9cAlTWC4HW3U/jtmNCTHbl0X93y51i45Bm6a9v9JNmq4K8UI5f7xWnHiiB/OwBt9KBQoaA3BRf29gxgklf/yXpTJdtueQ1owYcHxz4WhcMJ+SyIV/f3RWLkRcfAUyMIFQNSMwNUedCqx9j24Axy44a466fET7sQHnL6NW6hF1uBLCdmkAthYO0HwXfHFv3ZR0A+h8eyGLOu5i3Ar21QtGeT8z+0N/jBhD7ytgDzs5CcJ0OOtYva7SLALLga+Xor50NmH5BvcWOjeAnVxNAuDnY6WdPRl79TvKQCDcFYbyBvjTVbUc7hgQe5X2I0fA2G/PGIeCLh707pQ062rwEoiD9ty0HbxHrneuvFpFTpdBaN6P50IH27UVQfOCgClM0SNHJ0cmf5iBTt3XI2UATrn9LU2IKWSoFYaunT7gEc7Z7Ra5ESDFmZOQ2guo7bM4TORKngMThwHRSaecCZ0AaVY0vk6xLyQkC8bWPyOajZurXvBRy0NtnWDWgpHRSq590/w6QsfHtltoT6ZskHdUt8EwgR0iBAsCHr1jmkTJeJKLXryb08MnFPQm8/tkREB2hK5BF3juNgyQqVPxnDkHvgnVCUSLeDIiwLKXUwhwZlgJm1VdV4FH2A5ntZMIhRLJNRcHnyf6YTZ9vdu9tWuyf3C4p3NODUekvjsAPokFjQ0I7ZByMSexfvvQHsI/SOaWNLtpC9b91/FOZOyQ/HS3f7VQCgMvnxRnItuGGKAsct4gAsy6kEYNpYDHe65FyLApy9x1b6YBTgI+uMr47Ff6Fs9sPNOE5qMcVW03YWW809Xz0s+f7xr8VPxk9Dl6J/ewR3H5xuFKvCN0w/VXpLCDmA9IVrBBsiEGptQYKOjjnIgqT58+z51tokkm+gTj6UeZKXypOwihVKZPuIHv5FVl3t0Itmva9CAQg7Huk6Ioc3eRCtDf1ZcjboA8eXgPKfj96qUEIbji3Fdvn0xbiWLOuAUYEG5MvtUfXoJx8l0SAKDGB0tyjktCqoJAnCWYAGJZGOAN6C0KaFv7BkoKlpNLEfPb2RxTPD/baDkbny0mk0zGU7XSHd4xjQdAW4ZoMOaOTXPc36UDGIPB6cQ+kX6DEGjTL53fq971bB976ud4hJJctbNHLv4Bdw4xYyGWst5TQ3wk7gzM7H+uw7wvJbsGI8Su3M7fRxT6djWHpQPCz9mo+mQomMpAaOoyDvJMdQOvDBKB3kUny2uaViHW4UYxEKjoBUbV9/JQlTLXbMvd/iu/P+B3EPUMAme7DLG97UHtpQFjtiD179PrzJLTlLJdD9gvHrApQd0x3rgu6m6Q29gJdXlC83ZVsLrsgwcdafUmN19cFUMw3n40lVGSknQoU+f9IoPaUsLAh+JMByDGrILEAWCo9o4fyC6OONYpdSsSkM373yEbFDX1zQxu0YWqFNFAnLoqafCOQ1A34b02GdZmCwfdkTFNGNM8Q/Vekn0r8ZLyM7pPKNuD2WbGqWFw9g2PnGx8Rmt01h2KP4pZystQi2fi1xs6amkBMaNzOp49wDq8YVAMTU1dtWx7R6+4aEdFXc+O6k/CEJEotXCEZzHujt22Pq7boZEFSrgJNov242ofZrAAF42CVUp3rETJFeyJDT/2y5k5fVWssBCZ2+XDKLpZjwP92bCOy4rWYEnv2Gl3uZBmmpXWAWNkOx0v0Uuxi0e3R/lfZkaUfcKAEA8JJgbZZ5iUGPx+SHv1t6Isq5EB3DDUPAei4Oy5W2qm9l9/ou380c5tUWT0XJMTG/JDnzPi+SkBU7n3JlcT7WPefftfAFRtkxP6zd2fsLSGRVGonaiwe+xf7Yg0sfVWnAt7bHFg/APf1fjgK/E0pnja1iKNLB106vr66gy/e9dSWJyKAuJBinScQHCX20eFafdT4LCfrgsXR3y6/SsGM5QiE/3hzg6qTYivTVPTXJqieJv6+1+amaH+aeKaqy7Rtm9ZzSIqartHnDTmwEqlYfaCh5IQNQLqmguzSmUqy6LQoibEl8aIqq1iGMeuhJfEjbBuCuK8ggfr0aQdsJ26vi5oZbR8BwONEeg+UjWjmklWN6clArR7VyWOa4LKXnt4mCOTB0X1AUNB/Z7plbOY+Yy4eGRi4Xsk3Q2LDax2M6aCcOIL9Kdmy+Ewck6eGLQKeM4RFdwd0W285uy4xXtt7XdK03iylEuL7b7n617hG5+xwTFMoBYv/VKEYc8SkgtFhosppzQIBwHyR9cxKOz43Uh1MumgWyWSD+yc1WnAbASlMUovUFLdn4gVbYOjR33/iB1viccsWQHWkqHNOxrweUb23mXDCaiX0Wm9hkYB/ZI7b+VA8TY/clDZem9b7xW8anm7Z6CLTPmWnfWx6RXEl7h7RbVqbAJz/KcLfkflFTCDrRQytqlVUYEBULa2h9xY1EpMlgiMf5mjImQrzdL3ky3ngomfbcE7o1CU/5/PeFhvMt0ZGZ3ekB9pouYTI/TDj+p0ZrvStNFptY9ZM2nTToiqVhfthxPjrq/N8ZrXjad+GNG81vGpvGF7WkytFHOHh32+6R+7n/Q2tbNNRsRVXHIH6JXX16lHf7valOK6o9/0OQnNMjTR4yx9Aufe4zey34h7POJMxdr+6lBcWX3rK19XyZ5b+dZh1byoia411fGk9ltI5tDH4oUywdV9PRAGHYvAx/87Rj2S/VKb0JLQTjqB10raf+QNcNVYxvRWfhv8Y9NGvpnrBXK6DCLWjmL03MjF1m9gcEJJo906bjuayxhUj/8l+8OWZt4OYas8mcW1Oz5l643C96jZA5lGLufc43fv4qfW386lKnDaaTdT9VvQJbX7NtLHTXqeg0lxFsI5oh2hxbGrK+7l3P8UkjbDOfrU8LqAHkB1kd0t23vpP89te664ZRSxBMuxZytUzT1uvZdQ7pDMgz4fvPRTE87cpgkoFjBiS+4Jeg3Yp0TMeQOqCOOYQ3u6TvLN9JPW8vXwA2O6+usH/ktHB6/BPBrwedQT2wpioMcDFT9WSfXva3Qc5jLesEIZO6h4t5tcPpdD3SaLOIQZYttxkzNaj0jjgQ6KxKsmfXFLLAFQOi3OxVK/6afhZ4pjfdFOivpCxBFW6un2EupXSvApvTnVSnawyo2e29YavqYnFXebOKuqiwiGgg4Ee29I3sElah0f8tG2MsLdIDW7rreaGMKFxUngezXpdNHS94Cw375Afmr6Vrl7URW8YqmuHT/4tjF2uikTRFuKCGpaSx8qfnJosPhx//43927LsAoT/mCA1iMGK0wcAK7hzZ3Ptw1//suuf03HM4vD9w0naGXNCGIcvhf/RO0bjf8b859mPXkSSfui4eT6K52JVUGx1jutZymlLdURIqlvPOMaHr0Z08C56uoa8HCtJjLB7Hm/Gx7xxDeDNciwUb2RFsazNmbfeHGbMWmonvzJg1/d/pWbujZ8xaUSSVnjWrT72anrULaqvVnLWSeMBhzpptBEZuzNpEZ6o5PWs62t7Ts5a6Y86a2K1nzNpxZTRj1sQDjmUQk6mbLNNOVtVP/mu3ULHP8cjjD8zRDWrNQjrn4Lj9f2ikac0nU/d5ivbz3zzSDEVpmaq0urL0Xw69aUV8mSq+mjLOdjT9d+zoeqqqmej3GUHrjB99I5o7mimkGO/FRIS8mDdj7GKnSCWm4yYwfsnY4juxzVBh7b+VKyOdEFG8tGBmPOXRUKhUAItTL4lbpjFWja+0B018rBkrRimycv2Mlqzizft1faErh5XLvxzU0tllsgb1+A6jC508bHXdRuelpD5fiGVn0Awcx7IWvkDMbm662Mxw2m76s9KWUoO5v8L9Taw3UqeFk1W+tVHsSG9r/Gt6dTgsfSeyfLjJaXIobwlEzzuGo9fEv65XuR2dW97t2Y8WXlVWk5gnN7ul3vbtdJX4182VtXmoyPqD/x0QpBOYuEoHbd3aGdpn7x8v+itXsGpBGyDhZLv0nw+ZF7NsocI2xkJax5WFBJMRbZ9/ufXV94/eNnNq74iiTp61GO/UUjwvlPFmfTg/p0NDb4CFAgAwMhkku07BVjstAbHS4pIUU3U/9IUyimIdPt1GjwzzstmvWxbh71LsJpBB6J0e1zDWTJ8rpj6qPeuuU/q2XemMLjyU59Xv9vtortwQOvf3xQ1PXXlt2DDP1/s6mVz/wePIcQPImjXTRGEfNxL2Znz+ug16mNzyUZ7hTTJjLS5rreLr08miccnLbB6bJ9SARo6y9Q+k7966K3VkvSadPD3+iXhpfD/jwr9um2ORTzUzVWLj9PHXYya3T2OKUeEKGZBol/EL2KJH4XACxcOxgLgX0KSQ+1AiDd8BMr/2AKQNPy1215sZk8gD7168gRol8dMj+9/+laQUv1AVrhUbw1VPvvlFGaC/7gTxsIS4Uqem9T9SmCZQtaQvVWLhjW71wbDGgiDOb1+b7WObDECoU/tmUbm9oOrhnyXMWAFEGIToPMxGcUTWqwW2o47nDamG4hG2saBTD2/Km9WnNyO96xWxBZUoLrRbJRnElmR3w7V5/t7HtpZ6LFaqciqhEQRWdW36eMQU9M3peaW7BUH3r06bCMcM46i1n8UcrcS8v2JnIXNcI6Iw3j7Gt9Cgc99nfX06abiK0s4iYVV30V/ZSet1b/maVrYbRl1MLaQuM0Dc8HHwB+vrMpOGc4sURObfwk6udHH91TC53hBLP5S1elM24jEIOV0PNj8lYe/yxL729WYcnpFnxK0brTj8y+XtLHPEX0f3GiPWW0U24m9TxohzAyk+Ykcg0MRG/K1vrpiPODF/0Bhx6gIf8XXdGPHehQFzxJvE9IjfHpC+TVsWTBVT7DB20a2k8a9+p+YKC1+Gfe9J695f2cl0lJ7Vpw0lQ7jIllvzd63now/WvyadfPYAhL92ub82OQOTo3icU88KEfz13Uj6V+aDee6t9c8uub++C2Cr4tnXq2zm1lvJub6Lb6LpWm/eYH4y7tT1Jx8xP13roRuj2cB1+tBiSFs53YIrhZu5MaPR2kmg1tBUHcvfVKXBiBpwckprCmkIeKoKzAI+70qNZlMbnVDBdjaE0GoLbVGIH7a3ExOsY5sOo12zbbN98x3me+hduMEkfZ16z0PRA+zaoOvdtPFEYFdiVSwNP7tmo701idubKHBxxOZhE0LHG0ULimbJACKsu+ktZymQOu0rbfLu5T84Jf/yH9wrp9vNT2NY/vF7njD3+1eJeDZcedw9MlecUdHN3F9UMatQmZpOTrAqVJKfBuVPA1jFylF7UC7bjH91DCVx5xHA8uaJuRIhh/ZhI93aVGQMNnMX5p+tGT92PjEH7lWmBWT9VWZw+RKsfMyzYgkEZ9pr9PxjyN1v1Da/C7qqUs1qxXOJw9afNPks1kDCmdiMYex8rikxP1Tnw4mAkaUfqjAWM2U/Vs0KVjMqzOK24pPf+p8MlrvOr4+8k779KzX69/Qf90bV+jTnebtXf5hxBMKJRLIyzXNy+vSejKibjxLz+4kAuUZbcwRbj9ZGHfl/lxT15pt/l+a5VSv7lqZc+Y/WY5BXxz19UV78NN3XUV0f25hutnxB+/aNDPHws0sbzMonYY2d73U0qS/SAQsg7vVbLbdB2itNoPkTr64UaoFASdxu0geBin9bpWEJyIibnciBWePgWZ9VbMW8FudUbYHtAaZz6gw7yXo2i8ioPZsDdDKF1g0UquwTaJPlB3+2q60dhO+RbejRUGOLeqRoR1wuR7W8IfeLKhu605p5lkMGSsVhtrOsJMx2LpVGtZ+stF+Gfb+/wEyo3UYUmF+wNsFvIZYPUYsnMOdEQTR1Fh1lbWzvFXCKNopPqPGqDP8BfSQ3rX520WVJ6+ZrXQ30rUJh53aBdj6u44Z0InZe3CV+nR3otzF9uE/E2MrVo+vGJi5ZpZJjQv2G7e9s1vqLqXQoGS+hPZ+2Ea2R7f6swrOhcJdoxh7RVXeTeo1gUj8v0HlVBhauR/8W3Dil6cmzYv6cFmwa1/Cxty2C9aGR+c+j+12peAmGTDTD4BYGokjwcGc0t3JY8hmw4jFdUfRZYPTGIoIbMO4+m+UWaG874sm9lvx3nPmWCiX8FYgZAw6YzqWk9w4epuam4I/qVLTgwCI7uQf35yJ8XdGA9V1QtWCQsjSSJ245uHmMzBtmgWfXYE7KvBB3kTbyH/LDz/DDZf4Dj4OwXAdgCSMmNWvqoxkNhEzAjWsIIn2YN6Ma1UcvxqnEb3YqCCw5HbD0iToLpjwZPVAZBXBw4SlLBCwuEmCuwXJgHkr/EG373oybvOUWJs5s5JjUDHgwsYtce1p8Eee9NLF0CrOL9sKkocexGiSkCgXjSzqwvWk9XO9VEHtxF2fiG5lt7yyw1sEiwLMnhdxJPLYP22N59zHfO9u3mc4B6EcZLn1pY4cTpi8eTF4EenjOJ+twjR9mAyTDEMS1mAen0w2vgnuqND/0zlMm+2Glm/HroO/gtrvnWLinV6hQGljiigahTqNtyldTXzp75M/seFaAsc0h9/zUQv3VaLi2EOlRYE1RDD0F+64nJ2o8ysBnb1HOWLIBrxE31mt1zi7ZhZPXtnhAHKC9MfiFT3KRsNTkbR6+rxdeFgMkxSCaVAqllIKodq21tTaFp4ymzno8GCmDp3At/UwaL0uE9CtNmb0nKlNOzsosj+dgEtNHJY96PKeWFg9g6qSlJQ2NjZt/waQhPgq/sLS3GwyE9eWdcV25QgY31E89cvUDFF+vLGBgoj9orwaA3rNAhoLO+M6u+A77OPPz75rXtR/F5rltRuLljeLRhUt0jhnPceqzHk7wzQti61xx9ujkvkmKai1PYHJqPzaPGAKKkvE3ixdhAmOh2HOchVKbiHwLVTzGV42pLHQSD6BrY4LEOa1vywSTzQHtVIqqAi4/oqeuiXZnNu31VROHO+HuKi20sUmNS4sHaKs/zj3M0/aYHvYvMPn8SASnbJidMmbnoUoAXQBZ1Zhsofkgnq+Z13WnAPeeDGJyOUTbMhmxrI4MRl5SRfxp5gLA4hGM39kRVjviqBcAELfTsCIFJ3x1InQEBgc/evA2mKQj2CqezIMNUTzyt6xEv9GRV+2gdTmmU9WgYvFjIOirtuffE9U/2yyXpJSu9uYKeT+p+ky5KPZEf5p5MJsvvP2WJ7iOUrPajWPgzPzvPEyCF9Y7S6ZiB2CV/UbShHpff2JpgFSxlTfDkCpZXhAPECa/dm5NYL6wIDqcVXCnimk7XH+RnLwmrPuMkBlmshuPWEVkZuyPYkTbyO6dPUq7nHbEaWznNGIRXkqusiFEXskymSCsjh1n20/pezaFKflshULJaD3U0htN3ndJ2lr9rav/wfZPGFs3OyL0Aei4ay3LhFUkwTBUAccq4GAFHK0p17EM5dt/VmH1xfJxTVfbxAtRdnYXBmAfwFdn39f0BZbBxNkN1HeM0CCnfsHYim852+IxhIhGZsW7mxb6qpjSVy+Y95QydfGk5aChhHi0sf30xCWQRmbcCN900hZtZRaEv42SGIbJULTx9e/6VXZ4l0eXb1G3aq8aUiUyXGCGgDWJBU09V61POGDGqJ/AvpFO5vTigbUY5BPZT+vGIjG23WXqu+XAVu6uZIfzulS+0XoNpyHi7u4ZSRp2s2ouED6UbdgJXYDNmLVEsNYl8R0ewFrb53UHPqw8npx2ADSUYI4RVQ6u6025DiwOZ7WrrUUNKuZEBuzttAeBe+HheGUyZk4h1SNrOGh3pTwVij9rMP6BUAJQ1pRy7YiHNrhURVIUNaCE8rIGQRA4MRntKrwn9W0mTZTlt+eZ68/KuvLI5wtInNA7u5MU1ZrWQKwFaWkf5s7UTepMMQoV9cEUZ/474gl3nDv76vTJd0aZYISpxEEtpmPwShPAuT5dJDwwzvPtFVHafgRL34ummb1JcqfQVxMzUKAwLhISKjONVjOzTzuT3UBKn1zLpLhqgHinpHhqbkigI7H71B4n3vcZHgQRzmf3FcW0KlYql7GkLeAGPfwrBU4NbiHbscxFGZZGB/kdcG/pQ9H2I0DB9hjJPo4R6VyFEisyNW8zYsuVMn/dEachwb21WX14xLClOQVkE48dW1BI22dCTYzLazyhLvZGWxg7azHTeKPQTv3OviobIi4S9WkHLyJR58W/xdjpkjzfNS9eg3Vgs1aVhDU0pnLIwYqUENH81Z1R4IufUYBpxvza/J6gPeOXJ2/GPC2S5FrZMnbRLDz5lFngJIK30y385s0mnK7rSvIlLGJH2QGjNZcwsYcvob3dd2LlEuZPos+BlhCKoKXxJQRwb4YljO7HmDkAdVrCRZ00HFpCrsQYWgzvDDvl83eLqYdAEQeWzElh00Kzej35TC0sKz7rf84TuEnjDxe4qBu0N+iYzkcfEPck9OikhpOLByTT2ewG12bvQ/mmmIpWHx6yjkXMh3SU3XBBJbclFBuSLAMBzM0iz+4Pc6MNyxQfB+DQ/lYrnQdAjYHiNcQRBI+su8KLeFz5B88CMX+wQLA4yYkGz23P+LxRQ26OBxxjLaUg1TuTSip0gvIjoHugFfnpoEP55lKkoCAiN7KaK5RyuVLBT1mtrF3v7ERWwGrHliam0ucwo2V7zGbU2N5is7Y3CTImq+HEkGwHRZw7XsT4d1aUA8xOjsWLsaWT/AZ3szZRxIuOCa2FF2F6pVleFHu0FC+ixm4U8UABTyYOmkgI9M7EQuglR0Mg8qjqMDz0ACE3aGm1jszO+D1o56gIjNuLQbZzJYLlpoqhuZn0lGNjrUXIREdGSRYnPxdNitrMVtTWp6MSg2t7RDbQ4kwK6Sbg301Mr8mfCBdz2ldm+b1zZl/W6C3t4ePm6IvnVG310Ssad/nD6K0+Qgcc/cxFMzke3Myw8BYo26zfZebor2jISwDI4xh7y5+4n0PMtHgc45/pPhfaY1grQwBlBvaB5r9MYUwkDIVRFA2FkW1IQYVx/MqTCuMZv6EwCoKhMAqCoTDG408qjCxkz2NaejyZNp90Pr8XWuKrbeLccxZJMsL4GTzmLFfIDWAye4q3UlkA+TIPezYLTvSwRA0gaA1mcfFpRxzACtpAr04dpRdowpuAf/WTnwtuAsg7x1har3hOY56gOhCC+rkraHtMORfioS0gLQ/Jhhv7nJbOE1E4zzPudJKA10wRl+c0dKiDAsTz4HbS0Wp6NW7mAQq5JgAX+BMCfHRxNBtA2sXOzWjWu8KxIna6BmiSu4ZTnw+nXG3kptsUTtWG5wsb9P8Mc5VJ7wNh+Yj862OW+rROT2iY0S6psBM8ShMYuIppg/uNsyTQzFFHR9IguuCu1EaQA4QLSD72JCj3p0+feGeI7YQtntMGuO6OcUtGYL9jSp8vNsItdZaIUDiaVDuWiGsBE7pv7UELFPAduOaHQd7/nJ03fk5NXbpChkPiTwDHQJat9DrCGMClcrJ2BvNH2AaDbdHUvRSPCm/UzD7gBlmHuTEgI1PobxQNky47KERPH0NUPMejzYEtM4oBCMv1m3KZ+4lhGCxSM+u+cTjMBW7tFbbO6YssvW/6HfU+ULf9OgXye7ht4dkvj/mzlpPsMPFOlnvElfIyoCNlImuY4WIzC1KsUCjNyKYwM2b1G1DhEbh3kOaNRQR7WXtHfBcx/99Bt7F5+o9dDvYn2JldwO4+9t0QiY6nCGLOKE0xgC/byLSjPV4Cq9sE3LhDNKVd7v68u+2NG/dKEXz6sgTxu8cldGQoaV1GNsA9CUoPCJp9NkhQbajTv38fI7CD9kNJbeSbf5MINiASd7isSTd+QLJJmRZ/ZCevMXPJ01gVi9mE0h2F8pZCF5PtD1B6A+J/GXnMcBaeUM8MPE40RBGuwn1NCLuomWnqCNl6oZVvErjxhKkxGw1ZmOW0pRRb0YOoZIcbHoEyVqEs7BTyBrreMZNYutNH9AAmFMyEdmN6O38Yt+HkeAH+WVZIw/wqGFYrs4AMBQuHEyAXjeievJAutYyDJEKm5I2PUY7CV/gzChUKoP+2KOjW47o8PIcOd+iq1ktWLWG93dzlQhNJo5gqwlGMftmvjdsrlMcbhQ0hpcKdPsaN5YUoPLY0t1WwVr6XDfMk/TyGSEUHS1O6YOirEWn2cc0udjI1kaoqLd5YCu+7qk+V2Pl2Ho+QVdkFzZZmpPx81m+cEwyvPCitNidQ87A7aU4wYHJA9xxbWjpFc4L7oeKLuNEypysaufQAM6FuEOmezbNaXw2miCQTpL5wVjRnaXW5mUCNQ9pPVgQ0/5S+dAy43hlcGqZlImI9S0F0nbGgE8TWfc+Bdvvf1XLwAYHcJvgIc13RH57hsiMulx0MAwd1I+99ilaAh/1iAUA2gAr54RaMpcXxqkyz0dnJrlBIEKGHP53w4R4xfAoYUg+Txs6B+vNH9DeUezb+I1uwvHuIlHepTdXn/j4Gasc+VJzCfnjf1AMZ+C0evc4Or2oDuYJeViYH7UQ6/mwN384XVvqWL2wq9hwLuzrBaAqArlKhMIBtMADWayCxl8Sxgoh2mRMMmFVGMLalCQZL4fsyyB0tA5G71fUAAwVfKFlaTvl4AZbjjY0boZNSj4a91pRH/1kDC3FpuTXiaV8UhokzJF7FPMmDDuCcakvllN6OJPljMdvODidQpulEjJqBBYyv2nzxqig262ijVL31YfiF3N38cOT13E7PQ7ypt+LGIb2Dp8WRnFmVE/oAWuH5SFb96tMlb54g7IvylPdrrNz9vNGZ/YcAjGajJK678YCdWGIBA4JmYRkfkNr6NvdC0yxQJtN2lMFBuG5HqyRARmkCXTcgXDcAjFBYVZ0+ah9mifS3x+Kn40ywEvu1+yl+rNyZlG5EiW8Ks0MU4RV2nJRsVuoEHp6NJlPQkc4BkoY37YgDswedcDsIPKj+LMZ3NKAoDCJtXWfiTsrVT4JPnd43Gjq12B0ih1JpIprw2oOSbBwQD6/tNJKmedCe6kqhZEQjA4qVvQd9naDlO+Aysi+OqqRjIjXLxM9/KnQOK0m6tGdnTyJ2CzM0AyZ47EEQRy+lwvkgt8ZLnpgxkPCXzxiMrgFwiWbssja6fpR5j2HG3Ak+Y73aA0O1OJPSjVz0lJtNMDIisJyb4n9GNL7tFlSLpOG9sOaKZuZviiXmEpcHBa6RFO3Q52Ku2MoE2QsqydcgwV9gMQeY6skIgEDJOcftRQfaJMl5rlR0r4i81c9YrJ+ldIOv/CkNXYFAY/8NI7TsmmBxeNXL7nSz4ZcmtMmD6CiM6W7UcN10BOLz/cf83we97hwWf4HeSXH3HDvNwc/cjvyrj+XjOZzQpuAm278VtDS061P7LHhLdfpOWEH4OhjUcPXcPJyg+LE+tU28O5kv9upJNTmihDbYg6ENR6K6ejXTbcBd3gAI3YbLW2jEOG4POlw8yz3laRd6va8+MxLt/Z75vnJ2CCeZK7bOaV1DLWqitEqLlPi3ljktMHQn3Vylhbf7t+7b9AYotv8ZuIgxfI2qYXEpfyCPn7Vktoy2BhoRkDMlTvmTKTzgivboTSY6Wd8EePsyNVVCgO024SH7Mzn8ARPSDywa9v5yedgmbLvbtf60ihE0CVoUpmWdEoQ3eORRtvBW/6O2mK7+fDN+HAaleGtuSvqDHOHMEtMdB7OEPnbI5Ln4jjWBO0wYlsrMX2FUlxfLCYpHbOIlVW4tPhMCQLb6/LeBo001Pj6Ihz9QqEkWqpLQItkB6YIJ+cmHWxChzezHliLNGkGkbUSjM3lgDECTlwaqsAT6zbhvWWnfbEzVEzwcP2iExfGSB00u6X9mNPyK32nqcVbw7J8hhTZ/A2n++sdCjEL3MU/b9lldG4eVL4rcLQP23AzzD4j/QTxoL4poD3bvjAN92jWm6albtKp5Pyigd9ncjTaCXRjN2AMZfVGYbZspJOJG2qUuHvhILHbZCqKLJwK623r9e+un3UXcVKdHL1JrJXN68peAgm4RDDS6WXT+9P8NIEW8rHVcAL4oVUzAOvgTH4cDuaXR/it63ZqA5JarGYZ7BH+8POy3FU/eqgam0KB7EHqTJ0ZgAfLDysWdPzbv6NbQbgDiQPTnX4zTVrdkBLYImXt5UeCmU0lgIPsn6oseAAPc2vAz8LNzyuQ3A3JZd2KfDZTho0ldmUXvEJ94R0BrvYRKOshEvhPofATFPvD3OYT1w/LikX46jS79W/q+9LON/OAeT2bMkD/z+0RIY6lwLiSuOxytCxVC/uNLjVLo+JUpR8XEBo/nor/gcVkVaKmxRPWG8bLSyF5Pa0d72ex2sSdQbM/eO9kqCjn9S4GAQkeuiNvOn99q5OpzZ1WiIiXsmkqB6ueYjg8kR2xlMNMeyVk4P/rRBDCGaoXRf24NM/z5hoffdP6b4QBmgEBGyAB9loUM0KclW3j/jq9tHTA+6aEy7YB3WB+igWGdX2kShSPRmcNIBbQFmCI8eRUDlZ23fLecsNLJR3I4H5S8nAfI3VwLi03WwppSoV2rKoqocS15SMi56L0CIo2t1bsTUPF+lqUTBmkKNUJaJH72EuBwQUQn56c6KqX0RRf89UHi0T8jGV681aXUdV2SRgkKQb5TY6rkxLPFgk6YoW91XjB/QV2VCsLmvkVWWC0UY2VhlViFrBO5woZfq62WXdMNrtiiW9jSeXfb3xMpuezdCjxqhqUFT19K0SIk5lr52TtaXw5IphP7OrR2EFKqQLoCQAPBAb+PLQ0Vf5PqpjM/UJRFEFlP6K3W+pip62gyeTrAbOYX1LlP5igju6VP769OJfYav13AOB9+HzzwL/YsCOMkl5Yn4O+feRc8FM2DaG0pxP32YgdFCMEMQ8UluwSjUhThSdihknV/woGyFDOoxY8KrsUlqhRHWsS8kHeCMvX1A4FonNsdxMqVxCGGGjAZH8QaSY48EOuwPld3SGzi3QF0iu7j3cluqSuKeHh3eDwU6w7G6eJP2B2MJegpWs+6I1AIJR5/kLbRCI4RTetEBlaCtxz07qtHqLCNkJ01uzWKiN8cI3EuyzU6v0s4tEAmSZe37N9aFQZoUTzohKb1rWmJFANb49ZNHG3ZYsrP8/Xdz7IODU352V4rGG917MMg1WjAfh7JXTzXIbIwoSHso+3DHCOT8UPW5279AU9uvIE7kBoZy0O5l4yN26ZYbHt+WB/ONd39GFqGAa6WWyCnIxrRCVjCuh8jEp5/BZ+Lvo/NSaEDl4zP+ZmDq+Wj/svDLlj83b2mN8y80ObefWHH9Hfi69rnTzE6jX3OLgZknrbkg8d33/pu7tTr2eecT/JyPuOZpwaX5Oko25yzvqP0cPeJ13OexY8OiCWfMi/jroS0M8D0DcH6/fdvnW93vKbz3CMNGhsdqh4X+3n4tlA59/mEsSVGPO+7YXud4HlvclkAe7s+P5+uot3Ln87ifVmLFKSroD23A+kbYMIJRiDs5VHuMCrU0LY6h4y/Gm/hz7br4dqamgleBbmiicg7OXsam5j/yG8+8RdGTK0dv2mKze8GjY2s+8KLTEFx+c4ZGV3D+cbO17PL91C99JAn/PVdb9vbjDlxt1rhZKf4yrtx6F4dP09zTsHQ4w8eD2/4bu7Sq+0G18PwJG/SxHDDvaGLvbpuvNOFVKoZ9/bbhwbezpCJ//gl62bv+4BB/VahzThX5E+2BHjfTBNh9cmxcLFgaY+eL54c2pg5NWxIL/Z7U9U6m6sYxB9PlRZYs9+7f3Pnpytf+WK+KEpDiXY7M3emnOUxLmLJIzI7sjAztPfjuchTJaAXeWmvLqNdeGCY8XunEYRG9K6Bl/PuaUijPBkJWqmsjLBDbmz/3Gr7fDPbYaSqU7FStmf3yPt3zosrtlj+ZQFUwrH7N8zO2MczDiAqnalfRSp4vQKueGTxLTNzS1M6s519+K71v6H/u7zfG/0sjU4nfrLyTLDDjvNz117//ls/Lsmq5Pu82xVlKsFo4ZvNVVs6z4uvf/9LKiybzuTwfPNqZR/f0ijWtTvKu3e89v3PMi4HI9XsIjapSvVBiY6SbFJVhW3FFJt1/YGVpVzktAru6W6AewXxmvKTlS6CeARdF0fSiGNdnx71QHy3OWr9vjHqzWHdGLXYEZ8zRu2SLxijLp7zWfmod/iAovFR+xdPmqPWb4l81DsX0qNuXBw3R20rzhj1Gzd8R4we7kmk8+Z/HV6p5r4OAXQCCwb6yswWyz1LsAThS69pfz/1nebIe0GJKCzNU2Xkbs7dFert69FfS8U+YV+FhYX2zO6JMheuo73OUtea818ND39ZPN/q1lwUcAd389IiJpZksSndFNeA97mNEI6cn/VBQ2xOHGZPTN/Me4p1OKcH3Usn2BbbhUW2D1y7QvFDXP4d1lk0deltlaKkcatxqm8L++3LUT29/UapMLbfsFI3rEYoD0t+Xf1liyC83cZ2vWHMJM/VK3aOG5tvX06EHL3Nd54rzVxRdEVOveSXinX3WL6DFmWOXC/X5r/bMDz9/kt9qfOftkIMrM4PK1IIrZ7b2t+50L3tpY7U+mkx2ovQUxs4kkBe+07np9bvSbh6eSO1uJqYgaE5MixRMK+/be0Emanci4lxwjrrzgMbX87baVyW/IzBPYiy88r5O+jSzl7kQYPQ3mR4k7B7ZME4ES9ReiipPqgbIBNNzRe59N42JDSY4oG/XJv0hfW+LOyB2vOPKY2yE0GjFHmTbWTnfuFxmM7pGibvWA5FWCYAV1z+ZBn7TzN/g/W/xEBKk2abFNug1y/xpQYjNthwmgm/tJea3NTkpWlO+rJeWnCn3S6IrSgKO4Z7LH8sfnIVD+ez20Ic7WQLcQxJ4YnNgqU9fLh4sv/t9FHWz2nzOJODu6VBrfhaf/Tm/u4dRd2fmm3/WfOG7SQsxZuYNT2i3HsQQlzcMaNLeI7K32ZbvpTRgfbGR3duNac6RLFZ0+KLS/OF22OAa0eTKZebh0/BAmFWMr/urlBggQoichmIGBFA9X6rsPmhpvMM34bUYpgzTGOGYcp4CTGa532fGklajNissxHXU+nMiwjfzPREmW4owwf1El/6dHPpy3up4QsyHUFpL9BLe6npzDE9OWk3zssbqXie7UrHKArmPHKM0YF4vxWTIVivc7XBfMre/6szDX7vbmZdjccuLOuqYzxFXX2nzzeIXQ0NDTmoq3rAQV19wLt6QhplXRU3mV19qhvh5cWh/obT4KW+dP+9q1Q8nJhTUNEvmQCYn7/2Ml/6dIfAy51eEBpuvGEkQU0e4j/unvl21Tjgvzaq9dVu4vm9qHbxo31onWNoGN0zq1Yy3+/X1jokGq/Z3BmZKUsN54/H/GVF0sonMlby3C+ZdwhCJ2gwjmkgemUycOkGvVN4owlDJI/IIMC7McWKB3eZ9cV3wB9uohAMT5QmGuihBqBa8JwVt7DBo4XzkhOf3n+/1YYNvDupVGAbRTFoC5opk/tFbMkNLWJjLnScYDBV+kJhVekLC7BKXyj+Nn2xXCD/Nm5g7BHXBIi8UhI00AVBsphCgcstZWaD9CyrLU+Q6HkymWTHykySlrSp2THaZFQcAV3lFUdAVVIqBWpg8KH0IIAVKLYfk/3Fc1A5mvjJiumxiufGHMJWq4BlYa2NKyrpDIzPUjag2b2i7knvH34i8ePyJI/Ojj/wOrO8Nsvs9uGFFhHLRaE+2+YJreWw7McvEQ+fuCkNjXO78oSqJFItbI6KQtzazCot4mrTy+qHK8xsh//votJrHsG9KzxlZvB8u2fpnHng+6mkz7S55w5Lpp1WPCu343kIlprhk6JjMEIJcCznk/P/+Gey6y1m7a1Xph4mPjJyHAMcPTSTK26ZTF1P09qiw2ZQiLP53sUcEqvYrnrHtHrp7U5A9goFCMKOeLTAjZHiGBvsxgSDrOBnCZRKE5EC4eTStCPrvj5zK452tJw+zcg1wXFePFBTmZXTq2EstnJfDtoPlNK2JeEZrmLZnTuVFiGrV6M0uGXyZG7uZMqFAZxGFq/iOdobAQtEGwdCFDtYwiJp343qsyCqVPFtBB5B3PWj96yNndtI53CUJuZKbH69joXL86xRLUMK5cIH/qXcZ6FAllswJQdxd+k/lcCaPQm1D4bbzOyWfLjZZpqefTUs9nVblPkysu5rlLRScKn9E1jIC0VKMRATT5iiiMwyefxvJVOxv1vdxs7dQqBvVyLJa7lm1CbF4ELLzPOKZyvimWD5FqfkpxjvA4uLCuZW2dY/cqA8ysNgGjDx2Dn0oDTsScjNP8hAm7HPkSPpXLhCXj2DAdwXhTBgG1NrN3RoXujW8Fv5KPy59XshXR/DJIjqzdJfgJS5cQ8MJtYqkwFO9iYDa6xD2mBDGGEAtdZlMFDKybXHOHwbJMpPF4l3iD0sneGeBGUz/J1qL899COt4rwbXEbS2JexyQTS603Jbb4ZCaIPQhL7nhqXJBzJ0WbwUU2V1IBc038lc8RsNUxH6kUEE7TglX06gTRjnhG+UyuaJzcrueF2AMlldCCpDdnh8eoutD3gjxpbT+Ku0IcwwGrQD/YzORBBg1/0403l6cThrP7qLm4An5oe1kSEF93FjDHyE4TNLoZ9V+UUuopPxXxwANoTB3H5ml88eSh6yjcBCWt4Wyr8orKrM8kv7w0G7X3ovGFjTBle/UFq1tpJUsOIqW+4X5F9sC22oSFZQJsHhrDrcac3BXz4iFDYmuy5OTZyFgR5ZqFg7ifFR3QltIa4t9lqyW2YT/SPKw3Ps0WrapI33tIUYHclv4b7A9tn6PIEfF5U4DEgknf2pMTWcVSY/3t+BEfOgIC/Nnsnmyf95cj7n3VYV89N0OjuKYtGCPQktOBtGJomTrw2RWflv3ZxKWZjlK+emF5/NqUI7dHd8PP4+4C6aC+1BD7KNo3OHSProO7ngcgSZHf9wQrv/VkfGscknIyXrLp5nx33nP75kH49I9Nes+z7rgXgF47AXZouFHdKZaxhWXjx7gRaepwRkAYCEK+/fueVkqcd0dtZr38LOfypIiDrlE52EdHsSKWKmuHdvwlF1d1ulGrit9Q1KTo9+EQnPw+xKVbpxW9WvC2KjrGujoSQQbcsmQpjyL3L9khP+tYGYgsljeyhJBODKZUxf5wWJZU0AYGTtJPR6Qwgu9iCCSx8qYJcBZ0HXwxQRX8+fvax5sAayi9iPIlsd0u8KBcQkoq10OSIDGtdhJohBkHXoPbVRismyuZD0nDg0eeJf1jW9S7ISzBf3T+rx8xZrb2ewdSQfk/vuY2IIF0bpO7vFN5566Kvnxxf6V/4dBUMKHv8IoR9TkHiFA8lzwmZ3rrDtLm7G3zeLWx1sI4k9+J9nQN4QotQBZTJJOA2UViBrmLbwbQpTcB5P0Enh6QARBLOYuBP9CalzLCElQrTlJJrchRXtc4Nl+res+8onWfdBmBbTxxCt1mf+JKFeXhd27/0Hmh45vn5cU/27UXxP3dtKuelAEwhRajq4jQfz2a7W8nnywLzhpyqdd3r579s5YvgX6olMfbWgKxS4Yzk+pmn3s/END4EWYSI+/INfZze2nOUP5lzBgykxtEVPHm3Qb71xU/dimstrKCRbbiqJAyjcT2DWNQ/L2sAH3b2E8n7WA+1iBmJCB4/WnM4XzV1chv+ueE75YYvFfjKECXX7kketH3/1QeU/hue2liZ61zzA0+U8x5bqqlJVLqVWPLUEWOwFuPhawvxwUOlzKxVu0Ep2LniKYs0VStPWuYgvknrUv7AHvUG7gtEfTodmTuOmgBkMH7IwpRxwLSFJCrdy51z0zTz6cRcPm65h2StQR2G7Q4FdRZIETkCiW4K0Y9M2Jpd1kg6zN4kHcaE25NEVKa6RIosOvUXafosg50pNO9rpyJNN4aVjtOPktMOgs8mjUJgqKboNchPhmCuFhciVNmn2PUpFGk+FP2F/OpgMb6JCQSSxq0xO9PvCtH8XQ9YfoYHGUnnnzimrGUURKkI98VFp8YDeh9R2I1DVqcl8t95nagRtRkythnvV0OZIFk2hXJ52UD7gIQVlo2WpiZkcQOP42GomKUmcEmgkxbSri0llTTQHWDoij2anJTVWcqX87K+gL1zGhHf9jxR91M7ai31r4++ABWFvwCQoKMQUWNnzlCSvH21L7O9Gu4ZMaPz97jbjedZelY3V8TBqJhSl78f6ej4iP61DTU258GS7Ri759IxPpWf8l/dgxgdhxlEMyevqWr98xhFX5TKc8azKK5HFA2zGS+IAKzjjv8goKrIZ56EpxowjKcLryplMl5wSlnDUA0scNDJmZrUnShOZzxqt/95/1p/yDDXpaV94+gJ8YWLUI7L/TEq6GdKbMIs67gJim4LYwPbVuN6kps/KhBRB+1Ha1galvUxAL5OnH+pdogFytJmjTNZaO2lvD4j7a/OQjnA8wk10TC3o0fut7B3CU5fy97+XgxqAEppdUcwuxHMzyyjmHM/QTCrtwD21FJYDEdB2huMhLE88XDpWmhgdwnLwhtZZpzf4qfz+/OhAopSV1/UuxkazWdl+JgWTxsubwrHtWM79AumTrYqAmPJ50ZnqSyd5aVaacLBSsFNr5qV3w4mKEJXs59ShKVay9qcUVkKLGy8JjZQc/ylg/d4vOu7kI7AOoWbXx1LnuTkhAb2BERJxQKtiYH00Cf1uJEKSBusnCUlltkFHQIklMuIjDIIuIYWlAupSRCVUUBvwZnX2oiWNi08Wnncc7xS8xcdRPDdtjCOom+NQeAiP3W69+/gDRthzTHp4mBH2Do1Tvp0Ls3lYEGtTOA7aaHSB4CoCsh31+uQSdAkLWX00RJK7MPKkpMW3Uob+x1tcht4ocBk66V8pQwsbuQyN8RskQ4/4VsrQzN/ME/miq5qrlKB1DelX2SnRfsOk8icKDPEKa8rZsoA29YcKO+Is/bs36z4MMEg7MHtUI/qHZV39R80+UpBqCi9ryKjhmR2F82g5vQyrKPaCfpc1iGbKMyk8bwatlMBBS8+xg1e9ep2X2TBdqUdTLKfahpBiZIFo1ynvVDkaESixna3K7Whg++HnkbQhdIY2gLgAYlCd3i9uCCVKC+ejINv1oQDh1atdqZ+iw++EsEPVurdZ+edw9FSC0d7Rh+FNHrmdck3bgwvzw1kVyiyzmHZihvhNYZ0FNdKZOlnDaJIVNlejSlZc89lboEW3iI4AxueWxJOHmln2wCva5AJL8wXQzk8TBM2Te9/KZVmlwCUgllWUVwCmtkfcFlVdICYhTlxKhTf/o+YjERfP7diTAH4OK9AjQhu94p4EGmzzw/LhHhYJVZ2KyuxsAXwbN/ZWKDwYIQMbnygVMqDDVAqcTeF5cn97yLoMyg/bQng0yTw3gKrlLJsZt6ddUNkm7OK51AgdRuQ2TL/IYt32YPJ6IRlDnumyNHBwzaAkubiewLUG6GszgwjazC/286R7bI1KE52ZZ8t0UgJxWCvaVV4U8zLFnQYYtBd6FEOR+v3LwYepoN0l0cljLBlLQYRCWpsij0AdzQ837Y27hLPTxVn3eyzM9C2XMYv3F7lW39xWmJUmntcBua0+VfRhYnQf2mmE7KLvUv7Nv2raouZv1r8H8Qe+br87oPuLH+taTwxY9/rsNIM3DaHwTdGiZfLIGLosgPf23JBSZ8ioU4ceOX8aeZuwnuprTqouuB8g43taMwdLVreRzCaYajZ6EBin3Nwsz+ahZim6Dls9XN+pw3yrX36RUxRR2ppRID0YMYwi603zGTtJCrHRTWEz8I17h8M+3Y0Gm2ZZd9exvK1uDzvyyaxvj6m3by9hwv4k3c937af38Z8Cjj5bKmydi1ZeSgbtdXoHEMXFA4xGplyUpy4bc+klbnwlzRVSTi1KH270z3EvObmJkj22Z3OCSJoaUPjKCb0LKePyPB+rfrFgINRalo/KKzwxqn7dvXVOGxlR4MlDk5K/C74LdiExccMsDo1gmMiW3mik0/k7q9sJrEXpbJYXD2DSe3EApRDK9/HU712x3qxS+G+0+bT/At4oGHfy9//2BQRHwaUAuTnou2BZreM4PQjGRBLg+2iShL+ZKBMgdR1pFtS9eLwx/N0+Dkwc79+TSB6F9qD4tH/8DBjiNDA1AT1+OoSG0zJ5qcWrLd6+rUYLymR01U2uRTMLPoKSNFKVZykbged/dtqXI9MRjkxzWwmZevVqDh4eBvTauaKI3NymAEV9EpmeC7nx7w5relW23TV2unDwfR6k9HCZwviuwuMxBOsmWPPJWZxP7eG19dRT/v4nr0YiAczJgfGCrtRo8TeJm5i1yY2GCett3Jt4EhB0zZYOkBMua4/2fZcazMLfWegcPwpozeBkvmAwDUp38XkEGA3PmrD8utSKp+IoesPvraKx2ttj2sgwur2vKNoIIK/YRTKEdkU4tjSctSE0X0j5MCsU9K826XWdfPfpwwmeXpO0D3LNUhptWHA8VwT0exBnzgF4oQTRiW1ujwXem5+N76ZDL7KGY7XJ5sRX82TnaFDGwu+FHjLjGaxzv+jlBwzCgOugGXzXqaVxe7kccIg9Wv0IohkAlorpRsVuPfEByRkzcr9VsI/LH/ap50B27hBL4nPFbESdfESk1uCIXCk+Inbgxo74yGOy+GcNpjQPT2HLTl8xk4YGFSZtbOvzf2wqhboH5PDaWdzCL7aQC+6CitvAiCKgPSvo89Ub+r1QFMTGvVo/zIGf7H1+inENRQvwnKYLorBvLtVrRYMPYaqfXakjTfrCefjDjelGgEYPAvKz/nNrU/Cqwn5cE6IDxnVTOLqTMBlTHzZr0X3MQcH/WhQLrBHKougZds18n5NxlBIe89W97iFqiVMeGU9ZsFWVCu/ewXibbNANgmplVs/MLhIk0F3gZ+6DTHcCuhdQJGPM1NHWhsEATSrtatmbjP7QLmEe3za3ez2mEPS43bQig9XVd735joA2N4eUdXOfsthHWoWb4amt1ncTLel5U1xfb9Bvv1GLGWERk63wjsIBvYvSRbsSeoIdne42FIhqzIdcVZlVJsV/9g2rfu7Bjh0WHL/U/hiJfHvFix6VUF39TOqmKPbgOjsmmgQeKlw3mr1phiCkGvjtIOaPsPYDz2xUWx0F99TLQt5YD9DA0P0xO7pi7dmUhOkkyBNZlW60oLfDyIGk5IUiBe0k0jVpzSxybutcKoW+JZek6BqeX5TtvKoPOe2Gk92TkT7MKJRX2Zitnv1jf6rvWm/+0g10xVZzr3kTxesvakCjndJDKx4/6phOXiQiWr1E2YZcC3vff6D1iFXqv62WwdFsa68mj0zuZ9ZiaYdg+/wTUDXcmNDGH0HU3RVcnAOt+0D0ujtWzq64PDX3vKeZQftgVFdUPVJKjsiZsqOHXRJ6LLzxypCuBz6TYVYn+1tD7xRPfj06ml0Q7ewbEIEg+kFmHx27LN++nTz3GHSiy+r1y5FIbRseo2n1aeN1deOprix+2ld6q5XBcsRyr7dcPLN4u18+JhS476TcjUsPflDcfeH936Tcu4LRH6dciLuB00kQ9Yvn5j6ZAygqnkteorD7krk51xxpOsYtiO59V9Q+m99Maex7zHevsi87nQvXzMZ9dv7Wl4MTm4WtU5913TvZ0i26Bwr9n478fXyTu885PXvdGXTunf6w+1Tf1/4sseXxyaLrcweg1x+PRXu3wCC6gJbQRVhXX79OcAQWFwMOV2znzuix0vi//x0vFRtl+SrSpkKX7UmnB6gbCW1B6l7p7EA1BI/M4Mb8eseApnRaXaGyr5Nl6OcHwPhk/uFm4YymxiQXaVGF1k5F7cl5/+5x99IZ9NL1rcsueKtsof5AIpEAYphz2p2Hx5tesQ4NWU3R4TcLoDkPUEH1dip6Xy6w2olYfRStsP4VARJpXm8pY17nIaaHYdpxtx99WezbqDvG2LdgqfjmoIEMq3xnuljQlVA2sk9458qIvcp3Xlzn1fz/Goj4LeuCfjrwC/9nVZYL79+5d9YiHEpODSvNgidWJPaHNmAkjDe8yaV4m5KgyF6YiXblLAMaX61KCQZbUq2tnG2wn4yb0rDG1LHCf60TLlMqOcc43P/xNfTQk7ORgnCYqxGL6GjER9xSYceIbQ9tokAnY1ZfcNZJLkbBFa6ecZKDUfArC6m75F4U1juvxt7PeFvm65a9L/OF/I2W0lKLwF659kE8TgnP4Z2AsXV4eBi+VBAOxy4wt6YgfBS/wL2aJNSgbPMPahpffCixgBmD6MV25/7ZKuPF+yKB9/mLSz/Nn7oPL0ZhFGhJ9B8daGFjajG82Do0S2V88bnQdSzji9cElIT6DXuxuNF5daognb35/QVKCi4UPR61Z+iMQuE8HXe3zq+hpFM0q/1gIxrbL6LWVqVNbqwFiQUhjKxgAJHuPBarZISfCYcWzpja40t299ChzTIP+hF7jVPaX8V7L6hzLLSuOlnr48VX8F7nr8FzD8lg4+q/FvlOfFXvBZb23WOylaVLr2Seq+Z5Hij7yJHrXu4FnZ7jccelN6689B6IH/EYx1KKxFQVUwhp0tv98meYVBUDMJN7jSyAwhlQbfx0eNFBwToEWoSfh84dW1oa8M3zM1x7yU/KKttjaDhlz9Q3UYCn33DavxrHY+ftdVxR3enloYvZLV52TKuXJ1nID7cLL95xSCIfs0Y2IQ948e0fTiRRVD280Nqa+OjFt2+IfPz7RbfPaCuRVbL3P6ORk12QPBJl/OOPMhrH6RvzNDxztGDmlVEMTjeeox30E9TUnN8qGDn1aNsgFq86T4ZoOzGZbAoise1kAawpFAdSLha94WZ/o0v2KAIFFS+oOLd+ffzniEaxQ9iuOrqbbaAIuh5S6cDQxdBd8qW+Ev+g18xpwhBOeLU+QyEDITlKvvoefMSyrsLdCUDao6++B9a7oYu/UFDV7sGLoX77q++BqUlmlF5pD0QLz9RxYSxXKJk8KlhatcBRSgmKwXPMk7Qzqk/mshyiAS2QK7z9UL0Agv5PFsGxz0jMQb2+cF3kvU5dZr3e+9jodXNMYr0+GnexXjsCrSLr9Q8BB+s1hvMs77WlVzJ6nbPOnLf88BCHXVfy5KtxktLLlCs8I+aQ9RW9VGxSQkX0HT/hA037lbzUPX/gP9O0HWL+0A8mRX9l08vnV3gVPtQn41rU2KNbTvpegJ7S75dSXvquUPwY0nRTSlUT516otTwR0mQpv0O2ERZFT0Hv0dLcSbIEwC2dtAkmuM+3OOjAQqWMx2bTn06A6MbD04hb7Ls7pY/Z6B4tuC0N9Azkn881ac42n+rna4wBqQGiz9lY0fjg2cFc4RxXfp6vsetz+3sm842v552zqogfRImCyb0/NYCMMNQzTWdbf/Wh+Bxt4k5zpppTUNfoLLl0WUPJS2ck9N2vd34m9zqdXSnX5kB6l06zXi+WyZ20r481Lt/8MprYgxR46p9wofPYyRpgHh/+nG7jtp5NXH/oztAerEN8a2hJ7MoLdf+CMkcHsIkdLFPMi2r3bHLhMCVuuNgyf/DFtXsqdvb6XGn6+wW165gAbbY84RJ+zzPMzZO/60AWKUG+eEmu9YWLV27XetYvxiM4g8hs4Vmux6e/eMeDB3HcSilfgwjySbg1ioZEx9fe7xY7RPuoFnycmi90pbpE60P5CLkt4dIAGr7Yu3igJC7daJZHbAjJzLfIs3CBRFf9y3EAxLFzQb9DeG7fMAWCsBAZvuHyRfmKl6nDL65dLlObEvULateQlE05+QW1e2DeDXCU/n5x82ukzzNS51WNU24I5hDI2C1h3cRcKl6W41w4P3PYWI9cgi89HjmK/WhVqB+OUQn7cet+Syv2I3H/LejH1jmlFfvxr3CU+mGx8n6sMKssQ7U/4c2mKCaG4FdiJS+iveVmk+dub4WZ5AWMF2CghwcS0vE+FiPDYCdzg19+4W7wFx219fTADpNP8MIrGBw/g7OKDnuXpEZeZ5+u9fxn9gGBKauyFF5TLVxeqhD7E6VvfVZlE4prdgptSoXgD2+yB3vFtz4rxhi9Un4rv5s9sIc9s4E/hqFm8KQfz5EJbwuM2eD5/LBHgCZsVYW0H0j4OtUs7IqN2KhSn7ol7EmM29muIZSXrDBFu0YmT2AmSz1SgAeNJ0oxrBep6Xp23Hptxn7loHPV7dsewRO0Z27tXr6T+8NtRhp9mdv9zy6mUlQ8K/8yPRSG4lk5UBT0l0bnSj4acqDrrPRhMUV4+M0tRb9VYp/azB3j2FnPqtuw7cGMLdkeVs6qLP/9F1m3UslSIrYvVUPJ0qp87hiTLwcftKTarYJjLBadVdqtlIBE19qtfKt66/8v7tt+2jq2h+cf2K/nyU958MN+6EMkS5YqJGQpQhaKEIqoLBSriCgVIILSRCUXpZA6wCENtDRxSzmYhiSEQO5Qyql/GMqhOFAoFBOMXW6mtrkY369gDOabtWZvLilJE5Loa2PP4L33zOyZNWvWfUn287D+l9apE/8+7VkW50g+uFYn/lwdHRQdAdMsKxjN7kMVx92I/QgtHh5ab1I+dzOMmD0X7Qr3Clz8pcBWRAyxcKhj05Im9nn48xNiKIOffKBzlI4l3YtJCw+WBfzEemUj4Hf6hSSCDlByG5Zo4UOZLYV945k+TlFH7kISiDCYFHQxO2ENabWrm5SMHmza0n3A6VgD+3+dGA9BCnF63kyCz21d8iHJDTMVxIVVNHTOcGAYjeMLiE/zvOBkSD4NbA7u2OdSiDs+GvD9BBYRianF5DB3ZrSBHHUBJPLZDXHAc5StxJDXfHbTqld1+jYl0yxxTeV6gNI7vmyU4DBn4hRh5jJtmKWDssVIvZevOsHsSGH31kHZnmgAH76iCHhzG8QtYACVtyFhKSaCjx9TmLdDYHEsc8M9Aup7PgfvXR6Pmr0oQpLQA2jxCGzkI4uLRyrXl74bou+BSWlG8MA9gqDDd+BicX3JgDc5xRfHpnhp5/BJ3VZ77/KvaRTKI3NzSbuiDc9xyPNH++uJlymXVx8+BGvlM0tJIRCXTIaZSHB8UMTwNaXjqFOoXB+XAvrb7w9AlfwUQ4dZQzjtCntiaj45JoXgFUcW/SqF3V9yfXVMSj7zU049m6KapZK6iFelB0WSeVNbswF/R8+yaf96Z9q3wSExtZTs4lq2HlJ8BPY9jZigWRMP3H+2vn6VPpe0/Abe5cX0ujerIBx6OgGYui66lWQbBc15M+YXVWDxpnAGssEUWvwAEaKYmlJkejyZkE4txWr92vdYZnPWxHXEzL7NJRjvf2iIGJ5TuE5x9EinKf+Zb4o7J2IrD8g3kc1odBNMbOpCuWUBuuSP7atm+7rp5IeWRCKUSFAsL7MZOcs4qsNajiwyeokZIGUNRaNDWRRek+Fw0pWp3RxqaOjxaUrWQuONaFIbFyaGv29bjZ1vD1zXgYEWhddxS2T1wi6JY2OjoHDCTfOly0WpdH74IYy4r8+RQlqZm7FO51QUeS6jlbrPN3Zklo1GrtXK9bNy0em/3FOkE0IIK+2txGws8xWC7cFzXqxXLFNkTO8Z3Y7Ci89up8EYSsxEwHuBTmaaCfXhho2/Np7ApS9Wt/SHVApyeTUQ8V+P6IkhXlYc2FyvBG+IOsuWIxVtIOMziDiFsAt0aVJoWwq6LPQDuDc9m6fzRkmE2k0KTSMUKlMcgWxmioZf2QH6JbOBe4IBrBDPxygi4VoGPoInazfaMZ0TGaeHZRvFQOAGzz560pOsk8+HCy6sjnI3MWToxUmKMOXmfodRz3GVvi2rmqTZY9Ho4CFSUD6bk7WLzn3cxcQOME4I0ji1VKQS7N9DavFv8bp4f4p1Ji3LO0LP9wZS5SuACI/qYxyIwJwKVlfMzm6D6bZMT7MGOtw2sAuUTgO2zAkaUPAEKNIIjgAyG5AphRGgvRvWYOOOgL0eO9teaHPP39LhRZVshtJFgj/xiyNEGv7cMKUYJdddyUA2/zC6FXiYSmc+9j2HCCoquAfzOveWH8UX3KUxFeMQBtIZRlxUidAj/r19ffBLnM8Ux1bgLmvGlVztQhRYAdpf8BOO/5KaMr8Vuk93ung/d8sdGj9BtBu+qYQns27kFIYhAL6Nvhdp2ECkZ/ah4UYF85wes2Vi9B/GSYhZFU/qTaZBDb5KbzKx/Gc46aCArU9gZNFLAbuC0s2Hn22szg9Vr9XWUYJsTd+xaj6yaNSHc80WjoyYKTlA6+MWjrPQeg/ck3imX3X9HKvJ9k2k6gOL/7H7HnGkffOPTCGCDnrQr4U32PiIrOZui45V5ZOhMUGOuskGakiMorpe4fR+FZtGheBEyjwYy/Ztjd2eoihWuYyn0bwciFyKIFcv9CET1rFlAKd0968ghV0LRZMgFjUQfgYwKz3M3HbmmBmEjMz96uKF9Up0VG/cMlDMfrsnArQ5uAZ3SMDULOS+T+fkv8mN8BqlEgvClHZTOtYrmdpEoZaJNtGkRWABIbeakvKbm1oItNIxBblf5b8mRIcvdsTBCB1gmwlDjG2hubwBJCvgdWdgFszMfd7Mhg02wrlYKYmBVHfF9Gec0uYwUgwxDA1iDtY1VgjGw/1qtVpQI+vg7dBdUrgGf1UItrjs8Xb2ePv2LUKxBj7+0CfqnPdeY6QvThRkgry/sT1ReURhEydKMB6HmAF1LRQHXfTEttourIJczoAT5e0cB7coeV/8xYkC+mQIut89G6wCbzcFlS1Rj779y99vFiq0QVrf9x9GmPZc3GWt+LcCkCstAB/3xDUUo98PhAvkg0kWPAvd7XHEV7x/qaDFPD860xqUDuYZ22tC5ylN3LcJNtLgPzsZA0tPCt58O2S4hKfkc+iaaACs7s9jU0D2nfiXFMJC47LTpYXYd/dDIcpTQs4k4H0oMHg2Gyj16YHq88gE15EYhmqfLXjW6XoAVUPLZnvEqMNqQXhlvYpVpdatZSWrUoaxDwjcix6HCsgs6KAfgvLSi2vxClYZ99sVWDG3b7azitLhH2Hse0Xcwypc/5bA0Ku8kOBDhKddGy+PyKc3axk8aTdH2Majj7CNVxYfZxuvP9ksbDzHvvBEUYM+Tdh3gAooaphKVOHUUYKfglDek9VwAVPQ1OEWOsbtALpYHniEGd0+CxthN2YxxRFuMGneEUm9/yGOsDdNQF4ORF6SsU3tzghXTLMbPooM1iLJ8EWCS4KowCD4lRsgEEkTkK0BRn05nYz64jhGfTUq9lJf53sZ9YWbFLiFzr3UF+pNGRUNZHSY7nFgG0S/uzcrEZzBi2MvovuHSsnaDE+ZB0rh2pMzHW6wrYBAFkIYIEZG9YaASmwM9YJ1FYg1tCj7yAlSomAk2QAOGCDamJBMgG+hcjlY4kUBSN+mkbltVK7Hth0t4MpAoJrIzL5yophaKYKILDlBupLgsl8cs8m0icS0FNpUOIM5hojdm2UEwYoxobU52+PfOIA0oxskYCxZ6w8xx1j3WsNRlxtBSGbbopTNyBZzczRgjP0kmJdd908piHJ65TNuOKnnuoFvUnkjzXG0XdRubLJ4ixIrZKvAKPqbKLPFeyj886ZkK91M3dytjQkgIderO0M/0Rka92tzfZEiev+cnLQmh1PmAKkp7Kvdmygho+0wYTU3KkQH3Hb3ebGiZZCghRKTsm47e1AGEoP1ZLowrKB0jFkA3GLwneePPAHEJPCROkovIvEISQdEsvafvjaWVOkDlIJivhfcBOpoW9DxszjGCAZkqI3oRdqwXoWSLj0RiUViQA/SKszmKa8rlqnVkHXaIGyfl3600aShFCK+yLshMGHtWvQCRdNG6yhXHThfGTiffoyjVCujhM3MEbRCokqnO5ECUuazS5DithNy4sbtCa9KVp5D+uKa1C5fgh5b+s3/qulpYkzoipfpARXYis2CBwom0tQKWTR1gnaNIpdb4IWn2wrG1jDH8xSfH/BmwXXOElLrtoy4X3Rbq0/6N2Zkui2Tl/kVEC5F8AnpVUPMj0hi9XJDsp+7Yr8jrK+eMqCpl9LPx9xXBzcfc/WrGBhTyBwLILaIDiLCUvSO1NBS5U1GoxuoleIq4hDjXYcyCT2l3nLDkFA2cNeUpMzoj4mkZRllFR+Q3ePYWIn78xqSD+lZ1MfEV2Z94iY9xtM1cVtxW7gTpQOQilYcB8bXvBIDt3pz/E6WrCuIvsOJZES77SK0/zcd8+ddXTIYcyDy8jFr4hH93CYdc30gbFhwKiTFeXTn22R5/oQTU96nPTb9WrS9NjrR3wRXRrcVi20k23VbE3xF3KzDTA/7F/R9ajaGlL2XSk119KcX/uHr0uuE3kDoHbtYuL2fRrh9vTrn5sl9gMVAbpjuHqJlORjRU2BAG3rYgL0clC43IA+8vgDXDVwXBQnazt8HxP6VmiqejJw5bhw5Q++OxJI+D/1k67ZGuY6thyCMYcPSAZX0j9VSk+bNpkyAGgF8ywnAb0XcKt1ZwUb000IQTtI5rBclSwx4Sk31B9o9MDEpuyYWYxriVng9iNq9C2xRtguYV5noXUbQ1arUdOvpY93W2hZ94Wm5GLxuzxcp+pUhktA0PRLoT5LCT/hcz9aG5xZHJKd+DK1Xtm7+nDu5Xqm0JFked3Y6Mm0y9ibrW48NpadMJSk23eMBx76zA75GOXjAxYoB8NIHHu+7GCB2K4yYi2OhH21bZojuNxw2H+oYUlMc2Z9G2ui7a+KjHCkIT0shvzQ9kcPn5vHEbdj8JdbHEmSuCSqHEYxpC49LOwfSsQUDHOBxTTHwf1UJQ1U80dKQbAE503THyl/jzN1/JLgw0uZDKShtdrNBs7ZpYPkwI4XKZTxdb5q+ImWmmyTX5ypcAGK7acOOsUMOz23NHUZfyRicYgq7/3pkTs51bLZStrX1Va/CTYQvOfFVajeMcTwvKa0hvErflhDaqoMFtigMJlmkhOP3WGA7YIPoGUYpvUgSvCy4LhzOrQ04iXGn0rNyXX/7hwzdtgG0AuQqT8D8M8XhzhGJKYy0lv/X5n8xAA9w/0URCydeVTjheORGk3Z6WKY+Gkjf7e/26n9DiyrmOKxDx0FJ66Z+m/rVxJH3tSWMMsLdBb1VcWyCF6+WwXf1KiiOSkcqOLR7sR7PH8jP8vp8EVesOH2gNHXY94iDPCbZAfr+6ekoANuRXYIcE+SZIFJCcVIXSFLSfo1NnSOqlY212Hp1c7Kjv2b0Ot1oEDaXLlD/Zz+cpJticYHO+dc+IaQlhtQFoQlEBtVuhObBTXKc0rY/D2aLjk4GQVmi6I9/h1kBDGsl4KZW4Z1puj0r2FkkR3mSxcnUCi6bKwx4N2Mb3lgl6Vk4QTKmTdzRBQCzC64cNVoTqN0a5QK4uGjcOZwhUMqddk0c+njheXwrYXJ9DvaSKRz9lgy7i8n1QCcdWJ/USmFHPu/TroLguXkTvQZLLetbdJ9SrLWyZR9Ip+h9bSsGAld1/9Z9nBc8uPfOHcQ2YDZpUO7+XUYRFUQlsB6nsOqbjEMmowHunIqTjm6CMKvE2zVPAbUt8hn5Zq0mz7th/m1KCZIpStrJvy+UTjsVub7QDMRtO2P1/ZoGQh2ZjDxIfG9zpBx/mnrjJkZJXg4t34KxpU4mnSYnBhFpj3034BtoXGunRPfg7+7H7t//gEDaD5w6ojh2C6Kj/mK1nBr97RfXUW7Yf21pa6Um0Ad7IPTwVFvEkytErRMZOcKreBbeIeT+lnzZf2rYm5U/GYmH1OOz8o6Q+kkivAwuVEzcm3Xzi4u3T9Mjw99EdxZvWm3yLxwntYmhAUhhSrfeEwecgXgBQJ3PTleoWeD8uiEgcA1iCnh065X/5ANlAD/Wp1T2j/Fay62VBe1kpd6g+3Pma4NOO9zwy9i/zVrF3E/KjP9NyUnGWCIxnEZANaqlBK9aRsFx/Di5EOmzW6V1rmMtlIZaWU3QDZz6LQuCNwRUlihLPbLIZKvcsW3ZKspUs5x7JYrsgDvjS1BUcPoGP+hIIXxXrJ4cdyyVqyRaQRoJRmPqRYifkFOfBnLzFGtIz99zVz+LnZF3Bdz0CAGDsj5fbmrvV0RA/kQ8fP9e9tDTDcrA/ZGk66JuazZsHUoadpMfe07Lc+YAnYXzsa31mXWdnr7WFyGxuLyKBTlnXSxhEabVLxa7Bbh8CqmbzEvvHKu4mUUJZp2mNazDW+GjWtTy+gDFMM2x663xmnGnItNt0AfyL8SaQU9JaUnkHk7fLqy8mQXEhP9rP8O5jm9XAV6fJP87maimmHi8l8I+/fkJMDAK9YfHACUWL8ECKS2UXb77Pwhw1o/cjQ6inrmZBxRdUxczuNI5j44Ncz8xqdKthcyiQBiwed5yhXTMOKsBuOvmSiLT4UEZaZpXkowfnoQfS9+8i5OOmJFnXWifVoBnI+0CdCT0QeiCAqyZ0kHQBZB/cU2uI9ZP0UTe3DlyuuqaV034CTCW08YHD7G7FOmf+CqLPOUsSOucelbPGZlEu/0vRUXID8jzsu+8fLbbqWaKFG10OtbJ9Cg5t5/4KAbMHbzGvIzQ2lxSgsz6RQ9YMMjH4RzN9SQ6JSA77SCKnNSG5PNTU2sl4DyYNErFaPg7KeaUDiGuPbnU/9H7EBzv7o0bZYc2+aj/0nvqS0wzdXwhgn0d7vz5o75Hh95HXw0JzwrKp4vzLjJn6HfeBd2TTL85HihHr2py7VHq/TPvpy9jCMGdXJz7ZA73Fj8SHeHfzzLlLTOBxxCSdFOJNVZXLvvqRWlgc7JPELbj04LolVSuu5sNCZTGcqM7oQkwdK8ggDeAtEWMiPEe9A1HJrRCLrmyZWYukvFDCeMH0lkaZ+aXCzKKt9Ui7KTbao6eR1uMhnfQJtMG8r/Fwlcofho62dJ79G3bZJSLQLoQRru8ZZvw4rwJBCMDZ1HG+Q8CJZypjkD+hz03Xut2Ng28KQbsqPHjlp4PBd2ISJO/5rfkCCjRH8evO2yy3M4jN2+Qg7WDOk2fqQKZjWQkgvG3MfBOxiKlZVkIHgqjQsAtyhVwnyyFSyA8vYFdBfsbDPxGeWw4XuHGaBfgpn+vBl3IHhYXF0GsvXevX9jJRcdPzAs2R7kDp99GsP+Pqoc9KfD6fMzsGeis99zpdoKsyYgPrYVv9n30c9eH77XTy7GgE32xcy7mAY31PvuiEP0BYn79mtYDiTnJmZ+P1F97r50eP4WfI5R1yPP7c0mRP7GGNlvla5u3WGDttdgVcmRh9WoRXD9+lpATp4QR/c+uFkY0dxhGxHUl2YgqvD/iiDIXm3BEHWMmHkcU9+QJI/LiiD6Y8gkjypCL06AQiAmS6xONhb/CJPLvT+Gyu1v53ASrnB7Ifd+dinRTnicyCVc+7KJ0091D77VTSt3gREsmPdeivyB1c/cwUDf/X6b3net0BCUzHHjZ21HFfRAMuD/3/iozoSDm1YtwIFYEIJTuFtrYv6gfJWesIeCPIZWGl76VJj7AMZm6QfGfdfjbYNgKfAmlomRJ0Ms3bj2iZfHTwEYNXgfTC4EMz3T141Rc6ktDceuBlCiMckJTZ1RPZfefepvGGPgxol0k2Q/amECXC1T5QVsRSG+B8BbI7oM2ppiMACWtsR63qvEv4BMP2hh97ml2p+menD27FV/YcCjhefoEad6sk/eFhbS9hrbNH9X9kS8gWga2lRw81b/2JW0v+dv/JbY2fuGZjIOlV3tzNZbcgpZWkk8XMd1oHWRhE45jkQZgFMDBNVNCc+xsZyf7gRtjp7ZyPuLKJYfu0hO78/CBGxNOSOF8FE7HAzaGdC7MF0zWq/RSLwiSXnYbe1FGmwiESSkm1mDSo1d+s/diryW81Ws/K9AM5MFa7xBFfWljmyBVr9fmLEzxoKZxxWsr14dTW+mBHC8DcRfk2cgNj1eu2y72bzSC9QSApJAKJi8YmzdLyfF/f1yvJm+u1hLJQmQgcNsMnN1Rkh5Y3bWL4OwIoME9cBvvol3B65qxI0RkSN663fPRsN0EVGOFuj7rHYyTMTtEYHfoBmcMz9u/v3RoSsGyKoLhJXdMXVCgVknEbF/070/yMfurcsGTuet+6LfyqQT73QSDlP6jji3o99bI/1agX1fXSew3XAP9/sfL+j10SHyffXj2N1bR7culH6CVffjyN29lP078IG/0Ms7q/Snx3pVlxkt1yy9nbN7fS4GGguWp2zbpxr9Zyf4Tr4rR3vdeI+r5i6QBTJu8Waid/Ml9jphdRxVOI1H76QDp9Wp7FtxihASa3tPLZlBjXgNC8Cd3xdS0gt5+LgSTZdTEtPBQnTcLzbfFD2gBxA+aNIsf/qszHG/aSrbDCM38FK3R30DcNaLZL4FlTWi3u9TOPchcjAjx6vK8W0Kge28gin7GeV7vk9htv0MpGfbm6eKa0mAn4VMximQ6vzt51v41pLNHdvuPAVRsT2jNi/domfbgle1yD0fltKacqSfaVdOt+A0HpKPUxmaDiywvpXZ1S0xQqV0WIYxId7/xh9uJJks8g4fFtJauUwI03o/d+herfrwwKbhY/et+4EsQXBtsn6tsvwGIcv8OzT31svwX//phdXPyiJjicvPBv8QUpp+nCl5ZTOfEDQd0T2INpCBQS7gBm0wbKmSTAjIkjZuVdUF1caiKgq7rqHzOkXHWb8AYN5jHRjzNj1M2rDriC6kfaRvgGd2rvo6PhxavUW7JVtEX01xcvEgE55WiYC3qGoBVkI87MuCYgyAqvClciVQRkyEeH8Wx0+c1nXEtOfnsFOS5Q/RcBZZIdF8mupk7w5XAMIeJPgxOxYVoO1+wPC4F80pBpiAmVubHmX6lKREESW2u2+hWA7KfqISy2ldEp4JAPEsodclIKLrpuwg5G77zfc7c9unv3O8hsNZWOEchwliWF4E91/ts2XqOnFkZuzuLOJ/uOD2MSGYzXIjcO/R9TAsEDVqLgHIaXbB8kIRUH3yqX2sn2R46Jw36D6p8X4/N3+SIbsPz1LjeTnSJbyhWcaRowmWEtIGe+XnqWT9lPvPpyliU2yvDjEe1kOekB03vuGEkuc+Ha1/9xyhLDUjXCiZbPrekAUCBPczPgKH0uPyLEG0PXLCGD9P37OHP+vs5HbrFOBWVa62SDIeVTfZfx3dcsNjaU+7SVtNHxwMAx01MsOdGnIqT/pHcerDT+dqXnOpyKlLVlPkYHSEFoR5ZXaiaQHR0NlmMzUmxttO13/2PO6bgf76R6Rmil3qHiNlgMLOvIW07ZMH9qZ206zHvo46Mt9B9dnfFcM/dV0wuWWIW+o9u1bI5RYn9KCEVAwu3jSsN8g/0nhXTykqXbPshcmROkzKv3eXXmTawGu5LpUDpnh4PWKUloTKDrnQgXeX9cwSvn1lKrg7g3jWYvHZFTp3coCsLlVSutffef+7NAlKfxY7gegWPKwVs6lKXqxQ2w3Uy/rCWnkT6USkx6+gAy50UYp94PXrlLEaVST8m6fpVkF1UzCnrouBWn2krT5kpezhCuF+7OMWsptFMvrcpCdcVi4moGBTTyj9Cf6A57sLVBZBgmY3cAOgNGyK2CDj3nwmtgTIyz/fM7Djv0bS5jiodQXeWNnqVu5MY4yBSfPw5BFJlaSToxEOKZfpvSMvy7Eit9KBEHNICbkLjFo7CP6bJRf8KyCViBCoR0Y9yGc7OGz0fQvCTC74L9NY+hCm8XBLUELX3KtGENKQhkMONWqWwRa71oZFuLyH5wx39Q1rum1DiEX/WvbwUuM5n6QyQeVnwQisWcLDBsKtkBCTyAAYmV9m+jvfLzL1HXQ2FrsIzPTLwnAITotAQAzQi8iLF7kIWfhuIAXg6ywXmKVUuoKj1YB0rnQbgyXHDwaFdgsj1xhFMtpiHyciF1gSkhH9KH7VwaYszst3HBR0MRSMNhA6H0PEIDqXfwhE7mhglBRMbk2fpTmXxpyqW1noxK+wle4I5hJy9X8HSzwwBarCK2ZaHZk0Q+lC4VsdUVgObY1ihbU4UsETOFIGnda0uVXA9Cfsl8G84SwQPqUCfnJ52D8CEpI5ultEpQMLFYFBCz0r0mzRjFjNJNfPnfYIsUaZnUpDu1HR11bC4RPfjTpPJGauHeMR+9Jst8zdQmr6RN8adM9pWm7zMndPuOHUtqstdLtQ5FeZG2Jtqt4bVZTadeqmQ3tO8UPxgvf3BnFy51MrrA+Z6F0QLb4htp5Im3PVAbNYcMAsi7RO6m6cZp8H97maQUuCLMlBodvUwv/3pkQshMFcvjmjl8wC9HatPbk9RWkk+vxYKRRK1BOgnOooJrg/NoVvCBWCHAtHr1aGB209D41Kkhlr8Kijl5kkpknYLhYhhlf3eXLz+SVCrDt1X5/bHWCb4MMzIlTlm6HMrMXnXuN7FZfvAfqA1VFAVKt4ma4QooxjbH94hBBF+vlwsonu/SV3hA+MEfVx0/mUnD4wQXaNgiD0hbx6j2JQLwxwj3ejGZxUdGI5DpQWyfIJSZwpsmu6rPzXFygmMdGPhHGswFGqH5jCV/V6zErSvg4fU2fz2j/dBCMWoXAN7XMsez3nx8RgY1uC1Orl6P8OV150oiAmaAxOV/7tlXccmqjJcghOlcdNTECfKmrGjhFAtAubXLqpemI29FYFIZ+90f/97hAptkNb3/UdYT7tW9WUFYMjGQHbD2jW52SYDV9dAqZj6E0ect9gngRZzXfNKmCSlA0hqdeiWCWPZqEMrSTzz1SF/fbQLTVYbSSuEo8FTf2JByaY2z/9MwqaAqF9iMbRfAdMgY9OiUy2Cny/JW6Zz6E+0oMKfzod7llIg8nnKq6mDfRTBVwZ7YJbuOfvo+XAHJ6zDW+xtZnNX4Ov15rAqN5EY5YQZrY1XsTkdkiIFrw6Bwxcl0dV+m0eFldAjFyVVoKJriFWwCm+y3whDheT7R0NYIa2ba6wiR/n+/vDUFL3A4Omoq42tBj3W6oS00HoGT5polbDxRHgaSdu78ZafynHfdUbGGGqgbBqW2kgxzF3J0pwcJi9hPb4N4C+WBx6hpO/3SjbCz1wJcYTrN4QR3rSpcISdqcIIp9h+bYgV75Ti77FQwvohjqiV0rtQZroE/NEYq6BkUsDHyKSWp4xMojQOkkksXtMOmST7gZFJFPUjmcQr95JJx3a5RgF5u7BwGzNimYVQGm9UMnBuF3e0Qdybr65kBwByexaPSMcCZSn1q31gObajSkLP7d5J2FLXl4u0MchLM2GTHQUNe06QPtpHT5OGED1WNZFKbgIEBPQ0yQlgAO2O5QdedNEt9ruZ7RW4VcGJbJZRAoCe2tKxgZZgDppbQ3P0LCJan6+B0KVWOHuIceK5Eakmoy3L2dLiPApkljbwtU9T7L0OQbpzvMalo0cWBgPo5r1G6ZARFhqjDRL/NYD15XXvdQorU8MSbtirIno/BjMc1mDMfFIdCTDnOcn4GgvBXx2L4C/0nopoE3ghuTK5rnARuUVXHvg1RSeY6UqHbHTBKatSvdoAUKFR2EHFxk/NPWYs0vlIgoUPIEYIoY1Emhj25YWKVoh4A5Ex7nmYpq7djxLX8yEcpsLORqlnKvjySAMzaBv7GIQmdcltwSHki1kDg16DQJj+05elRirrohQTdwyJlhFMideAwypE0kFmw7jnBrTBq1o4uk36SafRgqgN/aI0IP6GROIyehBqI6saMGN+1SfnSUBTavo9n/A3LuI6uLOJ1HLf1E4qXdmVjkxFOqRsEg5REPLzJgpxufpU0Andk5PCoPWBe1wq+ZiH84K/txzdCKkfB68rKG7QrxyTj4fUed7NNSskWkDpJ7DoyOWGNEL5fS8knNfEAn2TCS3RrDVT6LRK8TpdF4rt6NJBss9xeZG/j94/mEA9Bp0SxY4UXD7t7bB7c7XRC0TRmqNnQT8bIRG7hNIrP3JXw9XA7u0S9iOZyoR/lLtI6/zjmtmgsHs7poIGaCXbZ4BWOIsBWklxGM5Hf5LUrxsgg9z8nEHMys2S88inXY8dKyptoIBkexvaNsKh8AZ97jMgQM4Gm7ii2cvQO3PEOibEfATt8JUY7MPxGZ7IuoLB3/PZt4hQXvJNx/rVkzoY62wUx3psBcdK36MRc1sZCkL9uU8SBvL1SsxoNRD6vjKb4WOf48Fk7DL4PenV4joIy4CrcI4ugvdXR/SsJvoNyQ3r4BrFT/sW5bGa1Bup+QP59O8X/hF6jdCLhF7F9l/yyV/2ZGZ8n/a3kWjIpeHTtLgS7uA0obs+Sj1o1jaBm9PEfnTQDUivlce6OLjzVuQqffzvY2D/8gcqjd1HvvqDHjaa2ILeu2Izrg7B+Ukaw7mkhwIXbQIOMcNrVPMHyt5sihAysxhk+vJgnZTLwjqZETIt4wiZ0dq9kIkdHWBH0Ou4I8SdlfGfDCEKE4Xtx3VvCNtmg7g1UApQnD9QbSzVxMLmUnh11skLX4gSeJLrnWhx93MaSjEqZERh3QwagW1R1LptMu1CPtftVCitya3wdYKJtbaTwuFRKOtbtz8WvgVaZPf3UddDgil8p6UAXekDpa9agZxgr+xuAlJZJseWdQ86pUTefQcOScpxZXub6bmoJMbpo84OSqT7myAKm9zmRNk6StbpA9xTOpiuO83+7Cyv62huQFsYqKpyuTRafw7dUfqR8TYjvb/NPtfQMu1sAdrMV1DsA/pI4ZyWymz0DKx8lgkArXT0H13o50h5wLwMyqahxA20ffGjaffUGe8tcit0nrT682GctE8cZ7jqH8aJqetICwTjSB8Mw0mX0VQjEb2TumS0MK8ix8/OrHKQe5E82FT03Jr5RDy2kJKDeIX6xDAde4sQb+zIoo5kDAWLycXQKCVEsr11wgX8klqXvgTwelz32GB+2f/tQyg01a5NU+LlAqR+gqOHfSlstouEexz7kah93wi/5Xo0JGeC0iulgyAMODPyhJfXUJbb4rINu7Pk11PlP81eowspyDf2FdrRn4fQ2ag/Nn6KntqupdlwkTb0ubmwKlXvVJvpyzYGz5pzjnAn/QNTTyOVEG4VveQVZsHZKAA2jNp4180AhJukh+u3P0iJgdF52rVECOSVsv6lIuZsxAiJ1MnfPla3OxWoBtjoJFwWWKzQ7/Mh33p4bdl9nBsdl5MMx0OSvQAUc9VcimYWHj29mJ3lALOF2hkZN+bIIJcCneTcwu/r8dXB+eNEim6/4Gw0JqWMTj2pDl0gHZSM4qdGjy3DIaYN9AKJl9Ec2JqqkVI+dGD1CS0vemLJCKXv+Oz7YLUp8ngpVi1dQPaPnUfSmpztX2RdpeSME8R4Mtvvp2x9tOjmZEqOu7MxS5ek0pHTtabntMsqctpTzA/EOmuAEjs7G7lE+BNy+opEuTT03VIfmBEBgV4XLnUMpVUdLsnb61kUavu0LYSeRQunx0Jjl1ba6YO3xi2fmqz3nQq1u375Aj0lr7jVWa5vbQ8lT4a/WzrKDU9UBLYCFVPgWeSzfHKkcMb9Ms+i1OGQq5Rc+FH+0CY7PuqPOhUjv0kanYqG6IyPLpgQDzmr4eOr1RK6GUz3IuV0Kzq+BZK0dk03B16FXJfva9ws4gWwdEg/xqFkK+exyTSB0k8WnA+mUvQsGulOTe0e4bXDV+wL2vHKRwbNH1b0LLryo/HKsFYx9+BIys9zCpI2HIsNp6EdbjslHNUKrs49lkGKXdeeDZGqGXntpEQxF4VOxBjpdXUC0dqzS9ZIH+2fYrLG9iEiRjntkollnjsS1ZKcq1wfpc0lna48kjoxf/YYp0WpnXohGmskOTWHDRC2ut9zhyvtP0UBOXUgAAJ2mE/DBJ+qzyUC1iXiGffSMt57qGnjDtGE7cm/hlc1O+TeC4fTiSEf3Uu5vq1Vw6pOx9yIhIL+iN5Ep8acL/MmEsScoHsZvs1LcqwaMAkqdFGqs9ypxmiK8FFZVeBL1MQ1uvKv+i8YBmHB2mb4Y8tapS0aG+MFX6L6Y99dJCIoUPjq7gak+STypWeAEvv3TA6l8JtCxcNq5NthrvOsOeQmoAbeBBnBdLTWNMqzWpEDOQWdWT51i/8d5aPto1KtF9JxXXaqMuce0P0PYvfHwdlwGQdid4X6u7+O7bS8u+ldbe9qnLV+aRHCqWHzd240AqK6DGJ9pY3yKCjXV1o6IKEDCPb1G2FritIcaKf1QouKwkW+Q0XkUw2Yj2xAAo2A79BHzhJshFwJRGc/ma2WTw6xVJm/co0RxIXlM0rV/P9BVTlTTto89mApKgjoW9yfgmgZRtTj307U7FabgiaMyMdBAKp0LOVjkgFMmlkXLuJGbTLlfCIK0bklKmY/iCcIyncXxaRjlx4deh+SVbITKJ3wfwqc42m9/D31JRO6ypj3dkDt0NOWrF9Pv5e+GlbNszjHxSfOFL2X1/lgerw8VIL+PEWpVfL38hbK6cBNBvN9nzz/ELnv2eDN97Q6/LggEgIpbqgzariNYiFKjP9bQGOUkqwWURo3CiG9GC1VFBm7bYh+h6YUfZHnppmYhgX8BjehMOQSk3WAUa/oYvQe5PAK03UBjM/OZO92GRLCpBO9Be0GFF0n31q6vpPYuRoVRYeelr2DNhllwjX5HbloMd3w9NDbtlkd0UyOU8qgSn6x6G0HCG/MdY9LiVKfjUI/0XP1ZR8QnzTblR90Xnqt25Eb5Lr8NuX53qMtN5lcES/881dNsGRmKPUafzHvjR5DMgRi7qFI6vZjJwjsUUIlGaE7iVlGlIVKUv7cWAsNkVxfJffJUr86HTWp7GoHWPk4+9V5/4s/YeRJ0ncLwO6M32lDOX9OsfqqBMDjncvUBRgE6wMhXjjJ7P7obYTZ/yhu3+2aA4uLJazwe+5UlGEhfKCEEmHkvXaa61m0Iu2Z8VlmueT99oU8AwovGN/wHvv6kJL58iPk0MSUMnVy4hBlz2yzkMGGnAyETiOj4nJNKSnvtXA8wzH+AcnIwDRUbES//JmGI3q2MYQj4roCbEQVy6U4osyFBhxREQVJHNEsBRIYUffmzzgiurOEEfES8e0FBgbk2ILHB4pc36duYQ+Z1cd8b0maQf2+OxUr/JiX0bU3f07trHy/nXa6z7gx/n5WuaQ8633PqkhOym1CZquzHfCi71x9cdEjZofRCeUWDkYTv6++6B5iv4P8kJYZjlEM3QvmVHBdtus5NKv6iqO00dLNoXgl07u52Zl0zY96OPNmP+jvdOnFC1amYdQFsuHvvCmbDK4nl85sg5PS2cwo9rZDgozwAGqDbeiUWjBEXbZe+TaNsbaUNj80drPlaM/pgzemjc1CXpHinCIgrw7aytC8Bmzl8i7y11IP3opiIgQGbZo/cv5MB9I48vNbDAmeS0P9Bjy7EnE9djiU8LzOlw2iQL7dHYqhrkPfHPy3+r7rU6COoC3zRhN3ceESbS8xfofpPZCYY6fWm2tsUqYw6YDk1BQyjnUl3Papi/qobfz4FuoYhnLbZ4Fv+VB/6m0aY6/JD3pNtLHKztSfbx68scf+8x49w1TqA7eyR3H3KrXMC5Kel92G9/FjvjGevh3F1nDvviqB7S98DfYS/3wzszMhpMjzv75OkOKuzVLo+Kjs6AQ9YO/HI05PwYXQdVIRb2yhYJPiMEAhs7WdDzVx1wIX0Aj2Pic6mamWlycpBsr4Iu0rOXlzLY5AVIokJSMoD6ja2SEWGanICMUDN8YKgQQUCMCDNrZnlg7cikC2CUTbgV+Mu3mTRdR3OFJ2MgbsknWq1SjrlM0wY0aupYUTJrKuXQIdOtbKCe1RObtmIPrFYyO3zGaK9ia7JJZxUhCeOwyd/sfPeuU/EOdzn6V+YzXTvmt8gFb2Wdw3b2XPehzkVYQznBGESA6+G/XUuzIDeKnSU6DxGIX3bsYsutEwucGiaq/LzO60SYsqUjKeQtrMRDrdQ7Kc3xPIzCGdNpAjf1WRdLChLRnPHLlI7yhczqX3NLiO0rt66SwbnmuXi+idLU4FvddopHerOivAgBdlqTd+5kHgDfLUMU8md+YrnjRsbM3JzSB2upOkNe4MRgba5QZj2ElsZNiT/mh3wqO9bjPX2yViat0NlghI6x+Ys0NVG27UmfN83RLung6E0HndclEyLQiqXln7m9sMHcWeUe0zak27MJih1+jj1AmoKbWUFfMV6czKqXkwUvd1Wk0QD4tWN+KMY6bVKdFXRsy6hRh68DqrtAuJQ4rszBeNG2wShH4W+spfPuG5YfQxsbcXsIbyFsYEdk0bLBSTRgAquRKG0PKZ7i7JvUBBlTeHcquNEAaMsxiEQuuHH+tUHrxFeEB4XNKZMOsdf5iLeNAE7PdB1zO6acFKs9h/DVTsGRQKmFW40q5HaTsEWeBNYKbJjyFmEoWVLK8DnJq0T3j8IcpAyTdga15qDsBAclocOH0kN6DVRq+SayENpCqNgwaBE6y+tRuwZJkL4APGdU9YaXHZU2Q2EPlsKxgS1MXMpuDKYzCvWrBUCNYFiNTNJHsF4a/QdR54Ec1qJLreCG4ZXMeSUlAZaH2nJb3zcv6XDZbDSSAS2AyXBrbi9zmFs1tyPVhctpStSuckOu9SOPEbr1uvJqWhyrqgGtTZ2e4m7lbwLOTjjN8WHWUM2wb70nHE/ZWB82glh2ELKn1FL/7BT7GABsLM348n6NQVBOpJkzurwFsGU8d1+C3hejr7E3dt80qd6yhOXfAzci1Syf+SiO+eOrQlEtZPE7rG3VtRAVyofVXaYBFp2tTxY+uRRfACSj920l1Cb5F0z8rhgZU8cSZ2G/pm+7TSaxl8TdbZyGOF8x5ode866xSzNbMaLuMwUPcp05rsv0AsnO1Ql0zRUVSHEuHouoW2VDdB9MhzthkI+GF/8DwRCyfAGUpmER7KsqlLZpD5ZClBiGIAQpQ20fP0GX/W32q87NOW+KuNaccU/NhG6B6FPO4zZ8ifpziWZsycMWij3xlLfY9hKdHz68hvR3c1Jqa1IkS/QldaOnyLdmAt7NGRW8OQrmfqprkubwb35Anjk9QxrXh766+HnxhRB1w0k9k89MDEQSCxCopKNNMpdLtTZu+yZ+m/e7IB1a1iDqEUx8NjDg2scs9dVCiXxdHRTmn3Rcrr6kjbfMpDn6uNNC9nHpqInqacfiLhsoHymkXCRs05nWEN8E7c1c9JgUdDWiB3D7hANhJw4QIlMZwBtDE5KXBfoGTLzdNXRb+UNkJ6pqVER+mjMo+agLdkm00G98N17VOOSHiDTkbOrzjmplP5R+veSi7vkeFOOx1DHfoEppgushQ5GsjxNxGeMZXK61pXDKCzwB/pK3dQ2jvLXVu4mFM1KaUnJ7ieaMEJRcbXHqfQAotbDWivDXC0FDWCOYtoqzCFlg6gPZHNYOrBRVAnNUzB3SNtRGRYttvZrnPfnMr06Lfx6+7ud5Cv5DqdfskPPt9NPtPlgkQ58oHYEKo8Mgfjf6IUN+3nc5iDx3MRScBsXlpzG/zjUMn9e76AxBWzmIKHOzez1otyPvVYnFK28vvBpTPS/66Bn0mKgfmZoFubci6xMcDr7YpiH8zR6SClNEr9FChSHCBhTP8f6CfpXgLJTeQ2Gha0LoeW78Gxl+ce+1T96Zg7j+K4rkMaXXZGl0WqsWd2eZ6HzNn2Qp1FZS8kjWY6/TozaKJ1UNXOZ049yJuynJ7Pyl6qyLf7Hll8dZinWky3x9UFYqGoO0sMeyLkvCoIsxAXLSteDdqEdliB+6UA0gy+UB0LhZV0nyqcsVBobZSrDFI02eDN7gNLXEr0AL5PGHkglb4Ia6GgJ44mNmNaGOQ1UcNtY/AKklHGcShUi30oIUKjCoZrM6xJyKYhn5tWSR6Hzmt8pymIdGp1ghxKLMXf5XOzJ/gnofP4HJp7sXag2dWBVKDZ+M+XTDggsAzCkqLoPeUFXzHc94UbUrbR5/rk4rm1y2AQCvoepscE3uN2BXnhmvBAHLH8C9fggWL2eN2Lz4laPgjNsvf9+SeRy5WBnfdXqNXpUNKTxIzvz1l65HMLBZJ7MQ17rp7bfn9oVqC595QwBsu6Dof0ZFX30vteYu5IBF3N/t/tQ/Dd4voo31+ntCaTE3IG/vCgkh7uMMtAHWhClX1g9aAJmRNf4+QF/gKigfLKlDrh8P7qCJtirtv1Eev4tb5h/s2aOKr9l+MaTTSqBcVqVKNxT45yZGTGTWu3fQXZK01QK15qblgqhpq6Y9piUGOtMBC8wGpS67QUapAzkb6ZJhapxtI/uJwJZajG3p0GpaYi8BWWXP/CigFheyWBJaWdDTuUhLC8AnxHLjNABo9RBtkUnW9Xti/RRT4BwLDz+HaD8L5xPYA3VxauxfmyYLf5/mbNWjScsCpBF5y+D7wdbEQf9Z4URtQUEUe0OclY2w8hDq2AAaAj7otgM3YooXQaLSVl8De9vtmN1zPBrAyuN/tPYqlwIskDTkvbB/0P3Gk46EtBp8yPJabvwEGf3xWAg54D314gFYt2HfTI1BrExMto7/f6xSv2w56y2A9mhN4ycGOW/+pRkTMuMWMmHjnb+DBwgZBaf458HjKKUwDscCpy/LXgJqFpABJYOt09iob4zStezCiTEwTYHbFwlNZtINzEKAf3GTeN9HxvIIW+shT7iJ2uRCGc54Wub93FbbA8BsvNufS5mxNGdOB83kN/WIITNGslpiVVQU8O2igEcyAmvhTcCrSUWFnOJHqPqjyA3g4jdpR6n48tIDVcHUdjzlxv58QEmo/y+b4mcstLT6qmYFErWBY0hs8r7MNc5jI9eCtCGgM8yPW70ZaBnPSPM90/c3V/odTtZJ5tBVtVchmNRhV2OA65LuQdyv3lSB3hsIQH6GlcrHAOFAOd8/LPQDE9rusk1WiLYsR+tSjFYWSGdAq3UBvOexWYSVH+uRizxKADRAsmhuEzuJzbdfIcg0+N1kQv+cD1UoirXbhUT1v49c9pSnjNn7TUEm0zKXt6SNDyK7/PINxwBzabOp6Izynz+j5SLrSQkw8m4oaGYOcVq8Fg4T5yGS4Go8vLBiZYZFZM24VvNkl5AE3sHjBw8KParxWLWBNpoj/6bOuJRjHjJ6TRo9RtZeQa12X+QM8SyerRi4PixHL4kfsxvoZ+d7xyj+UUFNzPGwmzW03vlDwLqeFxSjUKhfAjP7C5PgUPoPSM+3E9sZRL+ZfPJM+sUn1o3DTny9Yv3QGfR/ojqeuU0P5iLAUa+l4TiQbCa/WFhiBHpYEJ4OCse/khQ7v5eWNDGFfv38YVqU97HlJzTcFgH32DfJ5UebLzPIvmVQpG/JWb+dzfJtWWXNUSTeS65JlNhgETKKLbr5BfktQEz7InX/hH6DXCLpKXAgsd3cXwmBS0di8Mwa3O8zzkNO5KdvgFhkC/ErOAgaDGnQPblN6idFAy5m89C/3Tc5ZDNgFex96VSOo5TahCuTDBU24RbqBssHDrK2qvPRUAWL5SClgVwC7D1Bd7i4UCAKt7WbkXsF4fdOFN+e2pEVScbwaJe2eXNqHxzVPSLNYkrP0LX0pLfM2q0Lo+UnbnFYedrRQdZd6ZWv33NSN/dOmuenFtbU4phDeQazPYQQFgqttlZc6+PGUg+Ei2vWrqSsLX5Wangh9Iro7Riey3SknzkpLumQLk2+TzLYWeC+hs1YEbtWDFh7pNbgKkhcYJDszIaFHlymoLGxXOZpLlqp7omS5zF9ITZmDuwqwB9nlDeEL6m6+NsKynmPO02leUudxKMfT4L1N0hlWeEIgZ8r2hyxDYrv9HinSbVlSKv7q4PaOy7jsq18quUTF6AQxwPOFiFhpdsU0UA5JvB4SO+DzPcwsMMoc5AaXTxtMoalQujWUQTIoBHAyndWcTTRDwaa0nW0ivjlfxPpJZr35c9zJvgMd1OSCd/MQGotayOToQtH8vgfFLOigyxxr+BlfxvsK61G39nOKvmPfqZNu5SWVd0KGmxMjZaBtlzZj1+l5XZ3X/jkylMnTTqB87pLcVwI9CAT92T34g3CI8IAZyOe7A0CB90/JSXzFsQcpcFi9YKK4UC9JjYOFeHhHF1DdI03hyj7dfGhypbC5SLrhra92uj7PtdBlPOgo0z2FhxrXFEEBMbi05PK2XcE1O5XmHPRZbwDQjKEfghukL91EIoCxdga8a1q7/J3SkvhAIoqN4aC0BZ3ipaSVoKoU4U5YFdFVlvgkU1Nk/xrVScmvvL9AJp0/Ex/ispbLMiXPSMQvdUXlTw9Euuc7CpVkMuSsTJx4OEfmjVeuHfC6f7ypRuX7pCpwlpwZHzhFZOj369I5rXsSmCoGB/9on5CCsCtar1LX+BkqDto75vXN3FpSFS/+xH5N3li2WYE3R9cViCbs61kBpknI/faI+3LC7lZ2W4Zy94rcHS/jsPLvuv67HC7qRZ1qfLu3h7S9yCP/7QNP8CUqQfEuB9z++Kk8zUCTd3QDPeZ57JpDOKFSybJkQvXoXj5qDyVWYtc3XnnlfF0+KrNdul/Y0k7bG9hsjX4xI2xp/GusvGpGet1aoKvpbCbnunvb+wm+Hj1MthhZVSuuqTVk3Jhmpq3TWOiu5r0a+4l4lgWE55XLSB8AqvxTjmX7tt3rPKBT5C2dJ2fJ18nV8zlQqwxj0sFEveyz+y0R+Z+YMkVQ4ddrwn2Ed28F51RIw2AY3wlehNDwwFo9oQr7kiNfwN+yLhZ6iAnPNxpZZZwb+VWfWbsI3pdrNTKi+91uQ8NDZk5l0nLzxjxR0TaAkf3PQrsgsp7tXI3xgdDhdee5IrJEYBrhvbIoReqAtZml996cppinq0UrIdXvV/E2kLWEpCT852ur5mMLGt5S+FFZScexDCNgq1wrmsIPLQJBYy4nUYGCiPdAYlFsx6opBOXU+6zs5iKHbuaEAxF2a4KsnrGqimsmXWfsXayVDTURa07pYu93gTou7mtxpkzVadBPdg7rbC63/N52CzRKUStJ2CamfURJomJyNxhZVEo1WiElebx4YIuyWUuctQY8g003JGiBEa30oqs+fzctDkORHGhTzCKNNg1zr72ZwMphRZdq6FovoOVBNAegJrZEhDEiYdxVF348x+OkIxBmF/M9taCjUsnIs36cFRh/MqlhgrS8q2HjynQU7AXveWqwEvV9nnjfaObZvP3o37XJXUaDJPZycq2fYjNnLv227snSU53Cj9vDt3Xqpt2yXAjrirQxX3fj4O2wX8hDjfu280Y3M0fGxgdR3sm7VqwmB1ZNa+wRXT0LZYqEKTLF5BRMTnQ9UEqHatxYKJVgGjECt6CEK+lRiFvnGdyUuLPmTha3L+bVzV8zmw1omF9UPMS4lW0oOLBOEXaSFk1dqbtpOS3nwxrJqYWGOzwxaMPcOSMQP3BifIYGwZHWuikD1tib0gI0d/o6eQK0jHL7fK3yy8ZPz3WFupPW1biVZdWlEOTVoU8rYy77SrGf7i75aBl/nL/ELrmyv+RhS9ht/nkB1C/fxHx4Vc7xTLrdIv/cN0FPWbNyusEuZy83sZvC6ZY+jPPsYjyzuE2dgiNvRM7+1KPH49wLK502NogI4570IF9HNppJpPJS2ckGjlcu9p77ybjClXOlfo8weIF2pQAOVd9+X5NghpgdsXghMSN9CHPoafckUQpD+X6a1LETK++pL8qiUpP6UwXX4r46DRcLjTRCqy8ejlL04PDF/9KH3FMlOJVyRhD0CXAg24qnBMRyaWGVj0M5bcQy5i+U4hsyxdjaGXyZ5Noa1chzDYatXGMOHEvF9C6+zZVTOVu4itN+VYBWOjKun99hMyN5x+xeR5CFXZyx6DjWOKLF5Z+0rZKgU/mD6l8f2lB288O7GL/+eUUq1Fn6HAntbga1I1Qee3NbQ2YhYnodQHAW+9FCc9pVoViOh9TFe4y1BfYNb8CXbNhwJXwHzBUmvIwWutUBRPOy7Qu8s1FndufAjaORoAapD9xchMJxj0CTAEkDS6wtPRUhhcCJ7w2cRCm7ZJ28JEPAGz/JZgPk7nEGjsLhv8CxCx27YeJMxI39f7LDT9azre6qHkIXdqW/QAtxpmUnEn/F4t1n+0VItd6bLFLSbTEFDg//p7QX0Y6NPqtwTvLz3Gn06+mhubX1OyVqAFXoTqXOuDZ05cwwjO7SXUouUp3S0dceE4gCSZeHoEg4uxMEHaIUdSsKRlK5Af7k3bkU4AnYOgIO8EWoQuvvINju+30fg2OXHXn4LEpBIPgrE4z4i0e0vFMY8sF8LlOxSDrz0K6OQ4+k0fTK3NKsipGraBGYvJ4x9c40k191d4ZqWtlBuotadRQEI+PMabz7XbZXyvYnVpdxtHQTlcUADWLPsEVi9N5LS4zyhmKtjJwbSG8urYR9VY/qQ6eq3aUWNxHre4iBw4XS+wXLnjVvZb0LevBVcSroRRcnawVoheYE7LBhqrhgBohRNZED+pPqrHk9+oxA/nHzAQQ9B58e0h4xl6OHY749oD/bxW9DDeq+E9vAfD/Zw6BDrYb/1e32B+r4L9yaPsw2fN/sHwK4CN/wbPL7fJL/R4MnFL/ChbOf5XdjvXcnjD6hO3FcnA7PDKNuh8Tr/Nl57Z4MVXKeZBbRil+nzHqtolYS09HATjVmuKl7FS6c7jto1uTNHAZRaeqompfSSwUiyXFpbG+FVMpWM0HtGRo7ar8015M58ai/klRS78rXHyZVSklXPn408ls1MSm5HzpJSz8qnMtvW1gg3DHqK1Ef3eKIwKv5uzbxjW2zebeO8617R5gpOvMUmjpUxjAaU6zLO3fY0cYctXWmzmuKVK/ROkONKt+3s9q/gH2b1boPn3eP4mxm0MJqXtij2ujOO3OU+HNnOWHdGv/M+ZNvClOtmNtTVLKyh4k8Gwc1dFA3mEuXkSThH82oxJVrFnxMX2DNIHvw/jLHDIRTOAgA=';
let cachedInterAtlas = null;

function getInterAtlas() {
  if (cachedInterAtlas) return cachedInterAtlas;
  const buf = zlib.gunzipSync(Buffer.from(INTER_ATLAS_GZ_B64, 'base64'));
  const nl = buf.indexOf(10);
  const meta = JSON.parse(buf.subarray(0, nl).toString('utf8'));
  const alpha = new Uint8Array(buf.buffer, buf.byteOffset + nl + 1, buf.length - (nl + 1));
  cachedInterAtlas = { styles: meta.styles, alpha };
  return cachedInterAtlas;
}

function resolveAtlasChar(ch, st) {
  if (!ch) return ' ';
  const code = String(ch.codePointAt(0));
  if (st.glyphs[code]) return ch;
  if (ch === '\u00a0') return ' ';
  if (ch === '\u2212' || ch === '\u2010' || ch === '\u2011') return '-';
  const folded = ch.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  if (folded && folded[0]) {
    const fc = String(folded.codePointAt(0));
    if (st.glyphs[fc]) return folded[0];
  }
  return st.glyphs['63'] ? '?' : ' ';
}

export function measureInterText(text, styleKey = 'b13') {
  const atlas = getInterAtlas();
  const st = atlas.styles[styleKey] || atlas.styles.b13;
  const s = String(text ?? '');
  if (!s.length) return 0;
  let w16 = 0;
  let prev = '';
  for (const rawCh of s) {
    const ch = resolveAtlasChar(rawCh, st);
    const g = st.glyphs[String(ch.codePointAt(0))];
    if (prev && st.kern) {
      w16 += st.kern[prev + ch] || 0;
    }
    w16 += g ? g[0] : 0;
    prev = ch;
  }
  return Math.ceil(w16 / 16);
}

export function wrapInterText(text, maxWidthPx, styleKey = 'b13', maxLines = 3) {
  const clean = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (!clean) return [''];
  const words = clean.split(' ');
  const lines = [];
  let cur = '';

  const clipWordWithEllipsis = (word) => {
    const ell = '…';
    if (measureInterText(word, styleKey) <= maxWidthPx) return word;
    let out = '';
    for (const ch of word) {
      if (measureInterText(out + ch + ell, styleKey) > maxWidthPx) break;
      out += ch;
    }
    return out ? out + ell : ell;
  };

  for (const w of words) {
    const cand = cur ? cur + ' ' + w : w;
    if (measureInterText(cand, styleKey) <= maxWidthPx) {
      cur = cand;
    } else if (!cur) {
      // Single long token across remaining width
      let chunk = '';
      for (const ch of w) {
        if (measureInterText(chunk + ch, styleKey) > maxWidthPx && chunk) {
          lines.push(chunk);
          chunk = ch;
          if (lines.length >= maxLines) break;
        } else {
          chunk += ch;
        }
      }
      cur = lines.length >= maxLines ? '' : chunk;
    } else {
      lines.push(cur);
      if (lines.length >= maxLines) {
        cur = '';
        break;
      }
      cur = measureInterText(w, styleKey) <= maxWidthPx ? w : clipWordWithEllipsis(w);
    }
    if (lines.length >= maxLines) break;
  }
  if (cur && lines.length < maxLines) lines.push(cur);
  if (!lines.length) lines.push('');
  if (lines.length > maxLines) lines.length = maxLines;
  const last = lines[lines.length - 1];
  if (measureInterText(last, styleKey) > maxWidthPx) {
    lines[lines.length - 1] = clipWordWithEllipsis(last);
  }
  return lines;
}

export function drawInterText(rgba, imgW, imgH, text, x0, yTop, styleKey, color, clipRect = null) {
  const atlas = getInterAtlas();
  const st = atlas.styles[styleKey] || atlas.styles.b13;
  const alphaBuf = atlas.alpha;
  const s = String(text ?? '');
  const baseLineY = Math.round(yTop + st.ascent);
  const clipX0 = clipRect ? Math.max(0, clipRect.x | 0) : 0;
  const clipY0 = clipRect ? Math.max(0, clipRect.y | 0) : 0;
  const clipX1 = clipRect ? Math.min(imgW, (clipRect.x + clipRect.w) | 0) : imgW;
  const clipY1 = clipRect ? Math.min(imgH, (clipRect.y + clipRect.h) | 0) : imgH;

  let cursor16 = Math.round(x0 * 16);
  let prev = '';
  for (const rawCh of s) {
    const ch = resolveAtlasChar(rawCh, st);
    const g = st.glyphs[String(ch.codePointAt(0))];
    if (!g) continue;
    if (prev && st.kern) cursor16 += st.kern[prev + ch] || 0;
    const [adv, ox, oy, gw, gh, dataOff] = g;
    if (gw > 0 && gh > 0) {
      const gx = Math.round(cursor16 / 16) + ox;
      const gy = baseLineY + oy;
      for (let py = 0; py < gh; py++) {
        const y = gy + py;
        if (y < clipY0 || y >= clipY1) continue;
        for (let px = 0; px < gw; px++) {
          const x = gx + px;
          if (x < clipX0 || x >= clipX1) continue;
          const a = alphaBuf[dataOff + py * gw + px];
          if (!a) continue;
          const o = (y * imgW + x) * 4;
          if (a >= 254) {
            rgba[o] = color[0];
            rgba[o + 1] = color[1];
            rgba[o + 2] = color[2];
            rgba[o + 3] = 255;
          } else {
            const inv = 255 - a;
            rgba[o] = Math.round((color[0] * a + rgba[o] * inv) / 255);
            rgba[o + 1] = Math.round((color[1] * a + rgba[o + 1] * inv) / 255);
            rgba[o + 2] = Math.round((color[2] * a + rgba[o + 2] * inv) / 255);
            rgba[o + 3] = 255;
          }
        }
      }
    }
    cursor16 += adv;
    prev = ch;
  }
}

function fillRectRgba(rgba, imgW, imgH, x0, y0, w, h, color) {
  const xStart = Math.max(0, x0 | 0);
  const yStart = Math.max(0, y0 | 0);
  const xEnd = Math.min(imgW, (x0 + w) | 0);
  const yEnd = Math.min(imgH, (y0 + h) | 0);
  for (let y = yStart; y < yEnd; y++) {
    for (let x = xStart; x < xEnd; x++) {
      const o = (y * imgW + x) * 4;
      rgba[o] = color[0];
      rgba[o + 1] = color[1];
      rgba[o + 2] = color[2];
      rgba[o + 3] = color[3] ?? 255;
    }
  }
}

function strokeRectRgba(rgba, imgW, imgH, x0, y0, w, h, color) {
  fillRectRgba(rgba, imgW, imgH, x0, y0, w, 1, color);
  fillRectRgba(rgba, imgW, imgH, x0, y0 + h - 1, w, 1, color);
  fillRectRgba(rgba, imgW, imgH, x0, y0, 1, h, color);
  fillRectRgba(rgba, imgW, imgH, x0 + w - 1, y0, 1, h, color);
}

function decodeImageInput(imgInput) {
  if (!imgInput) return null;
  if (typeof imgInput === 'object' && !Buffer.isBuffer(imgInput) && !(imgInput instanceof Uint8Array)) {
    if (imgInput.rgba && Number(imgInput.width) > 0 && Number(imgInput.height) > 0) {
      return { width: Number(imgInput.width), height: Number(imgInput.height), rgba: imgInput.rgba };
    }
    const inner = imgInput.img || imgInput.dataUrl || imgInput.src || imgInput.url;
    if (!inner) return null;
    imgInput = inner;
  }
  let buf = imgInput;
  if (typeof imgInput === 'string') {
    const trimmed = imgInput.trim();
    if (!trimmed) return null;
    const m = /^data:image\/png(?:;[^,]+)*;base64,([\s\S]+)$/i.exec(trimmed);
    if (!m && /^(https?:\/\/|\/)/i.test(trimmed)) return null;
    const b64 = (m ? m[1] : trimmed).replace(/\s+/g, '');
    buf = Buffer.from(b64, 'base64');
  }
  return decodePng(buf);
}

function blitSpriteContainedNearestRgba(dstRgba, dstW, dstH, srcRgba, srcW, srcH, boxX, boxY, boxW, boxH, bgColor) {
  if (srcW <= 0 || srcH <= 0 || boxW <= 0 || boxH <= 0) return;
  let drawW, drawH;
  if (srcW <= boxW && srcH <= boxH) {
    const intScale = Math.max(1, Math.floor(Math.min(boxW / srcW, boxH / srcH)));
    drawW = srcW * intScale;
    drawH = srcH * intScale;
  } else {
    const fitRatio = Math.min(boxW / srcW, boxH / srcH);
    drawW = Math.max(1, Math.min(boxW, Math.round(srcW * fitRatio)));
    drawH = Math.max(1, Math.min(boxH, Math.round(srcH * fitRatio)));
  }

  const ox = boxX + ((boxW - drawW) >> 1);
  const oy = boxY + ((boxH - drawH) >> 1);
  const clipX0 = Math.max(0, boxX, ox);
  const clipY0 = Math.max(0, boxY, oy);
  const clipX1 = Math.min(dstW, boxX + boxW, ox + drawW);
  const clipY1 = Math.min(dstH, boxY + boxH, oy + drawH);

  for (let y = clipY0; y < clipY1; y++) {
    const dy = y - oy;
    const sy = Math.min(srcH - 1, Math.floor(((dy + 0.5) * srcH) / drawH));
    for (let x = clipX0; x < clipX1; x++) {
      const dx = x - ox;
      const sx = Math.min(srcW - 1, Math.floor(((dx + 0.5) * srcW) / drawW));
      const si = (sy * srcW + sx) * 4;
      const a = srcRgba[si + 3];
      if (a < 8) continue;
      let r = srcRgba[si], g = srcRgba[si + 1], b = srcRgba[si + 2];
      if (a < 250 && bgColor) {
        const inv = 255 - a;
        r = Math.round((r * a + bgColor[0] * inv) / 255);
        g = Math.round((g * a + bgColor[1] * inv) / 255);
        b = Math.round((b * a + bgColor[2] * inv) / 255);
      }
      const o = (y * dstW + x) * 4;
      dstRgba[o] = r;
      dstRgba[o + 1] = g;
      dstRgba[o + 2] = b;
      dstRgba[o + 3] = 255;
    }
  }
}

/**
 * Render a comparison grid PNG (~1024px wide) with crisp nearest-neighbour contained sprites
 * on a light grey background and readable Inter labels for title, meta, column headers (+ train counts), and row subjects.
 */
export function renderGridPng(grid, options = {}) {
  const rows = Array.isArray(grid?.rows) ? grid.rows : (Array.isArray(grid?.subjects) ? grid.subjects : []);
  const columns = Array.isArray(grid?.columns)
    ? grid.columns.map((c) => (typeof c === 'string' ? { phrase: c, train_count: null } : c))
    : [];
  const numRows = Math.max(1, rows.length);
  const numCols = Math.max(1, columns.length);
  const samples = Math.max(1, Math.min(8, Number(grid?.samples) || 3));

  const atlas = getInterAtlas();
  const targetWidth = Math.max(480, Math.min(2400, Number(options.targetWidth) || 1024));
  const bgColor = parseBgColor(options.bg) || [220, 220, 220, 255];       // #dcdcdc
  const cellBg = [212, 212, 212, 255];                                     // #d4d4d4
  const headerBg = [204, 204, 204, 255];                                   // #cccccc
  const borderCol = [174, 174, 174, 255];                                  // #aeaeae
  const slotBg = [196, 199, 206, 255];                                     // #c4c7ce
  const errCellBg = [226, 202, 202, 255];                                  // #e2caca
  const textDark = [24, 26, 30, 255];                                      // #181a1e
  const textMut = [78, 86, 98, 255];                                       // #4e5662
  const textErr = [165, 42, 42, 255];                                      // #a52a2a

  const padX = 16;
  const padY = 16;
  const colGap = 6;
  const rowGap = 6;

  // Measure natural subject widths to pick clean initial rowLabelW
  const maxSubjW = rows.reduce((m, s) => Math.max(m, measureInterText(String(s ?? ''), 'b13')), 80);
  let rowLabelW = Math.max(114, Math.min(numCols <= 6 ? 156 : 132, maxSubjW + 24));

  // Choose largest integer spriteScale (1..5) so total width stays around ~1024px
  let spriteScale = 1;
  let sampleGap = 4;
  let cellPad = 5;
  for (let sc = 5; sc >= 1; sc--) {
    const spSz = 16 * sc;
    const sGap = sc >= 3 ? 4 : 2;
    const cPad = sc >= 3 ? 5 : 4;
    const cW = samples * spSz + (samples - 1) * sGap + cPad * 2;
    const wNeed = padX * 2 + rowLabelW + colGap + numCols * cW + (numCols - 1) * colGap;
    if (wNeed <= targetWidth) {
      spriteScale = sc;
      sampleGap = sGap;
      cellPad = cPad;
      break;
    }
  }

  const spriteSize = 16 * spriteScale;
  const minCellW = samples * spriteSize + (samples - 1) * sampleGap + cellPad * 2;
  const minTotalW = padX * 2 + rowLabelW + colGap + numCols * minCellW + (numCols - 1) * colGap;
  const W = Math.max(targetWidth, minTotalW);
  let colW = minCellW;
  if (minTotalW < W) {
    const extra = W - minTotalW;
    const desiredExtraLabel = Math.max(0, Math.min(48, (maxSubjW + 26) - rowLabelW));
    const extraLabel = Math.min(desiredExtraLabel, Math.floor(extra * 0.25));
    rowLabelW += extraLabel;
    const perColExtra = Math.floor((extra - extraLabel) / numCols);
    colW += perColExtra;
    const rem = W - (padX * 2 + rowLabelW + colGap + numCols * colW + (numCols - 1) * colGap);
    rowLabelW += Math.max(0, rem);
  }

  // Title & metadata block in Inter
  const titleText = String(grid?.title || 'Comparison Grid').trim() || 'Comparison Grid';
  const maxTitleW = Math.max(120, W - padX * 2);
  const titleStyle = measureInterText(titleText, 'b18') <= maxTitleW ? 'b18' : 'b15';
  const titleLines = wrapInterText(titleText, maxTitleW, titleStyle, 2);
  const titleLineH = atlas.styles[titleStyle].lineHeight;

  const metaParts = [
    `${numRows}×${numCols} (${numRows * numCols} cells)`,
    `${samples} sample${samples === 1 ? '' : 's'}/cell`,
    grid?.seed != null ? `seed: ${grid.seed}` : 'seed: random',
    grid?.cond?.palette ? `palette: ${grid.cond.palette}` : null,
  ].filter(Boolean);
  const metaText = metaParts.join(' · ');
  const metaStyle = 'm12';
  const metaLines = wrapInterText(metaText, maxTitleW, metaStyle, 2);
  const metaLineH = atlas.styles[metaStyle].lineHeight;

  const titleBlockH = titleLines.length * titleLineH + 4 + metaLines.length * metaLineH + 14;

  // Measure column headers (Inter bold phrase + Inter medium train_count)
  const colTextW = Math.max(44, colW - 14);
  const colLayouts = columns.map((c) => {
    const phrase = String(c?.phrase ?? '');
    let stKey = colW >= 116 ? 'b13' : 'b12';
    let lines = wrapInterText(phrase, colTextW, stKey, 3);
    if (stKey === 'b13' && lines.length > 2) {
      const l12 = wrapInterText(phrase, colTextW, 'b12', 3);
      if (l12.length <= 2) {
        stKey = 'b12';
        lines = l12;
      }
    }
    const tc = (c && c.train_count !== null && c.train_count !== undefined && Number.isFinite(Number(c.train_count)))
      ? `train: ${Number(c.train_count)}`
      : 'train: n/a';
    const lineH = atlas.styles[stKey].lineHeight;
    return { lines, stKey, lineH, trainText: tc };
  });

  const trainLineH = atlas.styles.m11.lineHeight;
  const maxColPhraseH = Math.max(...colLayouts.map((cl) => cl.lines.length * cl.lineH), 16);
  const colHeaderH = Math.max(48, maxColPhraseH + 6 + trainLineH + 14);

  // Measure row subject labels in Inter bold
  const rowTextW = Math.max(48, rowLabelW - 18);
  const rowLayouts = rows.map((subj) => {
    const text = String(subj ?? '');
    let stKey = rowLabelW >= 120 ? 'b13' : 'b12';
    let lines = wrapInterText(text, rowTextW, stKey, 3);
    if (stKey === 'b13' && lines.length > 2) {
      const l12 = wrapInterText(text, rowTextW, 'b12', 3);
      if (l12.length <= 2) {
        stKey = 'b12';
        lines = l12;
      }
    }
    const lineH = atlas.styles[stKey].lineHeight;
    return { lines, stKey, lineH };
  });

  const maxRowTextH = Math.max(...rowLayouts.map((rl) => rl.lines.length * rl.lineH), 16);
  const cellH = Math.max(spriteSize + cellPad * 2, maxRowTextH + 14);
  const H = padY * 2 + titleBlockH + colHeaderH + rowGap + numRows * cellH + (numRows - 1) * rowGap;

  const out = new Uint8Array(W * H * 4);
  fillRectRgba(out, W, H, 0, 0, W, H, bgColor);

  // Draw title and metadata
  let curY = padY;
  for (const tl of titleLines) {
    drawInterText(out, W, H, tl, padX, curY, titleStyle, textDark, { x: padX, y: curY, w: maxTitleW, h: titleLineH + 4 });
    curY += titleLineH;
  }
  curY += 4;
  for (const ml of metaLines) {
    drawInterText(out, W, H, ml, padX, curY, metaStyle, textMut, { x: padX, y: curY, w: maxTitleW, h: metaLineH + 4 });
    curY += metaLineH;
  }
  curY += 14;

  // Draw top-left corner cell and column headers
  const tableX0 = padX;
  const colsX0 = tableX0 + rowLabelW + colGap;
  fillRectRgba(out, W, H, tableX0, curY, rowLabelW, colHeaderH, headerBg);
  strokeRectRgba(out, W, H, tableX0, curY, rowLabelW, colHeaderH, borderCol);
  const cornerText = 'Subject \\ Variant';
  const cornerLines = wrapInterText(cornerText, rowTextW, 'm11', 2);
  const cornerLineH = atlas.styles.m11.lineHeight;
  let cty = curY + colHeaderH - 8 - cornerLines.length * cornerLineH;
  for (const cln of cornerLines) {
    drawInterText(out, W, H, cln, tableX0 + 8, Math.max(curY + 6, cty), 'm11', textMut, {
      x: tableX0 + 4, y: curY + 4, w: rowLabelW - 8, h: colHeaderH - 8,
    });
    cty += cornerLineH;
  }

  for (let c = 0; c < numCols; c++) {
    const cx = colsX0 + c * (colW + colGap);
    fillRectRgba(out, W, H, cx, curY, colW, colHeaderH, headerBg);
    strokeRectRgba(out, W, H, cx, curY, colW, colHeaderH, borderCol);
    const cl = colLayouts[c] || { lines: [''], stKey: 'b13', lineH: 16, trainText: 'train: n/a' };
    const clip = { x: cx + 4, y: curY + 4, w: colW - 8, h: colHeaderH - 8 };
    const phraseBlockH = cl.lines.length * cl.lineH;
    const trainAreaTop = curY + colHeaderH - trainLineH - 7;
    const availPhraseH = trainAreaTop - (curY + 5);
    let ty = curY + 5 + Math.max(0, Math.floor((availPhraseH - phraseBlockH) / 2));
    for (const ln of cl.lines) {
      const lw = measureInterText(ln, cl.stKey);
      const lx = cx + Math.max(6, (colW - lw) >> 1);
      drawInterText(out, W, H, ln, lx, ty, cl.stKey, textDark, clip);
      ty += cl.lineH;
    }
    const tw = measureInterText(cl.trainText, 'm11');
    const tx = cx + Math.max(6, (colW - tw) >> 1);
    drawInterText(out, W, H, cl.trainText, tx, trainAreaTop, 'm11', textMut, clip);
  }

  curY += colHeaderH + rowGap;

  // Build quick cell lookup
  const cellLookup = new Map();
  if (Array.isArray(grid?.cells)) {
    for (let r = 0; r < grid.cells.length; r++) {
      const rowItem = grid.cells[r];
      if (Array.isArray(rowItem)) {
        for (let c = 0; c < rowItem.length; c++) {
          cellLookup.set(`${r}:${c}`, rowItem[c]);
        }
      } else if (rowItem && Number.isInteger(rowItem.row) && Number.isInteger(rowItem.col)) {
        cellLookup.set(`${rowItem.row}:${rowItem.col}`, rowItem);
      }
    }
  }

  // Draw rows & cells
  for (let r = 0; r < numRows; r++) {
    const ry = curY + r * (cellH + rowGap);
    fillRectRgba(out, W, H, tableX0, ry, rowLabelW, cellH, headerBg);
    strokeRectRgba(out, W, H, tableX0, ry, rowLabelW, cellH, borderCol);

    const rl = rowLayouts[r] || { lines: [''], stKey: 'b13', lineH: 16 };
    const rBlockH = rl.lines.length * rl.lineH;
    let rty = ry + Math.max(4, (cellH - rBlockH) >> 1);
    const rowClip = { x: tableX0 + 4, y: ry + 3, w: rowLabelW - 8, h: cellH - 6 };
    for (const ln of rl.lines) {
      const lw = measureInterText(ln, rl.stKey);
      const rx = tableX0 + Math.max(8, rowLabelW - 10 - lw);
      drawInterText(out, W, H, ln, rx, rty, rl.stKey, textDark, rowClip);
      rty += rl.lineH;
    }

    for (let c = 0; c < numCols; c++) {
      const cx = colsX0 + c * (colW + colGap);
      const cell = cellLookup.get(`${r}:${c}`);
      const isErr = cell?.status === 'error';
      const curCellBg = isErr ? errCellBg : cellBg;
      fillRectRgba(out, W, H, cx, ry, colW, cellH, curCellBg);
      strokeRectRgba(out, W, H, cx, ry, colW, cellH, isErr ? [180, 72, 72, 255] : borderCol);

      const imgs = Array.isArray(cell?.images) ? cell.images : [];
      const groupW = samples * spriteSize + (samples - 1) * sampleGap;
      const sxStart = cx + Math.max(cellPad, (colW - groupW) >> 1);
      const syStart = ry + ((cellH - spriteSize) >> 1);

      if (isErr && !imgs.length) {
        const ew = measureInterText('error', 'm11');
        const ey = ry + ((cellH - atlas.styles.m11.lineHeight) >> 1);
        drawInterText(out, W, H, 'error', cx + Math.max(4, (colW - ew) >> 1), ey, 'm11', textErr, {
          x: cx + 2, y: ry + 2, w: colW - 4, h: cellH - 4,
        });
        continue;
      }

      for (let s = 0; s < samples; s++) {
        const sx = sxStart + s * (spriteSize + sampleGap);
        const imObj = imgs[s];
        let rendered = false;
        if (imObj) {
          try {
            const dec = decodeImageInput(imObj);
            if (dec && dec.rgba && dec.width > 0 && dec.height > 0) {
              blitSpriteContainedNearestRgba(out, W, H, dec.rgba, dec.width, dec.height, sx, syStart, spriteSize, spriteSize, curCellBg);
              rendered = true;
            }
          } catch {}
        }
        if (!rendered) {
          fillRectRgba(out, W, H, sx, syStart, spriteSize, spriteSize, isErr ? errCellBg : slotBg);
        }
      }
    }
  }

  return encodePng(W, H, out);
}

/**
 * Render a 1200x630 OG card matching Pixie's og.png aesthetic.
 */
export function renderOgPng(rgba16, prompt = '') {
  const W = 1200, H = 630;
  const out = new Uint8Array(W * H * 4);
  for (let i = 3; i < out.length; i += 4) out[i] = 255;

  const tileSize = 480;
  const tx = (W - tileSize) >> 1;
  const ty = 48;
  for (let y = ty; y < ty + tileSize; y++) {
    for (let x = tx; x < tx + tileSize; x++) {
      const o = (y * W + x) * 4;
      out[o] = 17; out[o + 1] = 17; out[o + 2] = 17;
    }
  }

  const spScale = 28;
  const spSize = 16 * spScale; // 448
  const sx0 = (W - spSize) >> 1;
  const sy0 = ty + ((tileSize - spSize) >> 1);
  for (let py = 0; py < 16; py++) {
    for (let px = 0; px < 16; px++) {
      const si = (py * 16 + px) * 4;
      const a = rgba16[si + 3];
      if (a < 16) continue;
      for (let dy = 0; dy < spScale; dy++) {
        const y = sy0 + py * spScale + dy;
        for (let dx = 0; dx < spScale; dx++) {
          const x = sx0 + px * spScale + dx;
          const o = (y * W + x) * 4;
          if (a >= 250) {
            out[o] = rgba16[si]; out[o + 1] = rgba16[si + 1]; out[o + 2] = rgba16[si + 2];
          } else {
            const inv = 255 - a;
            out[o] = Math.round((rgba16[si] * a + 17 * inv) / 255);
            out[o + 1] = Math.round((rgba16[si + 1] * a + 17 * inv) / 255);
            out[o + 2] = Math.round((rgba16[si + 2] * a + 17 * inv) / 255);
          }
        }
      }
    }
  }

  drawBitmapText(out, W, H, 'PIXIE', 36, 28, 4, [182, 220, 60], { uppercase: true });

  const cleanPrompt = String(prompt || '').trim().slice(0, 34);
  if (cleanPrompt) {
    const scale = cleanPrompt.length > 22 ? 4 : 5;
    const textW = cleanPrompt.length * 6 * scale - scale;
    const px = Math.max(24, (W - textW) >> 1);
    drawBitmapText(out, W, H, cleanPrompt, px, 558, scale, [255, 255, 255], { uppercase: true });
  }

  return encodePng(W, H, out);
}
