// Pixie Inline 16x16 Pixel Editor & Recolour Studio (zero dependencies).
(() => {
  'use strict';

  const MAX_UNDO = 64;
  const MAX_RECENT_COLORS = 10;
  const MAX_IMAGE_COLORS = 64;
  const MAX_TEXT_PALETTE_COLORS = 256;
  const LS_PALETTES_KEY = 'pxl_editor_palettes_v1';
  const DESKTOP_BP = 860;

  const BUILTIN_PALETTES = [
    {
      id: 'pico8',
      name: 'PICO-8',
      colors: [
        '#000000', '#1d2b53', '#7e2553', '#008751',
        '#ab5236', '#5f574f', '#c2c3c7', '#fff1e8',
        '#ff004d', '#ffa300', '#ffec27', '#00e436',
        '#29adff', '#83769c', '#ff77a8', '#ffccaa',
      ],
    },
    {
      id: 'endesga32',
      name: 'Endesga 32',
      colors: [
        '#be4a2f', '#d77643', '#ead4aa', '#e4a672',
        '#b86f50', '#733e39', '#3e2731', '#a22633',
        '#e43b44', '#f77622', '#feae34', '#fee761',
        '#63c74d', '#3e8948', '#265c42', '#193c3e',
        '#124e89', '#0099db', '#2ce8f5', '#ffffff',
        '#c0cbdc', '#8b9bb4', '#5a6988', '#3a4466',
        '#262b44', '#181425', '#ff0044', '#68386c',
        '#b55088', '#f6757a', '#e8b796', '#c28569',
      ],
    },
    {
      id: 'sweetie16',
      name: 'Sweetie 16',
      colors: [
        '#1a1c2c', '#5d275d', '#b13e53', '#ef7d57',
        '#ffcd75', '#a7f070', '#38b764', '#257179',
        '#29366f', '#3b5dc9', '#41a6f6', '#73eff7',
        '#f4f4f4', '#94b0c2', '#566c86', '#333c57',
      ],
    },
    {
      id: 'gameboy',
      name: 'Game Boy 4',
      colors: [
        '#0f380f', '#306230', '#8bac0f', '#9bbc0f',
      ],
    },
  ];

  // ---- Oklab + HSL + Hex color helpers ----
  function srgbToLinear(c) {
    const u = c / 255;
    return u <= 0.04045 ? u / 12.92 : Math.pow((u + 0.055) / 1.055, 2.4);
  }
  function rgbToOklab(r, g, b) {
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
  function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s = 0;
    const l = (max + min) / 2;
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h /= 6;
    }
    return [h * 360, s * 100, l * 100];
  }
  function hslToRgb(h, s, l) {
    h = ((h % 360) + 360) % 360;
    s = Math.max(0, Math.min(100, s)) / 100;
    l = Math.max(0, Math.min(100, l)) / 100;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;
    if (h < 60) [r, g, b] = [c, x, 0];
    else if (h < 120) [r, g, b] = [x, c, 0];
    else if (h < 180) [r, g, b] = [0, c, x];
    else if (h < 240) [r, g, b] = [0, x, c];
    else if (h < 300) [r, g, b] = [x, 0, c];
    else [r, g, b] = [c, 0, x];
    return [
      Math.round((r + m) * 255),
      Math.round((g + m) * 255),
      Math.round((b + m) * 255),
    ];
  }
  function rgbToHex(rgb) {
    if (!rgb) return 'transparent';
    return '#' + rgb.map((v) => (v & 255).toString(16).padStart(2, '0')).join('');
  }
  function normalizeHex(raw) {
    const s = String(raw || '').trim().replace(/^(?:0x|#)/i, '');
    if (/^[0-9a-f]{6}$/i.test(s)) return '#' + s.toLowerCase();
    if (/^[0-9a-f]{8}$/i.test(s)) return '#' + s.slice(0, 6).toLowerCase();
    if (/^[0-9a-f]{3}$/i.test(s)) {
      return '#' + s.split('').map((c) => c + c).join('').toLowerCase();
    }
    return null;
  }
  function hexToRgb(hex) {
    const norm = normalizeHex(hex);
    if (!norm) return [255, 255, 255];
    const n = parseInt(norm.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  // ---- Palette 4096 loader (cached) ----
  let palettePromise = null;
  function buildPaletteStruct(raw) {
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
  function loadPalette() {
    if (!palettePromise) {
      palettePromise = fetch('/palette4096.json', { cache: 'force-cache' })
        .then((r) => {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .catch(() => {
          const fallback = [null];
          for (let r = 0; r < 16; r++)
            for (let g = 0; g < 16; g++)
              for (let b = 0; b < 16; b++)
                fallback.push([r * 17, g * 17, b * 17]);
          return fallback;
        })
        .then(buildPaletteStruct);
    }
    return palettePromise;
  }

  function nearestPaletteIdx(pal, r, g, b) {
    const key = ((r & 255) << 16) | ((g & 255) << 8) | (b & 255);
    const exact = pal.exactMap.get(key);
    if (exact !== undefined) return exact;
    const [L, oa, ob] = rgbToOklab(r, g, b);
    const lab = pal.oklab;
    let best = 1;
    let bestD = Infinity;
    for (let i = 1; i <= 4096; i++) {
      const dL = lab[i * 3] - L;
      const da = lab[i * 3 + 1] - oa;
      const db = lab[i * 3 + 2] - ob;
      const d = dL * dL + da * da + db * db;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      if (!String(src).startsWith('data:')) im.crossOrigin = 'anonymous';
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error('Could not load sprite image'));
      im.src = src;
    });
  }

  // ---- User Palettes Persistence (localStorage) ----
  function makePaletteId() {
    if (window.crypto && crypto.randomUUID) return 'usr_' + crypto.randomUUID().slice(0, 8);
    return 'usr_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }
  function loadUserPalettes() {
    try {
      const raw = window.localStorage.getItem(LS_PALETTES_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      const out = [];
      for (const item of parsed) {
        if (!item || typeof item !== 'object') continue;
        const name = String(item.name || 'Custom Palette').trim().slice(0, 40) || 'Custom Palette';
        const id = String(item.id || makePaletteId());
        const colors = [];
        const seen = new Set();
        if (Array.isArray(item.colors)) {
          for (const c of item.colors) {
            const hex = normalizeHex(c);
            if (hex && !seen.has(hex)) { seen.add(hex); colors.push(hex); }
            if (colors.length >= MAX_TEXT_PALETTE_COLORS) break;
          }
        }
        out.push({ id, name, colors });
      }
      return out;
    } catch {
      return [];
    }
  }
  function saveUserPalettes(palettes) {
    try {
      window.localStorage.setItem(LS_PALETTES_KEY, JSON.stringify(palettes));
    } catch {}
  }

  // ---- Palette Import Parsers (.hex, .gpl, .pal, .txt, image, Lospec) ----
  function dedupeHexColors(list, limit = MAX_TEXT_PALETTE_COLORS) {
    const seen = new Set();
    const out = [];
    for (const raw of list) {
      const hex = normalizeHex(raw);
      if (hex && !seen.has(hex)) {
        seen.add(hex);
        out.push(hex);
        if (out.length >= limit) break;
      }
    }
    return out;
  }

  function parseGplPalette(text) {
    const lines = String(text || '').split(/\r?\n/);
    let name = '';
    const colors = [];
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#') || /^GIMP Palette/i.test(line) || /^Columns:/i.test(line)) continue;
      const nameMatch = /^Name:\s*(.+)$/i.exec(line);
      if (nameMatch) { name = nameMatch[1].trim(); continue; }
      const rgbMatch = /^(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})(?:\s+.*)?$/.exec(line);
      if (rgbMatch) {
        const r = Math.max(0, Math.min(255, Number(rgbMatch[1])));
        const g = Math.max(0, Math.min(255, Number(rgbMatch[2])));
        const b = Math.max(0, Math.min(255, Number(rgbMatch[3])));
        colors.push(rgbToHex([r, g, b]));
      }
    }
    return { name, colors: dedupeHexColors(colors) };
  }

  function parseJascPalPalette(text) {
    const lines = String(text || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (!lines.length || !/^JASC-PAL$/i.test(lines[0])) return null;
    const colors = [];
    for (let i = 2; i < lines.length; i++) {
      const m = /^(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})$/.exec(lines[i]);
      if (!m) continue;
      const r = Math.max(0, Math.min(255, Number(m[1])));
      const g = Math.max(0, Math.min(255, Number(m[2])));
      const b = Math.max(0, Math.min(255, Number(m[3])));
      colors.push(rgbToHex([r, g, b]));
    }
    return { name: '', colors: dedupeHexColors(colors) };
  }

  function parseHexList(text) {
    const raw = String(text || '');
    const isPaintNet = /^;\s*paint\.net/im.test(raw);
    const lines = raw.split(/\r?\n/);
    const colors = [];

    for (const rawLine of lines) {
      // Strip comments: //..., ;..., or # followed by whitespace
      const clean = rawLine
        .replace(/\/\/.*$/, '')
        .replace(/;.*$/, '')
        .replace(/(?:^|\s)#(?=\s|$).*$/, '')
        .trim();
      if (!clean) continue;

      // Match rgb(r, g, b)
      const rgbRe = /rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)/gi;
      let rm;
      while ((rm = rgbRe.exec(clean)) !== null) {
        const r = Math.max(0, Math.min(255, Number(rm[1])));
        const g = Math.max(0, Math.min(255, Number(rm[2])));
        const b = Math.max(0, Math.min(255, Number(rm[3])));
        colors.push(rgbToHex([r, g, b]));
      }
      const noRgb = clean.replace(rgbRe, ' ');

      // Match hex tokens separated by delimiters
      const tokRe = /(?:^|[\s,;[\]"'()])((?:0x|#)[0-9a-f]{3}|(?:0x|#)?[0-9a-f]{6}|(?:0x|#)?[0-9a-f]{8})(?=$|[\s,;[\]"'()])/gi;
      let tm;
      while ((tm = tokRe.exec(noRgb)) !== null) {
        let tok = tm[1].replace(/^(?:0x|#)/i, '');
        if (tok.length === 8) {
          tok = isPaintNet ? tok.slice(2) : tok.slice(0, 6);
        }
        const hex = normalizeHex(tok);
        if (hex) colors.push(hex);
      }
    }
    return dedupeHexColors(colors);
  }

  function stripFileExtension(filename) {
    return String(filename || '').replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim();
  }

  function parsePaletteText(text, filename = '') {
    const trimmed = String(text || '').trim();
    const ext = (filename.split('.').pop() || '').toLowerCase();
    const fallbackName = stripFileExtension(filename);

    if (ext === 'gpl' || /^GIMP Palette/i.test(trimmed)) {
      const res = parseGplPalette(trimmed);
      return { name: res.name || fallbackName, colors: res.colors };
    }
    if (ext === 'pal' || /^JASC-PAL/i.test(trimmed)) {
      const jasc = parseJascPalPalette(trimmed);
      if (jasc && jasc.colors.length) return { name: fallbackName, colors: jasc.colors };
    }
    return { name: fallbackName, colors: parseHexList(trimmed) };
  }

  function extractColorsFromImageFile(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const im = new Image();
      im.onload = () => {
        try {
          const maxDim = 256;
          const scale = Math.min(1, maxDim / Math.max(im.width || 1, im.height || 1));
          const w = Math.max(1, Math.round(im.width * scale));
          const h = Math.max(1, Math.round(im.height * scale));
          const cv = document.createElement('canvas');
          cv.width = w;
          cv.height = h;
          const ctx = cv.getContext('2d');
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(im, 0, 0, w, h);
          const data = ctx.getImageData(0, 0, w, h).data;
          const counts = new Map();
          for (let i = 0; i < data.length; i += 4) {
            if (data[i + 3] < 128) continue;
            const hex = rgbToHex([data[i], data[i + 1], data[i + 2]]);
            counts.set(hex, (counts.get(hex) || 0) + 1);
          }
          const totalUnique = counts.size;
          let colors;
          if (totalUnique <= MAX_IMAGE_COLORS) {
            colors = Array.from(counts.keys());
          } else {
            colors = Array.from(counts.entries())
              .sort((a, b) => b[1] - a[1])
              .slice(0, MAX_IMAGE_COLORS)
              .map(([hex]) => hex);
          }
          resolve({
            name: stripFileExtension(file.name) || 'Image Palette',
            colors,
            totalUnique,
          });
        } catch (err) {
          reject(err);
        } finally {
          URL.revokeObjectURL(url);
        }
      };
      im.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Could not decode image file.'));
      };
      im.src = url;
    });
  }

  function normalizeLospecSlug(raw) {
    return String(raw || '')
      .trim()
      .toLowerCase()
      .replace(/^https?:\/\/(?:www\.)?lospec\.com\/palette-list\//, '')
      .replace(/\.json$/i, '')
      .replace(/\/+$/, '')
      .replace(/\s+/g, '-');
  }

  async function fetchLospecPalette(rawSlug) {
    const slug = normalizeLospecSlug(rawSlug);
    if (!slug) throw new Error('Enter a Lospec palette slug (e.g. resurrect-64).');
    const url = `https://lospec.com/palette-list/${encodeURIComponent(slug)}.json`;
    const res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error(`Lospec returned HTTP ${res.status} for "${slug}".`);
    const j = await res.json();
    const colors = dedupeHexColors(Array.isArray(j.colors) ? j.colors : []);
    if (!colors.length) throw new Error(`No colours returned for Lospec palette "${slug}".`);
    return {
      name: String(j.name || slug).trim(),
      colors,
    };
  }

  // ---- Palette Export Helpers ----
  function slugifyName(name) {
    return String(name || 'palette')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'palette';
  }

  function triggerBlobDownload(blob, filename) {
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function exportPalette(kind, name, colors) {
    const stem = slugifyName(name);
    if (kind === 'hex') {
      const content = colors.map((h) => h.replace('#', '').toLowerCase()).join('\n') + '\n';
      triggerBlobDownload(new Blob([content], { type: 'text/plain;charset=utf-8' }), `${stem}.hex`);
      return `Exported ${stem}.hex (${colors.length} colours).`;
    }
    if (kind === 'gpl') {
      const cols = Math.min(8, Math.max(1, colors.length));
      const lines = ['GIMP Palette', `Name: ${name}`, `Columns: ${cols}`, '#'];
      for (const h of colors) {
        const [r, g, b] = hexToRgb(h);
        lines.push(`${String(r).padStart(3)} ${String(g).padStart(3)} ${String(b).padStart(3)}\t${h}`);
      }
      const content = lines.join('\n') + '\n';
      triggerBlobDownload(new Blob([content], { type: 'text/plain;charset=utf-8' }), `${stem}.gpl`);
      return `Exported ${stem}.gpl (${colors.length} colours).`;
    }
    if (kind === 'png1') {
      const cv = document.createElement('canvas');
      cv.width = colors.length;
      cv.height = 1;
      const ctx = cv.getContext('2d');
      colors.forEach((h, i) => {
        ctx.fillStyle = h;
        ctx.fillRect(i, 0, 1, 1);
      });
      cv.toBlob((blob) => triggerBlobDownload(blob, `${stem}-1px-strip.png`), 'image/png');
      return `Exported ${stem}-1px-strip.png (${colors.length}×1).`;
    }
    // png8: 8px swatches grid
    const cols = Math.min(colors.length, 8);
    const rows = Math.ceil(colors.length / cols);
    const cv = document.createElement('canvas');
    cv.width = cols * 8;
    cv.height = rows * 8;
    const ctx = cv.getContext('2d');
    colors.forEach((h, i) => {
      const x = (i % cols) * 8;
      const y = Math.floor(i / cols) * 8;
      ctx.fillStyle = h;
      ctx.fillRect(x, y, 8, 8);
    });
    cv.toBlob((blob) => triggerBlobDownload(blob, `${stem}-8px-swatches.png`), 'image/png');
    return `Exported ${stem}-8px-swatches.png (${cv.width}×${cv.height}).`;
  }

  // ---- Singleton Editor Modal ----
  let modalEl = null;
  let state = null;

  function buildModalDom() {
    const wrap = document.createElement('div');
    wrap.className = 'px-ed-backdrop';
    wrap.id = 'pxEditorModal';
    wrap.hidden = true;
    wrap.innerHTML = `
      <div class="px-ed-modal" role="dialog" aria-modal="true" aria-labelledby="pxEdTitle">
        <div class="px-ed-head">
          <div class="px-ed-title-wrap">
            <svg class="px-mark" width="20" height="22" aria-hidden="true"><use href="#pxMark"/></svg>
            <h2 class="px-ed-title" id="pxEdTitle">Pixel Editor</h2>
            <span class="px-ed-prompt" data-ed="prompt"></span>
          </div>
          <span class="px-ed-diff" data-ed="diff">0 px changed</span>
          <button type="button" class="px-ed-close" data-ed="close" aria-label="Close editor">Close [Esc]</button>
        </div>

        <div class="px-ed-body">
          <div class="px-ed-main-col">
            <div class="px-ed-stage-col">
              <div class="px-ed-toolbar" role="toolbar" aria-label="Canvas tools">
                <button type="button" class="px-ed-tool is-active" data-tool="pencil" aria-pressed="true" title="Pencil (B)">Pencil</button>
                <button type="button" class="px-ed-tool" data-tool="eraser" aria-pressed="false" title="Eraser (E)">Eraser</button>
                <button type="button" class="px-ed-tool" data-tool="eyedropper" aria-pressed="false" title="Eyedropper (I or Alt+Click)">Picker</button>
                <button type="button" class="px-ed-tool" data-ed="mirror" aria-pressed="false" title="Mirror X (M)">Mirror X</button>
                <button type="button" class="px-ed-tool is-active" data-ed="grid" aria-pressed="true" title="Toggle 16x16 grid (H)">Grid</button>
                <button type="button" class="px-ed-tool" data-ed="undo" title="Undo (Ctrl+Z)" disabled>Undo</button>
                <button type="button" class="px-ed-tool" data-ed="redo" title="Redo (Ctrl+Y)" disabled>Redo</button>
                <button type="button" class="px-ed-tool" data-ed="reset" title="Reset to original sprite">Reset</button>
              </div>

              <div class="px-ed-canvas-wrap" data-ed="stageWrap">
                <canvas class="px-ed-canvas" width="16" height="16" data-ed="canvas"></canvas>
                <canvas class="px-ed-overlay" width="384" height="384" data-ed="overlay"></canvas>
              </div>

              <div class="px-ed-substage">
                <div class="px-ed-previews">
                  <div class="px-ed-prev-item">
                    <span>1×</span>
                    <div class="px-ed-prev-box"><canvas class="px-ed-prev-1x" width="16" height="16" data-ed="prev1"></canvas></div>
                  </div>
                  <div class="px-ed-prev-item">
                    <span>4×</span>
                    <div class="px-ed-prev-box"><canvas class="px-ed-prev-4x" width="16" height="16" data-ed="prev4"></canvas></div>
                  </div>
                  <button type="button" class="px-ed-tool" data-ed="compare" title="Hold to view original sprite">Original</button>
                </div>
                <div class="px-ed-nudge" role="group" aria-label="Nudge sprite 1 pixel">
                  <button type="button" class="px-ed-nudge-btn" data-nudge="-1,0" title="Nudge Left">←</button>
                  <button type="button" class="px-ed-nudge-btn" data-nudge="0,-1" title="Nudge Up">↑</button>
                  <button type="button" class="px-ed-nudge-btn" data-nudge="0,1" title="Nudge Down">↓</button>
                  <button type="button" class="px-ed-nudge-btn" data-nudge="1,0" title="Nudge Right">→</button>
                </div>
              </div>
            </div>

            <div class="px-ed-foot">
              <div class="px-ed-export-opts">
                <label>Size
                  <select data-ed="dlSize">
                    <option value="16">16×16</option>
                    <option value="32">32×32</option>
                    <option value="64">64×64</option>
                    <option value="512" selected>512×512</option>
                  </select>
                </label>
                <label>Bg
                  <select data-ed="dlBg">
                    <option value="transparent" selected>Transparent</option>
                    <option value="black">Black (#000)</option>
                    <option value="surface">Surface (#111)</option>
                    <option value="custom">Custom…</option>
                  </select>
                </label>
                <input type="color" class="px-ed-color-input" data-ed="dlCustomBg" value="#1a1a1a" hidden aria-label="Custom background colour">
                <label class="px-ed-optin">
                  <input type="checkbox" data-ed="optIn" checked>
                  <span>Allow edit to help train Pixie</span>
                </label>
              </div>
              <div class="px-ed-actions">
                <button type="button" class="px-ed-btn" data-ed="save">Save edit</button>
                <button type="button" class="px-ed-btn" data-ed="download">Download PNG</button>
                <button type="button" class="px-ed-btn px-ed-btn-accent" data-ed="share">Save &amp; Share</button>
              </div>
            </div>
            <div class="px-ed-status" data-ed="status" aria-live="polite"></div>
          </div>

          <div class="px-ed-sheet-scrim" data-ed="sheetScrim" hidden></div>

          <aside class="px-ed-sidebar" data-ed="sidebar" data-sheet="collapsed" aria-label="Palette, fill, outline and colour controls">
            <div class="px-ed-sheet-bar" data-ed="sheetBar">
              <div class="px-ed-sheet-grip" aria-hidden="true"></div>
              <div class="px-ed-sheet-bar-row">
                <span class="px-ed-sheet-dot" data-ed="sheetDot" aria-hidden="true"></span>
                <span class="px-ed-sheet-summary" data-ed="sheetSummary">Palette &amp; Tools</span>
                <button type="button" class="px-ed-sheet-toggle" data-ed="sheetToggle" aria-expanded="false">Palette ▲</button>
              </div>
            </div>

            <div class="px-ed-sidebar-scroll" data-ed="sidebarScroll">
              <section class="px-ed-panel" aria-label="Fill and outline tools">
                <div class="px-ed-panel-head">
                  <span class="px-ed-panel-title">Fill &amp; Outline</span>
                  <span class="px-ed-cur-readout" data-ed="toolHint">Click canvas for Fill</span>
                </div>
                <div class="px-ed-tool-row">
                  <button type="button" class="px-ed-tool" data-tool="fill" aria-pressed="false" title="Bucket Fill (G)">Bucket Fill (G)</button>
                  <button type="button" class="px-ed-tool" data-ed="fillGlobal" aria-pressed="false" title="Replace every pixel of the clicked colour">Global replace</button>
                </div>
                <div class="px-ed-tool-row">
                  <button type="button" class="px-ed-tool" data-ed="outlineApply" title="Add 1px outline around non-transparent shape in active colour">Add 1px Outline</button>
                  <div class="px-ed-seg" role="group" aria-label="Outline mode">
                    <button type="button" class="px-ed-seg-btn" data-outline-mode="outer" aria-pressed="true">Outer only</button>
                    <button type="button" class="px-ed-seg-btn" data-outline-mode="inner" aria-pressed="false">Inner</button>
                  </div>
                </div>
              </section>

              <section class="px-ed-panel" aria-label="Colour controls">
                <div class="px-ed-panel-head">
                  <span class="px-ed-panel-title">Active Colour</span>
                  <span class="px-ed-cur-readout" data-ed="curReadout">—</span>
                </div>
                <div class="px-ed-colour-row">
                  <input type="color" class="px-ed-color-input" data-ed="colorPicker" value="#b6dc3c" aria-label="Active colour picker">
                  <input type="text" class="px-ed-hex-input" data-ed="hexInput" maxlength="7" value="#b6dc3c" aria-label="Active hex code">
                  <button type="button" class="px-ed-tool" data-ed="pickEraser" title="Switch to Transparent / Eraser">Eraser</button>
                  <button type="button" class="px-ed-tool" data-ed="swapMode" aria-pressed="false" title="Replace all pixels of active colour with next picked swatch">Swap</button>
                </div>
                <span class="px-ed-cur-readout" data-ed="swapHint">Click swatch or pick colour</span>
                <div class="px-ed-recent-wrap">
                  <span class="px-ed-sub-label">Recent</span>
                  <div class="px-ed-recent" data-ed="recentSwatches" aria-label="Recent colours"></div>
                </div>
              </section>

              <section class="px-ed-panel" aria-label="Palette manager">
                <div class="px-ed-panel-head">
                  <span class="px-ed-panel-title">Palette</span>
                  <div class="px-ed-pal-actions">
                    <button type="button" class="px-ed-tool" data-ed="importBtn" title="Import palette from file, hex list, image, or Lospec">Import</button>
                    <button type="button" class="px-ed-tool" data-ed="exportBtn" aria-expanded="false" title="Export active palette">Export ▾</button>
                    <button type="button" class="px-ed-tool" data-ed="newPalBtn" title="Clone current palette to an editable custom palette">+ New</button>
                  </div>
                </div>
                <div class="px-ed-pal-select-row">
                  <select class="px-ed-select" data-ed="paletteSelect" aria-label="Select palette"></select>
                  <button type="button" class="px-ed-tool" data-ed="snapPalette" title="Remap sprite colours to selected palette">Snap sprite</button>
                </div>
                <div class="px-ed-swatches" data-ed="swatches" role="list" aria-label="Palette swatches"></div>
                <div class="px-ed-user-pal-bar" data-ed="userPalBar" hidden>
                  <div class="px-ed-user-pal-row">
                    <input type="text" class="px-ed-input-sm" data-ed="userPalName" maxlength="40" aria-label="Palette name">
                    <button type="button" class="px-ed-tool" data-ed="renamePalBtn">Rename</button>
                    <button type="button" class="px-ed-tool px-ed-tool-danger" data-ed="deletePalBtn">Delete</button>
                  </div>
                  <div class="px-ed-user-pal-row">
                    <button type="button" class="px-ed-tool" data-ed="addActiveToPal" title="Add active colour to this palette">+ Active</button>
                    <label class="px-ed-pick-add-label" title="Pick a colour to add to this palette">
                      <input type="color" class="px-ed-color-mini" data-ed="addPalPicker" value="#b6dc3c">
                      <span>+ Pick</span>
                    </label>
                    <button type="button" class="px-ed-tool" data-ed="moveLeft" title="Move selected swatch left">◀</button>
                    <button type="button" class="px-ed-tool" data-ed="moveRight" title="Move selected swatch right">▶</button>
                    <button type="button" class="px-ed-tool px-ed-tool-danger" data-ed="removeFromPal" title="Remove selected swatch">✕</button>
                  </div>
                </div>
              </section>

              <details class="px-ed-panel px-ed-details" data-ed="recolorDetails">
                <summary class="px-ed-panel-head">
                  <span class="px-ed-panel-title">Recolour &amp; Shift</span>
                </summary>
                <div class="px-ed-details-body">
                  <div class="px-ed-sliders">
                    <label class="px-ed-slider-row">
                      <span>Hue</span>
                      <input type="range" min="-180" max="180" step="5" value="0" data-ed="hueSlider">
                      <span class="px-ed-slider-val" data-ed="hueVal">0°</span>
                    </label>
                    <label class="px-ed-slider-row">
                      <span>Lightness</span>
                      <input type="range" min="-40" max="40" step="2" value="0" data-ed="lightSlider">
                      <span class="px-ed-slider-val" data-ed="lightVal">0%</span>
                    </label>
                  </div>
                  <div class="px-ed-tool-row">
                    <button type="button" class="px-ed-tool" data-ed="applyRecolor" disabled>Apply Shift</button>
                    <button type="button" class="px-ed-tool" data-ed="resetRecolor" disabled>Reset</button>
                    <button type="button" class="px-ed-tool" data-ed="strip" title="Download 64x16 4-colourway strip">4-Strip PNG</button>
                  </div>
                </div>
              </details>
            </div>
          </aside>
        </div>
      </div>

      <div class="px-ed-export-menu" data-ed="exportMenu" role="menu" aria-label="Export palette formats" hidden>
        <div class="px-ed-export-title" data-ed="exportTitle">Export Palette</div>
        <button type="button" class="px-ed-export-item" data-export="hex" role="menuitem"><span>Hex list</span><span class="px-ed-export-ext">.hex</span></button>
        <button type="button" class="px-ed-export-item" data-export="gpl" role="menuitem"><span>GIMP Palette</span><span class="px-ed-export-ext">.gpl</span></button>
        <button type="button" class="px-ed-export-item" data-export="png1" role="menuitem"><span>PNG 1px strip</span><span class="px-ed-export-ext">.png</span></button>
        <button type="button" class="px-ed-export-item" data-export="png8" role="menuitem"><span>PNG 8px swatches</span><span class="px-ed-export-ext">.png</span></button>
      </div>

      <div class="px-ed-dialog-backdrop" data-ed="importDialog" hidden>
        <div class="px-ed-dialog" role="dialog" aria-modal="true" aria-labelledby="pxEdImportTitle">
          <div class="px-ed-dialog-head">
            <h3 class="px-ed-dialog-title" id="pxEdImportTitle">Import Palette</h3>
            <button type="button" class="px-ed-close" data-ed="importClose">Close</button>
          </div>
          <div class="px-ed-field-group">
            <label class="px-ed-field-label" for="pxEdImportName">Palette name</label>
            <input type="text" id="pxEdImportName" class="px-ed-input-text" data-ed="importName" maxlength="40" placeholder="e.g. Dungeon 16">
          </div>
          <div class="px-ed-field-group">
            <span class="px-ed-field-label">From file (.hex, .gpl, .pal, .txt) or image (max 64 colours)</span>
            <label class="px-ed-dropzone" data-ed="importDropZone">
              <input type="file" data-ed="importFile" accept=".hex,.gpl,.pal,.txt,image/*">
              <span>Click or drop a palette or image file</span>
            </label>
          </div>
          <div class="px-ed-field-group">
            <label class="px-ed-field-label" for="pxEdImportText">Or paste hex codes (.hex / .gpl / .pal)</label>
            <textarea id="pxEdImportText" class="px-ed-textarea" data-ed="importText" placeholder="#1a1c2c #5d275d #b13e53 #ef7d57&#10;#ffcd75 #a7f070 #38b764"></textarea>
          </div>
          <div class="px-ed-field-group">
            <label class="px-ed-field-label" for="pxEdLospecSlug">Or Lospec palette slug (client-side fetch)</label>
            <div class="px-ed-lospec-row">
              <input type="text" id="pxEdLospecSlug" class="px-ed-input-text" data-ed="lospecSlug" placeholder="e.g. resurrect-64 or endesga-32">
              <button type="button" class="px-ed-tool" data-ed="lospecFetch">Fetch</button>
            </div>
          </div>
          <div class="px-ed-import-preview-box">
            <span class="px-ed-import-info" data-ed="importInfo">Add a file, paste hex codes, or fetch from Lospec.</span>
            <div class="px-ed-swatches" data-ed="importPreview"></div>
          </div>
          <div class="px-ed-dialog-foot">
            <button type="button" class="px-ed-btn" data-ed="importCancel">Cancel</button>
            <button type="button" class="px-ed-btn px-ed-btn-accent" data-ed="importConfirm" disabled>Import &amp; Select</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(wrap);
    return wrap;
  }

  function edRefs(root) {
    const r = {};
    root.querySelectorAll('[data-ed]').forEach((el) => { r[el.dataset.ed] = el; });
    return r;
  }

  function countDiffs(a, b) {
    let d = 0;
    for (let i = 0; i < 256; i++) if (a[i] !== b[i]) d++;
    return d;
  }

  function renderIndicesToCanvas(canvas, indices, pal) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(16, 16);
    const data = img.data;
    for (let i = 0; i < 256; i++) {
      const idx = indices[i];
      const o = i * 4;
      if (idx <= 0 || !pal.entries[idx]) {
        data[o] = 0; data[o + 1] = 0; data[o + 2] = 0; data[o + 3] = 0;
      } else {
        const rgb = pal.entries[idx];
        data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2]; data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function renderOverlay() {
    if (!state) return;
    const cv = state.r.overlay;
    const targetSize = Math.max(16, Math.round(cv.clientWidth || cv.width || 320));
    if (cv.width !== targetSize || cv.height !== targetSize) {
      cv.width = targetSize;
      cv.height = targetSize;
    }
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const cell = W / 16;
    ctx.clearRect(0, 0, W, H);

    if (state.showGrid) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < 16; i++) {
        const p = Math.round(i * cell) + 0.5;
        ctx.moveTo(p, 0); ctx.lineTo(p, H);
        ctx.moveTo(0, p); ctx.lineTo(W, p);
      }
      ctx.stroke();
    }

    if (state.hoverX >= 0 && state.hoverY >= 0 && !state.comparing) {
      const drawCursor = (gx, gy, col) => {
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.strokeRect(gx * cell + 1, gy * cell + 1, cell - 2, cell - 2);
      };
      drawCursor(state.hoverX, state.hoverY, '#b6dc3c');
      if (state.mirrorX && (state.tool === 'pencil' || state.tool === 'eraser' || state.tool === 'fill')) {
        const mx = 15 - state.hoverX;
        if (mx !== state.hoverX) drawCursor(mx, state.hoverY, 'rgba(182, 220, 60, 0.5)');
      }
    }
  }

  function extractSpritePaletteEntries(buf, pal) {
    const counts = new Map();
    for (let i = 0; i < 256; i++) {
      const idx = buf[i];
      if (idx > 0 && pal.entries[idx]) counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([idx, count]) => ({
        idx,
        count,
        hex: rgbToHex(pal.entries[idx]),
      }));
  }

  function findActivePaletteObj() {
    if (!state) return null;
    const id = state.activePaletteId;
    if (id === 'sprite') {
      const extracted = extractSpritePaletteEntries(state.buf, state.pal);
      return { id: 'sprite', name: 'Sprite colours', kind: 'sprite', colors: extracted.map((e) => e.hex), entries: extracted };
    }
    const builtin = BUILTIN_PALETTES.find((p) => p.id === id);
    if (builtin) return { ...builtin, kind: 'builtin' };
    const user = state.userPalettes.find((p) => p.id === id);
    if (user) return { ...user, kind: 'user' };
    state.activePaletteId = 'sprite';
    return findActivePaletteObj();
  }

  function recordRecentColor(idx) {
    if (!state || idx <= 0 || !state.pal.entries[idx]) return;
    state.recentColors = [idx, ...state.recentColors.filter((x) => x !== idx)].slice(0, MAX_RECENT_COLORS);
  }

  function selectColorByIndex(idx, { switchToPencil = true, recordRecent = true, swatchKey = null } = {}) {
    if (!state) return;
    if (state.swapMode && state.activeColor > 0 && idx > 0 && state.activeColor !== idx) {
      swapAllPixels(state.activeColor, idx);
      state.activeColor = idx;
      state.activeSwatchKey = swatchKey;
      state.swapMode = false;
      if (recordRecent) recordRecentColor(idx);
      updateColourAndPaletteUI();
      return;
    }
    state.activeColor = idx;
    state.activeSwatchKey = swatchKey;
    if (idx === 0) {
      setTool('eraser');
    } else {
      if (switchToPencil && state.tool === 'eraser') setTool('pencil');
      if (recordRecent) recordRecentColor(idx);
    }
    updateColourAndPaletteUI();
  }

  function selectColorByHex(hex, opts = {}) {
    if (!state) return;
    const [r, g, b] = hexToRgb(hex);
    const idx = nearestPaletteIdx(state.pal, r, g, b);
    selectColorByIndex(idx, opts);
  }

  function syncPaletteSelectOptions() {
    if (!state) return;
    const sel = state.r.paletteSelect;
    const spriteEntries = extractSpritePaletteEntries(state.buf, state.pal);
    sel.innerHTML = '';

    const spriteOpt = document.createElement('option');
    spriteOpt.value = 'sprite';
    spriteOpt.textContent = `Sprite colours (${spriteEntries.length})`;
    sel.appendChild(spriteOpt);

    const biGroup = document.createElement('optgroup');
    biGroup.label = 'Built-in Palettes';
    BUILTIN_PALETTES.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.name} (${p.colors.length})`;
      biGroup.appendChild(opt);
    });
    sel.appendChild(biGroup);

    if (state.userPalettes.length) {
      const uGroup = document.createElement('optgroup');
      uGroup.label = 'Imported & Custom';
      state.userPalettes.forEach((p) => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = `${p.name} (${p.colors.length})`;
        uGroup.appendChild(opt);
      });
      sel.appendChild(uGroup);
    }

    sel.value = state.activePaletteId;
    if (sel.value !== state.activePaletteId) {
      state.activePaletteId = 'sprite';
      sel.value = 'sprite';
    }
  }

  function updateSheetSummary() {
    if (!state) return;
    const { r, activeColor, pal, tool } = state;
    const palObj = findActivePaletteObj();
    const hex = activeColor > 0 && pal.entries[activeColor] ? rgbToHex(pal.entries[activeColor]) : 'Transparent';
    if (activeColor > 0 && pal.entries[activeColor]) {
      r.sheetDot.style.background = hex;
    } else {
      r.sheetDot.style.background = '';
    }
    const toolLabel = tool === 'fill' ? 'Fill' : tool === 'eyedropper' ? 'Picker' : tool === 'eraser' ? 'Eraser' : 'Pencil';
    r.sheetSummary.textContent = `${hex} · ${toolLabel} · ${palObj.name}`;
    const expanded = r.sidebar.dataset.sheet === 'expanded';
    r.sheetToggle.textContent = expanded ? 'Done ▼' : 'Palette ▲';
    r.sheetToggle.setAttribute('aria-expanded', String(expanded));
  }

  function setSheetExpanded(expanded) {
    if (!state) return;
    const isDesktop = window.innerWidth >= DESKTOP_BP;
    const open = !isDesktop && Boolean(expanded);
    state.r.sidebar.dataset.sheet = open ? 'expanded' : 'collapsed';
    state.r.sheetScrim.hidden = !open;
    if (!isDesktop) {
      state.r.sidebarScroll.inert = !open;
    } else {
      state.r.sidebarScroll.inert = false;
    }
    updateSheetSummary();
  }

  function updateDiffAndButtons() {
    if (!state) return;
    const diff = countDiffs(state.orig, state.buf);
    const el = state.r.diff;
    el.textContent = `${diff} px changed`;
    el.classList.toggle('is-polish', diff >= 1 && diff <= 24);
    el.classList.toggle('is-heavy', diff > 24);
    state.r.undo.disabled = state.undoStack.length === 0;
    state.r.redo.disabled = state.redoStack.length === 0;
  }

  function renderRecentSwatches() {
    if (!state) return;
    const box = state.r.recentSwatches;
    box.innerHTML = '';
    for (const idx of state.recentColors) {
      const rgb = state.pal.entries[idx];
      if (!rgb) continue;
      const hex = rgbToHex(rgb);
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'px-ed-swatch px-ed-swatch-sm' + (state.activeColor === idx ? ' is-active' : '');
      b.style.background = hex;
      b.title = `${hex} · #${idx}`;
      b.onclick = () => selectColorByIndex(idx);
      box.appendChild(b);
    }
  }

  function moveUserPaletteColor(fromIdx, toIdx) {
    const userPal = state.userPalettes.find((p) => p.id === state.activePaletteId);
    if (!userPal) return;
    if (fromIdx < 0 || toIdx < 0 || fromIdx >= userPal.colors.length || toIdx >= userPal.colors.length || fromIdx === toIdx) return;
    const [moved] = userPal.colors.splice(fromIdx, 1);
    userPal.colors.splice(toIdx, 0, moved);
    state.selectedPalSwatchIdx = toIdx;
    state.activeSwatchKey = `${userPal.id}:${toIdx}`;
    saveUserPalettes(state.userPalettes);
    updateColourAndPaletteUI();
  }

  function updatePaletteUI() {
    if (!state) return;
    syncPaletteSelectOptions();
    const palObj = findActivePaletteObj();
    const box = state.r.swatches;
    box.innerHTML = '';

    const isUser = palObj.kind === 'user';
    const isSprite = palObj.kind === 'sprite';

    if (isUser) {
      if (state.selectedPalSwatchIdx >= palObj.colors.length) {
        state.selectedPalSwatchIdx = palObj.colors.length - 1;
      }
      if (state.selectedPalSwatchIdx < 0 && palObj.colors.length > 0) {
        state.selectedPalSwatchIdx = 0;
      }
    }

    if (isSprite) {
      if (!palObj.entries.length) {
        const empty = document.createElement('div');
        empty.className = 'px-ed-swatches-empty';
        empty.textContent = 'No painted pixels yet. Pick a colour above or switch palettes.';
        box.appendChild(empty);
      } else {
        for (const entry of palObj.entries) {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'px-ed-swatch' + (state.activeColor === entry.idx ? ' is-active' : '');
          b.style.background = entry.hex;
          b.title = `${entry.hex} (idx ${entry.idx}, ${entry.count} px)`;
          b.onclick = () => selectColorByIndex(entry.idx, { swatchKey: `sprite:${entry.idx}` });
          box.appendChild(b);
        }
      }
    } else if (!palObj.colors.length) {
      const empty = document.createElement('div');
      empty.className = 'px-ed-swatches-empty';
      empty.textContent = 'Empty palette. Add active colour or pick a colour below.';
      box.appendChild(empty);
    } else {
      palObj.colors.forEach((hex, i) => {
        const [cr, cg, cb] = hexToRgb(hex);
        const mappedIdx = nearestPaletteIdx(state.pal, cr, cg, cb);
        const key = `${palObj.id}:${i}`;
        const isExactKey = state.activeSwatchKey === key;
        const isActive = isExactKey || (!state.activeSwatchKey && state.activeColor === mappedIdx);
        const isSelectedInUserPal = isUser && i === state.selectedPalSwatchIdx;

        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'px-ed-swatch'
          + (isActive ? ' is-active' : '')
          + (isSelectedInUserPal ? ' is-pal-selected' : '');
        b.style.background = hex;
        const snappedHex = rgbToHex(state.pal.entries[mappedIdx]);
        b.title = snappedHex !== hex
          ? `${hex} (snaps to ${snappedHex} · #${mappedIdx})`
          : `${hex} (#${mappedIdx})`;

        b.onclick = () => {
          if (isUser) state.selectedPalSwatchIdx = i;
          selectColorByIndex(mappedIdx, { swatchKey: key });
        };

        if (isUser) {
          b.draggable = true;
          b.dataset.palIdx = String(i);
          b.addEventListener('dragstart', (ev) => {
            state.dragSwatchIdx = i;
            if (ev.dataTransfer) {
              ev.dataTransfer.effectAllowed = 'move';
              ev.dataTransfer.setData('text/plain', String(i));
            }
          });
          b.addEventListener('dragover', (ev) => {
            if (state.dragSwatchIdx == null) return;
            ev.preventDefault();
            b.classList.add('is-drag-over');
          });
          b.addEventListener('dragleave', () => b.classList.remove('is-drag-over'));
          b.addEventListener('drop', (ev) => {
            ev.preventDefault();
            b.classList.remove('is-drag-over');
            const from = state.dragSwatchIdx;
            state.dragSwatchIdx = null;
            if (from != null && from !== i) {
              moveUserPaletteColor(from, i);
              setStatus(`Moved swatch to position ${i + 1}.`);
            }
          });
          b.addEventListener('dragend', () => {
            state.dragSwatchIdx = null;
            box.querySelectorAll('.is-drag-over').forEach((x) => x.classList.remove('is-drag-over'));
          });
        }
        box.appendChild(b);
      });
    }

    state.r.snapPalette.disabled = isSprite || !palObj.colors.length;
    state.r.userPalBar.hidden = !isUser;
    if (isUser) {
      if (document.activeElement !== state.r.userPalName) {
        state.r.userPalName.value = palObj.name;
      }
      const len = palObj.colors.length;
      const selI = state.selectedPalSwatchIdx;
      state.r.moveLeft.disabled = selI <= 0 || len < 2;
      state.r.moveRight.disabled = selI < 0 || selI >= len - 1 || len < 2;
      state.r.removeFromPal.disabled = selI < 0 || selI >= len;
      state.r.deletePalBtn.dataset.confirm = '0';
      state.r.deletePalBtn.textContent = 'Delete';
    }
  }

  function updateColourUI() {
    if (!state) return;
    const { activeColor, pal, r } = state;
    if (activeColor > 0 && pal.entries[activeColor]) {
      const hex = rgbToHex(pal.entries[activeColor]);
      r.curReadout.textContent = `${hex} · #${activeColor}`;
      r.colorPicker.value = hex;
      if (document.activeElement !== r.hexInput) r.hexInput.value = hex;
      r.pickEraser.classList.remove('is-active');
      r.pickEraser.setAttribute('aria-pressed', 'false');
    } else {
      r.curReadout.textContent = 'Transparent · #0';
      if (document.activeElement !== r.hexInput) r.hexInput.value = '';
      r.pickEraser.classList.add('is-active');
      r.pickEraser.setAttribute('aria-pressed', 'true');
    }
    syncSwapUI();
    renderRecentSwatches();
    updateSheetSummary();
  }

  function updateColourAndPaletteUI() {
    updateColourUI();
    updatePaletteUI();
  }

  function syncSwapUI() {
    if (!state) return;
    state.r.swapMode.setAttribute('aria-pressed', String(state.swapMode));
    state.r.swapMode.classList.toggle('is-active', state.swapMode);
    state.r.swapHint.textContent = state.swapMode
      ? 'Click any swatch or pick a colour to replace active pixels'
      : 'Click swatch to paint · Alt+Click canvas to pick';
  }

  function syncFillAndOutlineUI() {
    if (!state) return;
    state.r.fillGlobal.setAttribute('aria-pressed', String(state.fillGlobal));
    state.r.fillGlobal.classList.toggle('is-active', state.fillGlobal);
    modalEl.querySelectorAll('[data-outline-mode]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.outlineMode === state.outlineMode));
    });
    if (state.tool === 'fill') {
      state.r.toolHint.textContent = state.fillGlobal ? 'Global colour replace active' : '4-way bucket fill active';
    } else {
      state.r.toolHint.textContent = state.outlineMode === 'inner' ? 'Outline: inner perimeter' : 'Outline: outer border';
    }
  }

  function renderAll() {
    if (!state) return;
    const shown = state.comparing ? state.orig : state.buf;
    renderIndicesToCanvas(state.r.canvas, shown, state.pal);
    renderIndicesToCanvas(state.r.prev1, shown, state.pal);
    renderIndicesToCanvas(state.r.prev4, shown, state.pal);
    renderOverlay();
    updateDiffAndButtons();
    updateColourAndPaletteUI();
    syncFillAndOutlineUI();
  }

  function pushUndo(toolName) {
    if (!state) return;
    state.undoStack.push(new Uint16Array(state.buf));
    if (state.undoStack.length > MAX_UNDO) state.undoStack.shift();
    state.redoStack.length = 0;
    state.opsCount++;
    state.toolCounts[toolName] = (state.toolCounts[toolName] || 0) + 1;
  }

  function undo() {
    if (!state || !state.undoStack.length) return;
    clearRecolorPreview(false);
    state.redoStack.push(new Uint16Array(state.buf));
    state.buf = state.undoStack.pop();
    renderAll();
  }

  function redo() {
    if (!state || !state.redoStack.length) return;
    clearRecolorPreview(false);
    state.undoStack.push(new Uint16Array(state.buf));
    state.buf = state.redoStack.pop();
    renderAll();
  }

  function setTool(t) {
    if (!state) return;
    state.tool = t;
    modalEl.querySelectorAll('[data-tool]').forEach((b) => {
      const on = b.dataset.tool === t;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', String(on));
    });
    syncFillAndOutlineUI();
    updateSheetSummary();
    renderOverlay();
  }

  function paintPixel(x, y, val) {
    if (x < 0 || x >= 16 || y < 0 || y >= 16) return false;
    let changed = false;
    const i = y * 16 + x;
    if (state.buf[i] !== val) { state.buf[i] = val; changed = true; }
    if (state.mirrorX) {
      const mi = y * 16 + (15 - x);
      if (state.buf[mi] !== val) { state.buf[mi] = val; changed = true; }
    }
    return changed;
  }

  function bresenhamStroke(x0, y0, x1, y1, val) {
    let dx = Math.abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    let dy = -Math.abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    let any = false;
    while (true) {
      if (paintPixel(x0, y0, val)) any = true;
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
    return any;
  }

  function floodFillAt(sx, sy, newVal) {
    const target = state.buf[sy * 16 + sx];
    if (target === newVal) return false;
    const q = [[sx, sy]];
    const seen = new Uint8Array(256);
    seen[sy * 16 + sx] = 1;
    let changed = false;
    while (q.length) {
      const [x, y] = q.shift();
      const idx = y * 16 + x;
      if (state.buf[idx] !== target) continue;
      state.buf[idx] = newVal;
      changed = true;
      for (const [nx, ny] of [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]) {
        if (nx < 0 || nx >= 16 || ny < 0 || ny >= 16) continue;
        const ni = ny * 16 + nx;
        if (!seen[ni] && state.buf[ni] === target) {
          seen[ni] = 1;
          q.push([nx, ny]);
        }
      }
    }
    return changed;
  }

  function globalReplaceAt(sx, sy, newVal) {
    const target = state.buf[sy * 16 + sx];
    if (target === newVal) return 0;
    let count = 0;
    for (let i = 0; i < 256; i++) {
      if (state.buf[i] === target) {
        state.buf[i] = newVal;
        count++;
      }
    }
    return count;
  }

  function applyOutline(mode = state.outlineMode) {
    if (!state) return;
    if (state.activeColor <= 0) {
      setStatus('Select a non-transparent active colour before adding an outline.', 'err');
      return;
    }
    clearRecolorPreview(false);
    const src = new Uint16Array(state.buf);
    let opaqueCount = 0;
    for (let i = 0; i < 256; i++) if (src[i] > 0) opaqueCount++;
    if (!opaqueCount) {
      setStatus('Draw at least 1 non-transparent pixel before adding an outline.', 'err');
      return;
    }

    const next = new Uint16Array(src);
    const col = state.activeColor;
    let changed = 0;

    if (mode === 'inner') {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const i = y * 16 + x;
          if (src[i] === 0) continue;
          const isEdge =
            x === 0 || x === 15 || y === 0 || y === 15 ||
            src[i - 1] === 0 || src[i + 1] === 0 ||
            src[i - 16] === 0 || src[i + 16] === 0;
          if (isEdge && next[i] !== col) {
            next[i] = col;
            changed++;
          }
        }
      }
    } else {
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const i = y * 16 + x;
          if (src[i] !== 0) continue;
          const touchesOpaque =
            (x > 0 && src[i - 1] > 0) ||
            (x < 15 && src[i + 1] > 0) ||
            (y > 0 && src[i - 16] > 0) ||
            (y < 15 && src[i + 16] > 0);
          if (touchesOpaque) {
            next[i] = col;
            changed++;
          }
        }
      }
    }

    if (!changed) {
      setStatus(
        mode === 'inner'
          ? 'Inner edge is already drawn in the active colour.'
          : 'No transparent border pixels around shape (try Inner mode).',
        'err'
      );
      return;
    }

    pushUndo('outline');
    state.buf = next;
    recordRecentColor(col);
    renderAll();
    setStatus(`Added 1px ${mode === 'inner' ? 'inner' : 'outer'} outline (${changed} px).`, 'ok');
  }

  function nudge(dx, dy) {
    if (!state) return;
    clearRecolorPreview(false);
    pushUndo('nudge');
    const next = new Uint16Array(256);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        const sx = x - dx, sy = y - dy;
        if (sx >= 0 && sx < 16 && sy >= 0 && sy < 16) {
          next[y * 16 + x] = state.buf[sy * 16 + sx];
        }
      }
    }
    state.buf = next;
    renderAll();
  }

  function swapAllPixels(fromIdx, toIdx) {
    if (!state || fromIdx === toIdx || fromIdx <= 0) return;
    clearRecolorPreview(false);
    let count = 0;
    for (let i = 0; i < 256; i++) if (state.buf[i] === fromIdx) count++;
    if (!count) return;
    pushUndo('swap');
    for (let i = 0; i < 256; i++) {
      if (state.buf[i] === fromIdx) state.buf[i] = toIdx;
    }
    renderAll();
    setStatus(`Swapped ${count} pixels to ${rgbToHex(state.pal.entries[toIdx])}.`, 'ok');
  }

  function snapSpriteToActivePalette() {
    if (!state) return;
    const palObj = findActivePaletteObj();
    if (!palObj || palObj.kind === 'sprite' || !palObj.colors.length) return;
    clearRecolorPreview(false);
    const targetRgbs = palObj.colors.map(hexToRgb);
    const targetIndices = targetRgbs.map(([r, g, b]) => nearestPaletteIdx(state.pal, r, g, b));
    const targetLabs = targetRgbs.map(([r, g, b]) => rgbToOklab(r, g, b));
    pushUndo('quantize');
    const cache = new Map();
    for (let i = 0; i < 256; i++) {
      const idx = state.buf[i];
      if (idx === 0 || !state.pal.entries[idx]) continue;
      if (cache.has(idx)) { state.buf[i] = cache.get(idx); continue; }
      const [r, g, b] = state.pal.entries[idx];
      const [L, oa, ob] = rgbToOklab(r, g, b);
      let bestK = 0, bestD = Infinity;
      for (let k = 0; k < targetLabs.length; k++) {
        const dL = targetLabs[k][0] - L;
        const da = targetLabs[k][1] - oa;
        const db = targetLabs[k][2] - ob;
        const d = dL * dL + da * da + db * db;
        if (d < bestD) { bestD = d; bestK = k; }
      }
      const mapped = targetIndices[bestK];
      cache.set(idx, mapped);
      state.buf[i] = mapped;
    }
    renderAll();
    setStatus(`Snapped sprite to ${palObj.name} (${palObj.colors.length} colours).`, 'ok');
  }

  function computeHueLightShifted(baseBuf, pal, dHue, dLight) {
    const out = new Uint16Array(256);
    const cache = new Map();
    for (let i = 0; i < 256; i++) {
      const idx = baseBuf[i];
      if (idx === 0 || !pal.entries[idx]) continue;
      if (cache.has(idx)) { out[i] = cache.get(idx); continue; }
      const [r, g, b] = pal.entries[idx];
      const [h, s, l] = rgbToHsl(r, g, b);
      const [nr, ng, nb] = hslToRgb(h + dHue, s, Math.max(2, Math.min(98, l + dLight)));
      const mapped = nearestPaletteIdx(pal, nr, ng, nb);
      cache.set(idx, mapped);
      out[i] = mapped;
    }
    return out;
  }

  function clearRecolorPreview(restoreBase) {
    if (!state) return;
    if (restoreBase && state.recolorBase) {
      state.buf = new Uint16Array(state.recolorBase);
      renderAll();
    }
    state.recolorBase = null;
    state.r.hueSlider.value = '0';
    state.r.lightSlider.value = '0';
    state.r.hueVal.textContent = '0°';
    state.r.lightVal.textContent = '0%';
    state.r.applyRecolor.disabled = true;
    state.r.resetRecolor.disabled = true;
  }

  function previewRecolor() {
    if (!state) return;
    const dh = Number(state.r.hueSlider.value) || 0;
    const dl = Number(state.r.lightSlider.value) || 0;
    state.r.hueVal.textContent = `${dh > 0 ? '+' : ''}${dh}°`;
    state.r.lightVal.textContent = `${dl > 0 ? '+' : ''}${dl}%`;
    if (!state.recolorBase) state.recolorBase = new Uint16Array(state.buf);
    state.buf = computeHueLightShifted(state.recolorBase, state.pal, dh, dl);
    state.r.applyRecolor.disabled = (dh === 0 && dl === 0);
    state.r.resetRecolor.disabled = (dh === 0 && dl === 0);
    renderAll();
  }

  function commitRecolor() {
    if (!state || !state.recolorBase) return;
    const finalBuf = state.buf;
    state.undoStack.push(new Uint16Array(state.recolorBase));
    if (state.undoStack.length > MAX_UNDO) state.undoStack.shift();
    state.redoStack.length = 0;
    state.opsCount++;
    state.toolCounts.recolor = (state.toolCounts.recolor || 0) + 1;
    state.recolorBase = null;
    state.buf = finalBuf;
    state.r.hueSlider.value = '0';
    state.r.lightSlider.value = '0';
    state.r.hueVal.textContent = '0°';
    state.r.lightVal.textContent = '0%';
    state.r.applyRecolor.disabled = true;
    state.r.resetRecolor.disabled = true;
    renderAll();
    setStatus('Applied colour shift.', 'ok');
  }

  function exportFourColourwayStrip() {
    if (!state) return;
    const variants = [
      state.buf,
      computeHueLightShifted(state.buf, state.pal, 90, 0),
      computeHueLightShifted(state.buf, state.pal, 180, 0),
      computeHueLightShifted(state.buf, state.pal, 270, 0),
    ];
    const strip = document.createElement('canvas');
    strip.width = 64;
    strip.height = 16;
    const ctx = strip.getContext('2d');
    const tmp = document.createElement('canvas');
    tmp.width = 16;
    tmp.height = 16;
    variants.forEach((v, k) => {
      renderIndicesToCanvas(tmp, v, state.pal);
      ctx.drawImage(tmp, k * 16, 0);
    });
    strip.toBlob((blob) => {
      if (!blob) return;
      triggerBlobDownload(blob, 'pixie-4colourways-64x16.png');
      setStatus('Exported 64×16 4-colourway strip PNG.', 'ok');
    }, 'image/png');
  }

  function renderDownloadBlob(size, bgOpt) {
    return new Promise((resolve) => {
      const cv = document.createElement('canvas');
      cv.width = size;
      cv.height = size;
      const ctx = cv.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      if (bgOpt && bgOpt !== 'transparent') {
        const bgMap = { black: '#000000', surface: '#111111' };
        ctx.fillStyle = bgMap[bgOpt] || bgOpt;
        ctx.fillRect(0, 0, size, size);
      }
      const src = document.createElement('canvas');
      src.width = 16;
      src.height = 16;
      renderIndicesToCanvas(src, state.buf, state.pal);
      ctx.drawImage(src, 0, 0, 16, 16, 0, 0, size, size);
      cv.toBlob(resolve, 'image/png');
    });
  }

  function setStatus(msg, kind = '') {
    if (!state) return;
    const el = state.r.status;
    el.className = 'px-ed-status' + (kind === 'err' ? ' is-err' : kind === 'ok' ? ' is-ok' : '');
    if (typeof msg === 'string') el.textContent = msg;
    else { el.innerHTML = ''; el.appendChild(msg); }
  }

  async function saveEditToServer(action) {
    if (!state) return null;
    if (state.recolorBase) commitRecolor();
    if (!state.token) {
      throw new Error('Please sign in / enter your name first to save to Pixie.');
    }
    const payload = {
      token: state.token,
      job_id: state.opts.jobId || null,
      parent_share_id: state.opts.parentShareId || null,
      candidate_idx: state.opts.candidateIdx ?? null,
      prompt: state.opts.prompt || '',
      cond: state.opts.cond || null,
      original_indices: Array.from(state.orig),
      edited_indices: Array.from(state.buf),
      ops_count: state.opsCount,
      tool_counts: state.toolCounts,
      duration_ms: Math.round(performance.now() - state.startedAt),
      action,
      opt_in_train: Boolean(state.r.optIn.checked),
    };
    const r = await fetch('/api/chat/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
    return j;
  }

  // ---- Export Menu Popover ----
  function closeExportMenu() {
    if (!state || state.r.exportMenu.hidden) return;
    state.r.exportMenu.hidden = true;
    state.r.exportBtn.setAttribute('aria-expanded', 'false');
  }

  function openExportMenu() {
    if (!state) return;
    const palObj = findActivePaletteObj();
    if (!palObj.colors.length) {
      setStatus('Nothing to export — active palette has 0 colours.', 'err');
      return;
    }
    const menu = state.r.exportMenu;
    state.r.exportTitle.textContent = `Export: ${palObj.name}`;
    menu.hidden = false;
    state.r.exportBtn.setAttribute('aria-expanded', 'true');

    const btnRect = state.r.exportBtn.getBoundingClientRect();
    const menuW = menu.offsetWidth || 210;
    const menuH = menu.offsetHeight || 168;
    const left = Math.max(8, Math.min(window.innerWidth - menuW - 8, btnRect.right - menuW));
    let top = btnRect.bottom + 6;
    if (top + menuH > window.innerHeight - 8) {
      top = Math.max(8, btnRect.top - menuH - 6);
    }
    menu.style.left = `${Math.round(left)}px`;
    menu.style.top = `${Math.round(top)}px`;
  }

  // ---- Import Palette Dialog ----
  function renderImportPreview(colors, infoMsg, kind = '') {
    if (!state) return;
    state.importDraftColors = colors;
    state.r.importInfo.textContent = infoMsg;
    state.r.importInfo.className = 'px-ed-import-info' + (kind === 'err' ? ' is-err' : kind === 'ok' ? ' is-ok' : '');
    const box = state.r.importPreview;
    box.innerHTML = '';
    for (const hex of colors) {
      const sw = document.createElement('span');
      sw.className = 'px-ed-swatch px-ed-swatch-sm';
      sw.style.background = hex;
      sw.title = hex;
      box.appendChild(sw);
    }
    state.r.importConfirm.disabled = colors.length === 0;
  }

  function openImportDialog() {
    if (!state) return;
    closeExportMenu();
    state.importDraftColors = [];
    state.r.importName.value = '';
    state.r.importText.value = '';
    state.r.lospecSlug.value = '';
    state.r.importFile.value = '';
    renderImportPreview([], 'Add a file, paste hex codes, or fetch from Lospec.');
    state.r.importDialog.hidden = false;
    state.r.importName.focus();
  }

  function closeImportDialog() {
    if (!state || state.r.importDialog.hidden) return;
    state.r.importDialog.hidden = true;
    state.r.importBtn.focus();
  }

  async function handleImportFile(file) {
    if (!state || !file) return;
    const isImage = /^image\//i.test(file.type) || /\.(png|jpe?g|gif|webp|bmp)$/i.test(file.name);
    try {
      if (isImage) {
        const res = await extractColorsFromImageFile(file);
        if (!state.r.importName.value.trim()) state.r.importName.value = res.name;
        const note = res.totalUnique > MAX_IMAGE_COLORS
          ? `Extracted top ${res.colors.length} of ${res.totalUnique} unique colours (capped at ${MAX_IMAGE_COLORS}).`
          : `Extracted ${res.colors.length} unique colour${res.colors.length === 1 ? '' : 's'} from image.`;
        renderImportPreview(res.colors, note, res.colors.length ? 'ok' : 'err');
      } else {
        const text = await file.text();
        const res = parsePaletteText(text, file.name);
        if (!state.r.importName.value.trim() && res.name) state.r.importName.value = res.name;
        state.r.importText.value = res.colors.join('\n');
        renderImportPreview(
          res.colors,
          res.colors.length
            ? `Parsed ${res.colors.length} colour${res.colors.length === 1 ? '' : 's'} from ${file.name}.`
            : `No valid colours found in ${file.name}.`,
          res.colors.length ? 'ok' : 'err'
        );
      }
    } catch (err) {
      renderImportPreview([], err.message || 'Could not read file.', 'err');
    }
  }

  function confirmImportPalette() {
    if (!state || !state.importDraftColors.length) return;
    const name = state.r.importName.value.trim().slice(0, 40)
      || `Imported (${state.importDraftColors.length})`;
    const newPal = {
      id: makePaletteId(),
      name,
      colors: state.importDraftColors.slice(),
    };
    state.userPalettes.push(newPal);
    saveUserPalettes(state.userPalettes);
    state.activePaletteId = newPal.id;
    state.selectedPalSwatchIdx = 0;
    closeImportDialog();
    selectColorByHex(newPal.colors[0], { swatchKey: `${newPal.id}:0` });
    setStatus(`Imported palette "${newPal.name}" (${newPal.colors.length} colours).`, 'ok');
  }

  function createUserPaletteFromCurrent() {
    if (!state) return;
    const cur = findActivePaletteObj();
    let colors = cur.colors.slice();
    if (!colors.length && state.activeColor > 0 && state.pal.entries[state.activeColor]) {
      colors = [rgbToHex(state.pal.entries[state.activeColor])];
    }
    if (!colors.length) colors = ['#b6dc3c'];
    const baseName = cur.id === 'sprite' ? 'Custom Sprite' : `${cur.name} Copy`;
    const newPal = {
      id: makePaletteId(),
      name: baseName.slice(0, 40),
      colors,
    };
    state.userPalettes.push(newPal);
    saveUserPalettes(state.userPalettes);
    state.activePaletteId = newPal.id;
    state.selectedPalSwatchIdx = 0;
    updateColourAndPaletteUI();
    state.r.userPalName.focus();
    state.r.userPalName.select();
    setStatus(`Created custom palette "${newPal.name}".`, 'ok');
  }

  function bindEvents() {
    if (modalEl._bound) return;
    modalEl._bound = true;

    const r = state.r;
    r.close.onclick = close;
    modalEl.onclick = (e) => {
      if (!r.exportMenu.hidden && !r.exportMenu.contains(e.target) && e.target !== r.exportBtn) {
        closeExportMenu();
      }
      if (e.target === modalEl) close();
    };

    modalEl.querySelectorAll('[data-tool]').forEach((b) => {
      b.onclick = () => setTool(b.dataset.tool);
    });
    r.fillGlobal.onclick = () => {
      if (!state) return;
      state.fillGlobal = !state.fillGlobal;
      if (state.tool !== 'fill') setTool('fill');
      else syncFillAndOutlineUI();
    };
    modalEl.querySelectorAll('[data-outline-mode]').forEach((b) => {
      b.onclick = () => {
        if (!state) return;
        state.outlineMode = b.dataset.outlineMode === 'inner' ? 'inner' : 'outer';
        syncFillAndOutlineUI();
      };
    });
    r.outlineApply.onclick = () => applyOutline();

    r.mirror.onclick = () => {
      state.mirrorX = !state.mirrorX;
      r.mirror.setAttribute('aria-pressed', String(state.mirrorX));
      r.mirror.classList.toggle('is-active', state.mirrorX);
      renderOverlay();
    };
    r.grid.onclick = () => {
      state.showGrid = !state.showGrid;
      r.grid.setAttribute('aria-pressed', String(state.showGrid));
      r.grid.classList.toggle('is-active', state.showGrid);
      renderOverlay();
    };
    r.undo.onclick = undo;
    r.redo.onclick = redo;
    r.reset.onclick = () => {
      clearRecolorPreview(false);
      if (countDiffs(state.buf, state.orig) === 0) return;
      pushUndo('reset');
      state.buf = new Uint16Array(state.orig);
      renderAll();
      setStatus('Reset to original sprite.');
    };

    modalEl.querySelectorAll('[data-nudge]').forEach((b) => {
      b.onclick = () => {
        const [dx, dy] = b.dataset.nudge.split(',').map(Number);
        nudge(dx, dy);
      };
    });

    r.compare.addEventListener('pointerdown', () => { state.comparing = true; renderAll(); });
    const stopCompare = () => { if (state && state.comparing) { state.comparing = false; renderAll(); } };
    r.compare.addEventListener('pointerup', stopCompare);
    r.compare.addEventListener('pointerleave', stopCompare);
    r.compare.addEventListener('pointercancel', stopCompare);

    // Mobile bottom sheet controls
    r.sheetBar.onclick = () => {
      if (!state) return;
      setSheetExpanded(r.sidebar.dataset.sheet !== 'expanded');
    };
    r.sheetScrim.onclick = () => setSheetExpanded(false);
    window.addEventListener('resize', () => {
      if (!state) return;
      closeExportMenu();
      renderOverlay();
      if (window.innerWidth >= DESKTOP_BP) {
        state.r.sidebarScroll.inert = false;
        state.r.sheetScrim.hidden = true;
      } else {
        setSheetExpanded(state.r.sidebar.dataset.sheet === 'expanded');
      }
    });

    // Active colour controls
    r.pickEraser.onclick = () => selectColorByIndex(0, { switchToPencil: false, recordRecent: false });
    r.swapMode.onclick = () => {
      if (!state) return;
      if (state.activeColor <= 0 && !state.swapMode) {
        setStatus('Select a non-transparent colour first before swapping.', 'err');
        return;
      }
      state.swapMode = !state.swapMode;
      syncSwapUI();
    };
    r.colorPicker.oninput = () => selectColorByHex(r.colorPicker.value);
    r.colorPicker.onchange = () => selectColorByHex(r.colorPicker.value);
    const applyHexInput = () => {
      const norm = normalizeHex(r.hexInput.value);
      if (!norm) {
        updateColourUI();
        return;
      }
      selectColorByHex(norm);
    };
    r.hexInput.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); applyHexInput(); r.hexInput.blur(); }
    });
    r.hexInput.addEventListener('blur', applyHexInput);

    // Palette selection & actions
    r.paletteSelect.onchange = () => {
      if (!state) return;
      state.activePaletteId = r.paletteSelect.value;
      state.activeSwatchKey = null;
      state.selectedPalSwatchIdx = 0;
      updateColourAndPaletteUI();
    };
    r.snapPalette.onclick = snapSpriteToActivePalette;
    r.newPalBtn.onclick = createUserPaletteFromCurrent;

    // Export menu
    r.exportBtn.onclick = (ev) => {
      ev.stopPropagation();
      if (r.exportMenu.hidden) openExportMenu();
      else closeExportMenu();
    };
    r.exportMenu.querySelectorAll('[data-export]').forEach((b) => {
      b.onclick = () => {
        if (!state) return;
        const palObj = findActivePaletteObj();
        if (!palObj.colors.length) {
          setStatus('Nothing to export — active palette is empty.', 'err');
          closeExportMenu();
          return;
        }
        const msg = exportPalette(b.dataset.export, palObj.name, palObj.colors);
        closeExportMenu();
        setStatus(msg, 'ok');
      };
    });

    // User palette CRUD & reorder
    const commitRename = () => {
      const userPal = state && state.userPalettes.find((p) => p.id === state.activePaletteId);
      if (!userPal) return;
      const nextName = r.userPalName.value.trim().slice(0, 40);
      if (!nextName) { r.userPalName.value = userPal.name; return; }
      userPal.name = nextName;
      saveUserPalettes(state.userPalettes);
      updatePaletteUI();
      updateSheetSummary();
      setStatus(`Renamed palette to "${nextName}".`, 'ok');
    };
    r.renamePalBtn.onclick = commitRename;
    r.userPalName.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); commitRename(); r.userPalName.blur(); }
    });

    r.deletePalBtn.onclick = () => {
      const userPal = state && state.userPalettes.find((p) => p.id === state.activePaletteId);
      if (!userPal) return;
      if (r.deletePalBtn.dataset.confirm !== '1') {
        r.deletePalBtn.dataset.confirm = '1';
        r.deletePalBtn.textContent = 'Confirm?';
        setTimeout(() => {
          if (state && r.deletePalBtn.dataset.confirm === '1') {
            r.deletePalBtn.dataset.confirm = '0';
            r.deletePalBtn.textContent = 'Delete';
          }
        }, 4000);
        return;
      }
      state.userPalettes = state.userPalettes.filter((p) => p.id !== userPal.id);
      saveUserPalettes(state.userPalettes);
      state.activePaletteId = 'sprite';
      state.activeSwatchKey = null;
      updateColourAndPaletteUI();
      setStatus(`Deleted palette "${userPal.name}".`, 'ok');
    };

    const appendHexToUserPalette = (hex) => {
      const userPal = state && state.userPalettes.find((p) => p.id === state.activePaletteId);
      if (!userPal) return;
      const norm = normalizeHex(hex);
      if (!norm) return;
      const existing = userPal.colors.indexOf(norm);
      if (existing >= 0) {
        state.selectedPalSwatchIdx = existing;
        selectColorByHex(norm, { swatchKey: `${userPal.id}:${existing}` });
        setStatus(`${norm} is already in "${userPal.name}".`);
        return;
      }
      userPal.colors.push(norm);
      saveUserPalettes(state.userPalettes);
      const idxInPal = userPal.colors.length - 1;
      state.selectedPalSwatchIdx = idxInPal;
      selectColorByHex(norm, { swatchKey: `${userPal.id}:${idxInPal}` });
      setStatus(`Added ${norm} to "${userPal.name}".`, 'ok');
    };

    r.addActiveToPal.onclick = () => {
      if (!state) return;
      if (state.activeColor <= 0 || !state.pal.entries[state.activeColor]) {
        setStatus('Select a non-transparent colour first.', 'err');
        return;
      }
      appendHexToUserPalette(rgbToHex(state.pal.entries[state.activeColor]));
    };
    r.addPalPicker.onchange = () => appendHexToUserPalette(r.addPalPicker.value);
    r.moveLeft.onclick = () => {
      if (!state) return;
      moveUserPaletteColor(state.selectedPalSwatchIdx, state.selectedPalSwatchIdx - 1);
    };
    r.moveRight.onclick = () => {
      if (!state) return;
      moveUserPaletteColor(state.selectedPalSwatchIdx, state.selectedPalSwatchIdx + 1);
    };
    r.removeFromPal.onclick = () => {
      const userPal = state && state.userPalettes.find((p) => p.id === state.activePaletteId);
      if (!userPal) return;
      const idx = state.selectedPalSwatchIdx;
      if (idx < 0 || idx >= userPal.colors.length) return;
      const [removed] = userPal.colors.splice(idx, 1);
      state.selectedPalSwatchIdx = Math.min(idx, userPal.colors.length - 1);
      state.activeSwatchKey = null;
      saveUserPalettes(state.userPalettes);
      updateColourAndPaletteUI();
      setStatus(`Removed ${removed} from "${userPal.name}".`, 'ok');
    };

    // Import dialog events
    r.importBtn.onclick = openImportDialog;
    r.importClose.onclick = closeImportDialog;
    r.importCancel.onclick = closeImportDialog;
    r.importDialog.onclick = (ev) => {
      if (ev.target === r.importDialog) closeImportDialog();
    };
    r.importFile.onchange = () => {
      const file = r.importFile.files && r.importFile.files[0];
      if (file) handleImportFile(file);
    };
    r.importDropZone.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      r.importDropZone.classList.add('is-drag');
    });
    r.importDropZone.addEventListener('dragleave', () => r.importDropZone.classList.remove('is-drag'));
    r.importDropZone.addEventListener('drop', (ev) => {
      ev.preventDefault();
      r.importDropZone.classList.remove('is-drag');
      const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (file) handleImportFile(file);
    });
    r.importText.oninput = () => {
      const raw = r.importText.value;
      if (!raw.trim()) {
        renderImportPreview([], 'Add a file, paste hex codes, or fetch from Lospec.');
        return;
      }
      const parsed = parsePaletteText(raw, '');
      if (parsed.name && !r.importName.value.trim()) r.importName.value = parsed.name;
      renderImportPreview(
        parsed.colors,
        parsed.colors.length
          ? `Parsed ${parsed.colors.length} unique colour${parsed.colors.length === 1 ? '' : 's'}.`
          : 'No valid hex, .gpl, or .pal colours detected yet.',
        parsed.colors.length ? 'ok' : ''
      );
    };
    r.lospecFetch.onclick = async () => {
      r.lospecFetch.disabled = true;
      renderImportPreview([], 'Fetching palette from Lospec…');
      try {
        const res = await fetchLospecPalette(r.lospecSlug.value);
        r.importName.value = res.name;
        r.importText.value = res.colors.join('\n');
        renderImportPreview(res.colors, `Fetched "${res.name}" (${res.colors.length} colours) from Lospec.`, 'ok');
      } catch (err) {
        renderImportPreview(
          [],
          (err && err.message ? err.message : 'Lospec fetch failed.') + ' You can still paste hex codes or drop a file.',
          'err'
        );
      } finally {
        r.lospecFetch.disabled = false;
      }
    };
    r.lospecSlug.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); r.lospecFetch.click(); }
    });
    r.importConfirm.onclick = confirmImportPalette;

    // Recolour Studio
    r.hueSlider.oninput = previewRecolor;
    r.lightSlider.oninput = previewRecolor;
    r.applyRecolor.onclick = commitRecolor;
    r.resetRecolor.onclick = () => clearRecolorPreview(true);
    r.strip.onclick = exportFourColourwayStrip;

    r.dlBg.onchange = () => { r.dlCustomBg.hidden = r.dlBg.value !== 'custom'; };

    // Stage pointer drawing
    const stage = r.stageWrap;
    let drawing = false;
    let lastX = -1, lastY = -1;
    let strokePushed = false;

    const coordsFromEv = (ev) => {
      const rect = stage.getBoundingClientRect();
      const x = Math.max(0, Math.min(15, Math.floor(((ev.clientX - rect.left) / rect.width) * 16)));
      const y = Math.max(0, Math.min(15, Math.floor(((ev.clientY - rect.top) / rect.height) * 16)));
      return [x, y];
    };

    const pickColorAt = (x, y) => {
      const picked = state.buf[y * 16 + x];
      state.toolCounts.eyedropper = (state.toolCounts.eyedropper || 0) + 1;
      state.activeSwatchKey = null;
      state.activeColor = picked;
      if (picked > 0) recordRecentColor(picked);
      setTool(picked === 0 ? 'eraser' : 'pencil');
      updateColourAndPaletteUI();
      setStatus(picked === 0 ? 'Picked transparent (Eraser).' : `Picked ${rgbToHex(state.pal.entries[picked])} (#${picked}).`);
    };

    stage.onpointerdown = (ev) => {
      ev.preventDefault();
      if (!state) return;
      closeExportMenu();
      if (state.recolorBase) commitRecolor();
      stage.setPointerCapture(ev.pointerId);
      const [x, y] = coordsFromEv(ev);
      state.hoverX = x; state.hoverY = y;

      if (ev.altKey || state.tool === 'eyedropper') {
        pickColorAt(x, y);
        return;
      }
      if (state.tool === 'fill') {
        const fillVal = state.activeColor;
        const pre = new Uint16Array(state.buf);
        let changed = false;
        if (state.fillGlobal) {
          const replaced = globalReplaceAt(x, y, fillVal);
          changed = replaced > 0;
          if (changed) {
            setStatus(`Globally replaced ${replaced} px with ${fillVal ? rgbToHex(state.pal.entries[fillVal]) : 'transparent'}.`, 'ok');
          }
        } else {
          changed = floodFillAt(x, y, fillVal);
          if (state.mirrorX && (15 - x) !== x) {
            if (floodFillAt(15 - x, y, fillVal)) changed = true;
          }
        }
        if (changed) {
          state.undoStack.push(pre);
          if (state.undoStack.length > MAX_UNDO) state.undoStack.shift();
          state.redoStack.length = 0;
          state.opsCount++;
          state.toolCounts.fill = (state.toolCounts.fill || 0) + 1;
          if (fillVal > 0) recordRecentColor(fillVal);
          renderAll();
        }
        return;
      }

      drawing = true;
      lastX = x; lastY = y;
      strokePushed = false;
      const val = state.tool === 'eraser' ? 0 : (state.activeColor || 1);
      const pre = new Uint16Array(state.buf);
      if (paintPixel(x, y, val)) {
        state.undoStack.push(pre);
        if (state.undoStack.length > MAX_UNDO) state.undoStack.shift();
        state.redoStack.length = 0;
        state.opsCount++;
        const tk = state.tool === 'eraser' ? 'eraser' : 'pencil';
        state.toolCounts[tk] = (state.toolCounts[tk] || 0) + 1;
        if (val > 0) recordRecentColor(val);
        strokePushed = true;
        renderAll();
      } else {
        state._preStroke = pre;
      }
    };

    stage.onpointermove = (ev) => {
      if (!state) return;
      const [x, y] = coordsFromEv(ev);
      if (x !== state.hoverX || y !== state.hoverY) {
        state.hoverX = x; state.hoverY = y;
        renderOverlay();
      }
      if (!drawing) return;
      const val = state.tool === 'eraser' ? 0 : (state.activeColor || 1);
      const changed = bresenhamStroke(lastX, lastY, x, y, val);
      lastX = x; lastY = y;
      if (changed) {
        if (!strokePushed && state._preStroke) {
          state.undoStack.push(state._preStroke);
          if (state.undoStack.length > MAX_UNDO) state.undoStack.shift();
          state.redoStack.length = 0;
          state.opsCount++;
          const tk = state.tool === 'eraser' ? 'eraser' : 'pencil';
          state.toolCounts[tk] = (state.toolCounts[tk] || 0) + 1;
          if (val > 0) recordRecentColor(val);
          strokePushed = true;
        }
        renderAll();
      }
    };

    const endStroke = () => { drawing = false; if (state) state._preStroke = null; };
    stage.onpointerup = endStroke;
    stage.onpointercancel = endStroke;
    stage.onpointerleave = () => { if (!state) return; state.hoverX = -1; state.hoverY = -1; renderOverlay(); };

    // Save / Download / Share buttons
    r.save.onclick = async () => {
      r.save.disabled = true;
      setStatus('Saving edit…');
      try {
        const j = await saveEditToServer('save');
        state.lastEditId = j.edit_id;
        setStatus(`Saved edit (${j.diff_pixels} px changed).`, 'ok');
        if (typeof state.opts.onSaved === 'function') {
          const dataUrl = r.canvas.toDataURL('image/png');
          state.opts.onSaved({ ...j, dataUrl });
        }
      } catch (e) {
        setStatus(e.message || 'Could not save edit.', 'err');
      } finally { r.save.disabled = false; }
    };

    r.download.onclick = async () => {
      const size = Number(r.dlSize.value || 512);
      const bg = r.dlBg.value === 'custom' ? r.dlCustomBg.value : r.dlBg.value;
      const blob = await renderDownloadBlob(size, bg);
      if (blob) {
        triggerBlobDownload(blob, `pixie-edited-${size}${bg !== 'transparent' ? '-' + bg.replace('#', '') : ''}.png`);
      }
      if (state.token) {
        saveEditToServer('download')
          .then((j) => {
            state.lastEditId = j.edit_id;
            setStatus(`Downloaded ${size}×${size} PNG & logged edit (${j.diff_pixels} px diff).`, 'ok');
            if (typeof state.opts.onSaved === 'function') {
              state.opts.onSaved({ ...j, dataUrl: r.canvas.toDataURL('image/png') });
            }
          })
          .catch(() => setStatus(`Downloaded ${size}×${size} PNG.`, 'ok'));
      } else {
        setStatus(`Downloaded ${size}×${size} PNG.`, 'ok');
      }
    };

    r.share.onclick = async () => {
      r.share.disabled = true;
      setStatus('Saving & creating share link…');
      try {
        const ed = await saveEditToServer('share');
        state.lastEditId = ed.edit_id;
        if (typeof state.opts.onSaved === 'function') {
          state.opts.onSaved({ ...ed, dataUrl: r.canvas.toDataURL('image/png') });
        }
        const sr = await fetch('/api/share', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            token: state.token,
            job_id: state.opts.jobId || null,
            index: state.opts.topIndex ?? 0,
            edit_id: ed.edit_id,
            prompt: state.opts.prompt || 'edited sprite',
          }),
        });
        const sj = await sr.json().catch(() => ({}));
        if (!sr.ok) throw new Error(sj.error || ('Share failed: HTTP ' + sr.status));
        const url = (sj.url && sj.url.startsWith('http')) ? sj.url : (location.origin + (sj.path || ('/s/' + sj.id)));
        try { await navigator.clipboard.writeText(url); } catch {}
        const wrap = document.createElement('span');
        wrap.textContent = 'Shared & copied: ';
        const link = document.createElement('a');
        link.href = url;
        link.target = '_blank';
        link.rel = 'noopener';
        link.textContent = url;
        wrap.appendChild(link);
        setStatus(wrap, 'ok');
      } catch (e) {
        setStatus(e.message || 'Could not share edited sprite.', 'err');
      } finally { r.share.disabled = false; }
    };
  }

  function handleKeyDown(e) {
    if (!state) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      if (!state.r.importDialog.hidden) { closeImportDialog(); return; }
      if (!state.r.exportMenu.hidden) { closeExportMenu(); return; }
      if (window.innerWidth < DESKTOP_BP && state.r.sidebar.dataset.sheet === 'expanded') {
        setSheetExpanded(false);
        return;
      }
      close();
      return;
    }
    const tag = (e.target && e.target.tagName) || '';
    const isTextField = (tag === 'INPUT' && !['range', 'checkbox', 'color'].includes(e.target.type))
      || tag === 'SELECT' || tag === 'TEXTAREA';
    if (isTextField) return;

    const isMod = e.ctrlKey || e.metaKey;
    if (isMod && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
      return;
    }
    if (isMod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
    if (!state.r.importDialog.hidden) return;

    const k = e.key.toLowerCase();
    if (k === 'b') setTool('pencil');
    else if (k === 'e') setTool('eraser');
    else if (k === 'g') setTool('fill');
    else if (k === 'i') setTool('eyedropper');
    else if (k === 'm') state.r.mirror.click();
    else if (k === 'h') state.r.grid.click();
  }

  let keyListenerBound = false;

  async function open(opts = {}) {
    if (!modalEl) modalEl = buildModalDom();
    const r = edRefs(modalEl);
    modalEl.hidden = false;
    document.body.style.overflow = 'hidden';

    const pal = await loadPalette();
    let orig = new Uint16Array(256);
    if (Array.isArray(opts.indices) && opts.indices.length === 256) {
      orig = Uint16Array.from(opts.indices, (v) => Math.max(0, Math.min(4096, v | 0)));
    } else if (opts.imgSrc) {
      const im = await loadImage(opts.imgSrc);
      const off = document.createElement('canvas');
      off.width = 16; off.height = 16;
      const ctx = off.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(im, 0, 0, 16, 16);
      const data = ctx.getImageData(0, 0, 16, 16).data;
      for (let i = 0; i < 256; i++) {
        const o = i * 4;
        orig[i] = data[o + 3] < 128 ? 0 : nearestPaletteIdx(pal, data[o], data[o + 1], data[o + 2]);
      }
    }

    const counts = new Map();
    for (let i = 0; i < 256; i++) if (orig[i] > 0) counts.set(orig[i], (counts.get(orig[i]) || 0) + 1);
    const sortedTop = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([idx]) => idx);
    const topColor = sortedTop[0] || 2343;

    state = {
      opts,
      token: opts.token || null,
      pal,
      r,
      orig,
      buf: new Uint16Array(orig),
      undoStack: [],
      redoStack: [],
      opsCount: 0,
      toolCounts: {},
      startedAt: performance.now(),
      tool: 'pencil',
      activeColor: topColor,
      activeSwatchKey: null,
      recentColors: sortedTop.slice(0, MAX_RECENT_COLORS),
      activePaletteId: 'sprite',
      userPalettes: loadUserPalettes(),
      selectedPalSwatchIdx: 0,
      dragSwatchIdx: null,
      importDraftColors: [],
      fillGlobal: false,
      outlineMode: 'outer',
      mirrorX: false,
      showGrid: true,
      swapMode: false,
      comparing: false,
      hoverX: -1,
      hoverY: -1,
      recolorBase: null,
      lastEditId: null,
    };

    r.prompt.textContent = opts.prompt ? `"${opts.prompt}"` : '';
    r.mirror.setAttribute('aria-pressed', 'false');
    r.mirror.classList.remove('is-active');
    r.grid.setAttribute('aria-pressed', 'true');
    r.grid.classList.add('is-active');
    r.hueSlider.value = '0';
    r.lightSlider.value = '0';
    r.hueVal.textContent = '0°';
    r.lightVal.textContent = '0%';
    r.applyRecolor.disabled = true;
    r.resetRecolor.disabled = true;
    r.recolorDetails.open = false;
    r.exportMenu.hidden = true;
    r.exportBtn.setAttribute('aria-expanded', 'false');
    r.importDialog.hidden = true;
    setStatus('');
    bindEvents();
    setSheetExpanded(false);
    setTool('pencil');
    renderAll();

    if (!keyListenerBound) {
      document.addEventListener('keydown', handleKeyDown);
      keyListenerBound = true;
    }
    r.close.focus();
  }

  function close() {
    if (!modalEl || modalEl.hidden) return;
    if (state) {
      closeExportMenu();
      closeImportDialog();
    }
    modalEl.hidden = true;
    document.body.style.overflow = '';
    state = null;
  }

  window.PixieEditor = {
    open,
    close,
    isOpen: () => Boolean(modalEl && !modalEl.hidden),
    setToken: (t) => { if (state) state.token = t; },
  };
})();
