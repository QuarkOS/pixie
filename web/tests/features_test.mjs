import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { decodePng, encodePng, toNative16, renderSpritePng, renderOgPng, fromDataUri, parseBg } from '../api/_lib/png.js';
import { parseCond, publicCondOptions, NO_PALETTE, workerCond } from '../api/_lib/cond.js';
import { createShare, getShare, spritePngFor, ogPngFor, saveChatThumb, readChatThumb } from '../api/_lib/share.js';

const rgba = Buffer.alloc(256 * 256 * 4);
for (let y = 0; y < 256; y++) {
  for (let x = 0; x < 256; x++) {
    const o = (y * 256 + x) * 4;
    rgba[o] = x & 255;
    rgba[o + 1] = y & 255;
    rgba[o + 2] = 40;
    rgba[o + 3] = 255;
  }
}
const fixture = encodePng(rgba, 256, 256);
const im = decodePng(fixture);
assert.equal(im.width, 256);
const n16 = toNative16(im.rgba, im.width, im.height);
assert.equal(n16.length, 16 * 16 * 4);
const uri = 'data:image/png;base64,' + renderSpritePng(n16, 256, null).toString('base64');
assert.ok(uri.startsWith('data:image/png;base64,'));
const back = fromDataUri(uri);
assert.equal(back.width, 256);
assert.equal(toNative16(back.rgba, 256, 256).equals(n16), true);

const p16 = renderSpritePng(n16, 16, null);
assert.equal(decodePng(p16).width, 16);
const p512t = renderSpritePng(n16, 512, null);
assert.equal(decodePng(p512t).width, 512);
const p512b = renderSpritePng(n16, 512, parseBg('black'));
const im512b = decodePng(p512b);
assert.equal(im512b.rgba[3], 255); // opaque
const og = renderOgPng(n16, 'red potion');
const ogi = decodePng(og);
assert.equal(ogi.width, 1200);
assert.equal(ogi.height, 630);

const opts = publicCondOptions();
assert.ok(opts.palettes.some((p) => p.id === 'gold'));
assert.equal(opts.style.fill.length, 4);

let c = parseCond({});
assert.equal(c.default, true);
assert.equal(c.tensor.length, 6);
assert.equal(c.tensor[0], NO_PALETTE);
assert.equal(workerCond(c), undefined);

c = parseCond({ palette: 'gold', style: { fill: 'small', outline: 'dark', colours: 'low' } });
assert.equal(c.default, false);
assert.equal(c.palette_key, 'gold');
assert.equal(c.style.fill, 1);
assert.equal(c.tensor[3], 1);
assert.deepEqual(workerCond(c).style, [1, 1, 1]);

assert.throws(() => parseCond({ palette: [99999] }), /1–4096/);
c = parseCond({ palette: [376, 1635], style: ['medium', 'any', 'high'] });
assert.equal(c.palette[2], NO_PALETTE);
assert.equal(c.style.colours, 3);

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pixie-share-'));
process.env.DATA_DIR = dir;
const rec = createShare({ prompt: 'red potion', png: uri, index: 0, job_id: 'jobTEST0001', lid: 'abc', name: 'test' });
assert.ok(rec.id);
assert.ok(getShare(rec.id));
const s16 = spritePngFor(rec.id, 16, 'transparent');
assert.equal(decodePng(s16).width, 16);
const og2 = ogPngFor(rec.id);
assert.equal(decodePng(og2).width, 1200);
const thumb = saveChatThumb('jobTEST0001', uri);
assert.ok(thumb.includes('jobTEST0001'));
assert.ok(readChatThumb('jobTEST0001'));
assert.equal(readChatThumb('../etc/passwd'), null);
fs.rmSync(dir, { recursive: true, force: true });
console.log('features_test ok');
