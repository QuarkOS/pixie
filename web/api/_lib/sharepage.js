import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PUBLIC_PATH } from './share.js';

const TPL = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../public/share.html');
let cached = null;
function tpl() { return cached || (cached = fs.readFileSync(TPL, 'utf8')); }
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function escAttr(s) { return esc(s).replace(/'/g, '&#39;'); }

export function publicBase() {
  return String(process.env.AUTH_PUBLIC_URL || process.env.SHARE_PUBLIC_URL || '').replace(/\/$/, '');
}

export function renderSharePage(rec) {
  const base = publicBase();
  const path_ = PUBLIC_PATH(rec.id);
  const url = (base || '') + path_;
  const og = (base || '') + path_ + '/og.png';
  const title = rec.prompt ? (rec.prompt + ' — Pixie') : 'A Pixie sprite';
  const desc = rec.prompt
    ? ('A 16×16 sprite of “' + rec.prompt + '”, drawn by Pixie.')
    : 'A 16×16 sprite drawn by Pixie.';
  return tpl()
    .replaceAll('__TITLE__', esc(title))
    .replaceAll('__DESC__', esc(desc))
    .replaceAll('__URL__', esc(url))
    .replaceAll('__OG__', esc(og))
    .replaceAll('__ID__', esc(rec.id))
    .replaceAll('__PROMPT_ATTR__', escAttr(rec.prompt || ''))
    .replaceAll('__PROMPT__', esc(rec.prompt || 'Untitled sprite'))
    .replaceAll('__IMG512__', esc(path_ + '/png?size=512'))
    .replaceAll('__DL16__', esc(path_ + '/png?size=16'))
    .replaceAll('__DL512__', esc(path_ + '/png?size=512&bg=black'))
    .replaceAll('__DL512T__', esc(path_ + '/png?size=512&bg=transparent'));
}

export function renderShareNotFound() {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sprite not found — Pixie</title>
<meta name="robots" content="noindex">
<style>
  body{margin:0;background:#000;color:#ececec;font:16px/1.45 system-ui,Segoe UI,sans-serif;display:flex;min-height:100vh;align-items:center;justify-content:center}
  main{max-width:28rem;padding:24px;text-align:center}
  h1{font-size:22px;margin:0 0 8px}
  p{color:#9a9a9a;margin:0 0 16px}
  a{color:#c5e853}
</style>
</head>
<body>
<main>
  <h1>Sprite not found</h1>
  <p>This share link is missing or no longer available.</p>
  <p><a href="/">Back to Pixie</a></p>
</main>
</body>
</html>`;
}
