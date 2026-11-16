// Pixel AI chat page behaviour. API: /api/register, /api/chat/{info,submit,status,vote,event,edit}.
// Shares the labeller identity (localStorage pxl_token / pxl_name / pxl_lid) with the A/B rating page.
(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const store = (() => {
  const mem = new Map(); let ls = null;
  try { ls = window.localStorage; ls.setItem('__pxl_t', '1'); ls.removeItem('__pxl_t'); } catch (e) { ls = null; }
  return {
    get: (k) => { try { if (ls) return ls.getItem(k); } catch (e) {} return mem.has(k) ? mem.get(k) : null; },
    set: (k, v) => { mem.set(k, String(v)); try { if (ls) ls.setItem(k, String(v)); } catch (e) {} },
    del: (k) => { mem.delete(k); try { if (ls) ls.removeItem(k); } catch (e) {} },
  };
})();
const sstore = (() => {
  const mem = new Map(); let ss = null;
  try { ss = window.sessionStorage; ss.setItem('__pxl_s', '1'); ss.removeItem('__pxl_s'); } catch (e) { ss = null; }
  return {
    get: (k) => { try { if (ss) return ss.getItem(k); } catch (e) {} return mem.has(k) ? mem.get(k) : null; },
    set: (k, v) => { mem.set(k, String(v)); try { if (ss) ss.setItem(k, String(v)); } catch (e) {} },
  };
})();

const LS = { tok: 'pxl_token', name: 'pxl_name', lid: 'pxl_lid', view: 'pxl_chat_view' }; // view: 'top4' | 'grid'
const CHAIN_IDLE_MS = 10 * 60 * 1000;
const RETRY_WINDOW_MS = 90 * 1000;
const REASON_LABELS = {
  wrong_subject: 'Wrong subject',
  messy_pixels: 'Messy pixels',
  cut_off: 'Cut off',
  wrong_style: 'Wrong colour/style',
};

let token = store.get(LS.tok);
let active = null; // the job being generated / waiting for a pick
let lastNoneJobId = null;
let lastNoneAt = 0;
let tweakPrevJobId = null;
const POLL_MS = 800; // fallback interval when the long-poll errors or the server answers immediately
const FAST_WAIT_MS = 250;

function makeUuid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'c-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
}
function getChainId(forceReset = false) {
  const now = Date.now();
  const prevId = sstore.get('pxl_chain_id');
  const prevAt = Number(sstore.get('pxl_chain_at') || 0);
  if (!forceReset && prevId && (now - prevAt) < CHAIN_IDLE_MS) {
    sstore.set('pxl_chain_at', now);
    return prevId;
  }
  const fresh = makeUuid();
  sstore.set('pxl_chain_id', fresh);
  sstore.set('pxl_chain_at', now);
  return fresh;
}
function beaconEvent(payload) {
  if (!token) return;
  try {
    fetch('/api/chat/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, ...payload }),
      keepalive: true,
    }).catch(() => {});
  } catch (e) {}
}

async function api(path, body) {
  const r = await fetch(path, body === undefined ? { cache: 'no-store' } : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let j = {}; try { j = await r.json(); } catch (e) {}
  if (!r.ok) { const e = new Error(j.error || ('HTTP ' + r.status)); e.status = r.status; e.code = j.code; throw e; }
  return j;
}
function setWho() { const n = store.get(LS.name); $('who').textContent = n ? 'as ' + n : ''; }
function showQuota(q) {
  const el = $('quotaLeft');
  if (!el) return;
  if (!q || !q.limited || q.remaining == null) { el.hidden = true; el.textContent = ''; return; }
  const n = Math.max(0, q.remaining);
  el.hidden = false;
  el.textContent = n === 1 ? '1 left today' : (n + ' left today');
}
function showGate(msg) { $('gate').hidden = false; $('gateerr').textContent = msg || ''; setTimeout(() => $('name').focus(), 50); }
function scrollDown() { window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); }
function setBusy(b) { $('send').disabled = b; $('prompt').disabled = false; }
function tpl(id) { return $(id).content.firstElementChild.cloneNode(true); }
function refs(el) { const r = {}; el.querySelectorAll('[data-ref]').forEach((x) => { r[x.dataset.ref] = x; }); return r; }

function addUser(text) { const el = tpl('tplUser'); refs(el).text.textContent = text; $('thread').appendChild(el); }
function addBot() { const el = tpl('tplBot'); $('thread').appendChild(el); return { el, r: refs(el) }; }
function statusText(v) {
  if (v.status === 'queued') return v.position > 1 ? `In the queue: position ${v.position}, about ${v.eta_s} s…` : `Next in line, about ${v.eta_s} s…`;
  if (v.status === 'running') return `Drawing 20 sprites and picking the best 4… about ${Math.max(1, v.eta_s)} s`;
  return '';
}

function condValue(id, fallback) { const el = $(id); return (el && el.value) ? el.value : fallback; }
function currentCond() {
  return {
    palette: condValue('palettePick', 'auto'),
    style: {
      fill: condValue('styleFill', 'any'),
      outline: condValue('styleOutline', 'any'),
      colours: condValue('styleColours', 'any'),
    },
  };
}
function fillSelect(el, opts, current) {
  if (!el || !opts) return;
  el.innerHTML = '';
  opts.forEach((o) => { const opt = document.createElement('option'); opt.value = o.id; opt.textContent = o.label; el.appendChild(opt); });
  if (current) el.value = current;
}
function swatchesFor(p) {
  return p && p.id !== 'auto' && Array.isArray(p.swatches) ? p.swatches : [];
}
function renderSwatches(box, swatches) {
  if (!box) return;
  box.innerHTML = '';
  (swatches || []).forEach((s) => { const i = document.createElement('i'); i.style.background = s.hex; box.appendChild(i); });
}
function paintSwatches() {
  const sel = $('palettePick'), box = $('paletteSwatches'), val = $('paletteValue');
  if (!sel) return;
  const palettes = sel._palettes || [];
  const hit = palettes.find((p) => p.id === sel.value) || palettes[0];
  if (val) val.textContent = hit ? hit.label : 'Auto (no palette hint)';
  renderSwatches(box, swatchesFor(hit));
  $('paletteMenu')?.querySelectorAll('[role="option"]').forEach((opt) => {
    opt.setAttribute('aria-selected', String(opt.dataset.value === sel.value));
  });
}
function isPaletteOpen() {
  const pop = $('palettePop');
  return Boolean(pop && !pop.hidden);
}
function paletteOptions() {
  return [...($('paletteMenu')?.querySelectorAll('[role="option"]') || [])];
}
function setActivePaletteOpt(idx, kbd) {
  const menu = $('paletteMenu'), btn = $('paletteBtn'), opts = paletteOptions();
  if (!menu || !btn || !opts.length) return;
  const clamped = Math.max(0, Math.min(opts.length - 1, idx));
  opts.forEach((opt, i) => opt.classList.toggle('is-active', i === clamped));
  menu.classList.toggle('is-kbd', Boolean(kbd));
  btn.setAttribute('aria-activedescendant', opts[clamped].id);
  opts[clamped].scrollIntoView({ block: 'nearest' });
}
function setPaletteOpen(open, opts = {}) {
  const pop = $('palettePop'), btn = $('paletteBtn'), menu = $('paletteMenu');
  if (!pop || !btn || !menu) return;
  if (!open) {
    pop.hidden = true;
    btn.classList.remove('is-open');
    btn.setAttribute('aria-expanded', 'false');
    btn.removeAttribute('aria-activedescendant');
    menu.classList.remove('is-kbd');
    menu.querySelectorAll('.is-active').forEach((x) => x.classList.remove('is-active'));
    return;
  }
  const items = paletteOptions();
  if (!items.length) return;
  pop.hidden = false;
  btn.classList.add('is-open');
  btn.setAttribute('aria-expanded', 'true');
  const curId = $('palettePick')?.value;
  const curIdx = items.findIndex((o) => o.dataset.value === curId);
  const idx = opts.index != null ? opts.index : (curIdx >= 0 ? curIdx : 0);
  setActivePaletteOpt(idx, Boolean(opts.kbd));
}
function selectPalette(id) {
  const sel = $('palettePick');
  if (!sel) return;
  if ([...sel.options].some((o) => o.value === id)) sel.value = id;
  paintSwatches();
  updateOptionsIndicator();
}
function fillPaletteMenu(palettes) {
  const menu = $('paletteMenu');
  if (!menu) return;
  menu.innerHTML = '';
  const cur = $('palettePick')?.value || 'auto';
  (palettes || []).forEach((p, idx) => {
    const li = document.createElement('li');
    li.className = 'px-palette-opt';
    li.id = 'paletteOpt-' + String(p.id).replace(/[^a-zA-Z0-9_-]/g, '_');
    li.setAttribute('role', 'option');
    li.dataset.value = p.id;
    li.setAttribute('aria-selected', String(p.id === cur));

    const chk = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    chk.setAttribute('class', 'px-palette-check');
    chk.setAttribute('width', '10');
    chk.setAttribute('height', '10');
    chk.setAttribute('viewBox', '0 0 10 10');
    chk.setAttribute('shape-rendering', 'crispEdges');
    chk.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('fill', 'currentColor');
    path.setAttribute('d', 'M7 2h2v1H7zM6 3h2v1H6zM1 4h1v1H1zM5 4h2v1H5zM1 5h2v1H1zM4 5h2v1H4zM2 6h3v1H2zM3 7h1v1H3z');
    chk.appendChild(path);
    li.appendChild(chk);

    const name = document.createElement('span');
    name.className = 'px-palette-opt-name';
    name.textContent = p.label;
    li.appendChild(name);

    const sw = swatchesFor(p);
    if (sw.length) {
      const box = document.createElement('span');
      box.className = 'px-cond-swatches';
      box.setAttribute('aria-hidden', 'true');
      renderSwatches(box, sw);
      li.appendChild(box);
    }

    li.addEventListener('pointermove', (ev) => {
      if (ev.pointerType !== 'touch') setActivePaletteOpt(idx, false);
    });
    li.addEventListener('click', () => {
      selectPalette(p.id);
      setPaletteOpen(false);
      $('paletteBtn')?.focus();
    });
    menu.appendChild(li);
  });
}
function isOptionDirty() {
  return condValue('palettePick', 'auto') !== 'auto'
    || condValue('styleFill', 'any') !== 'any'
    || condValue('styleOutline', 'any') !== 'any'
    || condValue('styleColours', 'any') !== 'any'
    || (store.get(LS.view) === 'grid');
}
function updateOptionsIndicator() {
  const btn = $('optionsBtn'), reset = $('optionsReset');
  const dirty = isOptionDirty();
  if (btn) {
    btn.classList.toggle('is-dirty', dirty);
    btn.dataset.dirty = dirty ? 'true' : 'false';
    btn.setAttribute('aria-label', dirty ? 'Options (customised)' : 'Options');
  }
  if (reset) reset.hidden = !dirty;
}
function setOptionsOpen(open) {
  const panel = $('optionsPanel'), btn = $('optionsBtn');
  if (!panel || !btn) return;
  if (!open) setPaletteOpen(false);
  panel.hidden = !open;
  btn.classList.toggle('is-open', open);
  btn.setAttribute('aria-expanded', String(open));
}
function closeAllSpriteTools(except) {
  document.querySelectorAll('.px-sprite-wrap.is-tools-open').forEach((el) => {
    if (el !== except) el.classList.remove('is-tools-open');
  });
}
function openSpriteTools(wrap) {
  closeAllSpriteTools(wrap);
  if (wrap) wrap.classList.add('is-tools-open');
}
function wireSpriteClick(wrap, onVote) {
  const spBtn = wrap.querySelector('.px-sprite');
  if (!spBtn) return;
  spBtn.addEventListener('pointerdown', (ev) => { wrap._lastPtr = ev.pointerType || ''; });
  spBtn.addEventListener('click', () => {
    const isTouch = wrap._lastPtr === 'touch' || wrap._lastPtr === 'pen' || (!wrap._lastPtr && window.matchMedia && window.matchMedia('(hover: none)').matches);
    wrap._lastPtr = '';
    if (isTouch) openSpriteTools(wrap);
    else closeAllSpriteTools(null);
    onVote();
  });
}
async function downloadSprite(job, index, size, bg, editId, candidateIdx) {
  const body = { token, job_id: job, index, size, bg };
  if (candidateIdx != null) body.candidate_idx = candidateIdx;
  if (editId) body.edit_id = editId;
  const r = await fetch('/api/chat/download', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || ('HTTP ' + r.status)); }
  const blob = await r.blob();
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'pixie-' + size + (bg && bg !== 'transparent' ? '-' + bg : '') + '.png';
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
async function shareSprite(job, index, bot, editId, promptText, candidateIdx) {
  const body = { token, job_id: job, index };
  if (candidateIdx != null) body.candidate_idx = candidateIdx;
  if (editId) { body.edit_id = editId; body.prompt = promptText; }
  const j = await api('/api/share', body);
  const url = (j.url && j.url.startsWith('http')) ? j.url : (location.origin + (j.path || ('/s/' + j.id)));
  try { await navigator.clipboard.writeText(url); if (bot && bot.r.note) bot.r.note.textContent = 'Share link copied: ' + url; }
  catch (e) { if (bot && bot.r.note) bot.r.note.textContent = 'Share link: ' + url; }
}
function bindTools(wrap, { jobId, topIndex, candidateIdx, rank, prompt, cond, bot }) {
  wrap.querySelectorAll('[data-action]').forEach((b) => {
    b.addEventListener('click', (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      const editId = wrap.dataset.editId || null;
      if (b.dataset.action === 'download') {
        downloadSprite(jobId, topIndex, Number(b.dataset.size || 512), b.dataset.bg || 'transparent', editId, candidateIdx)
          .catch((e) => { if (bot && bot.r.note) bot.r.note.textContent = e.message; });
      } else if (b.dataset.action === 'share') {
        shareSprite(jobId, topIndex, bot, editId, prompt, candidateIdx)
          .catch((e) => { if (bot && bot.r.note) bot.r.note.textContent = e.message; });
      } else if (b.dataset.action === 'edit') {
        beaconEvent({
          job_id: jobId,
          event_type: 'edit_open',
          candidate_idx: candidateIdx ?? null,
          rank: rank ?? null,
          shown_view: bot.view || 'top4',
          session_chain_id: getChainId(),
          payload: { source: 'chat' },
        });
        if (!window.PixieEditor) {
          if (bot && bot.r.note) bot.r.note.textContent = 'Pixel editor failed to load.';
          return;
        }
        const img = wrap.querySelector('img');
        window.PixieEditor.open({
          token,
          jobId,
          topIndex,
          candidateIdx,
          prompt,
          cond,
          imgSrc: img ? img.src : null,
          onSaved: ({ edit_id, diff_pixels, dataUrl }) => {
            wrap.dataset.editId = edit_id;
            wrap.classList.add('is-edited');
            if (img && dataUrl) img.src = dataUrl;
            if (bot && bot.r.note) bot.r.note.textContent = `Saved edit (${diff_pixels} px changed).`;
          },
        }).catch((e) => { if (bot && bot.r.note) bot.r.note.textContent = e.message; });
      }
    });
  });
}

async function generate(prompt, opts = {}) {
  if (active) return;
  prompt = prompt.replace(/\s+/g, ' ').trim();
  if (prompt.length < 2) return;
  if (!token) { showGate(''); return; }

  const withinRetry = Boolean(lastNoneJobId && (Date.now() - lastNoneAt) <= RETRY_WINDOW_MS);
  const prevJobId = tweakPrevJobId || (withinRetry && !opts.freshTopic ? lastNoneJobId : null);
  const chainId = getChainId(!prevJobId && Boolean(opts.freshTopic));
  tweakPrevJobId = null;
  if (!prevJobId) { lastNoneJobId = null; lastNoneAt = 0; }

  addUser(prompt); const bot = addBot(); scrollDown();
  bot.r.status.textContent = 'Sending…';
  const cond = currentCond();
  active = { bot, prompt, cond, job: null, prevJobId, chainId, tSend: performance.now(), longPoll: true };
  setBusy(true);
  try {
    const submitBody = {
      token,
      prompt,
      palette: cond.palette,
      style: cond.style,
      session_chain_id: chainId,
    };
    if (prevJobId) submitBody.prev_job_id = prevJobId;
    const v = await api('/api/chat/submit', submitBody);
    active.job = v.job_id;
    bot.jobId = v.job_id;
    bot.prompt = prompt;
    bot.r.status.textContent = statusText(v);
    if (v.quota) showQuota(v.quota);
    poll();
  } catch (e) { fail(e); }
}
function fail(e) {
  const bot = active && active.bot;
  if (e.status === 401) { store.del(LS.tok); token = null; showGate('Session expired, please enter your name again.'); }
  if (bot) {
    bot.el.dataset.state = e.code === 'busy' ? 'busy' : 'error';
    if (e.status === 429) {
      bot.r.status.textContent = '';
      bot.r.status.append(document.createTextNode((e.message || "You've used your 20 free generations today. Sign in with Google for unlimited.") + ' '));
      const a = document.createElement('a');
      a.href = '/login';
      a.textContent = 'Sign in';
      bot.r.status.append(a);
    } else {
      bot.r.status.textContent = e.message || 'Something went wrong, please try again.';
    }
  }
  active = null; setBusy(false);
}
async function poll() {
  if (!active || !active.job) return;
  const job = active.job;
  const useWait = active.longPoll !== false;
  const t0 = performance.now();
  let v;
  try { v = await api(useWait ? '/api/chat/status?wait=1' : '/api/chat/status', { token, job_id: job }); }
  catch (e) {
    if (!active || active.job !== job) return;
    if (e.status === 401 || e.status === 404) return fail(e);
    active.longPoll = false;
    active.bot.r.note.textContent = 'Connection problem, retrying…';
    return setTimeout(poll, POLL_MS);
  }
  if (!active || active.job !== job) return;
  active.bot.r.note.textContent = '';
  active.bot.el.dataset.state = v.status;
  if (v.status === 'queued' || v.status === 'running') {
    active.bot.r.status.textContent = statusText(v);
    const fast = performance.now() - t0 < FAST_WAIT_MS;
    if (!useWait) active.longPoll = true; // a short poll succeeded; try the long-poll again
    // Immediate reply while still pending: waiter cap, or a server that ignores wait. Don't spin.
    if (!useWait || fast) return setTimeout(poll, POLL_MS);
    return setTimeout(poll, 0); // the ~25s hold elapsed; arm another long-poll
  }
  active.longPoll = true;
  if (v.status === 'error') return fail(new Error(v.error || 'Generation failed, please try again.'));
  showSprites(v);
}
// Grid mode: `candidates` (all 20, rank 1 = best) arrives with finished jobs; old jobs lack it and only get the top 4.
function candidatesOf(v) { return Array.isArray(v.candidates) && v.candidates.length ? v.candidates.slice().sort((a, b) => a.rank - b.rank) : null; }
// candidate behind a top-4 image: same img value, else the candidate whose rank matches its position
function topCandidate(cands, im) { if (!cands) return null; return cands.find((c) => c.img === im.img) || cands.find((c) => c.rank === im.index + 1) || null; }
const bots = []; // replies that have all 20 candidates (the global toggle switches them all)
function syncGlobal(mode) {
  $('viewDefault').querySelectorAll('[data-view]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === mode)));
  updateOptionsIndicator();
}
function pickView(mode) {
  store.set(LS.view, mode); syncGlobal(mode); bots.forEach((b) => setView(b, mode));
  beaconEvent({
    job_id: active ? active.job : (bots.length ? bots[bots.length - 1].jobId : null),
    event_type: 'grid_toggle',
    shown_view: mode,
    session_chain_id: getChainId(),
    payload: { view: mode, scope: 'global' },
  });
}
function setView(bot, mode) {
  closeAllSpriteTools(null);
  bot.view = mode; bot.el.dataset.view = mode;
  bot.r.sprites.hidden = mode === 'grid'; bot.r.grid.hidden = mode !== 'grid';
  bot.r.view.querySelectorAll('[data-view]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === mode)));
  hint(bot);
}
function hint(bot) {
  if (!active || active.bot !== bot || !active.waitingVote) return;
  bot.r.note.textContent = bot.view === 'grid' ? 'Tap any of the 20 (keys 1–4 pick the top 4) or "None are good" (key 0).' : 'Tap a sprite (keys 1–4) or "None are good" (key 0).';
}
function topSel(bot, index) { const c = bot.top.get(index); return { mode: 'top4', choice: index, candidate_idx: c ? c.idx : null, rank: index + 1 }; }
function gridSel(c) { return { mode: 'grid', choice: c.rank <= 4 ? c.rank - 1 : -1, candidate_idx: c.idx, rank: c.rank }; }

function wireNoneReasons(bot, { jobId, prompt, prevJobId }) {
  const strip = bot.r.noneReasons;
  if (!strip) return;
  strip.hidden = false;
  const ensureClearOfComposer = () => {
    const comp = document.querySelector('.px-composer') || document.getElementById('composer');
    const compTop = comp ? comp.getBoundingClientRect().top : window.innerHeight;
    const stripRect = strip.getBoundingClientRect();
    const noteRect = bot.r.note ? bot.r.note.getBoundingClientRect() : stripRect;
    const bottom = Math.max(stripRect.bottom, noteRect.bottom);
    if (bottom > compTop - 16) {
      window.scrollBy(0, Math.ceil(bottom - (compTop - 24)));
    }
  };
  requestAnimationFrame(ensureClearOfComposer);
  setTimeout(ensureClearOfComposer, 80);
  strip.querySelectorAll('[data-reason]').forEach((btn) => {
    btn.onclick = () => {
      const reason = btn.dataset.reason;
      strip.querySelectorAll('[data-reason]').forEach((x) => {
        const on = x === btn;
        x.classList.toggle('is-selected', on);
        x.setAttribute('aria-pressed', String(on));
      });
      beaconEvent({
        job_id: jobId,
        event_type: 'none_reason',
        shown_view: bot.view || 'top4',
        session_chain_id: getChainId(),
        prev_job_id: prevJobId || null,
        payload: { reason },
      });
      bot.r.note.textContent = `Noted: none were good (${REASON_LABELS[reason] || reason}). Thanks!`;
    };
  });
  const tweakBtn = strip.querySelector('[data-action="tweak"]');
  if (tweakBtn) {
    tweakBtn.onclick = () => {
      tweakPrevJobId = jobId;
      $('prompt').value = prompt || '';
      $('prompt').focus();
      const len = $('prompt').value.length;
      if ($('prompt').setSelectionRange) $('prompt').setSelectionRange(len, len);
      beaconEvent({
        job_id: jobId,
        event_type: 'tweak_prompt',
        shown_view: bot.view || 'top4',
        session_chain_id: getChainId(),
        payload: { prev_prompt: prompt || '' },
      });
      bot.r.note.textContent = 'Edit the prompt below and send to retry.';
    };
  }
}

function showSprites(v) {
  const { bot } = active; const box = bot.r.sprites; box.innerHTML = '';
  const jobId = v.job_id || (active && active.job);
  const cond = v.cond || (active && active.cond) || null;
  bot.jobId = jobId;
  bot.prompt = v.prompt;
  bot.r.status.textContent = `"${v.prompt}": which one is best?`;
  const cands = candidatesOf(v); bot.cands = cands; bot.top = new Map();
  v.images.forEach((im) => {
    const c = topCandidate(cands, im);
    bot.top.set(im.index, c);
    const b = tpl('tplSprite'); b.dataset.index = im.index;
    b.querySelector('img').src = im.img;
    b.querySelector('img').alt = `option ${im.index + 1} for ${v.prompt}`;
    b.querySelector('.px-sprite-key').textContent = String(im.index + 1);
    wireSpriteClick(b, () => vote(topSel(bot, im.index), bot));
    bindTools(b, {
      jobId,
      topIndex: im.index,
      candidateIdx: c ? c.idx : im.index,
      rank: im.index + 1,
      prompt: v.prompt,
      cond,
      bot,
    });
    box.appendChild(b);
  });
  const grid = bot.r.grid; grid.innerHTML = '';
  if (cands) {
    cands.forEach((c) => {
      const b = tpl('tplSprite'); delete b.dataset.index; b.dataset.cand = c.idx;
      const spBtn = b.querySelector('.px-sprite');
      const img = b.querySelector('img'); img.src = c.img; img.loading = 'lazy'; img.decoding = 'async';
      img.alt = `rank ${c.rank} of ${cands.length} for ${v.prompt}`;
      const key = b.querySelector('.px-sprite-key');
      if (c.rank <= 4) {
        b.classList.add('is-top');
        if (spBtn) spBtn.classList.add('is-top');
        key.textContent = String(c.rank);
      } else key.remove();
      wireSpriteClick(b, () => vote(gridSel(c), bot));
      bindTools(b, {
        jobId,
        topIndex: Number.isInteger(c.rank) && c.rank <= 4 ? c.rank - 1 : c.idx,
        candidateIdx: c.idx,
        rank: c.rank,
        prompt: v.prompt,
        cond,
        bot,
      });
      grid.appendChild(b);
    });
    bot.r.view.querySelectorAll('[data-view]').forEach((b) => {
      b.onclick = () => {
        store.set(LS.view, b.dataset.view);
        syncGlobal(b.dataset.view);
        setView(bot, b.dataset.view);
        beaconEvent({
          job_id: jobId,
          event_type: 'grid_toggle',
          shown_view: b.dataset.view,
          session_chain_id: getChainId(),
          payload: { view: b.dataset.view, scope: 'message' },
        });
      };
    });
    bots.push(bot);
  }
  bot.r.view.hidden = !cands;
  bot.r.actions.hidden = false;
  bot.r.none.onclick = () => vote({ mode: bot.view, choice: 'none', candidate_idx: null, rank: null }, bot);
  active.shownAt = performance.now(); active.waitingVote = true; active.done = v;
  if (active.tSend) active.sendMs = Math.round(performance.now() - active.tSend);
  setView(bot, cands && store.get(LS.view) === 'grid' ? 'grid' : 'top4');
  syncGlobal(store.get(LS.view) === 'grid' ? 'grid' : 'top4');
  setBusy(false); scrollDown();
}
async function vote(sel, bot) {
  if (!active || !active.waitingVote || (bot && active.bot !== bot)) return;
  const a = active; a.waitingVote = false;
  const { choice } = sel;
  const hit = (b) => (b.dataset.cand !== undefined
    ? sel.candidate_idx != null && b.dataset.cand === String(sel.candidate_idx)
    : typeof choice === 'number' && choice >= 0 && b.dataset.index === String(choice));
  const btns = [...a.bot.r.sprites.children, ...a.bot.r.grid.children];
  btns.forEach((b) => {
    const sp = b.querySelector('.px-sprite');
    b.classList.add('is-disabled');
    if (sp) sp.classList.add('is-disabled');
    if (hit(b)) {
      b.classList.add('is-picked');
      if (sp) sp.classList.add('is-picked');
    }
  });
  a.bot.r.none.disabled = true;

  if (choice === 'none') {
    lastNoneJobId = a.job;
    lastNoneAt = Date.now();
    wireNoneReasons(a.bot, { jobId: a.job, prompt: a.prompt, prevJobId: a.prevJobId });
  } else {
    lastNoneJobId = null;
    lastNoneAt = 0;
  }

  const response_ms = Math.round(performance.now() - a.shownAt);
  for (let tries = 0; ; tries++) {
    try {
      await api('/api/chat/vote', {
        token,
        job_id: a.job,
        choice,
        response_ms,
        ...(Number.isFinite(a.sendMs) ? { send_ms: a.sendMs } : {}),
        mode: sel.mode,
        shown_view: sel.mode,
        candidate_idx: sel.candidate_idx,
        rank: sel.rank ?? null,
        session_chain_id: a.chainId || getChainId(),
        prev_job_id: a.prevJobId || null,
      });
      if (choice === 'none') {
        if (!a.bot.r.noneReasons?.querySelector('.is-selected')) {
          a.bot.r.note.textContent = 'Noted: none were good. Thanks!';
        }
      } else {
        a.bot.r.note.textContent = choice >= 0 ? `Saved: you picked #${choice + 1}. Thanks!` : `Saved: you picked rank ${sel.rank} of 20. Thanks!`;
      }
      break;
    } catch (e) {
      if (e.status === 401 || e.status === 404 || tries >= 4) { a.bot.r.note.textContent = 'Could not save this pick: ' + e.message; break; }
      a.bot.r.note.textContent = 'Saving… (retrying)'; await new Promise((r) => setTimeout(r, 1500 * (tries + 1)));
    }
  }
  if (active === a) active = null;
  setBusy(false);
  if (choice !== 'none' && !(window.PixieEditor && window.PixieEditor.isOpen())) $('prompt').focus();
}

async function openShareForEdit(shareId) {
  if (!shareId || !window.PixieEditor) return;
  try {
    let meta = null;
    try {
      meta = await api('/api/share/' + encodeURIComponent(shareId));
    } catch (e) {
      if (shareId === 'preview00000001') {
        meta = { id: shareId, prompt: 'red potion', png_url: 'sprite16.png' };
      } else throw e;
    }
    const imgSrc = meta.png_url || (`/s/${encodeURIComponent(shareId)}/png?size=16`);
    beaconEvent({
      event_type: 'edit_open',
      session_chain_id: getChainId(),
      payload: { source: 'share', share_id: shareId },
    });
    await window.PixieEditor.open({
      token,
      parentShareId: shareId,
      prompt: meta.prompt || 'shared sprite',
      imgSrc,
    });
  } catch (e) {
    const b = addBot();
    b.el.dataset.state = 'error';
    b.r.status.textContent = 'Could not open shared sprite for editing: ' + e.message;
  }
}

$('composer').addEventListener('submit', (ev) => {
  ev.preventDefault();
  if (active && active.waitingVote) { active.bot.r.note.textContent = 'Please pick one first (or "None are good").'; return; }
  const p = $('prompt').value;
  generate(p).then(() => {});
  $('prompt').value = '';
});
$('chips').addEventListener('click', (ev) => {
  const c = ev.target.closest('[data-prompt]');
  if (c && !active) generate(c.dataset.prompt, { freshTopic: true });
});
document.addEventListener('keydown', (e) => {
  if (window.PixieEditor && window.PixieEditor.isOpen()) return;
  if (e.key === 'Escape' && isPaletteOpen()) {
    e.preventDefault();
    setPaletteOpen(false);
    $('paletteBtn')?.focus();
    return;
  }
  if (e.key === 'Escape' && $('optionsPanel') && !$('optionsPanel').hidden) {
    e.preventDefault();
    setOptionsOpen(false);
    $('optionsBtn')?.focus();
    return;
  }
  if ($('optionsPanel') && !$('optionsPanel').hidden) return;
  if (!active || !active.waitingVote) return;
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  const bot = active.bot;
  if (/^[1-4]$/.test(e.key)) {
    e.preventDefault(); const n = Number(e.key);
    if (bot.view === 'grid') { const c = bot.cands.find((x) => x.rank === n); if (c) vote(gridSel(c), bot); } else vote(topSel(bot, n - 1), bot);
  } else if (e.key === '0' || e.key.toLowerCase() === 'n') {
    e.preventDefault();
    vote({ mode: bot.view, choice: 'none', candidate_idx: null, rank: null }, bot);
  }
});
document.addEventListener('click', (ev) => {
  if (isPaletteOpen() && !$('paletteWrap')?.contains(ev.target)) {
    setPaletteOpen(false);
  }
  if ($('optionsPanel') && !$('optionsPanel').hidden && !$('optionsWrap')?.contains(ev.target)) {
    setOptionsOpen(false);
  }
  if (!ev.target.closest('.px-sprite-wrap')) closeAllSpriteTools(null);
});
document.addEventListener('focusin', (ev) => {
  if (isPaletteOpen() && !$('paletteWrap')?.contains(ev.target)) setPaletteOpen(false);
});
if ($('paletteBtn') && $('paletteWrap')) {
  let typeBuf = '', typeTimer = null;
  $('paletteBtn').onclick = () => setPaletteOpen(!isPaletteOpen());
  $('paletteWrap').addEventListener('keydown', (e) => {
    const opts = paletteOptions();
    if (!opts.length) return;
    const open = isPaletteOpen();
    if (!open) {
      if (e.target === $('paletteBtn') && (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        setPaletteOpen(true, { kbd: true });
      }
      return;
    }
    const activeIdx = Math.max(0, opts.findIndex((o) => o.classList.contains('is-active')));
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActivePaletteOpt(activeIdx + 1, true);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActivePaletteOpt(activeIdx - 1, true);
    } else if (e.key === 'Home' || e.key === 'PageUp') {
      e.preventDefault();
      setActivePaletteOpt(0, true);
    } else if (e.key === 'End' || e.key === 'PageDown') {
      e.preventDefault();
      setActivePaletteOpt(opts.length - 1, true);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const chosen = opts[activeIdx];
      if (chosen) selectPalette(chosen.dataset.value);
      setPaletteOpen(false);
      $('paletteBtn').focus();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      setPaletteOpen(false);
      $('paletteBtn').focus();
    } else if (e.key === 'Tab') {
      setPaletteOpen(false);
    } else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      clearTimeout(typeTimer);
      typeBuf = (typeBuf + e.key).toLowerCase();
      typeTimer = setTimeout(() => { typeBuf = ''; }, 500);
      const start = typeBuf.length === 1 ? (activeIdx + 1) % opts.length : 0;
      const ordered = [...opts.slice(start), ...opts.slice(0, start)];
      const match = ordered.find((o) => (o.querySelector('.px-palette-opt-name')?.textContent || '').toLowerCase().startsWith(typeBuf));
      if (match) setActivePaletteOpt(opts.indexOf(match), true);
    }
  });
}
if ($('optionsBtn')) {
  $('optionsBtn').onclick = () => setOptionsOpen(Boolean($('optionsPanel')?.hidden));
}
if ($('optionsReset')) {
  $('optionsReset').onclick = () => {
    const resetSelect = (id, def) => {
      const el = $(id);
      if (!el) return;
      if ([...el.options].some((o) => o.value === def)) el.value = def;
      else if (el.options.length) el.selectedIndex = 0;
    };
    resetSelect('palettePick', 'auto');
    resetSelect('styleFill', 'any');
    resetSelect('styleOutline', 'any');
    resetSelect('styleColours', 'any');
    setPaletteOpen(false);
    paintSwatches();
    if (store.get(LS.view) === 'grid') pickView('top4');
    else updateOptionsIndicator();
  };
}
$('changeName').onclick = () => { store.del(LS.tok); store.del(LS.name); store.del(LS.lid); token = null; setWho(); showGate(''); };
$('gateform').addEventListener('submit', async (ev) => {
  ev.preventDefault(); const name = $('name').value.trim();
  if (name.length < 2 || name.length > 30) { $('gateerr').textContent = 'Name must be 2–30 characters.'; return; }
  try {
    const j = await api('/api/register', { name });
    token = j.token; store.set(LS.tok, j.token); store.set(LS.name, j.name); store.set(LS.lid, j.labeler_id);
    store.set('pxl_done', 0); store.set('pxl_abdone', 0);
    if (window.PixieEditor) window.PixieEditor.setToken(token);
    $('gate').hidden = true; setWho();
    api('/api/chat/info', { token }).then((i) => { if (i && i.quota) showQuota(i.quota); }).catch(() => {});
    if (!(window.PixieEditor && window.PixieEditor.isOpen())) $('prompt').focus();
  } catch (e) { $('gateerr').textContent = e.message; }
});
$('viewDefault').querySelectorAll('[data-view]').forEach((b) => { b.onclick = () => pickView(b.dataset.view); });
syncGlobal(store.get(LS.view) === 'grid' ? 'grid' : 'top4');
(async () => {
  setWho();
  try {
    const i = await api('/api/chat/info', token ? { token } : undefined);
    if (i.prompt_max) $('prompt').maxLength = i.prompt_max;
    if (i.quota) showQuota(i.quota);
    if (i.cond) {
      fillSelect($('palettePick'), i.cond.palettes, 'auto');
      if ($('palettePick')) {
        $('palettePick')._palettes = i.cond.palettes;
        fillPaletteMenu(i.cond.palettes);
        $('palettePick').onchange = () => { paintSwatches(); updateOptionsIndicator(); };
        paintSwatches();
      }
      fillSelect($('styleFill'), i.cond.style.fill, 'any');
      fillSelect($('styleOutline'), i.cond.style.outline, 'any');
      fillSelect($('styleColours'), i.cond.style.colours, 'any');
      ['styleFill', 'styleOutline', 'styleColours'].forEach((id) => {
        if ($(id)) $(id).onchange = updateOptionsIndicator;
      });
      updateOptionsIndicator();
    }
    if (!i.worker_online) { const b = addBot(); b.el.dataset.state = 'error'; b.r.status.textContent = 'The sprite generator is offline right now. You can still rate on the A/B page.'; }
  } catch (e) {}
  const editShareId = new URLSearchParams(location.search).get('edit_share');
  if (editShareId) {
    openShareForEdit(editShareId);
  } else if (!token) {
    showGate('');
  } else {
    $('prompt').focus();
  }
})();
})();
