// Self-hosted server for the Pixel AI labeller: static files + the API handlers in api/.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUB = path.join(ROOT, 'public');
const PORT = Number(process.env.PORT || 8790);
const HOST = process.env.HOST || '0.0.0.0';
const routes = {};
for (const f of ['register', 'next', 'submit', 'admin/login', 'admin/data', 'admin/purge', 'admin/export', 'admin/override', 'admin/rate', 'admin/training', 'admin/training_ingest', 'admin/training_sample', 'admin/training_runs', 'admin/training_runs_ingest'])
  routes['/api/' + f] = (await import(path.join(ROOT, 'api', f + '.js'))).default;
routes['/api/admin/training/ingest'] = routes['/api/admin/training_ingest'];
routes['/api/admin/training/sample.png'] = routes['/api/admin/training_sample'];
routes['/api/admin/training/runs'] = routes['/api/admin/training_runs'];
routes['/api/admin/training/runs_ingest'] = routes['/api/admin/training_runs_ingest'];
routes['/api/config'] = (await import(path.join(ROOT, 'api', 'config.js'))).default;
routes['/api/ab/next'] = (await import(path.join(ROOT, 'api', 'ab_next.js'))).default;
routes['/api/ab/submit'] = (await import(path.join(ROOT, 'api', 'ab_submit.js'))).default;
for (const f of ['chat/submit', 'chat/status', 'chat/vote', 'chat/info', 'chat/download', 'worker/next', 'worker/result'])
  routes['/api/' + f] = (await import(path.join(ROOT, 'api', f + '.js'))).default;
routes['/api/share'] = (await import(path.join(ROOT, 'api', 'share/create.js'))).default;
routes['/api/admin/chat_thumb'] = (await import(path.join(ROOT, 'api', 'admin/chat_thumb.js'))).default;
const shareGet = (await import(path.join(ROOT, 'api', 'share/get.js'))).default;
const { getShare, spritePngFor, ogPngFor } = await import(path.join(ROOT, 'api/_lib/share.js'));
const { renderSharePage, renderShareNotFound } = await import(path.join(ROOT, 'api/_lib/sharepage.js'));
const SHARE_ID = /^\/s\/([A-Za-z0-9_-]{10,24})(?:\/(png|og\.png))?$/;
for (const f of ['blind/next', 'blind/submit', 'blind/undo'])
  routes['/api/' + f] = (await import(path.join(ROOT, 'api', f + '.js'))).default;
const authHandler = (await import(path.join(ROOT, 'api/_lib/auth.js'))).authHandler;
const { flywheelHandlers, gridHandlers } = await import(path.join(ROOT, 'lib/init.mjs'));
const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.png': 'image/png', '.json': 'application/json', '.woff2': 'font/woff2' };
const COMMON = { 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer' };

http.createServer(async (req, res) => {
  for (const [k, v] of Object.entries(COMMON)) res.setHeader(k, v);
  let u;
  try { u = new URL(req.url, 'http://x'); } catch { res.statusCode = 400; return res.end(); }
  const sm = SHARE_ID.exec(u.pathname);
  if (sm && req.method !== 'GET' && req.method !== 'HEAD') { res.statusCode = 405; return res.end(); }
  if (sm && !fs.existsSync(path.join(PUB, u.pathname))) {
    try {
      const rec = getShare(sm[1]);
      if (!rec) {
        res.statusCode = 404;
        if (!sm[2]) {
          res.setHeader('Content-Type', 'text/html; charset=utf-8');
          res.setHeader('Cache-Control', 'no-cache');
          if (req.method === 'HEAD') return res.end();
          return res.end(renderShareNotFound());
        }
        return res.end('not found');
      }
      if (!sm[2]) {
        const html = renderSharePage(rec);
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.setHeader('Cache-Control', 'public, max-age=300');
        if (req.method === 'HEAD') return res.end();
        return res.end(html);
      }
      if (sm[2] === 'og.png') {
        const buf = ogPngFor(rec.id);
        if (!buf) { res.statusCode = 404; return res.end('not found'); }
        res.setHeader('Content-Type', 'image/png');
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
        if (req.method === 'HEAD') return res.end();
        return res.end(buf);
      }
      const size = Number(u.searchParams.get('size') || 512);
      const bg = u.searchParams.get('bg');
      const buf = spritePngFor(rec.id, size, bg);
      res.setHeader('Content-Type', 'image/png');
      res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      if (req.method === 'HEAD') return res.end();
      return res.end(buf);
    } catch (e) {
      console.error('share', u.pathname, e && e.message);
      const perm = e && (e.code === 'EACCES' || e.code === 'ENOENT' || e.code === 'EPERM');
      if (!res.headersSent) {
        if (!sm[2] && perm) {
          res.statusCode = 404;
          res.setHeader('Content-Type', 'text/html; charset=utf-8');
          res.setHeader('Cache-Control', 'no-cache');
          if (req.method === 'HEAD') return res.end();
          return res.end(renderShareNotFound());
        }
        res.statusCode = perm ? 404 : 500;
        res.setHeader('Content-Type', 'text/plain');
      }
      return res.end(perm ? 'not found' : 'server error');
    }
  }
  if (u.pathname.startsWith('/api/share/') && u.pathname !== '/api/share') {
    const id = u.pathname.slice('/api/share/'.length);
    if (req.method !== 'GET' && req.method !== 'HEAD') { res.statusCode = 405; return res.end(); }
    try { return await shareGet(req, res, id); }
    catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'application/json'); }
      return res.end('{"error":"server error"}');
    }
  }
  if (u.pathname.startsWith('/api/auth')) {
    res.setHeader('X-Robots-Tag', 'noindex');
    let raw = '', size = 0;
    try {
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        for await (const c of req) { size += c.length; if (size > 65536) { res.statusCode = 413; return res.end('{"error":"too large"}'); } raw += c; }
        try { req.body = raw ? JSON.parse(raw) : {}; } catch { req.body = {}; }
      }
      return await authHandler(req, res);
    } catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'application/json'); }
      return res.end('{"error":"server error"}');
    }
  }
  const editPng = /^\/api\/chat\/edit\/([A-Za-z0-9_-]+)\.png$/.exec(u.pathname);
  if (editPng && (req.method === 'GET' || req.method === 'HEAD')) {
    res.setHeader('X-Robots-Tag', 'noindex');
    try { return flywheelHandlers.handleGetEditPng(req, res, editPng[1], Object.fromEntries(u.searchParams.entries())); }
    catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'text/plain'); }
      return res.end('server error');
    }
  }
  const gridPng = /^\/api\/admin\/grid\/([A-Za-z0-9_-]+)\/png$/.exec(u.pathname);
  if (gridPng && (req.method === 'GET' || req.method === 'HEAD')) {
    res.setHeader('X-Robots-Tag', 'noindex');
    try { return gridHandlers.handleGetGridPng(req, res, gridPng[1], u); }
    catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'application/json'); }
      return res.end('{"error":"server error"}');
    }
  }
  const gridOne = /^\/api\/admin\/grid\/([A-Za-z0-9_-]+)$/.exec(u.pathname);
  if (gridOne && (req.method === 'GET' || req.method === 'HEAD')) {
    res.setHeader('X-Robots-Tag', 'noindex');
    try { return gridHandlers.handleGetGrid(req, res, gridOne[1]); }
    catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'application/json'); }
      return res.end('{"error":"server error"}');
    }
  }
  const fwRoutes = {
    'POST /api/chat/event': (req, res) => flywheelHandlers.handleChatEvent(req, res),
    'POST /api/chat/edit': (req, res) => flywheelHandlers.handleChatEdit(req, res),
    'GET /api/admin/flywheel': (req, res) => flywheelHandlers.handleAdminFlywheel(req, res),
    'GET /api/admin/flywheel/cand_thumb': (req, res) => flywheelHandlers.handleAdminCandThumb(req, res, u),
    'POST /api/admin/flywheel/queue_ab': (req, res) => flywheelHandlers.handleAdminQueueAb(req, res),
    'POST /api/admin/grid': (req, res) => gridHandlers.handleCreateGrid(req, res),
    'GET /api/admin/grid': (req, res) => gridHandlers.handleListGrids(req, res, u),
  };
  const fw = fwRoutes[req.method + ' ' + u.pathname];
  if (fw) {
    res.setHeader('X-Robots-Tag', 'noindex');
    try {
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        let raw = '', size = 0;
        const LIMIT = (u.pathname === '/api/chat/edit' || u.pathname === '/api/admin/grid') ? 524288 : 65536;
        for await (const c of req) { size += c.length; if (size > LIMIT) { res.statusCode = 413; return res.end('{"error":"too large"}'); } raw += c; }
        try { req.body = raw ? JSON.parse(raw) : {}; } catch { req.body = {}; }
        req._parsedBody = req.body;
      }
      return await fw(req, res);
    } catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'application/json'); }
      return res.end('{"error":"server error"}');
    }
  }
  const handler = routes[u.pathname];
  if (handler) {
    res.setHeader('X-Robots-Tag', 'noindex');
    let raw = '', size = 0;
    const LIMIT = (u.pathname === '/api/worker/result' || u.pathname === '/api/admin/training/ingest' || u.pathname === '/api/admin/training/runs_ingest') ? 2097152 : 65536;
    try {
      for await (const c of req) { size += c.length; if (size > LIMIT) { res.statusCode = 413; return res.end('{"error":"too large"}'); } raw += c; }
      try { req.body = raw ? JSON.parse(raw) : {}; } catch { req.body = {}; }
      req._parsedBody = req.body;
      return await handler(req, res);
    } catch (e) {
      console.error('handler', u.pathname, e && e.message);
      if (!res.headersSent) { res.statusCode = 500; res.setHeader('Content-Type', 'application/json'); }
      return res.end('{"error":"server error"}');
    }
  }
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.statusCode = 405; return res.end(); }
  if (u.pathname === '/chat-preview') { res.statusCode = 301; res.setHeader('Location', '/chat'); return res.end(); }
  let p = u.pathname === '/' ? '/chat.html' : u.pathname === '/admin' ? '/admin.html' : u.pathname === '/chat' ? '/chat.html' : u.pathname === '/blind' ? '/blind.html' : u.pathname === '/login' ? '/login.html' : u.pathname === '/account' ? '/account.html' : u.pathname === '/privacy' ? '/privacy.html' : u.pathname === '/terms' ? '/terms.html' : (u.pathname === '/label' || u.pathname === '/home') ? '/index.html' : u.pathname;
  const fp = path.normalize(path.join(PUB, p));
  if (!fp.startsWith(PUB + path.sep) || !fs.existsSync(fp) || !fs.statSync(fp).isFile()) { res.statusCode = 404; return res.end('not found'); }
  res.setHeader('Content-Type', TYPES[path.extname(fp)] || 'application/octet-stream');
  res.setHeader('Cache-Control', (p.startsWith('/s/') || p.startsWith('/ab/') || p.startsWith('/fonts/') || p.startsWith('/blind/s/')) ? 'public, max-age=31536000, immutable' : 'no-cache, must-revalidate');
  if (req.method === 'HEAD') return res.end();
  fs.createReadStream(fp).pipe(res);
}).listen(PORT, HOST, () => console.log(`pixel-labeller on http://${HOST}:${PORT}`));
