// Email magic-link + Google + X (OAuth 2.0) sessions, plus HMAC name-gate tokens during a grace period.
// SQLite lives under AUTH_DIR (default ./data/auth). No IPs stored.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { send, body, verify, sign, newToken, encodeDone } from './core.js';
import { overrideMap, abSeen, gridSeen, compact } from './store.js';
import { abEncodeDone } from './ab.js';
import { sendMagicLinkEmail, mailerConfigured, mailerPubliclyEnabled } from './mailer.js';

export const OWNER_EMAIL = (process.env.OWNER_EMAIL || 'owner@example.invalid').trim().toLowerCase();
export const OWNER_LID = process.env.OWNER_LID || 'owner';
export const OWNER_NAME = (process.env.OWNER_NAME || 'owner').trim().slice(0, 80) || 'owner';

const AUTH_DIR = process.env.AUTH_DIR || path.join(process.cwd(), 'data', 'auth');
const PUBLIC_URL = (process.env.AUTH_PUBLIC_URL || '').replace(/\/+$/, '');
// OAuth redirect_uri origin. Staging keeps AUTH_PUBLIC_URL on localhost for magic-link
// tests, and sets AUTH_OAUTH_PUBLIC_URL to the public site so Google sees the registered URI.
function oauthPublicUrl() {
  return (process.env.AUTH_OAUTH_PUBLIC_URL || PUBLIC_URL || '').replace(/\/+$/, '');
}
const COOKIE = 'pxsess';
const OAUTH_COOKIE = 'pxoauth';
const SESSION_DAYS = Number(process.env.AUTH_SESSION_DAYS || 14);
const MAGIC_TTL_S = Number(process.env.AUTH_MAGIC_TTL_S || 600);
const OAUTH_TTL_S = 600;

function authSecret() {
  const s = process.env.AUTH_SECRET || '';
  if (s.length < 32) throw Object.assign(new Error('server not configured'), { status: 500 });
  return s;
}
function cookieSecure() {
  if (process.env.AUTH_COOKIE_SECURE === '0') return false;
  if (process.env.AUTH_COOKIE_SECURE === '1') return true;
  return (PUBLIC_URL || '').startsWith('https://');
}
function cookieSuffix() {
  return `Path=/; HttpOnly; SameSite=Lax${cookieSecure() ? '; Secure' : ''}`;
}

export function graceActive() {
  const until = (process.env.AUTH_TOKEN_GRACE_UNTIL || '').trim();
  if (!until || until === '1' || until.toLowerCase() === 'on') return true;
  if (until === '0' || until.toLowerCase() === 'off') return false;
  const t = Date.parse(until);
  return Number.isFinite(t) ? Date.now() < t : true;
}

export function providersStatus() {
  const googleId = process.env.AUTH_GOOGLE_CLIENT_ID || '';
  const googleSecret = process.env.AUTH_GOOGLE_CLIENT_SECRET || '';
  const xId = process.env.AUTH_X_CLIENT_ID || '';
  const xSecret = process.env.AUTH_X_CLIENT_SECRET || '';
  return {
    // Console mailer stays available for staged tests, but the login UI only
    // advertises email once Resend is configured (AUTH_MAILER=resend + a real key).
    magic_link: mailerPubliclyEnabled(),
    google: !!(googleId && googleSecret && googleId !== 'REPLACE_ME' && googleSecret !== 'REPLACE_ME'),
    x: !!(xId && xSecret && xId !== 'REPLACE_ME' && xSecret !== 'REPLACE_ME'),
  };
}

// ---- sqlite ----
let db = null;
function database() {
  if (db) return db;
  fs.mkdirSync(AUTH_DIR, { recursive: true, mode: 0o700 });
  try { fs.chmodSync(AUTH_DIR, 0o700); } catch { /* already tight or not ours */ }
  const file = path.join(AUTH_DIR, 'auth.sqlite');
  db = new DatabaseSync(file);
  try { fs.chmodSync(file, 0o600); } catch { /* ok */ }
  db.exec(`
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;
    PRAGMA busy_timeout=5000;
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT NOT NULL UNIQUE COLLATE NOCASE,
      name TEXT,
      is_admin INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      last_login_at TEXT
    );
    CREATE TABLE IF NOT EXISTS sessions (
      token_hash TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      expires_at INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      user_agent TEXT
    );
    CREATE TABLE IF NOT EXISTS lids (
      lid TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      is_primary INTEGER NOT NULL DEFAULT 0,
      attached_at TEXT NOT NULL,
      attach_source TEXT
    );
    CREATE INDEX IF NOT EXISTS lids_user ON lids(user_id);
    CREATE TABLE IF NOT EXISTS magic_links (
      token_hash TEXT PRIMARY KEY,
      email TEXT NOT NULL,
      callback TEXT,
      expires_at INTEGER NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS oauth_states (
      state TEXT PRIMARY KEY,
      provider TEXT NOT NULL,
      code_verifier TEXT NOT NULL,
      callback TEXT,
      expires_at INTEGER NOT NULL
    );
  `);
  return db;
}

function sha(s) { return crypto.createHash('sha256').update(s).digest('hex'); }
function newId() { return crypto.randomBytes(16).toString('hex'); }
function nowIso() { return new Date().toISOString(); }
function validEmail(raw) {
  if (typeof raw !== 'string') return null;
  const e = raw.normalize('NFKC').trim().toLowerCase();
  if (e.length < 5 || e.length > 200) return null;
  if (!/^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$/i.test(e)) return null;
  return e;
}
function safeCallback(raw) {
  if (raw == null || raw === '') return '/account';
  if (typeof raw !== 'string' || raw.length > 300) return '/account';
  if (raw.startsWith('/') && !raw.startsWith('//') && !raw.includes('\\')) return raw;
  if (PUBLIC_URL && raw.startsWith(PUBLIC_URL + '/')) {
    try { return new URL(raw).pathname + new URL(raw).search; } catch { return '/account'; }
  }
  return '/account';
}

function userByEmail(email) {
  return database().prepare('SELECT * FROM users WHERE email = ?').get(email) || null;
}
function userById(id) {
  return database().prepare('SELECT * FROM users WHERE id = ?').get(id) || null;
}
function lidsForUser(userId) {
  return database().prepare('SELECT * FROM lids WHERE user_id = ? ORDER BY is_primary DESC, attached_at ASC').all(userId);
}
function lidRow(lid) {
  return database().prepare('SELECT * FROM lids WHERE lid = ?').get(lid) || null;
}

export function accountLids(userId) { return lidsForUser(userId).map((r) => r.lid); }

function nameForLid(lid, fallback) {
  if (lid === OWNER_LID) return OWNER_NAME;
  try {
    // compact() is async but load is sync; use last known name from rows if already loaded
  } catch { /* ignore */ }
  return fallback || 'user';
}

async function latestName(lid, fallback) {
  if (lid === OWNER_LID) return OWNER_NAME;
  try {
    const { rows } = await compact();
    let name = null;
    for (const r of rows) if (r.lid === lid && r.name) name = r.name;
    if (name) return name;
  } catch { /* store not ready */ }
  return fallback || 'user';
}

export function provisionUser(email, profileName) {
  const d = database();
  let user = userByEmail(email);
  if (!user) {
    const isOwner = email === OWNER_EMAIL;
    const name = isOwner ? OWNER_NAME : (typeof profileName === 'string' && profileName.trim() ? profileName.trim().slice(0, 80) : email.split('@')[0]);
    const id = newId();
    d.prepare('INSERT INTO users (id, email, name, is_admin, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(id, email, name, isOwner ? 1 : 0, nowIso());
    if (isOwner) {
      d.prepare('INSERT INTO lids (lid, user_id, is_primary, attached_at, attach_source) VALUES (?, ?, 1, ?, ?)')
        .run(OWNER_LID, id, nowIso(), 'owner_map');
    } else {
      const lid = crypto.randomBytes(8).toString('hex');
      d.prepare('INSERT INTO lids (lid, user_id, is_primary, attached_at, attach_source) VALUES (?, ?, 1, ?, ?)')
        .run(lid, id, nowIso(), 'provisioned');
    }
    user = userById(id);
  }
  d.prepare('UPDATE users SET last_login_at = ? WHERE id = ?').run(nowIso(), user.id);
  return userById(user.id);
}

function setPrimary(userId, lid) {
  const d = database();
  d.prepare('UPDATE lids SET is_primary = 0 WHERE user_id = ?').run(userId);
  d.prepare('UPDATE lids SET is_primary = 1 WHERE user_id = ? AND lid = ?').run(userId, lid);
}

function primaryLid(user) {
  const rows = lidsForUser(user.id);
  return (rows.find((r) => r.is_primary) || rows[0] || {}).lid || null;
}

function createSession(userId, req) {
  const token = crypto.randomBytes(32).toString('base64url');
  const exp = Date.now() + SESSION_DAYS * 86400 * 1000;
  const ua = String(req.headers['user-agent'] || '').slice(0, 200);
  database().prepare('INSERT INTO sessions (token_hash, user_id, expires_at, created_at, user_agent) VALUES (?, ?, ?, ?, ?)')
    .run(sha(token), userId, exp, nowIso(), ua || null);
  return { token, exp };
}

function destroySession(token) {
  if (!token) return;
  database().prepare('DELETE FROM sessions WHERE token_hash = ?').run(sha(token));
}

function readCookie(req, name) {
  const m = new RegExp('(?:^|;\\s*)' + name + '=([^;]+)').exec(String(req.headers.cookie || ''));
  return m ? decodeURIComponent(m[1]) : '';
}
function appendCookie(res, value) {
  const prev = res.getHeader('Set-Cookie');
  res.setHeader('Set-Cookie', prev ? [].concat(prev, value) : value);
}
function setSessionCookie(res, token, exp) {
  const max = Math.max(0, Math.floor((exp - Date.now()) / 1000));
  appendCookie(res, `${COOKIE}=${token}; ${cookieSuffix()}; Max-Age=${max}`);
}
function clearCookie(res, name) {
  appendCookie(res, `${name}=; ${cookieSuffix()}; Max-Age=0`);
}

export function getSessionUser(req) {
  const token = readCookie(req, COOKIE);
  if (!token) return null;
  const row = database().prepare('SELECT s.expires_at, u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?').get(sha(token));
  if (!row || Number(row.expires_at) < Date.now()) {
    if (row) database().prepare('DELETE FROM sessions WHERE token_hash = ?').run(sha(token));
    return null;
  }
  return { id: row.id, email: row.email, name: row.name, is_admin: !!row.is_admin, last_login_at: row.last_login_at };
}

export function isSessionAdmin(req) {
  const u = getSessionUser(req);
  return !!(u && u.is_admin);
}

function payloadFor(lid, name) {
  const p = newToken(name);
  p.lid = lid;
  const seen = abSeen(lid);
  abEncodeDone(seen, p);
  p.an = seen.size;
  const gs = gridSeen(lid);
  p.done = encodeDone(gs, p);
  p.n = gs.size;
  return p;
}

export function issueLabellerToken(lid, name) {
  return sign(payloadFor(lid, name));
}

/**
 * Resolve the labeller identity for an API call.
 * Prefer a valid session; during the grace period a signed HMAC name-gate token still works.
 * If both are present, the HMAC lid must belong to the signed-in account.
 */
export function requireIdent(req, token) {
  const user = getSessionUser(req);
  const tok = typeof token === 'string' && token ? verify(token) : null;
  if (user) {
    const lids = accountLids(user.id);
    let lid = null;
    if (tok && lids.includes(tok.lid)) lid = tok.lid;
    else lid = primaryLid(user);
    if (!lid) return null;
    const name = tok && tok.lid === lid && tok.name ? tok.name : (user.name || nameForLid(lid, 'user'));
    const p = tok && tok.lid === lid ? tok : payloadFor(lid, name);
    p.lid = lid;
    p.name = name;
    return { p, user, source: 'session', lids };
  }
  if (tok && graceActive()) return { p: tok, user: null, source: 'token', lids: [tok.lid] };
  return null;
}

export const identUnauthorized = { error: 'invalid token, please enter your name again' };

function publicUser(user) {
  const lids = lidsForUser(user.id).map((r) => {
    const st = overrideMap()[r.lid] || 'auto';
    return { lid: r.lid, primary: !!r.is_primary, source: r.attach_source, status: st };
  });
  const primary = lids.find((l) => l.primary) || lids[0] || null;
  return {
    email: user.email,
    name: user.name,
    is_admin: !!user.is_admin,
    lid: primary ? primary.lid : null,
    lids,
    status: primary ? primary.status : 'auto',
    grace: graceActive(),
  };
}

async function sessionPayload(user) {
  const view = publicUser(user);
  const lid = view.lid;
  const name = lid ? await latestName(lid, user.name) : user.name;
  return {
    ...view,
    name,
    token: lid ? issueLabellerToken(lid, name) : null,
    labeler_id: lid,
  };
}

function redirect(res, loc) {
  res.statusCode = 302;
  res.setHeader('Location', loc);
  res.setHeader('Cache-Control', 'no-store');
  res.end();
}

function htmlError(res, code, msg) {
  res.statusCode = code;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(`<!doctype html><meta charset="utf-8"><title>Login</title><p>${escapeHtml(msg)}</p><p><a href="/login">Back to login</a></p>`);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function finishLogin(req, res, email, profileName, callback) {
  const user = provisionUser(email, profileName);
  const { token, exp } = createSession(user.id, req);
  setSessionCookie(res, token, exp);
  const dest = safeCallback(callback);
  if (wantsJson(req)) {
    const me = await sessionPayload(user);
    return send(res, 200, { ok: true, ...me });
  }
  return redirect(res, dest);
}

function wantsJson(req) {
  const a = String(req.headers.accept || '');
  return a.includes('application/json') && !a.includes('text/html');
}

// ---- route handlers ----
async function handleConfig(req, res) {
  return send(res, 200, { providers: providersStatus(), grace: graceActive(), owner_email: OWNER_EMAIL, login: '/login' });
}

async function handleMe(req, res) {
  const user = getSessionUser(req);
  if (!user) return send(res, 401, { error: 'not signed in', grace: graceActive(), providers: providersStatus() });
  return send(res, 200, await sessionPayload(user));
}

async function handleLogout(req, res) {
  destroySession(readCookie(req, COOKIE));
  clearCookie(res, COOKIE);
  if (req.method === 'GET' && !wantsJson(req)) return redirect(res, '/login');
  return send(res, 200, { ok: true });
}

async function handleMagicRequest(req, res) {
  if (!mailerConfigured()) return send(res, 503, { error: 'email login is not configured' });
  const email = validEmail(body(req).email);
  if (!email) return send(res, 400, { error: 'Please enter a valid email address.' });
  const callback = safeCallback(body(req).callbackURL);
  const token = crypto.randomBytes(32).toString('base64url');
  database().prepare('INSERT INTO magic_links (token_hash, email, callback, expires_at, created_at) VALUES (?, ?, ?, ?, ?)')
    .run(sha(token), email, callback, Date.now() + MAGIC_TTL_S * 1000, nowIso());
  if (!PUBLIC_URL) return send(res, 500, { error: 'server not configured' });
  const url = `${PUBLIC_URL}/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`;
  try {
    await sendMagicLinkEmail({ email, url });
  } catch (e) {
    return send(res, e.status || 502, { error: e.status ? e.message : 'could not send login email' });
  }
  return send(res, 200, { ok: true, message: 'Check your email for a login link. It expires in 10 minutes.' });
}

async function handleMagicVerify(req, res) {
  const u = new URL(req.url, 'http://x');
  const token = String(u.searchParams.get('token') || '');
  const fallback = safeCallback(u.searchParams.get('callbackURL'));
  if (!token) return htmlError(res, 400, 'Missing login token.');
  const row = database().prepare('SELECT * FROM magic_links WHERE token_hash = ?').get(sha(token));
  database().prepare('DELETE FROM magic_links WHERE token_hash = ?').run(sha(token));
  if (!row || Number(row.expires_at) < Date.now()) {
    if (wantsJson(req)) return send(res, 401, { error: 'This login link is invalid or has expired.' });
    return htmlError(res, 401, 'This login link is invalid or has expired.');
  }
  return finishLogin(req, res, String(row.email).toLowerCase(), null, row.callback || fallback);
}

function pkce() {
  const verifier = crypto.randomBytes(32).toString('base64url');
  const challenge = crypto.createHash('sha256').update(verifier).digest('base64url');
  return { verifier, challenge };
}

function beginOAuth(res, provider, verifier, callback) {
  const state = crypto.randomBytes(24).toString('base64url');
  database().prepare('INSERT INTO oauth_states (state, provider, code_verifier, callback, expires_at) VALUES (?, ?, ?, ?, ?)')
    .run(state, provider, verifier, safeCallback(callback), Date.now() + OAUTH_TTL_S * 1000);
  appendCookie(res, `${OAUTH_COOKIE}=${state}; ${cookieSuffix()}; Max-Age=${OAUTH_TTL_S}`);
  return state;
}

function consumeOAuth(req, res, provider, state) {
  clearCookie(res, OAUTH_COOKIE);
  if (!state) return null;
  const cookieState = readCookie(req, OAUTH_COOKIE);
  const row = database().prepare('SELECT * FROM oauth_states WHERE state = ?').get(state);
  database().prepare('DELETE FROM oauth_states WHERE state = ?').run(state);
  if (!row || row.provider !== provider || Number(row.expires_at) < Date.now()) return null;
  if (cookieState && cookieState !== state) return null;
  return row;
}

function oauthConfigured(provider) {
  const p = providersStatus();
  return provider === 'google' ? p.google : p.x;
}

async function handleLoginGoogle(req, res) {
  if (!oauthConfigured('google')) return htmlError(res, 503, 'Google login is not configured yet.');
  const origin = oauthPublicUrl();
  if (!origin) return htmlError(res, 500, 'server not configured');
  const u = new URL(req.url, 'http://x');
  const { verifier, challenge } = pkce();
  const state = beginOAuth(res, 'google', verifier, u.searchParams.get('callbackURL'));
  const dest = new URL('https://accounts.google.com/o/oauth2/v2/auth');
  dest.searchParams.set('client_id', process.env.AUTH_GOOGLE_CLIENT_ID);
  dest.searchParams.set('redirect_uri', origin + '/api/auth/callback/google');
  dest.searchParams.set('response_type', 'code');
  dest.searchParams.set('scope', 'openid email profile');
  dest.searchParams.set('state', state);
  dest.searchParams.set('code_challenge', challenge);
  dest.searchParams.set('code_challenge_method', 'S256');
  dest.searchParams.set('prompt', 'select_account');
  return redirect(res, dest.toString());
}

async function handleLoginX(req, res) {
  if (!oauthConfigured('x')) return htmlError(res, 503, 'X login is not configured yet.');
  const origin = oauthPublicUrl();
  if (!origin) return htmlError(res, 500, 'server not configured');
  const u = new URL(req.url, 'http://x');
  const { verifier, challenge } = pkce();
  const state = beginOAuth(res, 'x', verifier, u.searchParams.get('callbackURL'));
  const dest = new URL('https://twitter.com/i/oauth2/authorize');
  dest.searchParams.set('response_type', 'code');
  dest.searchParams.set('client_id', process.env.AUTH_X_CLIENT_ID);
  dest.searchParams.set('redirect_uri', origin + '/api/auth/callback/x');
  dest.searchParams.set('scope', 'tweet.read users.read offline.access users.email');
  dest.searchParams.set('state', state);
  dest.searchParams.set('code_challenge', challenge);
  dest.searchParams.set('code_challenge_method', 'S256');
  return redirect(res, dest.toString());
}

async function handleCallbackGoogle(req, res) {
  const u = new URL(req.url, 'http://x');
  if (u.searchParams.get('error')) return htmlError(res, 401, 'Google login was cancelled or failed.');
  const row = consumeOAuth(req, res, 'google', u.searchParams.get('state'));
  const code = u.searchParams.get('code');
  if (!row || !code) return htmlError(res, 401, 'Google login could not be completed (invalid state).');
  const tokenRes = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: process.env.AUTH_GOOGLE_CLIENT_ID,
      client_secret: process.env.AUTH_GOOGLE_CLIENT_SECRET,
      code,
      grant_type: 'authorization_code',
      redirect_uri: oauthPublicUrl() + '/api/auth/callback/google',
      code_verifier: row.code_verifier,
    }),
  });
  if (!tokenRes.ok) {
    console.error('google token exchange failed status=' + tokenRes.status);
    return htmlError(res, 401, 'Google login could not be completed.');
  }
  const tok = await tokenRes.json();
  const access = tok.access_token;
  if (!access) return htmlError(res, 401, 'Google login could not be completed.');
  const infoRes = await fetch('https://openidconnect.googleapis.com/v1/userinfo', { headers: { Authorization: 'Bearer ' + access } });
  if (!infoRes.ok) return htmlError(res, 401, 'Google login could not be completed.');
  const info = await infoRes.json();
  const email = validEmail(info.email);
  if (!email) return htmlError(res, 401, 'Google did not provide an email address.');
  if (info.email_verified === false) return htmlError(res, 401, 'Google email is not verified.');
  return finishLogin(req, res, email, info.name || null, row.callback);
}

async function handleCallbackX(req, res) {
  const u = new URL(req.url, 'http://x');
  if (u.searchParams.get('error')) return htmlError(res, 401, 'X login was cancelled or failed.');
  const row = consumeOAuth(req, res, 'x', u.searchParams.get('state'));
  const code = u.searchParams.get('code');
  if (!row || !code) return htmlError(res, 401, 'X login could not be completed (invalid state).');
  const basic = Buffer.from(process.env.AUTH_X_CLIENT_ID + ':' + process.env.AUTH_X_CLIENT_SECRET).toString('base64');
  const tokenRes = await fetch('https://api.x.com/2/oauth2/token', {
    method: 'POST',
    headers: { Authorization: 'Basic ' + basic, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      code,
      grant_type: 'authorization_code',
      redirect_uri: oauthPublicUrl() + '/api/auth/callback/x',
      code_verifier: row.code_verifier,
    }),
  });
  if (!tokenRes.ok) {
    console.error('x token exchange failed status=' + tokenRes.status);
    return htmlError(res, 401, 'X login could not be completed.');
  }
  const tok = await tokenRes.json();
  const access = tok.access_token;
  if (!access) return htmlError(res, 401, 'X login could not be completed.');
  const infoRes = await fetch('https://api.x.com/2/users/me?user.fields=confirmed_email,name,username', { headers: { Authorization: 'Bearer ' + access } });
  if (!infoRes.ok) return htmlError(res, 401, 'X login could not be completed.');
  const info = await infoRes.json();
  const data = info.data || {};
  const email = validEmail(data.confirmed_email);
  if (!email) return htmlError(res, 401, 'X did not provide an email address. Enable the users.email scope and a verified email on the X account.');
  return finishLogin(req, res, email, data.name || data.username || null, row.callback);
}

async function handleAttach(req, res) {
  const user = getSessionUser(req);
  if (!user) return send(res, 401, { error: 'Please log in first.' });
  const token = body(req).token;
  const tok = verify(token);
  if (!tok) return send(res, 400, { error: 'This device has no valid labeller identity to attach.' });
  const lid = tok.lid;
  if (!/^[0-9a-f]{16}$/.test(lid)) return send(res, 400, { error: 'bad labeller id' });
  if (lid === OWNER_LID && String(user.email).toLowerCase() !== OWNER_EMAIL) {
    return send(res, 403, { error: 'The owner labeller can only be attached to the owner email.' });
  }
  const existing = lidRow(lid);
  if (existing && existing.user_id !== user.id) {
    return send(res, 409, { error: 'That labeller identity is already attached to another account.' });
  }
  const d = database();
  if (!existing) {
    d.prepare('INSERT INTO lids (lid, user_id, is_primary, attached_at, attach_source) VALUES (?, ?, 0, ?, ?)')
      .run(lid, user.id, nowIso(), 'device');
  }
  // Prefer the attached device lid as primary. Drop an unused auto-provisioned lid with no history.
  const oldPrimary = primaryLid(user);
  setPrimary(user.id, lid);
  if (oldPrimary && oldPrimary !== lid) {
    const old = lidRow(oldPrimary);
    if (old && old.attach_source === 'provisioned') {
      const seen = abSeen(oldPrimary).size + gridSeen(oldPrimary).size;
      if (seen === 0) d.prepare('DELETE FROM lids WHERE lid = ? AND user_id = ? AND attach_source = ?').run(oldPrimary, user.id, 'provisioned');
    }
  }
  const fresh = userById(user.id);
  return send(res, 200, { ok: true, attached: lid, already: !!existing, ...(await sessionPayload(fresh)) });
}

const ROUTES = {
  'GET /api/auth/config': handleConfig,
  'GET /api/auth/me': handleMe,
  'POST /api/auth/logout': handleLogout,
  'GET /api/auth/logout': handleLogout,
  'POST /api/auth/magic-link': handleMagicRequest,
  'GET /api/auth/magic-link/verify': handleMagicVerify,
  'GET /api/auth/login/google': handleLoginGoogle,
  'GET /api/auth/login/x': handleLoginX,
  'GET /api/auth/callback/google': handleCallbackGoogle,
  'GET /api/auth/callback/x': handleCallbackX,
  'POST /api/auth/attach': handleAttach,
};

export async function authHandler(req, res) {
  try {
    authSecret();
    if (!db) database();
    // Drop expired leftovers occasionally
    if (Math.random() < 0.05) {
      const t = Date.now();
      database().prepare('DELETE FROM sessions WHERE expires_at < ?').run(t);
      database().prepare('DELETE FROM magic_links WHERE expires_at < ?').run(t);
      database().prepare('DELETE FROM oauth_states WHERE expires_at < ?').run(t);
    }
    const u = new URL(req.url, 'http://x');
    const fn = ROUTES[req.method + ' ' + u.pathname];
    if (!fn) return send(res, 404, { error: 'not found' });
    return await fn(req, res);
  } catch (e) {
    console.error('auth handler', e && e.message);
    if (res.headersSent) return;
    return send(res, e.status || 500, { error: e.status ? e.message : 'server error' });
  }
}

export function closeAuth() {
  if (db) { try { db.close(); } catch { /* ok */ } db = null; }
}
