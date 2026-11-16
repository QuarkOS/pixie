// Staged auth tests. Run on the container against 127.0.0.1 only.
// Never prints secrets, tokens, magic-link URLs, or passwords.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { inflightFor, enqueue } from '../api/_lib/chat.js';

const BASE = process.env.AUTH_TEST_BASE || 'http://127.0.0.1:8791';
const AUTH_DIR = process.env.AUTH_DIR || path.join(process.cwd(), 'data', 'auth');
const PROD_DATA = process.env.PROD_DATA || '';
const OWNER_EMAIL = process.env.OWNER_EMAIL || 'owner@example.invalid';
const OWNER_LID = process.env.OWNER_LID || 'owner';
const OWNER_NAME = process.env.OWNER_NAME || 'owner';
const LIVE_FILES = ['labels.jsonl', 'chat.jsonl', 'overrides.json'];

const results = [];
function ok(name, detail) { results.push({ name, ok: true, detail: detail || '' }); console.log('PASS  ' + name + (detail ? ' — ' + detail : '')); }
function fail(name, err) { results.push({ name, ok: false, detail: String(err && err.message || err) }); console.log('FAIL  ' + name + ' — ' + String(err && err.message || err)); }

function sha256File(p) { return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'); }
function prodSums() {
  if (!PROD_DATA) return null;
  const o = {};
  for (const f of LIVE_FILES) o[f] = sha256File(path.join(PROD_DATA, f));
  return o;
}

function cookieJar(setCookie) {
  const jar = new Map();
  for (const raw of [].concat(setCookie || [])) {
    const m = /^([^=]+)=([^;]*)/.exec(raw);
    if (m) jar.set(m[1], m[2]);
  }
  return jar;
}
function cookieHeader(jar) { return [...jar].map(([k, v]) => k + '=' + v).join('; '); }

async function req(method, url, { body, cookies, headers, redirect } = {}) {
  const h = { ...(headers || {}) };
  if (cookies && cookies.size) h.cookie = cookieHeader(cookies);
  if (body !== undefined && !h['content-type']) h['content-type'] = 'application/json';
  const r = await fetch(url, {
    method,
    headers: h,
    body: body !== undefined ? (typeof body === 'string' ? body : JSON.stringify(body)) : undefined,
    redirect: redirect || 'manual',
  });
  const list = typeof r.headers.getSetCookie === 'function' ? r.headers.getSetCookie() : [];
  const set = cookieJar(list);
  let text = '';
  try { text = await r.text(); } catch { text = ''; }
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch { json = null; }
  return { status: r.status, headers: r.headers, cookies: set, text, json, location: r.headers.get('location') };
}

function mergeCookies(jar, add) { for (const [k, v] of add) { if (v === '') jar.delete(k); else jar.set(k, v); } return jar; }

function readMagic() {
  const p = path.join(AUTH_DIR, 'last-magic-link.json');
  if (!fs.existsSync(p)) throw new Error('console mailer file missing');
  const rec = JSON.parse(fs.readFileSync(p, 'utf8'));
  if (!rec.url || !rec.email) throw new Error('console mailer file incomplete');
  return rec;
}

async function magicLogin(email) {
  const r = await req('POST', BASE + '/api/auth/magic-link', { body: { email, callbackURL: '/account' } });
  if (r.status !== 200) throw new Error('magic-link request failed: HTTP ' + r.status + (r.json && r.json.error ? ' ' + r.json.error : ''));
  const rec = readMagic();
  if (rec.email !== email) throw new Error('mailer wrote a different recipient');
  const verifyUrl = rec.url; // contains token; do not log
  if (!verifyUrl.startsWith(BASE) && !verifyUrl.includes('/api/auth/magic-link/verify?token=')) throw new Error('unexpected verify URL shape');
  const v = await req('GET', verifyUrl, { redirect: 'manual' });
  if (v.status !== 302) throw new Error('verify did not redirect, HTTP ' + v.status);
  const jar = mergeCookies(new Map(), v.cookies);
  if (!jar.has('pxsess')) throw new Error('no session cookie after verify');
  return jar;
}

async function me(jar) {
  const r = await req('GET', BASE + '/api/auth/me', { cookies: jar });
  if (r.status !== 200) throw new Error('me HTTP ' + r.status);
  return r.json;
}

const before = prodSums();
process.env._AUTH_TEST_BEFORE = JSON.stringify(before);

try {
  const health = await req('GET', BASE + '/login');
  assert.equal(health.status, 200);
  assert.match(health.text, /Log in to Pixie/);
  ok('staged login page serves on localhost');
} catch (e) { fail('staged login page serves on localhost', e); }

try {
  const r = await req('GET', BASE + '/api/auth/config');
  assert.equal(r.status, 200);
  assert.equal(r.json.providers.magic_link, false); // Resend not configured yet; console mailer still works in tests
  assert.equal(r.json.providers.google, true);
  assert.equal(r.json.providers.x, false);
  assert.equal(r.json.grace, true);
  ok('auth config (google on, x off, email off until Resend, grace on)');
} catch (e) { fail('auth config', e); }

try {
  const g = await req('GET', BASE + '/api/auth/login/google');
  assert.equal(g.status, 302, 'google should redirect, HTTP ' + g.status);
  assert.ok(g.location, 'missing Location');
  const loc = new URL(g.location);
  assert.equal(loc.protocol, 'https:');
  assert.equal(loc.hostname, 'accounts.google.com');
  assert.ok(loc.searchParams.get('client_id'), 'missing client_id');
  assert.equal(loc.searchParams.get('redirect_uri'), 'https://pixie.example.invalid/api/auth/callback/google');
  assert.equal(loc.searchParams.get('response_type'), 'code');
  assert.ok(loc.searchParams.get('state'), 'missing state');
  const x = await req('GET', BASE + '/api/auth/login/x');
  assert.equal(x.status, 503);
  ok('Google login redirects to accounts.google.com with the public callback; X stays disabled');
} catch (e) { fail('Google login redirect / X disabled', e); }

try {
  const privacy = await req('GET', BASE + '/privacy');
  const terms = await req('GET', BASE + '/terms');
  const login = await req('GET', BASE + '/login');
  assert.equal(privacy.status, 200);
  assert.equal(terms.status, 200);
  assert.match(privacy.text, /owner@example.invalid/);
  assert.match(privacy.text, /IP addresses/);
  assert.match(terms.text, /as-is/i);
  assert.match(login.text, /href="\/privacy"/);
  assert.match(login.text, /href="\/terms"/);
  ok('privacy and terms pages serve and are linked from login');
} catch (e) { fail('privacy and terms pages', e); }

// --- HMAC grace: name-gate token still works ---
let guestToken = null, guestLid = null;
try {
  const r = await req('POST', BASE + '/api/register', { body: { name: 'AuthTestGuest' } });
  assert.equal(r.status, 200, r.json && r.json.error);
  guestToken = r.json.token;
  guestLid = r.json.labeler_id;
  assert.match(guestLid, /^[0-9a-f]{16}$/);
  const next = await req('POST', BASE + '/api/ab/next', { body: { token: guestToken } });
  assert.equal(next.status, 200, next.json && next.json.error);
  assert.ok(Array.isArray(next.json.block));
  ok('HMAC name-gate token works during grace', 'lid issued and /api/ab/next accepted');
} catch (e) { fail('HMAC name-gate token works during grace', e); }

// --- Owner magic-link mapping + admin ---
let ownerJar = null;
try {
  ownerJar = await magicLogin(OWNER_EMAIL);
  const m = await me(ownerJar);
  assert.equal(m.email, OWNER_EMAIL);
  assert.equal(m.labeler_id, OWNER_LID);
  assert.equal(m.lid, OWNER_LID);
  assert.equal(m.is_admin, true);
  assert.equal(m.status, 'owner');
  assert.equal(m.name, OWNER_NAME);
  ok('owner email maps to existing lid and is admin');
} catch (e) { fail('owner email maps to existing lid and is admin', e); }

try {
  if (!ownerJar) throw new Error('no owner session');
  const r = await req('GET', BASE + '/api/admin/training', { cookies: ownerJar });
  assert.equal(r.status, 200, r.json && r.json.error);
  ok('owner session can access admin API (no password cookie)');
} catch (e) { fail('owner session can access admin API (no password cookie)', e); }

// Password admin still works (lockout path untouched). Password is read from the env, never printed.
try {
  const pw = process.env.ADMIN_PASSWORD || '';
  if (pw.length < 16) throw new Error('ADMIN_PASSWORD missing in test env');
  const r = await req('POST', BASE + '/api/admin/login', { body: { password: pw } });
  assert.equal(r.status, 200, r.json && r.json.error);
  assert.ok([...r.cookies.keys()].some((k) => k === 'pxadm'));
  const a = await req('GET', BASE + '/api/admin/training', { cookies: r.cookies });
  assert.equal(a.status, 200, a.json && a.json.error);
  ok('admin password login still works');
} catch (e) { fail('admin password login still works', e); }

try {
  const r = await req('POST', BASE + '/api/admin/login', { body: { password: 'definitely-wrong-password-xxxx' } });
  assert.equal(r.status, 401);
  ok('admin password rejection still 401 (lockout unchanged)');
} catch (e) { fail('admin password rejection still 401 (lockout unchanged)', e); }

// --- Attach flows ---
try {
  const guest = await req('POST', BASE + '/api/register', { body: { name: 'AuthAttachMe' } });
  assert.equal(guest.status, 200);
  const jar = await magicLogin('owner@example.invalid');
  const beforeMe = await me(jar);
  assert.notEqual(beforeMe.labeler_id, guest.json.labeler_id);
  const att = await req('POST', BASE + '/api/auth/attach', { body: { token: guest.json.token }, cookies: jar });
  assert.equal(att.status, 200, att.json && att.json.error);
  assert.equal(att.json.attached, guest.json.labeler_id);
  assert.equal(att.json.labeler_id, guest.json.labeler_id);
  const after = await me(jar);
  assert.ok(after.lids.some((l) => l.lid === guest.json.labeler_id && l.primary));
  ok('device lid attaches to a signed-in account and becomes primary');
} catch (e) { fail('device lid attaches to a signed-in account and becomes primary', e); }

try {
  if (!guestToken) throw new Error('no guest owner-lid token? using a fresh name-gate token for owner lid test');
  // Issue a forged? We cannot mint owner lid without TOKEN_SECRET + rate endpoint.
  // Use admin rate endpoint with password to get an owner HMAC token, then try attaching it to a non-owner account.
  const pw = process.env.ADMIN_PASSWORD || '';
  const login = await req('POST', BASE + '/api/admin/login', { body: { password: pw } });
  const rate = await req('POST', BASE + '/api/admin/rate', { body: {}, cookies: login.cookies });
  assert.equal(rate.status, 200, rate.json && rate.json.error);
  assert.equal(rate.json.labeler_id, OWNER_LID);
  const jar = await magicLogin('owner@example.invalid');
  const att = await req('POST', BASE + '/api/auth/attach', { body: { token: rate.json.token }, cookies: jar });
  assert.equal(att.status, 403);
  ok('owner lid cannot attach to a non-owner email');
} catch (e) { fail('owner lid cannot attach to a non-owner email', e); }

try {
  if (!ownerJar) throw new Error('no owner session');
  const pw = process.env.ADMIN_PASSWORD || '';
  const login = await req('POST', BASE + '/api/admin/login', { body: { password: pw } });
  const rate = await req('POST', BASE + '/api/admin/rate', { body: {}, cookies: login.cookies });
  const att = await req('POST', BASE + '/api/auth/attach', { body: { token: rate.json.token }, cookies: ownerJar });
  assert.equal(att.status, 200, att.json && att.json.error);
  assert.equal(att.json.labeler_id, OWNER_LID);
  ok('owner lid can attach (idempotent) to the owner email');
} catch (e) { fail('owner lid can attach (idempotent) to the owner email', e); }

// Trusted/excluded status is still served from overrides.json (owner already checked).
try {
  const ov = JSON.parse(fs.readFileSync(path.join(process.env.DATA_DIR || path.join(process.cwd(), 'data'), 'overrides.json'), 'utf8'));
  assert.equal((ov[OWNER_LID] || {}).status, 'owner');
  const trusted = Object.entries(ov).filter(([, v]) => v && v.status === 'trusted').length;
  const excluded = Object.entries(ov).filter(([, v]) => v && v.status === 'excluded').length;
  assert.ok(trusted >= 1 && excluded >= 1);
  ok('overrides.json trusted/excluded/owner status intact on staged data', `${trusted} trusted, ${excluded} excluded`);
} catch (e) { fail('overrides.json trusted/excluded/owner status intact on staged data', e); }

// --- one in-flight chat job per account (module-level, no worker needed) ---
try {
  const a = enqueue(OWNER_LID, OWNER_NAME, 'red potion', { accountLids: [OWNER_LID] });
  assert.ok(a.id, 'first enqueue should return an id');
  const b = enqueue(OWNER_LID, OWNER_NAME, 'blue hat', { accountLids: [OWNER_LID] });
  assert.equal(b.busy, true);
  assert.equal(b.reason, 'in_flight');
  assert.ok(inflightFor([OWNER_LID]));
  ok('one in-flight chat job per account');
} catch (e) { fail('one in-flight chat job per account', e); }

// --- grace off: spawn a second server on 8792 ---
try {
  const extraEnv = {
    ...process.env,
    PORT: '8792',
    HOST: '127.0.0.1',
    AUTH_TOKEN_GRACE_UNTIL: 'off',
    AUTH_PUBLIC_URL: 'http://127.0.0.1:8792',
    AUTH_DIR: path.join(AUTH_DIR, 'grace-off'),
    AUTH_COOKIE_SECURE: '0',
    AUTH_MAILER: 'console',
  };
  fs.mkdirSync(extraEnv.AUTH_DIR, { recursive: true, mode: 0o700 });
  const child = spawn(process.execPath, ['server.mjs'], {
    cwd: path.resolve(path.dirname(new URL(import.meta.url).pathname), '..'),
    env: extraEnv,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let ready = false;
  const boot = new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('grace-off server did not start')), 8000);
    const onData = (buf) => {
      const s = buf.toString();
      if (s.includes('pixel-labeller on http://')) { ready = true; clearTimeout(t); resolve(); }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('exit', (code) => { if (!ready) { clearTimeout(t); reject(new Error('grace-off server exited ' + code)); } });
  });
  await boot;
  try {
    if (!guestToken) throw new Error('no grace token to test');
    const r = await req('POST', 'http://127.0.0.1:8792/api/ab/next', { body: { token: guestToken } });
    assert.equal(r.status, 401);
    const reg = await req('POST', 'http://127.0.0.1:8792/api/register', { body: { name: 'AfterGrace' } });
    assert.equal(reg.status, 410);
    // session still works
    const ml = await req('POST', 'http://127.0.0.1:8792/api/auth/magic-link', { body: { email: 'owner@example.invalid' } });
    assert.equal(ml.status, 200);
    const rec = JSON.parse(fs.readFileSync(path.join(extraEnv.AUTH_DIR, 'last-magic-link.json'), 'utf8'));
    const v = await req('GET', rec.url, { redirect: 'manual' });
    assert.equal(v.status, 302);
    const jar = mergeCookies(new Map(), v.cookies);
    const m = await req('GET', 'http://127.0.0.1:8792/api/auth/me', { cookies: jar });
    assert.equal(m.status, 200);
    const next = await req('POST', 'http://127.0.0.1:8792/api/ab/next', { body: { token: m.json.token }, cookies: jar });
    assert.equal(next.status, 200, next.json && next.json.error);
    ok('after grace, HMAC tokens are rejected and login sessions still work');
  } finally {
    child.kill('SIGTERM');
    await new Promise((r) => setTimeout(r, 300));
  }
} catch (e) { fail('after grace, HMAC tokens are rejected and login sessions still work', e); }

// Production data must be byte-for-byte unchanged when PROD_DATA is set.
try {
  if (!before) {
    ok('production checksum skipped (set PROD_DATA to compare a live data dir)');
  } else {
    const after = prodSums();
    const changed = LIVE_FILES.filter((f) => after[f] !== before[f]);
    assert.deepEqual(changed, []);
    ok('production labels/chat/overrides checksums unchanged', LIVE_FILES.map((f) => f + '=' + after[f].slice(0, 12)).join(' '));
  }
} catch (e) { fail('production labels/chat/overrides checksums unchanged', e); }

const failed = results.filter((r) => !r.ok);
console.log('\n' + results.length + ' tests, ' + (results.length - failed.length) + ' passed, ' + failed.length + ' failed');
if (failed.length) process.exit(1);
