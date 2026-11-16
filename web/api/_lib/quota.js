// Daily free-generation quota for name-gate (unsigned) chat users.
// Signed-in sessions, the owner lid, and admin users are unlimited.
// Counts are stored under DATA_DIR (not labels.jsonl / chat.jsonl). IPs are hashed.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { OWNER_LID } from './auth.js';
import { getTrustTier } from './trust.js';

const DIR = process.env.DATA_DIR || path.join(process.cwd(), 'data');
const FILE = path.join(DIR, 'chat-quota.json');

export function freeDailyLimit() {
  const n = Number(process.env.CHAT_FREE_DAILY_LIMIT || 20);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 20;
}

const QUOTA_TZ = process.env.PIXIE_TZ || 'UTC';

export function quotaDay(now = Date.now()) {
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: QUOTA_TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date(now));
}

export const viennaDay = quotaDay;

export function clientIp(req) {
  const cf = String((req && req.headers && (req.headers['cf-connecting-ip'] || req.headers['CF-Connecting-IP'])) || '').trim();
  if (cf) return cf.split(',')[0].trim();
  const xff = String((req && req.headers && req.headers['x-forwarded-for']) || '').trim();
  if (xff) return xff.split(',')[0].trim();
  const ra = (req && req.socket && req.socket.remoteAddress) || (req && req.connection && req.connection.remoteAddress) || '';
  return String(ra).replace(/^::ffff:/, '');
}

function ipKey(ip) {
  if (!ip) return '';
  return crypto.createHash('sha256').update('px-quota-ip:' + ip).digest('hex').slice(0, 16);
}

export function quotaExempt(ident) {
  if (!ident) return false;
  if (ident.source === 'session') return true;
  if (ident.user && ident.user.is_admin) return true;
  const lid = ident.p && ident.p.lid;
  if (lid && lid === OWNER_LID) return true;
  if (lid && getTrustTier(lid, ident.source) === 'owner') return true;
  return false;
}

export function quotaMessage(limit) {
  return `You've used your ${limit} free generations today. Sign in with Google for unlimited.`;
}

function emptyState(day) { return { day, lids: {}, ips: {} }; }

function load() {
  const day = quotaDay();
  try {
    if (!fs.existsSync(FILE)) return emptyState(day);
    const st = JSON.parse(fs.readFileSync(FILE, 'utf8'));
    if (!st || typeof st !== 'object' || st.day !== day) return emptyState(day);
    return {
      day,
      lids: st.lids && typeof st.lids === 'object' && !Array.isArray(st.lids) ? st.lids : {},
      ips: st.ips && typeof st.ips === 'object' && !Array.isArray(st.ips) ? st.ips : {},
    };
  } catch {
    return emptyState(day);
  }
}

function save(st) {
  fs.mkdirSync(DIR, { recursive: true, mode: 0o750 });
  const tmp = FILE + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(st) + '\n', { mode: 0o600 });
  fs.renameSync(tmp, FILE);
}

function usedOf(st, lid, hash) {
  const lidUsed = lid && typeof st.lids[lid] === 'number' ? st.lids[lid] : 0;
  const ipUsed = hash && typeof st.ips[hash] === 'number' ? st.ips[hash] : 0;
  return { lidUsed, ipUsed, used: Math.max(lidUsed, ipUsed) };
}

export function peekQuota(ident, req) {
  const limit = freeDailyLimit();
  if (quotaExempt(ident)) return { limited: false, limit, remaining: null, used: 0 };
  const st = load();
  const lid = ident && ident.p && ident.p.lid ? String(ident.p.lid) : '';
  const hash = ipKey(clientIp(req));
  const { used } = usedOf(st, lid, hash);
  return { limited: true, limit, remaining: Math.max(0, limit - used), used };
}

// Call only after a job is successfully queued.
export function consumeQuota(ident, req) {
  const peek = peekQuota(ident, req);
  if (!peek.limited) return { ...peek, ok: true };
  if (peek.remaining <= 0) return { ...peek, remaining: 0, ok: false, code: 'quota', error: quotaMessage(peek.limit) };
  const st = load();
  const lid = ident && ident.p && ident.p.lid ? String(ident.p.lid) : '';
  const hash = ipKey(clientIp(req));
  if (lid) st.lids[lid] = (typeof st.lids[lid] === 'number' ? st.lids[lid] : 0) + 1;
  if (hash) st.ips[hash] = (typeof st.ips[hash] === 'number' ? st.ips[hash] : 0) + 1;
  save(st);
  const { used } = usedOf(st, lid, hash);
  return { limited: true, limit: peek.limit, remaining: Math.max(0, peek.limit - used), used, ok: true };
}

export { FILE as QUOTA_FILE, ipKey };
