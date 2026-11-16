// Outbound mail for magic links. Never log the link or token.
// AUTH_MAILER=console (default, staged/dev) writes the URL to AUTH_DIR/last-magic-link.json (mode 0600).
// AUTH_MAILER=resend sends via Resend when AUTH_RESEND_API_KEY and AUTH_FROM_EMAIL are set.
import fs from 'node:fs';
import path from 'node:path';

function redact(email) {
  const s = String(email || '');
  const i = s.indexOf('@');
  if (i < 1) return '(redacted)';
  return s[0] + '…' + s.slice(i);
}

function resendApiKey() {
  const k = process.env.AUTH_RESEND_API_KEY || process.env.RESEND_API_KEY || '';
  if (!k || k === 'REPLACE_ME') return '';
  return k;
}

export function mailerConfigured() {
  const kind = (process.env.AUTH_MAILER || 'console').toLowerCase();
  if (kind === 'console') return true;
  if (kind === 'resend') return !!(resendApiKey() && process.env.AUTH_FROM_EMAIL);
  return false;
}

// True only when email can actually be delivered to users.
export function mailerPubliclyEnabled() {
  const kind = (process.env.AUTH_MAILER || 'console').toLowerCase();
  return kind === 'resend' && !!(resendApiKey() && String(process.env.AUTH_FROM_EMAIL || '').trim());
}

export async function sendMagicLinkEmail({ email, url }) {
  const kind = (process.env.AUTH_MAILER || 'console').toLowerCase();
  if (kind === 'resend') return sendResend({ email, url });
  return sendConsole({ email, url });
}

function sendConsole({ email, url }) {
  const dir = process.env.AUTH_DIR || path.join(process.cwd(), 'data', 'auth');
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const dest = path.join(dir, 'last-magic-link.json');
  const rec = { email, url, at: new Date().toISOString() };
  const tmp = dest + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(rec) + '\n', { mode: 0o600 });
  fs.renameSync(tmp, dest);
  console.log(`auth mailer=console magic-link issued to ${redact(email)} (url written to last-magic-link.json)`);
}

async function sendResend({ email, url }) {
  const key = resendApiKey();
  const from = process.env.AUTH_FROM_EMAIL || '';
  if (!key || !from) throw Object.assign(new Error('email sender is not configured'), { status: 503 });
  const r = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + key, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      from,
      to: [email],
      subject: 'Your Pixie login link',
      text: `Sign in to Pixie with this link (expires in 10 minutes):\n\n${url}\n\nIf you did not request this, you can ignore the email.`,
    }),
  });
  if (!r.ok) {
    console.error('auth mailer=resend failed status=' + r.status);
    throw Object.assign(new Error('could not send login email'), { status: 502 });
  }
  console.log(`auth mailer=resend magic-link issued to ${redact(email)}`);
}
