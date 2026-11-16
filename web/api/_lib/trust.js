// Trust tiers for Flywheel v2. Derived from admin overrides + owner lid + auth source.
// Does not write overrides, votes, or auth records.
import { OWNER_LID } from './auth.js';
import { overrideMap } from './store.js';

export function getTrustTier(lid, source) {
  if (!lid) return 'anon';
  const st = overrideMap()[lid];
  if (lid === OWNER_LID || st === 'owner') return 'owner';
  if (st === 'excluded') return 'excluded';
  if (st === 'trusted') return 'trusted';
  if (source === 'session') return 'authed';
  return 'anon';
}

export function actorFromIdent(ident) {
  if (!ident || !ident.p || !ident.p.lid) return null;
  const lid = ident.p.lid;
  return {
    ok: true,
    lid,
    user_id: ident.user && ident.user.id ? ident.user.id : null,
    name: ident.p.name || ident.p.lid,
    trust_tier: getTrustTier(lid, ident.source),
  };
}
