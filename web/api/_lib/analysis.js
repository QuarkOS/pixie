// Pure analysis shared by the admin API (export filter) and the admin page (live threshold).
// data = { rounds: {rid: {prompt, kind, ck:[ckpt per cell], truth?:[0|1 per cell]}}, labels: [{lid,name,round_id,works,all_bad,response_ms,ts,flags}] }
export function dedupe(labels) {
  const seen = new Set();
  const out = [];
  for (const l of [...labels].sort((a, b) => String(a.ts).localeCompare(String(b.ts)))) {
    const k = l.lid + '|' + l.round_id;
    if (seen.has(k)) continue;
    seen.add(k); out.push(l);
  }
  return out;
}

function median(a) { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

export const STATUSES = ['auto', 'owner', 'trusted', 'excluded'];
export const isOwnerLid = (lid, ov) => lid === 'owner' || (ov && ov[lid] === 'owner');

// Reference truth per round: calibration rounds -> the owner's imported labels; any other round an owner labelled -> owner's clicks.
export function referenceTruth(data, ov) {
  const ref = new Map();
  for (const [rid, r] of Object.entries(data.rounds)) if (r.kind === 'cal' && r.truth) ref.set(rid, r.truth);
  for (const l of dedupe(data.labels)) {
    if (l.lid === 'owner' || !isOwnerLid(l.lid, ov)) continue;
    const r = data.rounds[l.round_id];
    if (!r || r.kind !== 'pool' || ref.has(l.round_id)) continue;
    const set = new Set(l.works || []);
    ref.set(l.round_id, r.ck.map((_, i) => (set.has(i) ? 1 : 0)));
  }
  return ref;
}

export function labellerStats(data, threshold = 0.3, minCal = 3, overrides = null) {
  const ov = overrides ?? data.overrides ?? {};
  const ref = referenceTruth(data, ov);
  const by = new Map();
  for (const l of dedupe(data.labels)) {
    const r = data.rounds[l.round_id];
    if (!r) continue;
    let s = by.get(l.lid);
    if (!s) { s = { lid: l.lid, name: l.name, n: 0, rts: [], cells: 0, works: 0, allBad: 0, fast: 0, allClicked: 0,
      cal: { n: 0, tp: 0, fp: 0, fn: 0, tn: 0 }, ck: {}, pck: {}, first: l.ts, last: l.ts }; by.set(l.lid, s); }
    s.n++; s.last = l.ts;
    if (Number.isFinite(l.response_ms)) s.rts.push(l.response_ms);
    const set = new Set(l.works || []);
    s.cells += r.ck.length; s.works += set.size;
    if (set.size === 0) s.allBad++;
    if ((l.flags || []).includes('fast')) s.fast++;
    if (set.size === r.ck.length) s.allClicked++;
    const owner = isOwnerLid(l.lid, ov);
    // agreement: owners only vs the imported calibration truth; everyone else vs calibration + owner-labelled rounds
    const truth = l.lid === 'owner' ? null : owner ? (r.kind === 'cal' ? r.truth : null) : ref.get(l.round_id);
    if (truth) {
      s.cal.n++;
      truth.forEach((t, i) => { const w = set.has(i); if (t && w) s.cal.tp++; else if (!t && w) s.cal.fp++; else if (t && !w) s.cal.fn++; else s.cal.tn++; });
    }
    if (r.kind !== 'cal') {
      r.ck.forEach((c, i) => {
        const w = set.has(i) ? 1 : 0;
        (s.ck[c] ||= { w: 0, t: 0 }).w += w; s.ck[c].t++;
        const pp = (s.pck[r.prompt] ||= {});
        (pp[c] ||= { w: 0, t: 0 }).w += w; pp[c].t++;
      });
    }
  }
  // ---- A/B preference answers ----
  const newS = (l) => ({ lid: l.lid, name: l.name, n: 0, rts: [], cells: 0, works: 0, allBad: 0, fast: 0, allClicked: 0,
    cal: { n: 0, tp: 0, fp: 0, fn: 0, tn: 0 }, ck: {}, pck: {}, first: l.ts, last: l.ts });
  const seenAB = new Set();
  for (const l of [...(data.ab || [])].sort((a, b) => String(a.ts).localeCompare(String(b.ts)))) {
    const k = l.lid + '|' + l.pair_id;
    if (seenAB.has(k)) continue; seenAB.add(k);
    const pr = (data.pairs || {})[l.pair_id];
    if (!pr) continue;
    let s = by.get(l.lid);
    if (!s) { s = newS(l); by.set(l.lid, s); }
    if (String(l.ts) > String(s.last)) s.last = l.ts;
    if (String(l.ts) < String(s.first)) s.first = l.ts;
    const a = (s.abx ||= { n: 0, fast: 0, rts: [], A: 0, B: 0, both_good: 0, both_bad: 0, cal: 0, calAgree: 0, h: {}, pp: {} });
    a.n++; if ((l.flags || []).includes('fast')) a.fast++; if (Number.isFinite(l.response_ms)) a.rts.push(l.response_ms);
    a[l.choice] = (a[l.choice] || 0) + 1;
    if (pr.kind === 'abcal') { a.cal++; if (l.choice === pr.truth) a.calAgree++; continue; }
    const mk = pr.m || 'default';
    const mm = (data.matchups || []).find((x) => x.key === mk);
    const first = mm ? mm.a : (data.compare || [])[0];
    const fl = pr.ck[0] === first ? 'A' : 'B';
    const res = l.choice === 'both_good' ? 'both_good' : l.choice === 'both_bad' ? 'both_bad' : l.choice === fl ? 'win' : 'lose';
    (a.h[mk] ||= { win: 0, lose: 0, both_good: 0, both_bad: 0 })[res]++;
    const q = ((a.pp[mk] ||= {})[pr.prompt] ||= { win: 0, lose: 0, both_good: 0, both_bad: 0 }); q[res]++;
  }
  // ---- chat picks (kind 'chat': best-of-20 -> 4 shown, pick one or none) ----
  const seenChat = new Set();
  for (const l of [...(data.chat || [])].sort((a, b) => String(a.ts).localeCompare(String(b.ts)))) {
    if (seenChat.has(l.job_id)) continue; seenChat.add(l.job_id);
    let s = by.get(l.lid);
    if (!s) { s = newS(l); by.set(l.lid, s); }
    if (String(l.ts) > String(s.last)) s.last = l.ts;
    if (String(l.ts) < String(s.first)) s.first = l.ts;
    const c = (s.chx ||= { n: 0, none: 0, fast: 0, rts: [], pos: [0, 0, 0, 0] });
    c.n++; if (l.none_good) c.none++; else if (Number.isInteger(l.chosen)) c.pos[l.chosen] = (c.pos[l.chosen] || 0) + 1;
    if ((l.flags || []).includes('fast')) c.fast++; if (Number.isFinite(l.response_ms)) c.rts.push(l.response_ms);
  }
  const out = [];
  for (const s of by.values()) {
    const c = s.cal, tot = c.tp + c.fp + c.fn + c.tn;
    const acc = tot ? (c.tp + c.tn) / tot : null;
    const prec = c.tp + c.fp ? c.tp / (c.tp + c.fp) : null;
    const rec = c.tp + c.fn ? c.tp / (c.tp + c.fn) : null;
    const f1 = prec !== null && rec !== null && prec + rec > 0 ? 2 * prec * rec / (prec + rec) : (tot ? 0 : null);
    let kappa = null;
    if (tot) {
      const po = acc;
      const pe = ((c.tp + c.fp) / tot) * ((c.tp + c.fn) / tot) + ((c.fn + c.tn) / tot) * ((c.fp + c.tn) / tot);
      kappa = pe < 1 ? (po - pe) / (1 - pe) : 0;
    }
    const med = median(s.rts);
    const worksRate = s.cells ? s.works / s.cells : 0;
    const flags = [];
    const status = s.lid === 'owner' ? 'owner' : (ov[s.lid] || 'auto');
    const owner = status === 'owner';
    if (status === 'excluded') flags.push('excluded');
    if (!owner) {
      if ((med !== null && med < 2500) || (s.n && s.fast / s.n > 0.3)) flags.push('too_fast');
      if (worksRate > 0.8 || (s.n >= 4 && s.allClicked / s.n > 0.5)) flags.push('clicks_everything');
      if (s.n >= 8 && s.allBad / s.n >= 0.95) flags.push('clicks_nothing');
      if (c.n >= minCal && kappa !== null && kappa < threshold) flags.push('low_agreement');
      if (c.n < minCal) flags.push('few_calibration');
    }
    const a = s.abx || null;
    const abMed = a ? median(a.rts) : null;
    const abAgree = a && a.cal ? a.calAgree / a.cal : null;
    if (a && !owner) {
      if ((abMed !== null && abMed < 700) || (a.n && a.fast / a.n > 0.3)) flags.push('ab_too_fast');
      const dec = a.A + a.B;
      if (dec >= 20 && (a.A / dec > 0.9 || a.B / dec > 0.9)) flags.push('ab_side_bias');
      if (a.cal >= 4 && abAgree < 0.4) flags.push('ab_low_agreement');
    }
    const ch = s.chx || null;
    const chMed = ch ? median(ch.rts) : null;
    if (ch && !owner) {
      if (ch.n >= 3 && ((chMed !== null && chMed < 700) || ch.fast / ch.n > 0.3)) flags.push('chat_too_fast');
      const picked = ch.n - ch.none;
      if (picked >= 10 && Math.max(...ch.pos) / picked > 0.8) flags.push('chat_position_bias');
    }
    const gridOk = c.n >= minCal && kappa !== null && kappa >= threshold;
    const abOk = a && a.cal >= 4 && abAgree >= 0.6;
    const bad = ['too_fast', 'clicks_everything', 'low_agreement', 'ab_too_fast', 'ab_side_bias', 'ab_low_agreement', 'chat_too_fast', 'chat_position_bias'].some((f) => flags.includes(f));
    const autoTrusted = !!((gridOk || abOk) && !bad);
    const trusted = owner || status === 'trusted' || (status === 'auto' && autoTrusted);
    const role = owner ? 'owner' : status === 'excluded' ? 'excluded' : status === 'trusted' ? 'trusted' : autoTrusted ? 'auto-trusted' : 'auto-untrusted';
    out.push({ lid: s.lid, name: s.name, status, role, owner, excluded: status === 'excluded', auto_trusted: autoTrusted, rounds: s.n, avg_ms: s.rts.length ? Math.round(s.rts.reduce((a, b) => a + b, 0) / s.rts.length) : null,
      median_ms: med, works_rate: worksRate, all_bad_rate: s.n ? s.allBad / s.n : 0, fast_rounds: s.fast,
      cal_rounds: c.n, cal_accuracy: acc, cal_precision: prec, cal_recall: rec, cal_f1: f1, agreement: kappa,
      ab_rounds: a ? a.n : 0, ab_median_ms: abMed, ab_fast: a ? a.fast : 0, ab_choices: a ? { A: a.A, B: a.B, both_good: a.both_good, both_bad: a.both_bad } : null,
      ab_cal: a ? a.cal : 0, ab_agreement: abAgree, ab_h2h: a ? a.h : null, ab_per_prompt: a ? a.pp : {},
      chat_votes: ch ? ch.n : 0, chat_none: ch ? ch.none : 0, chat_median_ms: chMed, chat_positions: ch ? ch.pos : null,
      flags, trusted, per_ckpt: s.ck, per_prompt: s.pck, first: s.first, last: s.last });
  }
  const rank = (s) => (s.lid === 'owner' ? 0 : s.owner ? 1 : 2);
  out.sort((a, b) => rank(a) - rank(b) || b.rounds - a.rounds);
  return out;
}

export function headToHead(stats, pick) {
  const ck = {}, pp = {};
  for (const s of stats) {
    if (!pick(s)) continue;
    for (const [c, v] of Object.entries(s.per_ckpt)) { (ck[c] ||= { w: 0, t: 0 }).w += v.w; ck[c].t += v.t; }
    for (const [p, m] of Object.entries(s.per_prompt)) for (const [c, v] of Object.entries(m)) {
      const x = ((pp[p] ||= {})[c] ||= { w: 0, t: 0 }); x.w += v.w; x.t += v.t;
    }
  }
  return { overall: ck, per_prompt: pp };
}

// A/B head-to-head for one matchup key: win = matchup's first model (a) preferred, lose = second (b) preferred.
export function abHeadToHead(stats, pick, key = 'default') {
  const t = { win: 0, lose: 0, both_good: 0, both_bad: 0 }, pp = {};
  for (const s of stats) {
    if (!pick(s) || !s.ab_h2h || !s.ab_h2h[key]) continue;
    for (const k in t) t[k] += s.ab_h2h[key][k];
    for (const [p, v] of Object.entries((s.ab_per_prompt || {})[key] || {})) { const q = (pp[p] ||= { win: 0, lose: 0, both_good: 0, both_bad: 0 }); for (const k in q) q[k] += v[k]; }
  }
  return { overall: t, per_prompt: pp };
}
