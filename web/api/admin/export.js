import { send } from '../_lib/core.js';
import { isAdmin, buildData } from '../_lib/admin.js';
import { compact, overrideMap } from '../_lib/store.js';
import { dedupe, labellerStats } from '../_lib/analysis.js';
import { flywheelHandlers } from '../../lib/init.mjs';

// Per-cell checkpoint provenance: rows saved since 2026-11-10 carry ck/ck_sha/cal themselves ('stored');
// older rows are mapped back through the pool (round id -> ck per cell), reported as 'derived:pool' without touching labels.jsonl.
function prov(l, r, m) {
  if (Array.isArray(l.ck) && l.ck.length === r.ck.length) {
    return { source: 'stored', ck: l.ck, sha: l.ck.map((c) => (l.ck_sha && l.ck_sha[c]) || (Array.isArray(m.sha) ? null : m.sha[c]) || null),
      cal: typeof l.cal === 'boolean' ? l.cal : r.kind === 'cal', mismatch: l.ck.some((c, i) => c !== r.ck[i]) };
  }
  return { source: r.kind === 'owner' ? 'imported' : 'derived:pool', ck: r.ck, sha: r.ck.map((c, i) => (Array.isArray(m.sha) ? m.sha[i] : m.sha[c])), cal: r.kind === 'cal', mismatch: false };
}

export default async function handler(req, res) {
  if (!isAdmin(req)) return send(res, 401, { error: 'login required' });
  const urlObj = new URL(req.url, 'http://x');
  if (urlObj.searchParams.get('kind') === 'flywheel') {
    return flywheelHandlers.handleAdminExportFlywheel(req, res, urlObj);
  }
  const q = urlObj.searchParams;
  const fmt = q.get('fmt') === 'csv' ? 'csv' : 'jsonl';
  const who = q.get('who') || 'all';
  const th = Number.isFinite(Number(q.get('th'))) && q.get('th') !== null ? Number(q.get('th')) : 0.3;
  const { rows } = await compact();
  let d = buildData(rows);
  if (q.get('archive')) {
    if (!d.archive) return send(res, 404, { error: 'no archived votes found' });
    const a = d.archive;
    d = { ...d, ab: a.ab, chat: a.chat, labels: a.labels, pairs: a.pairs, matchups: a.matchups, rounds: d.rounds, meta: d.meta };
  }
  const ov = overrideMap();
  const stats = labellerStats(d, th, 3, ov);
  const S = new Map(stats.map((s) => [s.lid, s]));
  const trusted = new Set(stats.filter((s) => s.trusted).map((s) => s.lid));
  const owners = new Set(stats.filter((s) => s.owner).map((s) => s.lid));
  const keep = (l) => who === 'all' || (who === 'trusted' && trusted.has(l.lid)) || (who === 'owner' && owners.has(l.lid)) || l.lid === who;
  const st = (lid) => S.get(lid) || { status: 'auto', role: 'auto-untrusted' };
  const fileWho = (q.get('archive') ? 'archived_pre_v3_' : '') + who;
  const labels = dedupe(d.labels).filter(keep);
  if (q.get('kind') === 'ab') return exportAB(res, d, keep, st, trusted, owners, fmt, fileWho);
  if (q.get('kind') === 'chat') return exportChat(res, d, keep, st, trusted, fmt, fileWho);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  res.statusCode = 200;
  res.setHeader('Cache-Control', 'no-store');
  if (fmt === 'csv') {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename=pixelai_web_grid_${fileWho}_${stamp}.csv`);
    const esc = (v) => { const s = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    const lines = ['round_id,kind,labeler_id,labeler_name,status,role,trusted,prompt,grid_pos,model,sha256,seed,works,all_bad,response_ms,flags,timestamp,sheet,cal,ck_source,pool_version'];
    for (const l of labels) {
      const r = d.rounds[l.round_id], m = d.meta[l.round_id], w = new Set(l.works);
      const pv = prov(l, r, m);
      pv.ck.forEach((c, i) => {
        const sha = pv.sha[i];
        lines.push([l.round_id, r.kind, l.lid, l.name, st(l.lid).status, st(l.lid).role, trusted.has(l.lid) ? 1 : 0, r.prompt, i, c, sha, m.seeds[i], w.has(i) ? 1 : 0,
          l.all_bad ? 1 : 0, l.response_ms, (l.flags || []).join('|'), l.ts, r.kind === 'owner' ? '' : `/s/${l.round_id}.png`, pv.cal ? 1 : 0, pv.source, l.pool_version || ''].map(esc).join(','));
      });
    }
    return res.end(lines.join('\n') + '\n');
  }
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename=pixelai_web_grid_${who}_${stamp}.jsonl`);
  const out = [];
  for (const l of labels) {
    const r = d.rounds[l.round_id], m = d.meta[l.round_id], w = new Set(l.works), pv = prov(l, r, m);
    out.push(JSON.stringify({
      schema: 'pixelai-grid-v1', id: `${l.round_id}:${l.lid}`, round_id: l.round_id, kind: r.kind, cal: pv.cal,
      ck_source: pv.source, ...(pv.mismatch ? { ck_mismatch: true } : {}), pool_version: l.pool_version || null, device: l.device || null, timestamp: l.ts, prompt: r.prompt,
      labeler: owners.has(l.lid) ? 'human:owner' : `web:${l.name}`, labeler_id: l.lid, labeler_name: l.name, status: st(l.lid).status, role: st(l.lid).role, trusted: trusted.has(l.lid),
      sampling: { temperature: 0.8, top_k: 32 }, n_works: w.size, all_bad: !!l.all_bad, response_ms: l.response_ms, flags: l.flags || [],
      sheet: r.kind === 'owner' ? null : `/s/${l.round_id}.png`, src_id: m.src_id || null,
      samples: pv.ck.map((c, i) => ({ model: c, checkpoint: { id: c, sha256: pv.sha[i] }, seed: m.seeds[i], works: w.has(i), grid_pos: i })),
    }));
  }
  return res.end(out.join('\n') + (out.length ? '\n' : ''));
}

function exportAB(res, d, keep, st, trusted, owners, fmt, who) {
  const seen = new Set();
  const rows = [...d.ab].sort((a, b) => String(a.ts).localeCompare(String(b.ts)))
    .filter((l) => { const k = l.lid + '|' + l.pair_id; if (seen.has(k)) return false; seen.add(k); return keep(l); });
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  res.statusCode = 200;
  res.setHeader('Cache-Control', 'no-store');
  const rec = (l) => {
    const pr = d.pairs[l.pair_id] || {};
    return { schema: 'pixelai-prefs-web-v1', id: `${l.pair_id}:${l.lid}`, pair_id: l.pair_id, kind: pr.kind === 'abcal' ? 'calibration' : 'ab', timestamp: l.ts, prompt: pr.prompt,
      labeler: owners.has(l.lid) ? 'human:owner' : `web:${l.name}`, labeler_id: l.lid, labeler_name: l.name, status: st(l.lid).status, role: st(l.lid).role, trusted: trusted.has(l.lid),
      choice: l.choice, label: { A: 'A_better', B: 'B_better', both_good: 'both_good', both_bad: 'both_bad' }[l.choice],
      matchup: pr.m || null, mode: l.mode || pr.mode || 'single', mode_source: l.mode ? 'stored' : 'derived:pool', ...(pr.pick ? { pick: pr.pick } : {}), winner_model: l.winner_model, calibration_truth: pr.kind === 'abcal' ? pr.truth : null,
      shown: { A_left: { model: l.left_model, sha256: l.left_sha, seed: l.left_seed }, B_right: { model: l.right_model, sha256: l.right_sha, seed: l.right_seed } },
      sampling: { temperature: 0.8, top_k: 32 }, response_ms: l.response_ms, flags: l.flags || [], img: `/ab/${l.pair_id}.png`, pool: l.pool || pr.pool || pr.m || null };
  };
  if (fmt === 'csv') {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename=pixelai_web_ab_${who}_${stamp}.csv`);
    const esc = (v) => { const s = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    const lines = ['pair_id,kind,matchup,labeler_id,labeler_name,status,role,trusted,prompt,choice,winner_model,left_model,left_sha256,left_seed,right_model,right_sha256,right_seed,calibration_truth,response_ms,flags,timestamp,img,mode,pool'];
    for (const l of rows) { const r = rec(l); lines.push([r.pair_id, r.kind, r.matchup, r.labeler_id, r.labeler_name, r.status, r.role, r.trusted ? 1 : 0, r.prompt, r.choice, r.winner_model,
      r.shown.A_left.model, r.shown.A_left.sha256, r.shown.A_left.seed, r.shown.B_right.model, r.shown.B_right.sha256, r.shown.B_right.seed, r.calibration_truth, r.response_ms, r.flags.join('|'), r.timestamp, r.img, r.mode, r.pool].map(esc).join(',')); }
    return res.end(lines.join('\n') + '\n');
  }
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename=pixelai_web_ab_${who}_${stamp}.jsonl`);
  return res.end(rows.map((l) => JSON.stringify(rec(l))).join('\n') + (rows.length ? '\n' : ''));
}

// chat picks: one row per vote (jsonl) or one row per shown sprite (csv)
function exportChat(res, d, keep, st, trusted, fmt, who) {
  const seen = new Set();
  const rows = [...(d.chat || [])].sort((a, b) => String(a.ts).localeCompare(String(b.ts)))
    .filter((l) => { if (seen.has(l.job_id)) return false; seen.add(l.job_id); return keep(l); });
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  res.statusCode = 200;
  res.setHeader('Cache-Control', 'no-store');
  if (fmt === 'csv') {
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename=pixelai_web_chat_${who}_${stamp}.csv`);
    const esc = (v) => { const s = v === null || v === undefined ? '' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
    const lines = ['job_id,kind,labeler_id,labeler_name,status,role,trusted,prompt,model,step,sha256,scorer,shown_index,seed,score,fill,ncolors,chosen,none_good,response_ms,gen_ms,flags,timestamp'];
    for (const l of rows) for (const x of l.shown || []) lines.push([l.job_id, 'chat', l.lid, l.name, st(l.lid).status, st(l.lid).role, trusted.has(l.lid) ? 1 : 0, l.prompt,
      l.model && l.model.key, l.model && l.model.step, l.model && l.model.sha, l.scorer, x.index, x.seed, x.score, x.fill, x.ncolors,
      l.chosen === x.index ? 1 : 0, l.none_good ? 1 : 0, l.response_ms, l.gen_ms, (l.flags || []).join('|'), l.ts].map(esc).join(','));
    return res.end(lines.join('\n') + '\n');
  }
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename=pixelai_web_chat_${who}_${stamp}.jsonl`);
  return res.end(rows.map((l) => JSON.stringify({ schema: 'pixelai-chat-v1', kind: 'chat', job_id: l.job_id, labeler_id: l.lid, labeler_name: l.name,
    status: st(l.lid).status, role: st(l.lid).role, trusted: trusted.has(l.lid), device: l.device, prompt: l.prompt, model: l.model, scorer: l.scorer,
    sampling: l.sampling || null, n_samples: l.n_samples, shown: l.shown, all: l.all, chosen: l.chosen, none_good: l.none_good, chosen_seed: l.chosen_seed,
    mode: l.mode || 'top4', candidate_idx: l.candidate_idx ?? null, chosen_rank: l.chosen_rank ?? (Number.isInteger(l.chosen) ? l.chosen + 1 : null), candidates: l.candidates || null,
    response_ms: l.response_ms, gen_ms: l.gen_ms, flags: l.flags, timestamp: l.ts })).join('\n') + (rows.length ? '\n' : ''));
}
