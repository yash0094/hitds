// HITDS analyst console. Plain JS, no build step, no CDN (works offline on demo day).
//
// Two independent review systems share this page:
//   - Host incidents (from sensor/hitds_host_sensor.py): the PRIMARY review queue (left) and detail (center).
//     Click an incident -> see what/risk/do and the raw measurement, then Confirm threat or mark it a False alarm.
//   - Flow alerts (from "Start traffic" replay or "Upload flows"): a compact queue + detail widget in the
//     right-hand column, with the full evidence/SHAP/verdict workflow, unchanged in substance from before.
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => (x == null ? "n/a" : (100 * x).toFixed(d) + "%");
async function api(url, opts = {}) {
  opts.headers = { ...(opts.headers || {}), "X-Requested-With": "hitds" };
  const r = await fetch(url, opts);
  if (r.status === 401) { location.href = "/login"; throw new Error("login required"); }
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const state = { flowCurrent: null, openedAt: 0, verdict: null, revealed: false, flowQueue: [], streaming: false,
                hostIncidents: [], hostCurrent: null };
const STATES = ["safe", "intrusion", "exploit", "impact"];
const SEV_CLASS = { CRITICAL: "bad", HIGH: "bad", MEDIUM: "warn", LOW: "ok" };

function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), 4000); }
function stack(belief) {
  return `<div class="stack">${STATES.map((s) => `<div class="s-${s}" style="width:${(100 * (belief[s] || 0)).toFixed(1)}%" title="${s} ${pct(belief[s], 1)}"></div>`).join("")}</div>`;
}

// ============================================================== PRIMARY: host-sensor incidents (left + center)
async function loadHostQueue() {
  const [st, inc] = await Promise.all([api("/api/host/status"), api("/api/host/incidents")]);
  const h = st.hosts[0];
  $("#host-line").textContent = h ? `${h.host} · CPU ${h.cpu.toFixed(0)}% · mem ${h.mem.toFixed(0)}% · ${h.conns} sockets · ${h.age_s < 30 ? "live" : "last seen " + Math.round(h.age_s) + "s ago"}` : "no sensor connected";
  state.hostIncidents = inc.incidents;
  $("#hqueue").innerHTML = inc.incidents.length ? inc.incidents.map((i) => `
    <div class="qitem ${i.severity === "HIGH" || i.severity === "CRITICAL" ? "ESCALATE" : "REVIEW"} ${state.hostCurrent === i.id ? "active" : ""}" data-id="${i.id}">
      <div class="row1"><span>#${i.id} · ${esc(i.host)}</span><span class="pill ${SEV_CLASS[i.severity] || "warn"}">${i.status === "open" ? i.severity : i.status}</span></div>
      <div class="row2">${esc(i.plain.title)} ${i.count > 1 ? "· ×" + i.count : ""}</div>
    </div>`).join("") : `<p class="empty">${h ? "No incidents. This host looks clean." : 'Waiting for a sensor. Run <span class="mono">sensor/hitds_host_sensor.py</span> on a machine to see incidents here.'}</p>`;
  document.querySelectorAll("#hqueue .qitem").forEach((el) => (el.onclick = () => openHostAlert(+el.dataset.id)));
  if (state.hostCurrent) {
    const still = inc.incidents.find((i) => i.id === state.hostCurrent);
    if (still) renderHostDetail(still); // keep an open detail in sync with status/count changes, without re-navigating
  }
}

function renderHostDetail(i) {
  const p = i.plain;
  const act = i.status === "open"
    ? `<div class="decide"><button class="btn" id="btn-confirm">Confirm threat</button><button class="btn ghost" id="btn-falsealarm">False alarm</button></div>`
    : `<p><span class="pill ${i.status === "CONFIRMED" ? "bad" : "ok"}">${esc(i.status)}</span> decided by <b>${esc(i.decided_by)}</b>${i.note ? " — " + esc(i.note) : ""}</p>`;
  $("#detail").innerHTML = `
    <div class="dhead"><h3>${esc(p.title)}</h3><span class="pill ${SEV_CLASS[i.severity] || "warn"}">${esc(i.severity)}</span>
      <span class="mono">${esc(i.host)}</span>
      <span class="muted mono">seen ${i.count}× · first ${new Date(i.first_ts * 1000).toLocaleTimeString()} · last ${new Date(i.last_ts * 1000).toLocaleTimeString()}</span></div>
    <p class="reason">${esc(i.details)}</p>
    <div class="verdictbox">
      <h2>What this means</h2>
      <p>${esc(p.what)}</p>
      <h2 style="margin-top:10px">Risk</h2>
      <p>${esc(p.risk)}</p>
      <h2 style="margin-top:10px">What to do</h2>
      <ol>${p.do.map((d) => `<li>${esc(d)}</li>`).join("")}</ol>
      <p class="muted small">${esc(p.caveat)}</p>
      <h2 style="margin-top:14px">Your decision</h2>
      ${act}
    </div>`;
  if (i.status === "open") {
    $("#btn-confirm").onclick = () => hostTriage(i.id, "CONFIRMED");
    $("#btn-falsealarm").onclick = () => hostTriage(i.id, "FALSE_POSITIVE");
  }
}

function openHostAlert(id) {
  const i = state.hostIncidents.find((x) => x.id === id);
  if (!i) return;
  state.hostCurrent = id;
  renderHostDetail(i);
  document.querySelectorAll("#hqueue .qitem").forEach((el) => el.classList.toggle("active", +el.dataset.id === id));
}

async function hostTriage(id, decision) {
  try {
    await post(`/api/host/incident/${id}/triage`, { decision });
    toast("Decision recorded in the audit log");
    await loadHostQueue();
    if (state.hostCurrent === id) openHostAlert(id);
    loadAudit();
  } catch (e) { toast(e.message); }
}

// ============================================================== SECONDARY: flow alerts (right-hand widget)
async function loadFlowQueue() {
  const q = await api("/api/queue");
  state.flowQueue = q.alerts;
  $("#fqueue").innerHTML = q.alerts.length ? q.alerts.map((a) => `
    <div class="qitem ${a.route} ${a.status} ${state.flowCurrent === a.id ? "active" : ""}" data-id="${a.id}">
      <div class="row1"><span>#${a.id} · ${esc(a.host)}</span><span class="pill ${a.status === "second_review" ? "second_review" : a.route}">${a.status === "second_review" ? "2ND REVIEW" : a.route}</span></div>
      <div class="row2">${esc(a.src_ip)} → ${esc(a.dst_ip)} · priority ${Number(a.final_priority).toFixed(2)}${a.source && a.source !== "replay" ? " · " + esc(a.source) : ""}</div>
    </div>`).join("") : `<p class="empty">Queue empty.${state.streaming ? " Waiting for uncertain events…" : " Start the traffic replay or upload flows."}</p>`;
  document.querySelectorAll("#fqueue .qitem").forEach((el) => (el.onclick = () => openFlowAlert(+el.dataset.id)));
  $("#hosts").innerHTML = q.hosts.map((h) => `
    <div class="hostrow"><b title="${esc(h.host)}">${esc(h.host).slice(-8)}</b>${stack(h.belief)}<span class="pill ${h.sprt === "compromised" ? "bad" : h.sprt === "safe" ? "ok" : "warn"}" title="${h.sprt === "compromised" ? "detected by the sequential test · " : ""}log R ${h.llr} · most likely ${h.stage}">${h.sprt === "compromised" ? "⚑ " + esc(h.hypothesis) : h.stage}</span></div>`).join("")
    + `<div class="legend">${STATES.map((s) => `<span><i class="s-${s}"></i>${s}</span>`).join("")}</div>`;
}

async function openFlowAlert(id) {
  const d = await api(`/api/alert/${id}`);
  state.flowCurrent = id; state.openedAt = Date.now(); state.verdict = null; state.revealed = !$("#blind").checked;
  const a = d.alert;
  const maxAbs = Math.max(...d.attribution.map((x) => Math.abs(x.contribution)), 1e-9);
  const topClass = (p) => Object.entries(p).sort((x, y) => y[1] - x[1])[0];
  const prior = d.prior_verdicts.length ? `<div class="sigbox" style="border-color:var(--info);background:transparent">
      <b>Second review requested.</b> ${d.prior_verdicts.map((v) => `${esc(v.analyst)} said <b>${esc(v.corrected_label || v.verdict)}</b> (confidence ${v.analyst_confidence})`).join("; ")}. Your verdict is final.</div>` : "";
  $("#fdetail").innerHTML = `
    <div class="dhead"><h3>Alert #${a.id}</h3><span class="pill ${a.route}">${a.route}</span>
      <span class="mono">${esc(a.src_ip)} → ${esc(a.dst_ip)} (${esc(a.host)})</span>
      <span class="muted mono">${esc(a.source)} · model v${a.model_version} · ${new Date(a.ts * 1000).toLocaleTimeString()}</span></div>
    <p class="reason">Why you are seeing this: ${esc(a.reason)}</p>
    ${prior}
    ${a.signature ? `<div class="sigbox"><b>Signature hit:</b> ${esc(a.signature)}</div>` : ""}

    <div class="cols">
      <div><h2>Raw flow evidence</h2>
        <table class="tbl">${Object.entries(d.raw_evidence).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${Number(v).toLocaleString(undefined, { maximumFractionDigits: 3 })}</td></tr>`).join("")}</table></div>
      <div><h2>What drove the score <small>${esc(d.attribution[0]?.method || "")}</small></h2>
        <div class="bars">${d.attribution.map((x) => `<div class="bar"><span class="name" title="${esc(x.feature)}">${esc(x.feature)}</span>
          <div class="track"><div class="fill ${x.contribution < 0 ? "neg" : ""}" style="left:0;width:${(100 * Math.abs(x.contribution) / maxAbs).toFixed(1)}%"></div></div>
          <span class="mono">${x.contribution > 0 ? "+" : ""}${x.contribution.toFixed(3)}</span></div>`).join("")}</div>
        <p class="muted small">Blue pushes toward the model's class, green pushes away from it.</p>
        <h2 style="margin-top:10px">Statistical layer (SMAD) <small>${d.smad.flag ? "flagged" : "not flagged"} · ${pct(d.smad.score)} of ANOVA features outside 3σ</small></h2>
        <div class="mono small">${d.smad.features.map((f) => `${esc(f.feature)} ${f.sigma > 0 ? "+" : ""}${f.sigma}σ`).join(" · ") || "all selected features inside the benign 3σ band"}</div></div>
    </div>

    <div class="cols sec">
      <div><h2>Similar past incidents</h2>
        <table class="tbl"><tr><th>label</th><th>source</th><th>distance</th></tr>${d.similar.map((s) => `<tr><td>${esc(s.label)}</td><td>${esc(s.source)}</td><td class="num">${s.distance}</td></tr>`).join("")}</table></div>
      <div><h2>Host context <small>attack-progression HMM</small></h2>
        ${stack(d.host.belief)}
        <div class="legend">${STATES.map((s) => `<span><i class="s-${s}"></i>${s} ${pct(d.host.belief[s], 1)}</span>`).join("")}</div>
        <p class="muted small">Most likely stage <b>${esc(d.host.stage)}</b> · log R = ${d.host.llr} · sequential test: <b>${esc(d.host.sprt)}</b>${d.host.hypothesis ? " (" + esc(d.host.hypothesis) + ")" : ""}</p></div>
    </div>

    <div class="verdictbox">
      <div id="modelview" class="${state.revealed ? "" : "hidden"}">
        <h2>Model verdict</h2>
        <p><b style="font-size:18px">${esc(a.pred)}</b> · confidence ${pct(a.confidence, 1)} · entropy ${a.entropy.toFixed(3)} · anomaly ${a.anomaly.toFixed(3)}</p>
        <div class="bars">${Object.entries(a.proba).sort((x, y) => y[1] - x[1]).slice(0, 5).map(([c, p]) => `<div class="bar"><span class="name">${esc(c)}</span><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div><span class="mono">${pct(p, 1)}</span></div>`).join("")}</div>
        <table class="tbl" style="margin-top:8px"><tr><th>ensemble member</th><th>top class</th><th>prob</th></tr>${Object.entries(d.members).map(([m, p]) => { const t = topClass(p); return `<tr><td>${esc(m)}</td><td>${esc(t[0])}</td><td class="num">${pct(t[1], 1)}</td></tr>`; }).join("")}</table>
      </div>
      <button id="btn-reveal" class="btn reveal ${state.revealed ? "hidden" : ""}">Reveal model verdict (R)</button>

      <h2 style="margin-top:14px">Your decision</h2>
      <div class="decide">
        <button class="btn" data-v="TP">Malicious (T)</button>
        <button class="btn" data-v="FP">Benign / false alarm (F)</button>
        <button class="btn" data-v="UNKNOWN">Need more info (U)</button>
      </div>
      <div class="form-row">
        <label>True class <select id="label">${d.classes.map((c) => `<option ${c === a.pred ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>
        <label>How sure are you? <select id="conf"><option value="1.0">certain</option><option value="0.7" selected>fairly sure</option><option value="0.4">unsure - ask a second reviewer</option></select></label>
        <label id="actwrap" class="${d.suggested_action ? "" : "hidden"}"><input type="checkbox" id="act" checked> authorise <b class="mono">${esc(d.suggested_action || "")}</b> on ${esc(a.src_ip)}</label>
      </div>
      <button id="btn-submit" class="btn primary" disabled>Submit verdict (Enter)</button>
    </div>`;
  $("#btn-reveal").onclick = reveal;
  document.querySelectorAll("#fdetail .decide button").forEach((b) => (b.onclick = () => choose(b.dataset.v)));
  $("#btn-submit").onclick = submitFlowVerdict;
  $("#fdetail").dataset.action = d.suggested_action || "";
  loadFlowQueue();
}

function reveal() { state.revealed = true; $("#modelview")?.classList.remove("hidden"); $("#btn-reveal")?.classList.add("hidden"); }

function choose(v) {
  state.verdict = v;
  document.querySelectorAll("#fdetail .decide button").forEach((b) => (b.className = "btn" + (b.dataset.v === v ? " sel-" + v : "")));
  if (v === "FP") $("#label").value = "BENIGN";
  $("#actwrap").style.opacity = v === "TP" ? 1 : 0.4;
  $("#btn-submit").disabled = false;
}

async function submitFlowVerdict() {
  if (!state.flowCurrent || !state.verdict) return;
  const body = { verdict: state.verdict, label: state.verdict === "UNKNOWN" ? null : $("#label").value,
    confidence: +$("#conf").value, seconds: (Date.now() - state.openedAt) / 1000, revealed: state.revealed,
    action: state.verdict === "TP" && $("#act").checked ? $("#fdetail").dataset.action : null };
  try {
    const r = await post(`/api/alert/${state.flowCurrent}/verdict`, body);
    const msg = r.status === "second_review" ? "sent for a second review" : r.status === "needs_info" ? "parked as needs-info" : "saved to the audit log";
    toast(`Alert #${state.flowCurrent}: ${state.verdict} ${msg}${r.retrain_started ? " - retraining started" : ""}`);
    nextFlowAlert();
  } catch (e) { toast("Error: " + e.message); }
}

function nextFlowAlert() {
  state.flowCurrent = null;
  loadFlowQueue().then(() => { if (state.flowQueue.length) openFlowAlert(state.flowQueue[0].id); else $("#fdetail").innerHTML = `<p class="empty">Queue clear.</p>`; });
}

// ============================================================== shared side panels
async function loadMetrics() {
  const m = await api("/api/metrics");
  const auto = m.total_events ? (m.by_route.AUTO || 0) / m.total_events : 0;
  $("#mv").textContent = "v" + m.model_version;
  $("#pol").textContent = m.policy;
  $("#k-events").textContent = m.total_events.toLocaleString();
  $("#k-auto").textContent = pct(auto, 1);
  $("#k-human").textContent = pct(m.human_share, 1);
  $("#k-open").textContent = m.open + (m.second_review ? ` (${m.second_review} 2nd)` : "");
  $("#k-verdicts").textContent = m.verdicts;
  $("#k-ttd").textContent = m.mean_seconds_to_decide + "s";
  $("#k-agree").textContent = `${pct(m.agreement_blind)} / ${pct(m.agreement_after_reveal)}`;
  $("#k-kappa").textContent = m.inter_annotator.cohen_kappa ?? (m.inter_annotator.double_reviewed ? "1 pair" : "n/a");
  $("#k-auto-err").textContent = pct(m.auto_error_rate, 2);
  $("#k-lat").textContent = m.latency_ms ?? "n/a";
  $("#fb-count").textContent = m.pending_feedback; $("#fb-every").textContent = m.retrain_every;
  $("#fb-bar").style.width = Math.min(100, (100 * m.pending_feedback) / m.retrain_every) + "%";
  $("#retraining").classList.toggle("hidden", !m.retraining);
  $("#retrain-msg").textContent = m.last_retrain?.error ? "Last retrain failed: " + m.last_retrain.error : "";
  state.streaming = m.streaming;
  $("#btn-stream").textContent = m.streaming ? "Pause traffic" : "Start traffic";
  $("#btn-stream").classList.toggle("on", m.streaming);
  $("#history tbody").innerHTML = m.model_history.map((h) => { const g = JSON.parse(h.metrics_json);
    return `<tr><td>v${h.version}</td><td class="num">${h.n_feedback}</td><td class="num">${g.val_macro_f1_old}→${g.val_macro_f1_new}</td>
      <td class="num">${pct(g.human_share_old)}→${pct(g.human_share_new)}</td><td><span class="pill ${h.promoted ? "ok" : "bad"}">${h.promoted ? "live" : "rejected"}</span></td></tr>`; }).join("")
    || `<tr><td colspan="5" class="muted">v1 trained on benchmark data. Retrains appear here.</td></tr>`;
}

async function loadFeed() {
  const f = await api("/api/feed");
  $("#feed").innerHTML = f.map((a) => `<div class="feedrow"><span>#${a.id}</span><span>${esc(a.host).slice(-8)}</span><span>${esc(a.pred)} ${pct(a.confidence)}</span><span class="pill ${a.route}">${a.route}</span></div>`).join("");
}

async function loadAudit() {
  const a = await api("/api/audit");
  const c = $("#chain");
  c.className = "pill " + (a.verify.ok ? "ok" : "bad");
  c.textContent = a.verify.ok ? `hash chain verified · ${a.verify.checked} rows` : `CHAIN BROKEN at #${a.verify.broken_at_seq}`;
  $("#audit").innerHTML = a.tail.map((r) => `<div class="auditrow" title="${esc(r.payload)}">#${r.seq} ${new Date(r.ts * 1000).toLocaleTimeString()} <b>${esc(r.event)}</b> ${r.alert_id ? "alert " + r.alert_id : ""} · ${esc(r.actor)} · ${esc(r.hash.slice(0, 10))}</div>`).join("");
}

// ------------------------------------------------------------------ wiring
$("#btn-stream").onclick = async () => { const r = await post("/api/stream", { running: !state.streaming, rate: +$("#rate").value }); state.streaming = r.running; loadMetrics(); };
$("#rate").onchange = () => post("/api/stream", { rate: +$("#rate").value });
$("#btn-inject").onclick = async () => { await post("/api/inject", { n: 30 }); refresh(); };
$("#btn-retrain").onclick = async () => { try { await post("/api/retrain", {}); toast("Retraining started"); } catch (e) { toast(e.message); } };
$("#upload").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    toast(`Scored ${r.ingested} flows from ${f.name} · feature coverage ${pct(r.coverage)}${r.missing_model_features.length ? " · missing " + r.missing_model_features.length : ""}`);
    refresh();
  } catch (err) { toast("Upload rejected: " + err.message); }
  e.target.value = "";
};
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (!state.flowCurrent) return; // shortcuts only act on an open flow alert; host incidents are click-driven
  const k = e.key.toLowerCase();
  if (k === "t") choose("TP"); if (k === "f") choose("FP"); if (k === "u") choose("UNKNOWN");
  if (k === "r") reveal();
  if (e.key === "Enter") submitFlowVerdict();
});

function refresh() { loadFlowQueue().catch(() => {}); loadMetrics().catch(() => {}); loadFeed().catch(() => {}); }
refresh(); loadAudit().catch(() => {}); loadHostQueue().catch(() => {});
setInterval(refresh, 1500);
setInterval(() => loadAudit().catch(() => {}), 4000);
setInterval(() => loadHostQueue().catch(() => {}), 3000);
