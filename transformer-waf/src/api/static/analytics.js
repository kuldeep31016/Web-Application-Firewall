/* Analytics & Insights page. All figures come from /analytics/* (aggregated server-side from detections.db). */
(function () {
  const { api, esc } = WAF;
  const $ = id => document.getElementById(id);
  const SVGNS = 'xmlns="http://www.w3.org/2000/svg"';

  /* ---------------- formatting ---------------- */
  const nf = new Intl.NumberFormat();
  const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
  const int = x => x == null ? "—" : (Math.abs(x) >= 1e6 ? compact.format(x) : nf.format(x));
  const dec = (x, d = 2) => x == null || isNaN(x) ? "—" : Number(x).toFixed(d);
  const pctOf = (x, d = 1) => x == null || isNaN(x) ? "—" : (x * 100).toFixed(d) + "%";
  const dFmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" });
  const dShort = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" });
  const hFmt = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" });
  const dtFmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  const at = s => new Date(s * 1000);
  const SOURCE_LABELS = { direct: "Direct (/detect)", batch: "Batch (/detect/batch)", replay: "Replay (/replay)" };
  const GROUP_LABELS = { path: "Endpoint", client: "Client (hashed IP)", method: "Method", source: "Source" };

  function keyLabel(group, k) {
    if (k == null || k === "") return group === "client" ? "not recorded" : "—";
    if (group === "source") return SOURCE_LABELS[k] || k;
    if (group === "client") return k.slice(0, 12) + "…";
    return k;
  }

  /* ---------------- state ---------------- */
  const state = {
    group: "path", sort: "anomalies", order: "desc", search: "", page: 0, pageSize: 10,
    overview: null, seq: 0, bseq: 0,
  };

  /* ---------------- filters → query ---------------- */
  function localMidnight(d) { const x = new Date(d); x.setHours(0, 0, 0, 0); return x; }
  function parseDateInput(v) { if (!v) return null; const [y, m, d] = v.split("-").map(Number); return new Date(y, m - 1, d); }

  function rangeFromUI() {
    const kind = $("fRange").value;
    const now = new Date();
    const nowS = now.getTime() / 1000;
    const days = { "7d": 7, "30d": 30, "90d": 90, "180d": 182 };
    if (kind === "all") return { start: null, end: null };
    if (kind === "today") return { start: localMidnight(now).getTime() / 1000, end: nowS };
    if (kind === "year") return { start: new Date(now.getFullYear(), 0, 1).getTime() / 1000, end: nowS };
    if (days[kind]) return { start: nowS - days[kind] * 86400, end: nowS };
    const from = parseDateInput($("fFrom").value), to = parseDateInput($("fTo").value);
    if (!from || !to) throw new Error("Choose both a start and an end date for the custom range.");
    if (to < from) throw new Error("The end date must be on or after the start date.");
    const endDay = new Date(to); endDay.setDate(endDay.getDate() + 1); // inclusive of the whole end day
    return { start: from.getTime() / 1000, end: endDay.getTime() / 1000 };
  }

  function baseParams() {
    const r = rangeFromUI();
    const p = new URLSearchParams();
    if (r.start != null) { p.set("start", r.start.toFixed(3)); p.set("end", r.end.toFixed(3)); }
    p.set("tz_offset", String(-new Date().getTimezoneOffset()));
    [["verdict", "fVerdict"], ["method", "fMethod"], ["source", "fSource"]].forEach(([k, id]) => { if ($(id).value) p.set(k, $(id).value); });
    const path = $("fPath").value.trim(); if (path) p.set("path", path);
    return p;
  }

  /* ---------------- skeletons / states ---------------- */
  const skel = (w, h = 12) => `<span class="skel" style="width:${w};height:${h}px"></span>`;
  function showSkeletons() {
    $("kpis").innerHTML = Array.from({ length: 6 }, () => `<div class="an-kpi">${skel("60%")}${skel("45%", 26)}${skel("80%")}</div>`).join("");
    ["trendChart", "histChart"].forEach(id => $(id).innerHTML = `<div class="skel an-skel-chart"></div>`);
    ["verdictChart", "topEndpoints", "mix", "insights"].forEach(id => $(id).innerHTML = [90, 75, 82, 60].map(w => `<div style="margin:8px 0">${skel(w + "%", 14)}</div>`).join(""));
    ["scoreStats", "thresholds", "urls", "breakdown"].forEach(id => $(id).innerHTML = `<div class="card-body">${[95, 88, 92].map(w => `<div style="margin:10px 0">${skel(w + "%", 14)}</div>`).join("")}</div>`);
  }
  const empty = (title, hint) => `<div class="empty"><b>${esc(title)}</b>${hint ? `<div class="small" style="margin-top:4px">${esc(hint)}</div>` : ""}</div>`;
  const NO_DATA = () => empty("No analytics data available", "There isn't any data for the selected period and filters. Try a wider date range or clear filters.");

  /* ---------------- KPI cards ---------------- */
  const ICONS = {
    requests: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    anomalies: '<path d="M12 3 4 6v6c0 5 3.4 8.4 8 9 4.6-.6 8-4 8-9V6l-8-3z"/><path d="M12 8v4M12 16h.01"/>',
    rate: '<path d="M19 5 5 19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    score: '<path d="M12 14l4-4"/><path d="M3.3 17a9 9 0 1 1 17.4 0"/>',
    endpoints: '<path d="M4 6h16M4 12h10M4 18h6"/>',
    clients: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M21.5 20a6.5 6.5 0 0 0-4-6"/>',
  };
  const icon = k => `<svg ${SVGNS} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[k]}</svg>`;

  /* tone: which direction is bad. "up" = an increase is a concern (anomalies), null = neutral volume. */
  function changeLine(ch, tone, hasPrev, isRate) {
    if (!hasPrev) return `<span class="an-ch neutral">All recorded data · no comparison</span>`;
    if (!ch) return `<span class="an-ch neutral">—</span>`;
    let v, text;
    if (isRate) {
      if (ch.pp == null) return `<span class="an-ch neutral">No previous data to compare</span>`;
      v = ch.pp; text = `${v > 0 ? "+" : ""}${v.toFixed(1)} pp`;
    } else if (ch.pct == null) {
      if (ch.previous === 0 && ch.abs > 0) return `<span class="an-ch neutral">New · none in previous period</span>`;
      if (ch.previous == null) return `<span class="an-ch neutral">No previous data to compare</span>`;
      return `<span class="an-ch neutral">No change</span>`;
    } else {
      v = ch.pct; text = `${v > 0 ? "+" : ""}${Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)}%`;
    }
    const arrow = v > 0 ? "↑" : v < 0 ? "↓" : "→";
    const cls = v === 0 || !tone ? "neutral" : (v > 0 ? "bad" : "good");
    return `<span class="an-ch ${cls}"><b>${arrow} ${esc(text)}</b> vs previous period</span>`;
  }

  function renderKpis(o) {
    const s = o.summary, c = o.change, hasPrev = !!o.previous;
    const cards = [
      ["requests", "Requests scored", int(s.requests), "Requests evaluated by the WAF model", c.requests, null],
      ["anomalies", "Flagged anomalous", int(s.anomalies), "Score above the decision threshold", c.anomalies, "up"],
      ["rate", "Anomaly rate", pctOf(s.anomaly_rate), "Flagged ÷ scored", c.anomaly_rate, "up", true],
      ["score", "Mean score", dec(s.mean_score), s.min_score == null ? "No scores in range" : `Range ${dec(s.min_score)} – ${dec(s.max_score)}`, c.mean_score, "up"],
      ["endpoints", "Endpoints hit", int(s.endpoints), "Distinct request paths", c.endpoints, null],
      ["clients", "Unique clients", int(s.clients), "Distinct hashed client IPs", c.clients, null],
    ];
    $("kpis").innerHTML = cards.map(([k, label, value, desc, ch, tone, isRate]) => `
      <div class="an-kpi">
        <div class="an-kpi-h"><span>${esc(label)}</span><i class="an-ico ${k === "anomalies" || k === "rate" ? "bad" : ""}">${icon(k)}</i></div>
        <div class="an-kpi-v">${esc(value)}</div>
        <div class="an-kpi-d">${esc(desc)}</div>
        ${changeLine(ch, tone, hasPrev, isRate)}
      </div>`).join("");
  }

  /* ---------------- tooltip ---------------- */
  function attachTooltip(host, svg, n, x0, bandW, content) {
    let tip = host.querySelector(".an-tip");
    if (!tip) { tip = document.createElement("div"); tip.className = "an-tip"; tip.setAttribute("role", "status"); host.appendChild(tip); }
    const hl = svg.querySelector(".an-hl");
    const hide = () => { tip.style.display = "none"; if (hl) hl.setAttribute("opacity", "0"); };
    svg.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect();
      const x = ev.clientX - r.left;
      const i = Math.floor((x - x0) / bandW);
      if (i < 0 || i >= n) return hide();
      tip.innerHTML = content(i);
      tip.style.display = "block";
      if (hl) { hl.setAttribute("x", x0 + i * bandW); hl.setAttribute("opacity", "1"); }
      const hostR = host.getBoundingClientRect();
      const left = Math.min(Math.max(ev.clientX - hostR.left + 14, 0), hostR.width - tip.offsetWidth);
      const top = Math.max(ev.clientY - hostR.top - tip.offsetHeight - 12, 0);
      tip.style.left = left + "px"; tip.style.top = top + "px";
    });
    svg.addEventListener("pointerleave", hide);
  }

  function niceMax(v) {
    if (v <= 0) return 1;
    const e = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * e) return m * e;
    return 10 * e;
  }

  /* Rect with only the top corners rounded (data end), anchored on the baseline. */
  function topRounded(x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h));
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  /* Stacked normal/anomalous columns. items: [{normal, anomalies}], opts: {label(i), tick(i), tip(i)} */
  function stackedColumns(host, items, opts) {
    const W = Math.max(host.clientWidth, 260);
    const H = host.classList.contains("short") ? 220 : (W < 520 ? 220 : 280);
    const padL = 40, padR = 8, padT = 10, padB = 28;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const n = items.length, band = plotW / n;
    const maxV = niceMax(Math.max(...items.map(d => d.normal + d.anomalies), 1));
    const y = v => padT + plotH - v / maxV * plotH;
    const barW = Math.max(1, Math.min(band * 0.72, 48));
    let out = `<svg ${SVGNS} width="${W}" height="${H}" class="an-svg" role="img" aria-label="${esc(opts.aria)}">`;
    for (let t = 0; t <= 4; t++) {
      const v = maxV * t / 4, yy = y(v);
      out += `<line class="grid-line" x1="${padL}" x2="${W - padR}" y1="${yy}" y2="${yy}"/>`;
      out += `<text x="${padL - 6}" y="${yy + 4}" text-anchor="end">${int(Math.round(v * 100) / 100)}</text>`;
    }
    out += `<rect class="an-hl" x="${padL}" y="${padT}" width="${band}" height="${plotH}" opacity="0"/>`;
    const every = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(plotW / (opts.tickW || 64)))));
    items.forEach((d, i) => {
      const x = padL + i * band + (band - barW) / 2;
      const gap = barW > 3 ? 2 : 0;
      const hN = d.normal / maxV * plotH, hA = d.anomalies / maxV * plotH;
      const r = barW > 6 ? 4 : 0;
      if (d.normal > 0) out += `<path class="an-bar normal" d="${d.anomalies > 0 ? `M${x},${y(0)}h${barW}v${-hN}h${-barW}Z` : topRounded(x, y(d.normal), barW, hN, r)}"/>`;
      if (d.anomalies > 0) {
        const top = y(d.normal + d.anomalies), h = Math.max(hA - (d.normal > 0 ? gap : 0), 1);
        out += `<path class="an-bar anom" d="${topRounded(x, top, barW, h, r)}"/>`;
      }
      if (i % every === 0) out += `<text x="${padL + i * band + band / 2}" y="${H - 8}" text-anchor="middle">${esc(opts.tick(i))}</text>`;
    });
    out += `<line class="axis" x1="${padL}" x2="${W - padR}" y1="${y(0)}" y2="${y(0)}"/></svg>`;
    host.innerHTML = out;
    attachTooltip(host, host.querySelector("svg"), n, padL, band, opts.tip);
  }

  const tipRows = rows => rows.map(([k, v, cls]) => `<div class="an-tip-r">${cls ? `<i class="an-sw ${cls}"></i>` : ""}<span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");

  function renderTrend(o) {
    const ts = o.timeseries, host = $("trendChart");
    const names = { hour: "per hour", day: "per day", week: "per week (Mon–Sun)" };
    $("bucketLabel").textContent = ts.bucket ? names[ts.bucket] + " · your local time" : "";
    if (!ts.points.length || o.summary.requests === 0) { host.innerHTML = NO_DATA(); return; }
    const pts = ts.points;
    const tick = i => { const d = at(pts[i].start); return ts.bucket === "hour" ? hFmt.format(d) : dShort.format(d); };
    const title = i => {
      const d = at(pts[i].start);
      if (ts.bucket === "hour") return dtFmt.format(d);
      if (ts.bucket === "week") return "Week of " + dFmt.format(d);
      return dFmt.format(d);
    };
    stackedColumns(host, pts, {
      aria: `Requests ${names[ts.bucket]}, split into normal and anomalous`, tick, tickW: ts.bucket === "hour" ? 52 : 64,
      tip: i => { const p = pts[i]; return `<div class="an-tip-t">${esc(title(i))}</div>` + tipRows([
        ["Total", int(p.requests)], ["Normal", int(p.normal), "normal"], ["Anomalous", int(p.anomalies), "anom"],
        ["Anomaly rate", pctOf(p.anomaly_rate)], ["Mean score", dec(p.mean_score)]]); },
    });
  }

  function renderHistogram(o) {
    const h = o.distributions.score_histogram, host = $("histChart");
    $("histLabel").textContent = h.step ? `bins of ${h.step}` : "";
    if (!h.bins.length || o.summary.requests === 0) { host.innerHTML = NO_DATA(); return; }
    const b = h.bins, digits = h.step < 1 ? 1 : 0;
    stackedColumns(host, b, {
      aria: "Distribution of anomaly scores", tickW: 40,
      tick: i => Number(b[i].from).toFixed(digits),
      tip: i => `<div class="an-tip-t">Score ${dec(b[i].from)} – ${dec(b[i].to)}</div>` + tipRows([
        ["Requests", int(b[i].requests)], ["Normal", int(b[i].normal), "normal"], ["Anomalous", int(b[i].anomalies), "anom"]]),
    });
  }

  function renderVerdict(o) {
    const s = o.summary, host = $("verdictChart");
    if (!s.requests) { host.innerHTML = NO_DATA(); return; }
    const R = 70, r = 48, C = 90;
    const segs = [["Normal", s.normal, "normal"], ["Anomalous", s.anomalies, "anom"]].filter(x => x[1] > 0);
    let a0 = -Math.PI / 2, paths = "";
    const gap = segs.length > 1 ? 0.03 : 0;
    const pt = (rad, a) => [C + rad * Math.cos(a), C + rad * Math.sin(a)];
    segs.forEach(([label, v, cls]) => {
      const sweep = v / s.requests * Math.PI * 2;
      if (segs.length === 1) {
        paths += `<circle class="an-ring ${cls}" cx="${C}" cy="${C}" r="${(R + r) / 2}" stroke-width="${R - r}" fill="none"><title>${label}: ${int(v)}</title></circle>`;
      } else {
        const a1 = a0 + gap, a2 = a0 + sweep - gap, large = a2 - a1 > Math.PI ? 1 : 0;
        const [x1, y1] = pt(R, a1), [x2, y2] = pt(R, a2), [x3, y3] = pt(r, a2), [x4, y4] = pt(r, a1);
        paths += `<path class="an-bar ${cls}" d="M${x1},${y1}A${R},${R} 0 ${large} 1 ${x2},${y2}L${x3},${y3}A${r},${r} 0 ${large} 0 ${x4},${y4}Z"><title>${label}: ${int(v)} (${pctOf(v / s.requests)})</title></path>`;
      }
      a0 += sweep;
    });
    host.innerHTML = `<div class="an-donut">
      <svg ${SVGNS} viewBox="0 0 180 180" width="180" height="180" role="img" aria-label="Verdict split: ${pctOf(s.anomaly_rate)} anomalous">${paths}
        <text x="${C}" y="${C - 2}" text-anchor="middle" class="an-donut-v">${pctOf(s.anomaly_rate, 0)}</text>
        <text x="${C}" y="${C + 16}" text-anchor="middle" class="an-donut-l">anomalous</text></svg>
      <ul class="an-donut-legend">
        <li><i class="an-sw normal"></i><span>Normal</span><b>${int(s.normal)}</b><em>${pctOf(s.normal / s.requests)}</em></li>
        <li><i class="an-sw anom"></i><span>Anomalous</span><b>${int(s.anomalies)}</b><em>${pctOf(s.anomalies / s.requests)}</em></li>
        <li class="an-total"><span>Total scored</span><b>${int(s.requests)}</b></li>
      </ul></div>`;
  }

  /* Horizontal bars: rows [{label, title, normal, anomalies}] scaled to the largest total. */
  function barList(rows) {
    const max = Math.max(...rows.map(r => r.normal + r.anomalies), 1);
    return `<ul class="an-bars">${rows.map(r => {
      const tot = r.normal + r.anomalies;
      return `<li title="${esc(r.title || r.label)}: ${int(r.anomalies)} anomalous of ${int(tot)} (${pctOf(tot ? r.anomalies / tot : null)})">
        <div class="an-bars-h"><span class="mono">${esc(r.label)}</span><span class="an-bars-n"><b>${int(r.anomalies)}</b> / ${int(tot)} · ${pctOf(tot ? r.anomalies / tot : null, 0)}</span></div>
        <div class="an-track">${r.anomalies ? `<i class="an-seg anom" style="width:${r.anomalies / max * 100}%"></i>` : ""}${r.normal ? `<i class="an-seg normal" style="width:${r.normal / max * 100}%"></i>` : ""}</div>
      </li>`; }).join("")}</ul>`;
  }

  function renderTop(o) {
    const rows = o.top_endpoints.filter(r => r.requests > 0);
    if (!rows.length) { $("topEndpoints").innerHTML = NO_DATA(); return; }
    $("topEndpoints").innerHTML = `<div class="legend an-legend-top"><span><i class="an-sw anom"></i>Anomalous</span><span><i class="an-sw normal"></i>Normal</span></div>` +
      barList(rows.map(r => ({ label: r.key, normal: r.requests - r.anomalies, anomalies: r.anomalies })));
  }

  function renderMix(o) {
    const m = o.distributions.methods, s = o.distributions.sources;
    if (!m.length) { $("mix").innerHTML = NO_DATA(); return; }
    const toRows = (arr, lbl) => arr.map(r => ({ label: lbl(r.key), normal: r.requests - r.anomalies, anomalies: r.anomalies }));
    $("mix").innerHTML = `<div class="an-sub">HTTP method</div>${barList(toRows(m, k => k))}
      <div class="an-sub" style="margin-top:18px">Source</div>${barList(toRows(s, k => SOURCE_LABELS[k] || k))}`;
  }

  function renderInsights(o) {
    const list = o.insights;
    if (!o.summary.requests) { $("insights").innerHTML = NO_DATA(); return; }
    if (!list.length) { $("insights").innerHTML = empty("No notable patterns", "None of the insight rules fired for this selection."); return; }
    const label = { warn: "Attention", good: "Improving", info: "Observation" };
    $("insights").innerHTML = `<ul class="an-insights">${list.map(i => `<li class="${esc(i.level)}"><span class="badge ${i.level === "warn" ? "warn" : i.level === "good" ? "ok" : "info"}">${label[i.level] || i.level}</span><p>${esc(i.text)}</p></li>`).join("")}</ul>`;
  }

  function renderScoreStats(o) {
    const rows = o.score_stats.filter(r => r.requests > 0);
    const nt = o.near_threshold;
    if (!rows.length) { $("scoreStats").innerHTML = `<div class="card-body">${NO_DATA()}</div>`; return; }
    const name = { all: "All", anomaly: "Anomalous", normal: "Normal" };
    $("scoreStats").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Group</th><th class="num">n</th><th class="num">Min</th><th class="num">Median</th><th class="num">Mean</th><th class="num">P95</th><th class="num">Max</th><th class="num" title="Mean of score − threshold">Margin</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td>${r.group === "all" ? "" : `<i class="an-sw ${r.group === "anomaly" ? "anom" : "normal"}"></i>`}${name[r.group]}</td><td class="num">${int(r.requests)}</td><td class="num">${dec(r.min)}</td><td class="num">${dec(r.median)}</td><td class="num">${dec(r.mean)}</td><td class="num">${dec(r.p95)}</td><td class="num">${dec(r.max)}</td><td class="num">${r.mean_margin == null ? "—" : (r.mean_margin > 0 ? "+" : "") + dec(r.mean_margin)}</td></tr>`).join("")}
    </tbody></table></div>
    <div class="card-body small muted">Near the threshold (within ±${Math.round(nt.band * 100)}%): <b>${int(nt.flagged)}</b> flagged and <b>${int(nt.passed)}</b> passed — the decisions most sensitive to a threshold change. Margin is the mean distance above (+) or below (−) each request's own threshold.</div>`;
  }

  function renderThresholds(o) {
    const th = o.distributions.thresholds;
    if (!th.length) { $("thresholds").innerHTML = `<div class="card-body">${NO_DATA()}</div>`; return; }
    $("thresholds").innerHTML = `<div class="table-wrap"><table><thead><tr><th class="num">Threshold</th><th class="num">Requests</th><th class="num">Flagged</th><th class="num">Rate</th><th>In use</th></tr></thead><tbody>
      ${th.map(t => `<tr><td class="num">${dec(t.threshold, 4).replace(/\.?0+$/, "")}</td><td class="num">${int(t.requests)}</td><td class="num">${int(t.anomalies)}</td><td class="num">${pctOf(t.anomaly_rate)}</td><td class="small">${esc(dFmt.format(at(t.first_seen)))}${t.last_seen - t.first_seen > 86400 ? " – " + esc(dFmt.format(at(t.last_seen))) : ""}</td></tr>`).join("")}
    </tbody></table></div>
    <div class="card-body small muted">${th.length > 1 ? "Several thresholds were active in this range, so anomaly rates across time are not strictly like-for-like." : "A single threshold was active for every request in this range."}</div>`;
  }

  function renderUrls(o) {
    const u = o.urls, host = $("urls");
    if (!u) { host.innerHTML = `<div class="card-body">${empty("URL classifier history unavailable", "The url_detections table has not been created yet.")}</div>`; return; }
    const s = u.summary;
    if (!s.analyses) { host.innerHTML = `<div class="card-body">${empty("No URL analyses in this period", "Analyse URLs on the Analyze page, or widen the date range.")}</div>`; return; }
    const prevLine = o.previous_range ? (u.previous && u.previous.analyses ? `${u.change.analyses > 0 ? "+" : ""}${dec(u.change.analyses, 0)}% vs previous period` : "none in previous period") : "all recorded data";
    const tiles = [
      [int(s.analyses), "URLs analysed", prevLine],
      [pctOf(s.phishing_rate), "classified phishing", `${int(s.phishing)} of ${int(s.analyses)}`],
      [pctOf(s.mean_probability), "mean P(phishing)", "average model output"],
      [s.mean_inference_ms == null ? "—" : dec(s.mean_inference_ms, 1) + " ms", "mean inference time", "per URL"],
      [int(s.incomplete), "with missing info", `${pctOf(s.incomplete / s.analyses, 0)} analysed with segments unavailable`],
      [int(s.waf_flagged), "also flagged by WAF", `${int(s.waf_scored)} scored by the WAF model · ${int(s.both_flagged)} flagged by both`],
    ];
    const maxMiss = Math.max(...u.missing_features.map(m => m.analyses), 1);
    host.innerHTML = `<div class="card-body"><div class="tiles">${tiles.map(([v, l, d]) => `<div class="tile"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div><div class="an-tile-d">${esc(d)}</div></div>`).join("")}</div></div>
      <div class="grid cols-2 an-url-grid">
        <div class="table-wrap"><table><thead><tr><th>Strategy / model</th><th class="num">URLs</th><th class="num">Phishing</th><th class="num">Mean P</th><th class="num">ms</th></tr></thead><tbody>
          ${u.strategies.map(r => `<tr><td class="mono small">${esc(r.strategy)} <span class="dim">/ ${esc(r.model_variant)}</span></td><td class="num">${int(r.analyses)}</td><td class="num">${int(r.phishing)} <span class="dim">(${pctOf(r.phishing_rate, 0)})</span></td><td class="num">${pctOf(r.mean_probability)}</td><td class="num">${dec(r.mean_inference_ms, 1)}</td></tr>`).join("")}
        </tbody></table></div>
        <div class="card-body"><div class="an-sub">Information units marked unavailable</div>
          ${u.missing_features.length ? `<ul class="an-bars">${u.missing_features.map(m => `<li><div class="an-bars-h"><span class="mono">${esc(m.feature)}</span><span class="an-bars-n"><b>${int(m.analyses)}</b> analyses</span></div><div class="an-track"><i class="an-seg normal" style="width:${m.analyses / maxMiss * 100}%"></i></div></li>`).join("")}</ul>` : `<p class="dim small">Every URL in this period was analysed with complete information.</p>`}
        </div>
      </div>`;
  }

  function renderRangeBar(o) {
    const r = o.range, p = o.previous_range, s = o.summary;
    const span = (a, b) => `${dFmt.format(at(a))} – ${dFmt.format(at(b - 1))}`;
    let html = r.start == null
      ? `<span><b>All time</b>${s.first_seen ? ` · ${esc(dFmt.format(at(s.first_seen)))} – ${esc(dFmt.format(at(s.last_seen)))}` : ""}</span>`
      : `<span>Period <b>${esc(span(r.start, r.end))}</b></span>`;
    if (p) html += `<span>Compared with <b>${esc(span(p.start, p.end))}</b></span>`;
    if (s.last_seen) html += `<span>Last scored request <b>${esc(dtFmt.format(at(s.last_seen)))}</b></span>`;
    $("rangeBar").innerHTML = html;
  }

  /* ---------------- filter options ---------------- */
  function syncOptions(opts) {
    [["fMethod", opts.methods, k => k], ["fSource", opts.sources, k => SOURCE_LABELS[k] || k]].forEach(([id, list, lbl]) => {
      const sel = $(id), cur = sel.value;
      const vals = new Set(list);
      if (cur) vals.add(cur);
      sel.innerHTML = `<option value="">All</option>` + [...vals].map(v => `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(lbl(v))}</option>`).join("");
    });
  }

  /* ---------------- breakdown table ---------------- */
  const COLS = [
    ["key", "", false], ["requests", "Requests", true], ["anomalies", "Anomalous", true], ["rate", "Rate", true],
    ["mean_score", "Mean score", true], ["max_score", "Max score", true], ["change", "Δ anomalous", true], ["last_seen", "Last seen", false],
  ];

  async function loadBreakdown() {
    const my = ++state.bseq;
    let p;
    try { p = baseParams(); } catch (e) { return; }
    p.set("group_by", state.group); p.set("sort", state.sort); p.set("order", state.order);
    p.set("limit", state.pageSize); p.set("offset", state.page * state.pageSize);
    if (state.search) p.set("search", state.search);
    $("breakdown").classList.add("an-dim");
    try {
      const b = await api("/analytics/breakdown?" + p.toString());
      if (my !== state.bseq) return;
      renderBreakdown(b);
    } catch (e) {
      if (my !== state.bseq) return;
      $("breakdown").innerHTML = `<div class="card-body"><div class="notice error">Failed to load the breakdown: ${esc(e.message)} <button class="btn small" type="button" id="bRetry">Retry</button></div></div>`;
      $("bPager").innerHTML = "";
      $("bRetry").addEventListener("click", loadBreakdown);
    } finally { $("breakdown").classList.remove("an-dim"); }
  }

  function renderBreakdown(b) {
    const host = $("breakdown");
    $("bCount").textContent = b.total ? `${int(b.total)} ${b.total === 1 ? "row" : "rows"}` : "";
    if (!b.rows.length) {
      host.innerHTML = `<div class="card-body">${state.search ? empty("No rows match your search", "Clear the search box to see every row.") : NO_DATA()}</div>`;
      $("bPager").innerHTML = ""; return;
    }
    const head = COLS.map(([k, label, num]) => {
      const text = k === "key" ? GROUP_LABELS[state.group] : label;
      const active = state.sort === k;
      const aria = active ? (state.order === "asc" ? "ascending" : "descending") : "none";
      if (k === "change" && !b.has_previous) return "";
      return `<th class="${num ? "num" : ""}" aria-sort="${aria}"><button type="button" class="an-sort${active ? " on" : ""}" data-sort="${k}">${esc(text)}<span aria-hidden="true">${active ? (state.order === "asc" ? " ▲" : " ▼") : ""}</span></button></th>`;
    }).join("");
    const rows = b.rows.map((r, i) => {
      const ch = r.anomalies_change;
      const chCell = b.has_previous ? `<td class="num ${ch > 0 ? "delta-neg" : ch < 0 ? "delta-pos" : "dim"}">${ch > 0 ? "↑ +" : ch < 0 ? "↓ " : ""}${int(ch)}<span class="dim small"> (was ${int(r.previous_anomalies)})</span></td>` : "";
      return `<tr>
        <td class="mono" style="word-break:break-all" title="${esc(r.key || "")}"><span class="dim small">${state.page * state.pageSize + i + 1}.</span> ${esc(keyLabel(state.group, r.key))}</td>
        <td class="num">${int(r.requests)}</td><td class="num">${int(r.anomalies)}</td>
        <td class="num"><span class="an-rate"><i style="width:${(r.anomaly_rate || 0) * 100}%"></i></span>${pctOf(r.anomaly_rate)}</td>
        <td class="num">${dec(r.mean_score)}</td><td class="num">${dec(r.max_score)}</td>${chCell}
        <td class="small">${r.last_seen ? esc(dtFmt.format(at(r.last_seen))) : "—"}</td></tr>`;
    }).join("");
    host.innerHTML = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
    host.querySelectorAll(".an-sort").forEach(btn => btn.addEventListener("click", () => {
      const k = btn.dataset.sort;
      if (state.sort === k) state.order = state.order === "asc" ? "desc" : "asc";
      else { state.sort = k; state.order = k === "key" ? "asc" : "desc"; }
      state.page = 0; loadBreakdown();
    }));
    const pages = Math.ceil(b.total / state.pageSize);
    $("bPager").innerHTML = pages > 1 ? `<button class="btn small" type="button" id="bPrev"${state.page === 0 ? " disabled" : ""}>Previous</button><span class="badge neutral">page ${state.page + 1} of ${pages}</span><button class="btn small" type="button" id="bNext"${state.page >= pages - 1 ? " disabled" : ""}>Next</button>` : "";
    if (pages > 1) {
      $("bPrev").addEventListener("click", () => { state.page--; loadBreakdown(); });
      $("bNext").addEventListener("click", () => { state.page++; loadBreakdown(); });
    }
  }

  /* ---------------- main load ---------------- */
  function renderAll(o) {
    renderRangeBar(o); renderKpis(o); renderTrend(o); renderVerdict(o); renderHistogram(o);
    renderTop(o); renderMix(o); renderInsights(o); renderScoreStats(o); renderThresholds(o); renderUrls(o);
    syncOptions(o.filter_options);
  }

  async function load() {
    const n = $("filterNotice");
    let p;
    try { p = baseParams(); n.classList.add("hidden"); } catch (e) { WAF.notice(n, "error", e.message); return; }
    const my = ++state.seq;
    $("errorState").classList.add("hidden");
    $("dash").classList.remove("hidden");
    showSkeletons();
    state.page = 0;
    loadBreakdown();
    try {
      const o = await api("/analytics/overview?" + p.toString());
      if (my !== state.seq) return;
      state.overview = o;
      renderAll(o);
    } catch (e) {
      if (my !== state.seq) return;
      $("dash").classList.add("hidden");
      $("rangeBar").innerHTML = "";
      $("errorText").textContent = e.status === 401
        ? "The API rejected the key. Check the API key field in the top bar, then retry."
        : "Something went wrong while retrieving the analytics data. " + e.message;
      $("errorState").classList.remove("hidden");
    }
  }

  async function exportCsv() {
    const btn = $("exportCsv"), n = $("filterNotice");
    let p;
    try { p = baseParams(); } catch (e) { WAF.notice(n, "error", e.message); return; }
    p.set("group_by", state.group); p.set("sort", state.sort); p.set("order", state.order);
    if (state.search) p.set("search", state.search);
    btn.disabled = true;
    try {
      const text = await api("/analytics/export?" + p.toString());
      const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
      const a = document.createElement("a");
      a.href = url; a.download = `waf-analytics-${state.group}-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      WAF.notice(n, "error", "Export failed: " + e.message);
    } finally { btn.disabled = false; }
  }

  /* ---------------- wiring ---------------- */
  function syncCustom() {
    const custom = $("fRange").value === "custom";
    document.querySelectorAll(".an-custom").forEach(el => el.classList.toggle("hidden", !custom));
    if (custom && !$("fFrom").value) {
      const t = new Date(), f = new Date(); f.setDate(f.getDate() - 29);
      const iso = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
      $("fFrom").value = iso(f); $("fTo").value = iso(t);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    try { const r = localStorage.getItem("waf_analytics_range"); if (r && [...$("fRange").options].some(o => o.value === r && r !== "custom")) $("fRange").value = r; } catch (e) { /* ignore */ }
    syncCustom();
    $("fRange").addEventListener("change", () => {
      syncCustom();
      try { localStorage.setItem("waf_analytics_range", $("fRange").value); } catch (e) { /* ignore */ }
      if ($("fRange").value !== "custom") load();
    });
    ["fVerdict", "fMethod", "fSource", "fFrom", "fTo"].forEach(id => $(id).addEventListener("change", load));
    $("filters").addEventListener("submit", ev => { ev.preventDefault(); load(); });
    $("fReset").addEventListener("click", () => {
      $("fRange").value = "30d"; ["fVerdict", "fMethod", "fSource", "fPath"].forEach(id => $(id).value = "");
      syncCustom(); load();
    });
    $("retryBtn").addEventListener("click", load);
    $("exportCsv").addEventListener("click", exportCsv);
    $("apiKey").addEventListener("change", load);
    document.querySelectorAll("#groupTabs button").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll("#groupTabs button").forEach(x => x.classList.toggle("active", x === b));
      state.group = b.dataset.group; state.page = 0; state.search = ""; $("bSearch").value = "";
      loadBreakdown();
    }));
    let t;
    $("bSearch").addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { state.search = $("bSearch").value.trim(); state.page = 0; loadBreakdown(); }, 300); });

    // Charts are drawn at their real pixel width (so labels stay readable on phones); redraw on resize.
    let lastW = 0, rt;
    new ResizeObserver(() => {
      const w = $("trendChart").clientWidth;
      if (!state.overview || Math.abs(w - lastW) < 8) return;
      lastW = w; clearTimeout(rt);
      rt = setTimeout(() => { renderTrend(state.overview); renderHistogram(state.overview); }, 120);
    }).observe($("dash"));

    load();
  });
})();
