/* Shared helpers for the Transformer WAF UI. No framework; same-origin API. */
(function () {
  const API_BASE = (location.protocol === "file:") ? "http://localhost:8000" : location.origin;

  function apiKey() {
    const el = document.getElementById("apiKey");
    if (el && el.value) return el.value;
    try { return localStorage.getItem("waf_api_key") || "dev-key"; } catch (e) { return "dev-key"; }
  }

  function initApiKey() {
    const el = document.getElementById("apiKey");
    if (!el) return;
    try { el.value = localStorage.getItem("waf_api_key") || "dev-key"; } catch (e) { el.value = "dev-key"; }
    el.addEventListener("change", () => { try { localStorage.setItem("waf_api_key", el.value); } catch (e) { /* ignore */ } });
  }

  async function api(path, opts = {}) {
    const headers = Object.assign({ "X-API-Key": apiKey() }, opts.headers || {});
    let body = opts.body;
    if (body !== undefined && typeof body !== "string") {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(API_BASE + path, { method: opts.method || "GET", headers, body });
    } catch (e) {
      throw new Error("Cannot reach the API at " + API_BASE + " — is the detection service running?");
    }
    let data = null;
    const text = await res.text();
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
    if (!res.ok) {
      const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText;
      const err = new Error(`${res.status}: ${detail}`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function pct(x, digits = 2) { return (x == null || isNaN(x)) ? "—" : (x * 100).toFixed(digits) + "%"; }
  function num(x, digits = 3) { return (x == null || isNaN(x)) ? "—" : Number(x).toFixed(digits); }
  function signed(x, digits = 2) {
    if (x == null || isNaN(x)) return "—";
    const v = x * 100;
    return (v > 0 ? "+" : "") + v.toFixed(digits) + " pp";
  }
  function fmtTime(ts) {
    const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
    return isNaN(d) ? String(ts) : d.toLocaleString();
  }

  function setBusy(btn, busy, label) {
    if (!btn) return;
    if (busy) {
      btn.dataset.label = btn.textContent;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>' + esc(label || "Working…");
    } else {
      btn.disabled = false;
      btn.textContent = btn.dataset.label || btn.textContent;
    }
  }

  function notice(el, kind, msg) {
    el.className = "notice " + kind;
    el.textContent = msg;
    el.classList.remove("hidden");
  }
  function clearNotice(el) { el.classList.add("hidden"); el.textContent = ""; }

  /* Render model-facing text with [MISSING] markers highlighted (escaped first). */
  function modelText(text) {
    return esc(text).replace(/\[MISSING\]/g, '<span class="missing">[MISSING]</span>');
  }

  /* Minimal SVG line chart: series = [{name, color, points:[{x,y}]}], x in [0,1] as %, y in [0,1]. */
  function lineChart(svg, series, opts = {}) {
    const W = 720, H = 300, padL = 52, padR = 16, padT = 14, padB = 36;
    const xs = series.flatMap(s => s.points.map(p => p.x));
    const ys = series.flatMap(s => s.points.map(p => p.y));
    if (!xs.length) { svg.innerHTML = ""; return; }
    const xMin = 0, xMax = Math.max(...xs, 0.1);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (opts.yMin != null) yMin = Math.min(yMin, opts.yMin);
    yMax = Math.min(1, yMax + 0.02);
    yMin = Math.max(0, Math.floor((yMin - 0.02) * 20) / 20);
    if (yMax - yMin < 0.1) yMin = Math.max(0, yMax - 0.1);
    const sx = x => padL + (x - xMin) / (xMax - xMin || 1) * (W - padL - padR);
    const sy = y => padT + (1 - (y - yMin) / (yMax - yMin || 1)) * (H - padT - padB);
    let out = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="${esc(opts.label || "chart")}">`;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = yMin + (yMax - yMin) * i / yTicks;
      out += `<line class="grid-line" x1="${padL}" x2="${W - padR}" y1="${sy(y)}" y2="${sy(y)}"/>`;
      out += `<text x="${padL - 8}" y="${sy(y) + 4}" text-anchor="end">${(y * 100).toFixed(0)}%</text>`;
    }
    const xTicks = [...new Set(xs)].sort((a, b) => a - b);
    xTicks.forEach(x => {
      out += `<text x="${sx(x)}" y="${H - padB + 18}" text-anchor="middle">${(x * 100).toFixed(0)}%</text>`;
    });
    out += `<line class="axis" x1="${padL}" x2="${W - padR}" y1="${H - padB}" y2="${H - padB}"/>`;
    out += `<line class="axis" x1="${padL}" x2="${padL}" y1="${padT}" y2="${H - padB}"/>`;
    out += `<text x="${(padL + W - padR) / 2}" y="${H - 4}" text-anchor="middle">${esc(opts.xLabel || "")}</text>`;
    series.forEach(s => {
      const pts = s.points.slice().sort((a, b) => a.x - b.x);
      const d = pts.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1)).join(" ");
      out += `<path class="series" d="${d}" stroke="${s.color}"/>`;
      pts.forEach(p => {
        out += `<circle class="pt" cx="${sx(p.x)}" cy="${sy(p.y)}" r="3.5" fill="var(--surface)" stroke="${s.color}"><title>${esc(s.name)} @ ${(p.x * 100).toFixed(0)}%: ${(p.y * 100).toFixed(2)}%</title></circle>`;
      });
    });
    out += "</svg>";
    svg.innerHTML = out;
  }

  function initTheme() {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
      const next = cur === "light" ? "dark" : "light";
      if (next === "light") document.documentElement.setAttribute("data-theme", "light");
      else document.documentElement.removeAttribute("data-theme");
      try { localStorage.setItem("waf_theme", next); } catch (e) { /* ignore */ }
    });
  }
  document.addEventListener("DOMContentLoaded", initTheme);

  window.WAF = { API_BASE, api, esc, pct, num, signed, fmtTime, setBusy, notice, clearNotice, modelText, lineChart, initApiKey };
  document.addEventListener("DOMContentLoaded", initApiKey);
})();
