"use strict";

// The local dev server (scripts/dev_server.py) serves the API from the same origin.
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const API_BASE = LOCAL ? "/api" : "https://cq094tylu1.execute-api.us-west-2.amazonaws.com/Prod/api";
const TICKER_RE = /^[A-Z0-9][A-Z0-9.\-]{0,9}$/;
const WINDOWS = [30, 90, 180, 365];
const NEWS_REFRESH_MS = 10 * 60 * 1000;
const STARTERS = ["AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "SPY"];
const COLORS = { up: "#35c6a0", down: "#f0716a", amber: "#f5b94a", text: "#8fa0b8", grid: "#1c2b45", ma20: "#f5b94a", ma50: "#7fb2ff", ma200: "#c79bff" };

const $ = (id) => document.getElementById(id);
const state = { ticker: null, window: 180, data: null, charts: [], lines: { ma20: true, ma50: true, ma200: false }, compare: false, requestId: 0 };

// ---------- small helpers ----------
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}
const num = (v, d = 2) => (v == null ? "n/a" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v) => (v == null ? "n/a" : `${v > 0 ? "+" : ""}${num(v)}%`);
const tone = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
function money(v, currency) {
  if (v == null) return "n/a";
  try { return v.toLocaleString("en-US", { style: "currency", currency: currency || "USD" }); }
  catch { return num(v); }
}
function ago(iso) {
  if (!iso) return "";
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  if (mins < 60 * 24) return `${Math.round(mins / 60)} hr ago`;
  return `${Math.round(mins / 1440)} d ago`;
}
function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* private mode or blocked storage: carry on without it */ }
  return null;
}
async function api(path) {
  const res = await fetch(`${API_BASE}${path}`);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) {
    if (res.status === 429) throw new Error("Too many requests right now. Wait a few seconds and try again.");
    throw new Error((body && (body.error || body.message)) || "The service did not respond. Try again in a minute.");
  }
  return body;
}

// ---------- watchlist ----------
function watchlist() { const w = store("mp.watchlist"); return Array.isArray(w) ? w.filter((t) => TICKER_RE.test(t)) : []; }
function renderWatchlist() {
  const list = watchlist();
  const ul = $("watchlist");
  ul.replaceChildren();
  if (!list.length) {
    ul.append(el("li", { class: "empty", text: "Symbols you add will stay here on this device." }));
  }
  for (const t of list) {
    const b = el("button", { type: "button", text: t });
    if (t === state.ticker) b.setAttribute("aria-current", "true");
    b.addEventListener("click", () => load(t));
    ul.append(el("li", {}, b));
  }
  const watching = list.includes(state.ticker);
  $("btnWatch").textContent = watching ? "In your watchlist" : "Add to watchlist";
  $("btnWatch").setAttribute("aria-pressed", String(watching));
}
$("btnWatch").addEventListener("click", () => {
  if (!state.ticker) return;
  const list = watchlist();
  const next = list.includes(state.ticker) ? list.filter((t) => t !== state.ticker) : [...list, state.ticker].slice(-12);
  store("mp.watchlist", next);
  renderWatchlist();
});
$("btnShare").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(location.href);
    $("btnShare").textContent = "Link copied";
  } catch {
    $("btnShare").textContent = "Copy the address bar";
  }
  setTimeout(() => { $("btnShare").textContent = "Copy link"; }, 2000);
});

// ---------- search with suggestions ----------
let TICKERS = [];
let active = -1;
const input = $("ticker");
const sug = $("suggestions");
fetch("tickers.json").then((r) => (r.ok ? r.json() : [])).then((rows) => {
  TICKERS = rows.map((r) => ({ s: String(r.s || r.symbol || ""), n: String(r.n || r.name || "") })).filter((r) => r.s);
}).catch(() => {});

function closeSuggestions() { sug.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; }
function showSuggestions() {
  const q = input.value.trim().toUpperCase();
  // With nothing typed the list shows every ticker, so it works as a plain dropdown too.
  const hits = q ? TICKERS.filter((t) => t.s.startsWith(q) || t.n.toUpperCase().includes(q)) : TICKERS;
  sug.replaceChildren();
  if (!hits.length) return closeSuggestions();
  hits.forEach((t, i) => {
    const li = el("li", { role: "option", id: `sug-${i}`, "aria-selected": "false" }, el("b", { text: t.s }), el("span", { text: t.n }));
    li.addEventListener("mousedown", (e) => { e.preventDefault(); input.value = t.s; closeSuggestions(); load(t.s); });
    sug.append(li);
  });
  sug.hidden = false;
  input.setAttribute("aria-expanded", "true");
}
input.addEventListener("input", showSuggestions);
input.addEventListener("focus", () => { input.select(); showAll(); });
input.addEventListener("click", () => { if (sug.hidden) showAll(); });
input.addEventListener("blur", closeSuggestions);
function showAll() { const typed = input.value; input.value = ""; showSuggestions(); input.value = typed; }
$("tickerToggle").addEventListener("mousedown", (e) => {
  e.preventDefault();
  if (!sug.hidden) return closeSuggestions();
  if (document.activeElement === input) showAll(); else input.focus();
});
input.addEventListener("keydown", (e) => {
  const items = [...sug.children];
  if (e.key === "Escape") return closeSuggestions();
  if (sug.hidden || !["ArrowDown", "ArrowUp"].includes(e.key)) return;
  e.preventDefault();
  active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
  items.forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
  items[active].scrollIntoView({ block: "nearest" });
  input.value = items[active].firstChild.textContent;
});
$("searchForm").addEventListener("submit", (e) => {
  e.preventDefault();
  closeSuggestions();
  const raw = input.value.trim().toUpperCase();
  const byName = TICKERS.find((t) => t.n.toUpperCase() === raw);
  load(byName ? byName.s : raw);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
});

// ---------- charts ----------
function baseChart(node, height) {
  return LightweightCharts.createChart(node, {
    autoSize: true, height,
    layout: { background: { type: "solid", color: "transparent" }, textColor: COLORS.text, fontFamily: "IBM Plex Sans, sans-serif" },
    grid: { vertLines: { visible: false }, horzLines: { color: COLORS.grid } },
    rightPriceScale: { borderVisible: false },
    timeScale: { borderVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: false, handleScale: false,
  });
}
function drawCharts() {
  state.charts.forEach((c) => c.remove());
  state.charts = [];
  const rows = state.data.chart;
  const chart = baseChart($("chart"), 380);
  const lineOpts = { lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };

  if (state.compare) {
    const firstSpy = rows.find((r) => r.spy != null);
    const rebase = (key, base) => rows.filter((r) => r[key] != null).map((r) => ({ time: r.t, value: (r[key] / base - 1) * 100 }));
    const fmt = { priceFormat: { type: "custom", formatter: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%` } };
    chart.addLineSeries({ ...lineOpts, ...fmt, color: COLORS.amber, title: state.data.ticker, lastValueVisible: true }).setData(rebase("c", rows[0].c));
    if (firstSpy) chart.addLineSeries({ ...lineOpts, ...fmt, color: COLORS.text, title: "S&P 500", lastValueVisible: true }).setData(rebase("spy", firstSpy.spy));
  } else {
    chart.addCandlestickSeries({ upColor: COLORS.up, downColor: COLORS.down, wickUpColor: COLORS.up, wickDownColor: COLORS.down, borderVisible: false })
      .setData(rows.map((r) => ({ time: r.t, open: r.o, high: r.h, low: r.l, close: r.c })));
    const vol = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "vol", priceLineVisible: false, lastValueVisible: false });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    vol.setData(rows.map((r) => ({ time: r.t, value: r.v, color: r.c >= r.o ? "rgba(53,198,160,.35)" : "rgba(240,113,106,.35)" })));
    for (const key of ["ma20", "ma50", "ma200"]) {
      if (!state.lines[key]) continue;
      chart.addLineSeries({ ...lineOpts, lineWidth: 1, color: COLORS[key] }).setData(rows.filter((r) => r[key] != null).map((r) => ({ time: r.t, value: r[key] })));
    }
  }
  chart.timeScale().fitContent();

  const rsi = baseChart($("rsiChart"), 120);
  const rsiLine = rsi.addLineSeries({ ...lineOpts, lineWidth: 1, color: COLORS.amber });
  rsiLine.setData(rows.filter((r) => r.rsi != null).map((r) => ({ time: r.t, value: r.rsi })));
  for (const level of [70, 30]) rsiLine.createPriceLine({ price: level, color: COLORS.text, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true });
  rsi.timeScale().fitContent();
  state.charts = [chart, rsi];
}

// ---------- rendering ----------
function renderQuote(d) {
  const f = d.facts;
  $("qSymbol").textContent = d.ticker;
  $("qName").textContent = [d.name, d.exchange].filter(Boolean).join(", ");
  $("qPrice").textContent = money(f.current_price, d.currency);
  $("qChange").textContent = pct(f.chg_1d);
  $("qChange").className = `change ${tone(f.chg_1d)}`;
  $("qAsOf").textContent = `Close on ${new Date(`${d.as_of}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}`;

  $("rLow").textContent = money(f.low_52w, d.currency);
  $("rHigh").textContent = money(f.high_52w, d.currency);
  $("rDot").style.left = `${Math.min(100, Math.max(0, f.range_52w_pos ?? 50))}%`;
  $("rText").textContent = f.range_52w_pos == null ? "" :
    `52-week range. The price sits ${num(f.range_52w_pos, 0)}% of the way from the low to the high and is above ${f.averages_above} of its 3 moving averages.`;

  const rows = [
    ["1 week", "5 trading days", pct(f.chg_5d), tone(f.chg_5d)],
    ["1 month", "", pct(f.chg_1m), tone(f.chg_1m)],
    ["3 months", "", pct(f.chg_3m), tone(f.chg_3m)],
    ["1 year", "", pct(f.chg_1y), tone(f.chg_1y)],
    ["Against S&P 500", "over 3 months", pct(f.rel_3m_vs_spy_pct), tone(f.rel_3m_vs_spy_pct)],
    ["RSI (14)", "momentum, 0 to 100", num(f.rsi14, 1), ""],
    ["Volatility", "30-day, annualised", f.vol_30 == null ? "n/a" : `${num(f.vol_30, 1)}%`, ""],
    ["Typical daily swing", "average true range", f.atr14 == null ? "n/a" : `${money(f.atr14, d.currency)} (${num(f.atr_pct, 1)}%)`, ""],
    ["Beta", "moves per 1% move in the S&P 500", num(f.beta_vs_spy), ""],
    ["Worst drop this year", "peak to trough", pct(f.max_dd_ytd), tone(f.max_dd_ytd)],
    ["Volume today", "against 20-day average", f.vol_ratio == null ? "n/a" : `${num(f.vol_ratio)}x`, ""],
    ["20-day range", "recent low to high", `${money(f.sr_low_20d, d.currency)} to ${money(f.sr_high_20d, d.currency)}`, ""],
  ];
  $("facts").replaceChildren(...rows.map(([label, hint, value, cls]) =>
    el("div", {}, el("dt", {}, label, ...(hint ? [el("small", { text: hint })] : [])), el("dd", { class: cls, text: value }))));

  $("tickerNewsTitle").textContent = `${d.ticker} in the news`;
  renderNews($("tickerNews"), d.news, `No recent headlines found for ${d.ticker}.`);
  document.querySelectorAll(".segmented button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.window) === state.window)));
  drawCharts();
}

function renderNews(ul, items, emptyText) {
  ul.replaceChildren();
  if (!items || !items.length) return ul.append(el("li", { class: "quiet", text: emptyText }));
  for (const item of items) {
    // Links come from third-party feeds: accept https only and never pass our URL along.
    if (typeof item.link !== "string" || !item.link.startsWith("https://")) continue;
    const a = el("a", { href: item.link, target: "_blank", rel: "noopener noreferrer", text: item.title });
    ul.append(el("li", {}, a, el("span", { class: "news-meta", text: [item.source, ago(item.published)].filter(Boolean).join(", ") })));
  }
}

const BRIEF_NOTICES = {
  no_key: "The written brief is switched off because no Claude API key is configured yet. The numbers and chart above are complete without it.",
  limit: "Today's allowance of written briefs has been used. It resets at midnight UTC. The numbers and chart above are complete without it.",
  unavailable: "The brief could not be written just now. Reload in a minute to try again.",
};
function renderBrief(b) {
  const body = $("briefBody");
  if (!b.story) return body.replaceChildren(el("p", { class: "notice", text: BRIEF_NOTICES[b.status] || BRIEF_NOTICES.unavailable }));
  const s = b.story;
  const list = (items) => el("ul", {}, ...(items || []).map((t) => el("li", { text: t })));
  const nodes = [
    el("span", { class: `picture ${s.picture}`, text: `Technical picture: ${s.picture}` }),
    el("h3", { text: s.title }),
    el("p", { class: "tldr", text: s.tldr }),
    el("p", { class: "narrative", text: s.narrative }),
    el("div", { class: "points" },
      el("div", { class: "for" }, el("h4", { text: "Working in its favour" }), list(s.positives)),
      el("div", { class: "against" }, el("h4", { text: "Worth watching" }), list(s.risks))),
    el("p", { class: "extra" }, el("b", { text: "Levels. " }), s.levels_to_watch),
  ];
  if (s.news_context) nodes.push(el("p", { class: "extra" }, el("b", { text: "In the news. " }), s.news_context));
  const used = b.headlines_used ? ` and ${b.headlines_used} recent headlines` : "";
  nodes.push(el("p", { class: "quiet byline", text: `Written by Claude from the numbers on this page${used}, ${ago(b.generated_at)}. Not investment advice.` }));
  body.replaceChildren(...nodes);
}

// ---------- loading ----------
async function load(raw, { push = true } = {}) {
  const ticker = String(raw || "").trim().toUpperCase();
  const error = $("error");
  error.hidden = true;
  if (!TICKER_RE.test(ticker)) {
    error.textContent = "Enter a ticker symbol such as AAPL or BRK-B.";
    error.hidden = false;
    return;
  }
  const id = ++state.requestId;
  state.ticker = ticker;
  input.value = ticker;
  document.title = `${ticker} | MarketPulse`;
  if (push) history.pushState(null, "", `?t=${encodeURIComponent(ticker)}&d=${state.window}`);
  store("mp.last", { t: ticker, d: state.window });
  renderWatchlist();
  $("briefBody").replaceChildren(el("div", { class: "skeleton" }, ...Array.from({ length: 5 }, () => el("span"))));

  try {
    const data = await api(`/quote/${encodeURIComponent(ticker)}?window_days=${state.window}`);
    if (id !== state.requestId) return;
    state.data = data;
    $("welcome").hidden = true;
    $("quote").hidden = false;
    renderQuote(data);
  } catch (e) {
    if (id !== state.requestId) return;
    $("quote").hidden = true;
    $("welcome").hidden = false;
    error.textContent = e.message;
    error.hidden = false;
    return;
  }
  try {
    const brief = await api(`/brief/${encodeURIComponent(ticker)}`);
    if (id === state.requestId) renderBrief(brief);
  } catch {
    if (id === state.requestId) renderBrief({ status: "unavailable" });
  }
}

async function changeWindow(days) {
  state.window = days;
  if (!state.ticker) return;
  history.replaceState(null, "", `?t=${encodeURIComponent(state.ticker)}&d=${days}`);
  store("mp.last", { t: state.ticker, d: days });
  const id = state.requestId;
  try {
    const data = await api(`/quote/${encodeURIComponent(state.ticker)}?window_days=${days}`);
    if (id !== state.requestId) return;
    state.data = data;
    renderQuote(data);
  } catch (e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
  }
}
document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => changeWindow(Number(b.dataset.window))));
document.querySelectorAll("[data-line]").forEach((box) => box.addEventListener("change", () => {
  state.lines[box.dataset.line] = box.checked;
  if (state.data) drawCharts();
}));
$("compare").addEventListener("change", (e) => { state.compare = e.target.checked; if (state.data) drawCharts(); });

async function loadMarketNews() {
  try {
    const news = await api("/news");
    renderNews($("marketNews"), news.items, "No headlines available right now.");
    $("newsUpdated").textContent = news.updated ? `Updated ${ago(news.updated)}. Refreshes every 30 minutes.` : "";
  } catch {
    if (!$("marketNews").children.length) renderNews($("marketNews"), [], "Headlines could not be loaded.");
  }
}

// ---------- start ----------
function fromUrl() {
  const q = new URLSearchParams(location.search);
  const d = Number(q.get("d"));
  if (WINDOWS.includes(d)) state.window = d;
  return (q.get("t") || "").toUpperCase();
}
$("starters").append(...STARTERS.map((t) => {
  const b = el("button", { type: "button", text: t });
  b.addEventListener("click", () => load(t));
  return el("li", {}, b);
}));
window.addEventListener("popstate", () => { const t = fromUrl(); if (t) load(t, { push: false }); });
renderWatchlist();
loadMarketNews();
setInterval(loadMarketNews, NEWS_REFRESH_MS);
const initial = fromUrl();
if (initial) load(initial, { push: false });
