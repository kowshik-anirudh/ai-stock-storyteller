# backend/market.py
"""Daily price history from Yahoo's chart API and the indicators computed from it."""
import json
import math
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125 Safari/537.36"
)
CHART_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
BENCHMARK = "SPY"
MAX_BODY_BYTES = 2_000_000


class TickerNotFound(Exception):
    pass


def valid_ticker(raw: str):
    ticker = (raw or "").strip().upper()
    return ticker if TICKER_RE.match(ticker) else None


def http_get(url: str, timeout: float = 8.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(MAX_BODY_BYTES)


def fetch_history(ticker: str) -> dict:
    """Two years of daily bars. Returns {"meta": {...}, "bars": [{d,o,h,l,c,v}, ...]}."""
    path = f"/v8/finance/chart/{urllib.parse.quote(ticker)}?range=2y&interval=1d"
    last_error = None
    for host in CHART_HOSTS:
        try:
            data = json.loads(http_get(f"https://{host}{path}"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise TickerNotFound(ticker) from e
            last_error = e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last_error = e
    else:
        raise RuntimeError(f"price source unavailable: {last_error}")

    result = (data.get("chart") or {}).get("result") or []
    if not result or not result[0].get("timestamp"):
        raise TickerNotFound(ticker)
    res = result[0]
    meta = res.get("meta") or {}
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    offset = int(meta.get("gmtoffset") or 0)

    bars = []
    for i, ts in enumerate(res["timestamp"]):
        row = [quote.get(k, [None])[i] if i < len(quote.get(k, [])) else None for k in ("open", "high", "low", "close")]
        if any(v is None for v in row):
            continue
        vol = quote.get("volume", [])
        day = datetime.fromtimestamp(ts + offset, timezone.utc).date().isoformat()
        bars.append({"d": day, "o": row[0], "h": row[1], "l": row[2], "c": row[3],
                     "v": (vol[i] if i < len(vol) and vol[i] is not None else 0)})
    if len(bars) < 2:
        raise TickerNotFound(ticker)
    return {
        "meta": {
            "name": meta.get("longName") or meta.get("shortName") or ticker,
            "currency": meta.get("currency") or "USD",
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName") or "",
        },
        "bars": bars,
    }


# ---------- indicators ----------

def sma(values, period):
    out, total = [None] * len(values), 0.0
    for i, v in enumerate(values):
        total += v
        if i >= period:
            total -= values[i - period]
        if i >= period - 1:
            out[i] = total / period
    return out


def rsi_wilder(closes, period=14):
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _pct(now, then):
    return (now / then - 1.0) * 100.0 if then else None


def _round(value, digits=2):
    return None if value is None or isinstance(value, float) and math.isnan(value) else round(value, digits)


def analyse(ticker: str) -> dict:
    """Fetch history for the ticker and the benchmark, and compute everything the app shows."""
    hist = fetch_history(ticker)
    bars = hist["bars"]
    closes = [b["c"] for b in bars]
    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]
    vols = [b["v"] for b in bars]
    n = len(bars)
    last = closes[-1]

    try:
        spy = {b["d"]: b["c"] for b in fetch_history(BENCHMARK)["bars"]} if ticker != BENCHMARK else {}
    except (TickerNotFound, RuntimeError):
        spy = {}

    def change(days):
        return _pct(last, closes[-1 - days]) if n > days else None

    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, n)]
    vol_30 = statistics.stdev(rets[-30:]) * math.sqrt(252) * 100.0 if len(rets) >= 30 else None

    year = bars[-1]["d"][:4]
    ytd = [b["c"] for b in bars if b["d"].startswith(year)]
    max_dd_ytd, peak = None, None
    for c in ytd:
        peak = c if peak is None else max(peak, c)
        dd = (c / peak - 1.0) * 100.0
        max_dd_ytd = dd if max_dd_ytd is None else min(max_dd_ytd, dd)

    ma20, ma50, ma200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    rsi = rsi_wilder(closes, 14)

    true_ranges = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, n)]
    atr14 = sum(true_ranges[-14:]) / 14 if len(true_ranges) >= 14 else None

    slope20 = None
    if n >= 20:
        ys = closes[-20:]
        mean_x, mean_y = 9.5, sum(ys) / 20
        slope = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(ys)) / sum((x - mean_x) ** 2 for x in range(20))
        slope20 = slope / last * 252 * 100.0

    # Relative strength and beta against the benchmark, on dates both traded.
    rel_3m = beta = None
    paired = [(b["c"], spy[b["d"]]) for b in bars if b["d"] in spy]
    if len(paired) > 63:
        rel_3m = _pct(paired[-1][0] / paired[-64][0], paired[-1][1] / paired[-64][1])
    if len(paired) > 60:
        window = paired[-253:]
        rt = [window[i][0] / window[i - 1][0] - 1.0 for i in range(1, len(window))]
        rm = [window[i][1] / window[i - 1][1] - 1.0 for i in range(1, len(window))]
        var_m = statistics.pvariance(rm)
        if var_m > 0:
            mt, mm = statistics.fmean(rt), statistics.fmean(rm)
            beta = sum((a - mt) * (b - mm) for a, b in zip(rt, rm)) / len(rt) / var_m

    avg_vol_20 = sum(vols[-20:]) / 20 if n >= 20 else None
    year_closes = closes[-252:]
    hi_52w, lo_52w = max(year_closes), min(year_closes)

    signals = []
    for label, ma in (("20-day", ma20[-1]), ("50-day", ma50[-1]), ("200-day", ma200[-1])):
        if ma:
            signals.append(f"{'above' if last > ma else 'below'} {label} average")
    if ma50[-1] and ma200[-1]:
        signals.append("50-day average above 200-day" if ma50[-1] > ma200[-1] else "50-day average below 200-day")
    above = sum(1 for ma in (ma20[-1], ma50[-1], ma200[-1]) if ma and last > ma)

    facts = {
        "current_price": _round(last),
        "prev_close": _round(closes[-2]),
        "chg_1d": _round(_pct(last, closes[-2])),
        "chg_5d": _round(change(5)), "chg_1m": _round(change(21)), "chg_3m": _round(change(63)),
        "chg_6m": _round(change(126)), "chg_1y": _round(change(252)),
        "vol_30": _round(vol_30), "max_dd_ytd": _round(max_dd_ytd),
        "dma20": _round(ma20[-1]), "dma50": _round(ma50[-1]), "dma200": _round(ma200[-1]),
        "dist20_pct": _round(_pct(last, ma20[-1])), "dist50_pct": _round(_pct(last, ma50[-1])),
        "dist200_pct": _round(_pct(last, ma200[-1])),
        "rsi14": _round(rsi[-1]), "atr14": _round(atr14),
        "atr_pct": _round(atr14 / last * 100.0 if atr14 else None),
        "slope20_pct_annual": _round(slope20),
        "rel_3m_vs_spy_pct": _round(rel_3m), "beta_vs_spy": _round(beta),
        "avg_vol_20d": _round(avg_vol_20, 0),
        "vol_ratio": _round(vols[-1] / avg_vol_20 if avg_vol_20 else None),
        "sr_high_20d": _round(max(closes[-20:])), "sr_low_20d": _round(min(closes[-20:])),
        "high_52w": _round(hi_52w), "low_52w": _round(lo_52w),
        "range_52w_pos": _round((last - lo_52w) / (hi_52w - lo_52w) * 100.0 if hi_52w > lo_52w else None, 1),
        "averages_above": above,
        "signals": signals,
    }

    # Keep the last ~13 months of bars so any window up to a year can be sliced from cache.
    keep = min(n, 280)
    series = []
    for i in range(n - keep, n):
        b = bars[i]
        series.append({
            "t": b["d"], "o": _round(b["o"], 4), "h": _round(b["h"], 4), "l": _round(b["l"], 4),
            "c": _round(b["c"], 4), "v": b["v"],
            "ma20": _round(ma20[i], 4), "ma50": _round(ma50[i], 4), "ma200": _round(ma200[i], 4),
            "rsi": _round(rsi[i]), "spy": _round(spy.get(b["d"]), 4),
        })

    return {
        "ticker": ticker,
        "name": hist["meta"]["name"],
        "currency": hist["meta"]["currency"],
        "exchange": hist["meta"]["exchange"],
        "as_of": bars[-1]["d"],
        "facts": facts,
        "series": series,
    }


def slice_window(series, window_days: int):
    cutoff = (datetime.fromisoformat(series[-1]["t"]) - timedelta(days=window_days)).date().isoformat()
    return [p for p in series if p["t"] >= cutoff]
