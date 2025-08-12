# backend/app.py
import os, json, math, time
from datetime import datetime, timedelta, timezone

import boto3, numpy as np, pandas as pd, yfinance as yf, requests, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from openai import OpenAI

# =========================
# Env / Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
S3_CACHE_BUCKET = os.getenv("S3_CACHE_BUCKET", "")
S3_CACHE_TTL_SECONDS = int(os.getenv("S3_CACHE_TTL_SECONDS", "1800"))

s3 = boto3.client("s3")
YA_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"

# =========================
# App & CORS
# =========================
app = FastAPI(title="AI Stock Storyteller", version="2.0-trader")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Simple S3 cache
# =========================
def _cache_key(ticker: str, window_days: int) -> str:
    return f"stories/{ticker.upper()}_{int(window_days)}.json"

def s3_get_cached_story(ticker: str, window_days: int):
    if not S3_CACHE_BUCKET:
        return None
    try:
        obj = s3.get_object(Bucket=S3_CACHE_BUCKET, Key=_cache_key(ticker, window_days))
        cached = json.loads(obj["Body"].read())
        ts = cached.get("_cached_at")
        if not ts:
            return None
        age = datetime.now(timezone.utc).timestamp() - float(ts)
        if age <= S3_CACHE_TTL_SECONDS:
            return cached.get("payload")
    except s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None
    return None

def s3_put_cached_story(ticker: str, window_days: int, payload: dict):
    if not S3_CACHE_BUCKET:
        return
    doc = {"_cached_at": datetime.now(timezone.utc).timestamp(), "payload": payload}
    try:
        s3.put_object(
            Bucket=S3_CACHE_BUCKET,
            Key=_cache_key(ticker, window_days),
            Body=json.dumps(doc).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception:
        pass

# =========================
# TA (no pandas_ta)
# =========================
def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# =========================
# Market data loaders
# =========================
def fetch_ohlcv_yfinance(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    sess = requests.Session()
    sess.headers.update({"User-Agent": YA_USER_AGENT})
    last_exc = None
    for _ in range(3):
        try:
            df = yf.download(
                tickers=ticker, start=start, end=end, progress=False,
                auto_adjust=True, session=sess, threads=False, interval="1d",
            )
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                return df
        except Exception as e:
            last_exc = e
        time.sleep(0.6)
    if last_exc: raise last_exc
    raise RuntimeError("yfinance returned empty data")

def fetch_ohlcv_yahoo_chart_api(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    total_days = max((end - start).days, 30)
    range_str = "6mo" if total_days <= 200 else ("1y" if total_days <= 380 else "2y")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_str, "interval": "1d"}
    headers = {"User-Agent": YA_USER_AGENT}
    with httpx.Client(timeout=10.0) as client:
        r = client.get(url, params=params, headers=headers); r.raise_for_status()
        data = r.json()
    result = data.get("chart", {}).get("result", [])
    if not result: raise RuntimeError(f"Yahoo chart API returned no result for {ticker}")
    res = result[0]; ts = res.get("timestamp", [])
    if not ts: raise RuntimeError(f"Yahoo chart API missing timestamps for {ticker}")
    ind = pd.to_datetime(ts, unit="s", utc=True)
    indicators = res.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame(
        {"Open": quote.get("open"), "High": quote.get("high"), "Low": quote.get("low"),
         "Close": adjclose if adjclose else quote.get("close"), "Volume": quote.get("volume")},
        index=ind,
    ).sort_index()
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    df = df.loc[mask]
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Yahoo chart API empty window for {ticker}")
    return df

# =========================
# Feature engineering
# =========================
def compute_trader_facts(ticker: str, window_days: int = 180) -> dict:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(window_days + 60, 260))
    try:
        df = fetch_ohlcv_yfinance(ticker, start, end)
    except Exception:
        df = fetch_ohlcv_yahoo_chart_api(ticker, start, end)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No data for ticker {ticker}")

    # Normalize index to UTC
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    else: df.index = df.index.tz_convert("UTC")

    close = df["Close"].dropna().copy()
    high  = df["High"] if "High" in df.columns else close
    low   = df["Low"]  if "Low"  in df.columns else close
    volume = (df["Volume"] if "Volume" in df.columns else pd.Series(0, index=close.index, dtype="float64")).fillna(0)

    # SPY (soft fail)
    try:
        spy_df = fetch_ohlcv_yfinance("SPY", start, end)
        if spy_df.index.tz is None: spy_df.index = spy_df.index.tz_localize("UTC")
        else: spy_df.index = spy_df.index.tz_convert("UTC")
        spy_close = spy_df["Close"].dropna()
    except Exception:
        spy_close = pd.Series(dtype="float64")

    def pct(a, b):
        try: return float(((a / b) - 1.0) * 100.0)
        except Exception: return None

    now_px = float(close.iloc[-1])
    chg_5d = pct(close.iloc[-1], close.iloc[-5])   if len(close) >= 5   else None
    chg_1m = pct(close.iloc[-1], close.iloc[-21])  if len(close) >= 21  else None
    chg_3m = pct(close.iloc[-1], close.iloc[-63])  if len(close) >= 63  else None
    chg_6m = pct(close.iloc[-1], close.iloc[-126]) if len(close) >= 126 else None
    chg_1y = pct(close.iloc[-1], close.iloc[-252]) if len(close) >= 252 else None

    ret = close.pct_change().dropna()
    vol_30 = float(ret.tail(30).std() * math.sqrt(252) * 100.0) if len(ret) >= 30 else None

    ytd = close[close.index >= pd.Timestamp(end.year, 1, 1, tz=end.tzinfo)]
    if len(ytd) >= 2:
        dd = (ytd / ytd.cummax() - 1.0) * 100.0
        max_dd_ytd = float(dd.min())
    else:
        max_dd_ytd = None

    dma20  = float(close.rolling(20).mean().iloc[-1])  if len(close) >= 20  else None
    dma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
    dma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    def dist(p, m): return float((p / m - 1.0) * 100.0) if (p and m) else None
    dist20, dist50, dist200 = dist(now_px, dma20), dist(now_px, dma50), dist(now_px, dma200)

    rsi14 = float(rsi_series(close, 14).iloc[-1]) if len(close) >= 15 else None
    atr14 = float(atr_series(high, low, close, 14).iloc[-1]) if len(close) >= 15 else None

    slope20 = None
    if len(close) >= 20:
        y = close.tail(20).values; x = np.arange(len(y))
        m, _ = np.polyfit(x, y, 1); slope20 = float((m / y[-1]) * 252 * 100.0)

    rel_3m = None
    if len(close) >= 63 and len(spy_close) >= 63:
        rel_3m = pct(close.iloc[-1] / close.iloc[-63], spy_close.iloc[-1] / spy_close.iloc[-63])

    beta = None
    try:
        rr = pd.concat([ret, spy_close.pct_change()], axis=1).dropna(); rr.columns = ["r_t", "r_m"]
        if len(rr) > 30:
            cov = np.cov(rr["r_t"], rr["r_m"])[0, 1]; var_m = np.var(rr["r_m"])
            beta = float(cov / var_m) if var_m > 0 else None
    except Exception:
        pass

    vol20 = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else None
    vol_ratio = float(volume.iloc[-1] / vol20) if (vol20 and vol20 > 0) else None
    sr_high = float(close.tail(20).max()) if len(close) >= 20 else None
    sr_low  = float(close.tail(20).min()) if len(close) >= 20 else None

    trend = []
    if dist20 is not None: trend.append("above 20DMA" if dist20 > 0 else "below 20DMA")
    if dma20 and dma50 and len(close) >= 51:
        prev20 = float(close.rolling(20).mean().iloc[-2]) if len(close) >= 21 else None
        prev50 = float(close.rolling(50).mean().iloc[-2]) if len(close) >= 51 else None
        if prev20 and prev50 and dma20 > dma50 and prev20 <= prev50: trend.append("bullish 20/50 cross")
    technicals = ", ".join(trend) if trend else None

    return {
        "current_price": now_px,
        "chg_5d": chg_5d, "chg_1m": chg_1m, "chg_3m": chg_3m, "chg_6m": chg_6m, "chg_1y": chg_1y,
        "vol_30": vol_30, "max_dd_ytd": max_dd_ytd, "technicals": technicals, "events": None,
        "dma20": dma20, "dma50": dma50, "dma200": dma200,
        "dist20_pct": dist20, "dist50_pct": dist50, "dist200_pct": dist200,
        "rsi14": rsi14, "atr14": atr14, "slope20_pct_annual": slope20,
        "rel_3m_vs_spy_pct": rel_3m, "beta_vs_spy": beta,
        "avg_vol_20d": vol20, "vol_ratio": vol_ratio,
        "sr_high_20d": sr_high, "sr_low_20d": sr_low,
        "start": str(close.index[0]), "end": str(close.index[-1]),
    }

# =========================
# Prompt builder (inline)
# =========================
def build_story_prompt(ticker: str, window_days: int, f: dict) -> str:
    return f"""
You are a trading desk assistant. Write a concise, practical brief for active traders on {ticker}.
Use the JSON schema: {{
  "title": str,
  "tldr": str,
  "narrative": str,
  "positives": [str],
  "risks": [str],
  "suggested_tags": [str]
}}
Focus on risk/return, momentum, volatility, trend, and key levels.
Data (do not invent):
current=${f.get('current_price')}
chg_5d={f.get('chg_5d')}%
chg_1m={f.get('chg_1m')}%
chg_3m={f.get('chg_3m')}%
chg_6m={f.get('chg_6m')}%
chg_1y={f.get('chg_1y')}%
vol30={f.get('vol_30')}%
maxDD_YTD={f.get('max_dd_ytd')}%
RSI14={f.get('rsi14')}
ATR14={f.get('atr14')}
beta_vs_SPY={f.get('beta_vs_spy')}
rel_3m_vs_SPY={f.get('rel_3m_vs_spy_pct')}%
20/50/200DMA={f.get('dma20')},{f.get('dma50')},{f.get('dma200')}
dist_to_20/50/200DMA={f.get('dist20_pct')}%,{f.get('dist50_pct')}%,{f.get('dist200_pct')}%
slope20_annual%={f.get('slope20_pct_annual')}
vol_ratio={f.get('vol_ratio')}
SR_20d_high/low={f.get('sr_high_20d')}/{f.get('sr_low_20d')}
WindowDays={window_days}

Rules:
- Output strictly valid JSON only (no code fences).
- Keep "tldr" to one crisp sentence.
- "narrative" 120–180 words, trader tone.
- Include 3–6 bullet "positives" and 3–6 "risks".
- Tags: lowercase, 3–7 items (e.g., "momentum","rsi","breakout").
"""

# =========================
# LLM story
# =========================
def generate_story_openai(ticker: str, window_days: int, facts: dict) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_story_prompt(ticker, window_days, facts)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=700,
    )
    parsed = json.loads(resp.choices[0].message.content)
    return {
        "title": parsed.get("title") or f"{ticker} trader brief",
        "tldr": parsed.get("tldr") or "",
        "narrative": parsed.get("narrative") or "",
        "positives": parsed.get("positives") or [],
        "risks": parsed.get("risks") or [],
        "suggested_tags": parsed.get("suggested_tags") or [],
    }

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).timestamp(), "build": "v2-trader-inline"}

@app.get("/api/story/{ticker}")
def get_story(ticker: str, window_days: int = 180):
    cached = s3_get_cached_story(ticker, window_days)
    if cached: return cached

    try:
        facts = compute_trader_facts(ticker, window_days)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute facts: {e}")

    try:
        story = generate_story_openai(ticker, window_days, facts)
    except Exception:
        story = {
            "title": f"{ticker} (data-only fallback)",
            "tldr": "OpenAI quota/rate limit hit — showing data-only summary.",
            "narrative": (
                f"{ticker} current ${facts.get('current_price'):.2f}. "
                f"5d {facts.get('chg_5d')}%, 1m {facts.get('chg_1m')}%, "
                f"3m {facts.get('chg_3m')}%, 6m {facts.get('chg_6m')}%. "
                f"Vol30≈{facts.get('vol_30')}%, MaxDD YTD {facts.get('max_dd_ytd')}%. "
                f"RSI14 {facts.get('rsi14')}, ATR14 {facts.get('atr14')}, "
                f"beta {facts.get('beta_vs_spy')}, rel3m vs SPY {facts.get('rel_3m_vs_spy_pct')}%."
            ),
            "positives": [], "risks": [], "suggested_tags": ["fallback","quota"],
        }

    payload = {
        "ticker": ticker.upper(),
        "window_days": int(window_days),
        "facts": facts,
        "story": story,
        "disclaimer": "Educational use only. Not investment advice.",
    }
    try: s3_put_cached_story(ticker, window_days, payload)
    except Exception: pass
    return payload

# =========================
# Lambda entrypoint (matches template Handler: app.handler)
# =========================
handler = Mangum(app)
