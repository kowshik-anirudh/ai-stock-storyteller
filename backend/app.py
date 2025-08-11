# backend/app.py
import os
import json
import math
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from openai import OpenAI

from prompts import build_story_prompt  # local module

# ---- Env / config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
S3_CACHE_BUCKET = os.getenv("S3_CACHE_BUCKET", "")
S3_CACHE_TTL_SECONDS = int(os.getenv("S3_CACHE_TTL_SECONDS", "1800"))  # 30 min default

s3 = boto3.client("s3")

app = FastAPI(title="AI Stock Storyteller", version="2.0-trader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Utils: S3 cache ----
def _cache_key(ticker: str, window_days: int) -> str:
    return f"stories/{ticker.upper()}_{int(window_days)}.json"

def s3_get_cached_story(ticker: str, window_days: int):
    if not S3_CACHE_BUCKET:
        return None
    key = _cache_key(ticker, window_days)
    try:
        obj = s3.get_object(Bucket=S3_CACHE_BUCKET, Key=key)
        body = obj["Body"].read()
        cached = json.loads(body)
        # TTL check
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
    key = _cache_key(ticker, window_days)
    doc = {
        "_cached_at": datetime.now(timezone.utc).timestamp(),
        "payload": payload,
    }
    s3.put_object(
        Bucket=S3_CACHE_BUCKET,
        Key=key,
        Body=json.dumps(doc).encode("utf-8"),
        ContentType="application/json",
    )

# ---- Data & features for traders ----
def compute_trader_facts(ticker: str, window_days: int = 180) -> dict:
    """
    Pull OHLCV for ticker & SPY. Compute returns, vol, drawdown,
    DMAs, RSI, ATR, slope, relative strength, beta, volume stats,
    simple support/resistance, and earnings (best-effort).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(window_days + 60, 260))

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for ticker {ticker}")

    close = df["Close"].dropna().copy()
    high = df.get("High", close)
    low = df.get("Low", close)
    volume = df.get("Volume", pd.Series(index=close.index, dtype="float64")).fillna(0)

    # SPY as market proxy
    spy_close = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)["Close"].dropna()

    # window slice
    win = close.last(f"{window_days}D")
    if win.empty:
        win = close.tail(window_days)

    # helpers
    def pct(a, b):
        try:
            return float(((a / b) - 1.0) * 100.0)
        except Exception:
            return None

    now_px = float(close.iloc[-1])
    chg_5d = pct(close.iloc[-1], close.iloc[-5]) if len(close) >= 5 else None
    chg_1m = pct(close.iloc[-1], close.iloc[-21]) if len(close) >= 21 else None
    chg_3m = pct(close.iloc[-1], close.iloc[-63]) if len(close) >= 63 else None
    chg_6m = pct(close.iloc[-1], close.iloc[-126]) if len(close) >= 126 else None
    chg_1y = pct(close.iloc[-1], close.iloc[-252]) if len(close) >= 252 else None

    # realized vol (30d, annualized)
    ret = close.pct_change().dropna()
    vol_30 = float(ret.tail(30).std() * math.sqrt(252) * 100.0) if len(ret) >= 30 else None

    # max drawdown YTD
    ytd = close[close.index >= pd.Timestamp(end.year, 1, 1, tz=end.tzinfo)]
    if len(ytd) >= 2:
        roll_max = ytd.cummax()
        dd = (ytd / roll_max - 1.0) * 100.0
        max_dd_ytd = float(dd.min())
    else:
        max_dd_ytd = None

    # DMAs
    dma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    dma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    dma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    def dist(p, m): return float((p / m - 1.0) * 100.0) if (p and m) else None
    dist20 = dist(now_px, dma20)
    dist50 = dist(now_px, dma50)
    dist200 = dist(now_px, dma200)

    # RSI & ATR
    rsi14 = float(ta.rsi(close, length=14).iloc[-1]) if len(close) >= 15 else None
    try:
        atr14 = float(ta.atr(high, low, close, length=14).iloc[-1])
    except Exception:
        atr14 = None

    # slope of last 20 closes (annualized %/yr proxy)
    slope20 = None
    if len(close) >= 20:
        y = close.tail(20).values
        x = np.arange(len(y))
        m, _ = np.polyfit(x, y, 1)
        slope20 = float((m / y[-1]) * 252 * 100.0)

    # relative strength vs SPY (3m)
    rel_3m = None
    if len(close) >= 63 and len(spy_close) >= 63:
        rel_3m = pct(close.iloc[-1] / close.iloc[-63], spy_close.iloc[-1] / spy_close.iloc[-63])

    # beta vs SPY
    beta = None
    try:
        rr = pd.concat([ret, spy_close.pct_change()], axis=1).dropna()
        rr.columns = ["r_t", "r_m"]
        if len(rr) > 30:
            cov = np.cov(rr["r_t"], rr["r_m"])[0, 1]
            var_m = np.var(rr["r_m"])
            beta = float(cov / var_m) if var_m > 0 else None
    except Exception:
        pass

    # volume stats
    vol20 = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else None
    vol_ratio = float(volume.iloc[-1] / vol20) if (vol20 and vol20 > 0) else None

    # simple support/resistance (20d)
    sr_high = float(close.tail(20).max()) if len(close) >= 20 else None
    sr_low = float(close.tail(20).min()) if len(close) >= 20 else None

    # earnings date best-effort
    earnings = None
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            dt = cal.loc["Earnings Date"].dropna()
            if len(dt):
                earnings = str(dt.iloc[0])
    except Exception:
        pass

    # simple technical blurb
    trend_bits = []
    if dist20 is not None:
        trend_bits.append("above 20DMA" if dist20 > 0 else "below 20DMA")
    if dma20 and dma50 and len(close) >= 51:
        prev20 = float(close.rolling(20).mean().iloc[-2]) if len(close) >= 21 else None
        prev50 = float(close.rolling(50).mean().iloc[-2]) if len(close) >= 51 else None
        if prev20 and prev50 and dma20 > dma50 and prev20 <= prev50:
            trend_bits.append("bullish 20/50 cross")
    technicals = ", ".join(trend_bits) if trend_bits else None

    return {
        # existing keys (compat with v1 UI)
        "current_price": now_px,
        "chg_5d": chg_5d, "chg_1m": chg_1m, "chg_3m": chg_3m, "chg_1y": chg_1y,
        "vol_30": vol_30, "max_dd_ytd": max_dd_ytd,
        "technicals": technicals, "events": None,

        # trader extras
        "chg_6m": chg_6m,
        "dma20": dma20, "dma50": dma50, "dma200": dma200,
        "dist20_pct": dist20, "dist50_pct": dist50, "dist200_pct": dist200,
        "rsi14": rsi14, "atr14": atr14, "slope20_pct_annual": slope20,
        "rel_3m_vs_spy_pct": rel_3m, "beta_vs_spy": beta,
        "avg_vol_20d": vol20, "vol_ratio": vol_ratio,
        "sr_high_20d": sr_high, "sr_low_20d": sr_low,
        "earnings_event": earnings,
        "start": str(close.index[0]),
        "end": str(close.index[-1]),
    }

# ---- OpenAI story ----
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

# ---- Routes ----
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).timestamp(), "build": "v2-trader"}

@app.get("/api/story/{ticker}")
def get_story(ticker: str, window_days: int = 180):
    # 1) cache
    cached = s3_get_cached_story(ticker, window_days)
    if cached:
        return cached

    # 2) build facts
    try:
        facts = compute_trader_facts(ticker, window_days)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute facts: {e}")

    # 3) LLM -> story (fallback on quota)
    try:
        story = generate_story_openai(ticker, window_days, facts)
    except Exception as e:
        story = {
            "title": f"{ticker} (data-only fallback)",
            "tldr": "OpenAI quota/rate limit hit — showing data-only summary.",
            "narrative": (
                f"{ticker} current: ${facts.get('current_price'):.2f} "
                f"5d {facts.get('chg_5d')}%, 1m {facts.get('chg_1m')}%, "
                f"3m {facts.get('chg_3m')}%, 6m {facts.get('chg_6m')}%. "
                f"Vol30≈{facts.get('vol_30')}%, MaxDD YTD {facts.get('max_dd_ytd')}%. "
                f"RSI14 {facts.get('rsi14')}, ATR14 {facts.get('atr14')}, "
                f"beta {facts.get('beta_vs_spy')}, rel3m vs SPY {facts.get('rel_3m_vs_spy_pct')}%."
            ),
            "positives": [],
            "risks": [],
            "suggested_tags": ["fallback","quota"],
        }

    payload = {
        "ticker": ticker.upper(),
        "window_days": int(window_days),
        "facts": facts,
        "story": story,
        "disclaimer": "Educational use only. Not investment advice.",
    }

    # 4) cache it
    try:
        s3_put_cached_story(ticker, window_days, payload)
    except Exception:
        pass

    return payload

# Lambda handler
lambda_handler = Mangum(app)
