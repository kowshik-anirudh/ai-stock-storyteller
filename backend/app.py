# ---------- AI Stock Storyteller (Lambda/FastAPI) ----------
import os, math, json, time
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

# If you split prompts to a separate file, import them; else inline simple defaults:
try:
    from prompts import SYSTEM_PROMPT, USER_TEMPLATE
except Exception:
    SYSTEM_PROMPT = (
        "You are a helpful markets narrator. Produce concise JSON with keys: "
        "title, tldr, narrative, positives[], risks[], suggested_tags[]. No advice."
    )
    USER_TEMPLATE = (
        "Ticker: {ticker}\nWindow: {window_days} days\n"
        "Current: {current_price}\n"
        "5d: {chg_5d}% | 1m: {chg_1m}% | 3m: {chg_3m}% | 1y: {chg_1y}%\n"
        "Vol(30d): {vol_30}% | MaxDD YTD: {max_dd_ytd}%\n"
        "Technicals: {technicals}\n"
        "Events: {events}\n"
        "Write an engaging, neutral summary for a broad audience."
    )

APP_BUILD = "linux-rebuild-2"

# -------- ENV --------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

# -------- FastAPI App --------
app = FastAPI(title="AI Stock Storyteller", version="1.0.1")

allowed = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple in-memory cache per container
CACHE = {}
CACHE_TTL = 60 * 10  # 10 minutes


class StoryRequest(BaseModel):
    ticker: str
    window_days: int = 180


def _pct_change(series, days):
    try:
        if len(series) < days + 1:
            return float("nan")
        return (series[-1] / series[-(days + 1)] - 1.0) * 100.0
    except Exception:
        return float("nan")


def compute_metrics(ticker: str, window_days: int):
    """
    Fetches from Yahoo's chart API (more reliable on Lambda).
    Lazy imports keep /health lightweight.
    """
    import pandas as pd
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "2y", "interval": "1d", "includeAdjustedClose": "true"}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Yahoo chart HTTP {r.status_code}")

    try:
        j = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Bad JSON from Yahoo chart")

    result = (j.get("chart") or {}).get("result") or []
    if not result:
        raise HTTPException(status_code=404, detail=f"No data for ticker {ticker}")

    res = result[0]
    timestamps = res.get("timestamp") or []
    ind = ((res.get("indicators") or {}).get("adjclose") or [{}])[0]
    closes = ind.get("adjclose") or []

    if not timestamps or not closes or len(timestamps) != len(closes):
        ind_quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        closes = ind_quote.get("close") or []
        if not timestamps or not closes or len(timestamps) != len(closes):
            raise HTTPException(status_code=404, detail=f"No data for ticker {ticker}")

    idx = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True).dt.tz_convert("UTC")
    close = pd.Series(closes, index=idx, dtype="float64").dropna()

    end = pd.Timestamp(datetime.now(timezone.utc))
    cutoff = end - pd.Timedelta(days=max(window_days, 200))
    close = close[close.index >= cutoff]

    if close.empty or len(close) < 30:
        raise HTTPException(status_code=400, detail="Insufficient price history")

    chg_5d = _pct_change(close.values, 5)
    chg_1m = _pct_change(close.values, 21)
    chg_3m = _pct_change(close.values, 63)
    chg_1y = _pct_change(close.values, 252)

    vol_30 = float(pd.Series(close).pct_change().rolling(30).std().iloc[-1] * 100.0)

    y = end.year
    ytd = close[close.index.year == y]
    if len(ytd) > 0:
        peak = ytd.cummax()
        dd = (ytd / peak - 1.0) * 100.0
        max_dd_ytd = float(dd.min())
    else:
        max_dd_ytd = float("nan")

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else float("nan")
    price = float(close.iloc[-1])

    trend_parts = []
    if not math.isnan(ma20):
        trend_parts.append(f"price {'above' if price > ma20 else 'below'} 20DMA")
    if not math.isnan(ma50):
        trend_parts.append(f"{'bullish' if ma20 > ma50 else 'bearish'} 20/50 cross")
    if not math.isnan(ma200):
        trend_parts.append(f"price {'above' if price > ma200 else 'below'} 200DMA")
    technicals = ", ".join(trend_parts) if trend_parts else "insufficient data"

    events = []
    if not math.isnan(chg_5d) and abs(chg_5d) > 8:
        events.append("sharp weekly move")
    if not math.isnan(max_dd_ytd) and max_dd_ytd < -20:
        events.append("deep YTD drawdown")
    if not math.isnan(chg_1m) and chg_1m > 10:
        events.append("momentum uptick 1M")
    if not events:
        events.append("no obvious anomalies")

    def nz(x):
        return None if (isinstance(x, float) and math.isnan(x)) else float(x)

    return {
        "current_price": price,
        "chg_5d": nz(chg_5d),
        "chg_1m": nz(chg_1m),
        "chg_3m": nz(chg_3m),
        "chg_1y": nz(chg_1y),
        "vol_30": nz(vol_30),
        "max_dd_ytd": nz(max_dd_ytd),
        "technicals": technicals,
        "events": ", ".join(events),
    }


def generate_story_openai(ticker: str, window_days: int, facts: dict):
    """
    Uses OpenAI SDK (v1.x). Handles rate limits/quota politely.
    """
    from openai import OpenAI
    import json as _json
    from fastapi import HTTPException
    from openai import RateLimitError, APIError

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    client = OpenAI(api_key=OPENAI_API_KEY)

    user = USER_TEMPLATE.format(
        ticker=ticker.upper(),
        window_days=window_days,
        current_price=f"${facts['current_price']:.2f}",
        chg_5d=facts["chg_5d"] if facts["chg_5d"] is not None else float("nan"),
        chg_1m=facts["chg_1m"] if facts["chg_1m"] is not None else float("nan"),
        chg_3m=facts["chg_3m"] if facts["chg_3m"] is not None else float("nan"),
        chg_1y=facts["chg_1y"] if facts["chg_1y"] is not None else float("nan"),
        vol_30=facts["vol_30"] if facts["vol_30"] is not None else float("nan"),
        max_dd_ytd=facts["max_dd_ytd"] if facts["max_dd_ytd"] is not None else float("nan"),
        technicals=facts["technicals"],
        events=facts["events"],
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.6,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except RateLimitError as e:
        # Friendly fallback payload
        return {
            "title": f"{ticker.upper()} story (temporary fallback)",
            "tldr": "OpenAI quota/rate limit hit — showing a concise data-only summary.",
            "narrative": (
                f"{ticker.upper()} current: ${facts['current_price']:.2f}. "
                f"5d {facts['chg_5d']}%, 1m {facts['chg_1m']}%, 3m {facts['chg_3m']}%, 1y {facts['chg_1y']}%. "
                f"Vol(30d) {facts['vol_30']}%, MaxDD YTD {facts['max_dd_ytd']}%. "
                f"Technicals: {facts['technicals']}. Events: {facts['events']}."
            ),
            "positives": [],
            "risks": [],
            "suggested_tags": ["fallback", "quota"],
        }
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e}")
    except Exception as e:
        # If model ever returns non-JSON / other exception
        return {
            "title": "Stock Story",
            "tldr": "",
            "narrative": str(e),
            "positives": [],
            "risks": [],
            "suggested_tags": [],
        }


@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "build": APP_BUILD}


@app.get("/api/story/{ticker}")
def get_story(ticker: str, window_days: int = 180):
    key = f"{ticker.upper()}:{window_days}"
    now = time.time()
    if key in CACHE and now - CACHE[key]["ts"] < CACHE_TTL:
        return CACHE[key]["data"]

    facts = compute_metrics(ticker, window_days)
    story = generate_story_openai(ticker, window_days, facts)
    payload = {
        "ticker": ticker.upper(),
        "window_days": window_days,
        "facts": facts,
        "story": story,
        "disclaimer": "Educational use only. Not investment advice.",
    }
    CACHE[key] = {"ts": now, "data": payload}
    return payload


# Lambda entrypoint
handler = Mangum(app)
