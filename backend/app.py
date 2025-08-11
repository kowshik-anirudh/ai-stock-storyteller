import os
import math
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mangum import Mangum
from openai import OpenAI

from prompts import SYSTEM_PROMPT, USER_TEMPLATE

# -------- ENV & Clients --------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")  # set to your site URL later

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------- FastAPI App --------
app = FastAPI(title="AI Stock Storyteller", version="1.0.0")

# CORS
allowed = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache to reduce cost/latency
CACHE = {}
CACHE_TTL = 60 * 10  # 10 min


class StoryRequest(BaseModel):
    ticker: str
    window_days: int = 180


def pct_change(series, days):
    try:
        if len(series) < days + 1:
            return np.nan
        return (series[-1] / series[-(days+1)] - 1.0) * 100.0
    except Exception:
        return np.nan


def compute_metrics(ticker: str, window_days: int):
    end = datetime.utcnow()
    start = end - timedelta(days=max(window_days, 200))
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)

    if data is None or data.empty:
        raise HTTPException(status_code=404, detail=f"No data for ticker {ticker}")

    close = data["Close"].dropna()
    if len(close) < 30:
        raise HTTPException(status_code=400, detail="Insufficient price history")

    chg_5d = pct_change(close.values, 5)
    chg_1m = pct_change(close.values, 21)
    chg_3m = pct_change(close.values, 63)
    chg_1y = pct_change(close.values, 252)

    vol_30 = float(pd.Series(close).pct_change().rolling(30).std().iloc[-1] * 100.0)

    ytd = close[close.index.year == end.year]
    if len(ytd) > 0:
        peak = ytd.cummax()
        dd = (ytd / peak - 1.0) * 100.0
        max_dd_ytd = float(dd.min())
    else:
        max_dd_ytd = np.nan

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

    price = close.iloc[-1]

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

    return {
        "current_price": float(price),
        "chg_5d": float(chg_5d) if not math.isnan(chg_5d) else None,
        "chg_1m": float(chg_1m) if not math.isnan(chg_1m) else None,
        "chg_3m": float(chg_3m) if not math.isnan(chg_3m) else None,
        "chg_1y": float(chg_1y) if not math.isnan(chg_1y) else None,
        "vol_30": float(vol_30) if not math.isnan(vol_30) else None,
        "max_dd_ytd": float(max_dd_ytd) if not math.isnan(max_dd_ytd) else None,
        "technicals": technicals,
        "events": ", ".join(events),
    }


def generate_story_openai(ticker: str, window_days: int, facts: dict):
    user = USER_TEMPLATE.format(
        ticker=ticker.upper(),
        window_days=window_days,
        current_price=f"${facts['current_price']:.2f}",
        chg_5d=facts["chg_5d"] or float("nan"),
        chg_1m=facts["chg_1m"] or float("nan"),
        chg_3m=facts["chg_3m"] or float("nan"),
        chg_1y=facts["chg_1y"] or float("nan"),
        vol_30=facts["vol_30"] or float("nan"),
        max_dd_ytd=facts["max_dd_ytd"] or float("nan"),
        technicals=facts["technicals"],
        events=facts["events"],
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.6,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=500
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # fallback if model didn’t return perfect JSON
        return {
            "title": "Stock Story",
            "tldr": "",
            "narrative": content,
            "positives": [],
            "risks": [],
            "suggested_tags": []
        }


@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}


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
        "disclaimer": "Educational use only. Not investment advice."
    }
    CACHE[key] = {"ts": now, "data": payload}
    return payload


# Lambda handler (API Gateway)
handler = Mangum(app)
