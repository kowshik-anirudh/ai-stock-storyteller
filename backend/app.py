# backend/app.py
"""Lambda entry point: routes API Gateway requests and the scheduled news refresh."""
import json
import os
from datetime import datetime, timezone

import brief
import market
import news
import store

ALLOWED_ORIGINS = {o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}
DAILY_BRIEF_LIMIT = int(os.getenv("DAILY_BRIEF_LIMIT", "10"))
DATA_TTL = int(os.getenv("DATA_TTL_SECONDS", "1800"))
NEWS_TTL = int(os.getenv("NEWS_TTL_SECONDS", "1800"))
BRIEF_TTL = int(os.getenv("BRIEF_TTL_SECONDS", "21600"))
WINDOWS = (30, 90, 180, 365)
DISCLAIMER = "Educational use only. Not investment advice."


def _response(status, body, origin=None, max_age=0):
    headers = {
        "Content-Type": "application/json",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": f"public, max-age={max_age}" if max_age else "no-store",
        "Vary": "Origin",
    }
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
    return {"statusCode": status, "headers": headers, "body": json.dumps(body, separators=(",", ":"))}


def _analysis(ticker):
    cached = store.get_json(f"data/{ticker}.json", DATA_TTL)
    if cached:
        return cached
    data = market.analyse(ticker)
    data["news"] = news.ticker_news(ticker)
    store.put_json(f"data/{ticker}.json", data)
    return data


def get_quote(ticker, params):
    try:
        window = int(params.get("window_days", 180))
    except ValueError:
        window = 180
    window = window if window in WINDOWS else 180
    data = _analysis(ticker)
    return {
        "ticker": data["ticker"], "name": data["name"], "currency": data["currency"],
        "exchange": data["exchange"], "as_of": data["as_of"], "window_days": window,
        "facts": data["facts"], "chart": market.slice_window(data["series"], window),
        "news": data["news"], "disclaimer": DISCLAIMER,
    }


def get_brief(ticker):
    cached = store.get_json(f"brief/{ticker}.json", BRIEF_TTL)
    if cached:
        return {**cached, "cached": True}
    data = _analysis(ticker)  # also confirms the ticker exists before any model call
    api_key = store.get_api_key()
    if not api_key:
        return {"ticker": ticker, "status": "no_key", "story": None}
    if not store.reserve_brief_slot(DAILY_BRIEF_LIMIT):
        return {"ticker": ticker, "status": "limit", "story": None}
    story, status = brief.generate(api_key, ticker, data["name"], data["as_of"], data["facts"], data["news"][:6])
    payload = {
        "ticker": ticker, "status": status, "story": story,
        "as_of": data["as_of"], "headlines_used": len(data["news"][:6]) if story else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }
    if story:
        store.put_json(f"brief/{ticker}.json", payload)
    return {**payload, "cached": False}


def get_news():
    cached = store.get_json("news/market.json", NEWS_TTL)
    if cached:
        return cached
    return refresh_news()


def refresh_news():
    payload = news.market_news()
    if payload["items"]:
        store.put_json("news/market.json", payload)
    return payload


def handler(event, context):
    # EventBridge schedule: keep the market headlines fresh.
    if event.get("task") == "refresh_news":
        payload = refresh_news()
        print(f"news refreshed: {len(payload['items'])} items")
        return {"ok": True, "items": len(payload["items"])}

    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    origin = headers.get("origin")
    path = (event.get("path") or "").rstrip("/")
    params = event.get("queryStringParameters") or {}
    if event.get("httpMethod") != "GET":
        return _response(405, {"error": "Only GET is supported."}, origin)

    try:
        if path == "/health":
            return _response(200, {"ok": True, "build": "marketpulse-v4"}, origin)
        if path == "/api/news":
            return _response(200, get_news(), origin, max_age=300)
        for prefix, fn in (("/api/quote/", lambda t: get_quote(t, params)), ("/api/brief/", get_brief)):
            if path.startswith(prefix):
                ticker = market.valid_ticker(path[len(prefix):])
                if not ticker:
                    return _response(400, {"error": "That does not look like a ticker symbol."}, origin)
                return _response(200, fn(ticker), origin, max_age=300)
        return _response(404, {"error": "Not found."}, origin)
    except market.TickerNotFound:
        return _response(404, {"error": "No price data found for that symbol."}, origin)
    except Exception as e:  # never leak internals to the client
        print(f"unhandled error on {path}: {type(e).__name__}: {e}")
        return _response(502, {"error": "The data source did not respond. Try again in a minute."}, origin)
