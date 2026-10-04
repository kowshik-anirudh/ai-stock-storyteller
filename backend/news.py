# backend/news.py
"""Headlines from public RSS feeds. Only the headline, source, time and link are kept."""
import html
import re
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from defusedxml import ElementTree  # refuses entity expansion and external entities

from market import http_get

MARKET_FEEDS = (
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
)
TICKER_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def _clean(text, limit):
    text = SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", text or ""))).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _source_from_link(link):
    host = urllib.parse.urlparse(link).hostname or ""
    host = host[4:] if host.startswith("www.") else host
    return host


def parse_feed(xml_bytes: bytes, source=None, limit=15):
    try:
        root = ElementTree.fromstring(xml_bytes)
    except Exception:  # malformed or hostile XML: treat as an empty feed
        return []
    items = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title"), 180)
        link = (item.findtext("link") or "").strip()
        # Only plain https links are passed to the browser.
        if not title or not link.startswith("https://"):
            continue
        published = None
        try:
            dt = parsedate_to_datetime(item.findtext("pubDate") or "")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            published = dt.astimezone(timezone.utc).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            pass
        items.append({
            "title": title,
            "link": link,
            "source": source or _source_from_link(link),
            "published": published,
            "summary": _clean(item.findtext("description"), 200),
        })
        if len(items) >= limit:
            break
    return items


def _fetch(url, source=None, limit=15):
    try:
        return parse_feed(http_get(url, timeout=6.0), source, limit)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"feed failed ({source or url[:40]}): {e}")
        return []


def _merge(groups, limit):
    seen, merged = set(), []
    for item in sorted((i for g in groups for i in g), key=lambda i: i["published"] or "", reverse=True):
        key = item["title"].lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged[:limit]


def market_news(limit=24):
    items = _merge([_fetch(url, source, 15) for source, url in MARKET_FEEDS], limit)
    return {"items": items, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def ticker_news(ticker: str, limit=8):
    return _merge([_fetch(TICKER_FEED.format(ticker=urllib.parse.quote(ticker)), None, 15)], limit)
