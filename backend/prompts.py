# backend/prompts.py
def build_story_prompt(ticker: str, window_days: int, facts: dict) -> str:
    """
    Trader-oriented prompt. Requires JSON-only output with specific keys.
    """
    return f"""
You are a disciplined sell-side strategist writing a concise trader's brief.

TICKER: {ticker}
WINDOW_DAYS: {window_days}

DATA (JSON):
{facts}

Write a SHORT report focused on tradable insights. Use the data above (do not invent numbers).
Return VALID JSON ONLY (no markdown) with keys:
- title (string)
- tldr (string, <= 220 chars)
- narrative (string, 3–6 sentences)
- positives (array of short bullets)
- risks (array of short bullets)
- suggested_tags (array of lowercase slugs)

Cover:
- Momentum/Trend: 5d/1m/3m/6m; price vs 20/50/200DMA; slope20%; relative strength vs SPY.
- Volatility/Risk: realized vol(30d), ATR(14) as stop sizing context; maxDD YTD; beta vs SPY.
- Mean reversion: RSI-14 zones and implications.
- Volume: 20d average and latest/avg ratio (vol_ratio).
- Levels: 20d support/resistance (sr_low_20d, sr_high_20d) — not guarantees.
- Catalysts: earnings_event if present.
- Tone: neutral, evidence-based, no advice or targets. <= 1200 chars total.
- Include 4–8 tags like ["momentum","rsi","atr","trend","beta","levels","liquidity"].
"""
