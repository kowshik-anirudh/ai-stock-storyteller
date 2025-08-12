# 📈 MarketPulse – AI-Powered Stock Insights

**MarketPulse** turns real-time market data into actionable, AI-generated insights — instantly.  
It combines **live stock data**, **technical analysis**, and **AI-powered summaries** to give traders, investors, and analysts an edge.

---

## 🚀 Why MarketPulse?
Most trading dashboards just show numbers.  
**MarketPulse** explains what’s happening — in plain language — and backs it with interactive visuals.

---

## 🌟 Features

### 🔍 AI-Powered Stock Narratives
- Uses OpenAI’s GPT models to generate deep, context-rich stock summaries.
- Covers technical indicators, news sentiment, and trend analysis.

### 📊 Real-Time Stock Data
- Prominent **current price** display.
- Automatic **green/red** coloring for gains/losses.

### ℹ️ Key Facts with Tooltips
- Displays market cap, P/E ratio, volume, volatility, and more.
- Hover over each stat to learn what it means.

### 📈 Interactive Price Charts
- Shows performance trends for customizable timeframes.
- Candlestick or line charts with green/red coloring.

### ✏️ Autocomplete Stock Search
- Start typing a symbol — get instant matching suggestions.
- Powered by a preloaded `tickers.json` for speed.

---

## 🛠 Architecture
<img width="1589" height="964" alt="image" src="https://github.com/user-attachments/assets/ee972441-ccc0-4667-8a18-b24a6404497d" />


**Flow:**
1. User searches stock on frontend.
2. API Gateway routes request to Lambda.
3. Lambda fetches live data, computes technical indicators, and generates AI summary.
4. Response is sent back → displayed with charts & key facts.

---

Landing Page: <img width="1374" height="888" alt="Landing page" src="https://github.com/user-attachments/assets/2fc0511f-f30f-44e0-9370-5cb4e368609a" />


Key Facts Panel: <img width="530" height="551" alt="Key Facts" src="https://github.com/user-attachments/assets/90ce67c4-4d3d-4290-939f-c094b4f7a491" />


Interactive Chart: <img width="1233" height="575" alt="Interactive Chart" src="https://github.com/user-attachments/assets/3f0d5913-66bf-4408-a731-416cbc4ff86e" />


AI Summary: <img width="546" height="884" alt="AI Summary" src="https://github.com/user-attachments/assets/fc728cca-8618-48bc-926c-721c8070bafe" />




