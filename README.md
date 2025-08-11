# 📈 AI Stock Storyteller

Turn raw market data into **engaging, human-readable stock narratives** powered by **OpenAI GPT** and **AWS Serverless**.  
Just type a ticker (e.g., `NVDA`) — get a **fact-rich story** with trends, positives, and risks.

🔗 **Live Demo (Frontend)**: [GitHub Pages Link](https://kowshik-anirudh.github.io/ai-stock-storyteller/)  
🔗 **API Base URL**: `https://<your-api-id>.execute-api.us-west-2.amazonaws.com/Prod`

---

## 🚀 Features
- **Serverless & Scalable** — AWS Lambda + API Gateway, deployable in minutes via AWS SAM.
- **Fast & Cost-Effective** — S3-based caching avoids unnecessary OpenAI calls.
- **Rich Market Insights** — Technicals, returns, volatility, and event highlights.
- **AI Narrative Generation** — GPT-4o-mini transforms stats into a concise story.
- **CORS Secured** — Restricted to your frontend domain.
- **Graceful Fallback** — If API quota is hit, still returns factual summaries.

---

## 🏗 Architecture Overview

```mermaid
flowchart LR
  A[User Browser\nGitHub Pages SPA] -->|HTTPS fetch| B(API Gateway\nREST /health, /api/story)
  B --> C(Lambda\nFastAPI + Mangum\nPython 3.11)
  C -->|GET/PUT| D[(S3 Bucket\nCache: stories/TICKER_WINDOW.json)]
  C -->|Prompt + facts| E[(OpenAI API\nModel: gpt-4o-mini)]
  C --> F(CloudWatch Logs)

  subgraph AWS
    B
    C
    D
    F
  end

  classDef ext fill:#111,stroke:#6cf,stroke-width:1px,color:#e8eef5;
  class E ext;
