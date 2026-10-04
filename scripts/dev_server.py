"""Local preview: serves docs/ and routes /api/* and /health to the Lambda handler.

Run from the repo root:  python scripts/dev_server.py   then open http://localhost:8787
Without S3_CACHE_BUCKET set there is no caching and the brief reports that no key is configured.
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

import app  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(ROOT, "docs"), **kwargs)

    def do_GET(self):
        url = urlsplit(self.path)
        if not (url.path.startswith("/api/") or url.path == "/health"):
            return super().do_GET()
        result = app.handler(
            {"httpMethod": "GET", "path": url.path, "headers": {}, "queryStringParameters": dict(parse_qsl(url.query))},
            None,
        )
        body = result["body"].encode("utf-8")
        self.send_response(result["statusCode"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("MarketPulse preview on http://localhost:8787")
    ThreadingHTTPServer(("127.0.0.1", 8787), Handler).serve_forever()
