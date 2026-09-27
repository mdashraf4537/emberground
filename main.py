"""
Emberground - Step 1: Ingest Router + Auto-Title
Render FastAPI backend. Called by n8n Webhook -> Switch -> HTTP Request.

n8n sends POST /ingest with JSON:
  { "url": "...", "text": "...optional raw chat/gmail/slack...",
    "intent": "Exam revision", "subject": "Core Concepts" }

Returns:
  { "status": "routed", "detected_type": "youtube|web|pdf|chat",
    "title": "auto-generated title", "source": url, ... }
"""
import os
import re
from typing import Optional
from urllib.parse import urlparse, unquote

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Emberground - Ingest Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None          # raw chat / gmail / slack paste from n8n
    content: Optional[str] = None       # alias for text (n8n flexibility)
    intent: str = "General Reference"
    subject: str = "General"
    source_hint: Optional[str] = None   # n8n Switch can pass "youtube"|"web"|"pdf"|"chat"


class IngestResponse(BaseModel):
    status: str
    detected_type: str
    title: str
    source: str
    intent: str
    subject: str


# ---------- Step 1a: Type detection / routing ----------

def detect_source_type(url: Optional[str], text: Optional[str], hint: Optional[str]) -> str:
    """Route based on format. n8n hint wins if provided, else infer from url/text."""
    if hint in ("youtube", "web", "pdf", "chat", "gmail", "slack"):
        # normalize gmail/slack -> chat for Step 1 (split out in Step 2)
        if hint in ("gmail", "slack"):
            return "chat"
        return hint
    if url:
        u = url.lower().strip()
        if "youtube.com" in u or "youtu.be" in u:
            return "youtube"
        # pdf: extension OR google-drive / dropbox pdf links handled in Step 2
        path = urlparse(u).path.lower()
        if path.endswith(".pdf") or "mimetype=pdf" in u or ".pdf?" in u:
            return "pdf"
        if u.startswith("http"):
            return "web"
    if text and text.strip():
        return "chat"
    return "web"


# ---------- Step 1b: Auto-title per type ----------

def _clean(s: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:max_len] if len(s) > max_len else s or "Untitled Resource"


def title_for_youtube(url: str) -> str:
    """Use YouTube oEmbed (no API key needed) -> real video title."""
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=8,
        )
        if r.status_code == 200:
            return _clean(r.json().get("title", ""))
    except Exception:
        pass
    # fallback: video id
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([\w-]{6,})", url)
    return f"YouTube Video {m.group(1)}" if m else "YouTube Video"


def title_for_web(url: str) -> str:
    """Fetch page <title> tag. Lightweight, no bs4 needed."""
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.IGNORECASE | re.DOTALL)
            if m:
                return _clean(m.group(1))
    except Exception:
        pass
    # fallback: domain + path
    try:
        p = urlparse(url)
        return _clean(f"{p.netloc}{p.path}".strip("/") or url)
    except Exception:
        return "Web Article"


def title_for_pdf(url: Optional[str]) -> str:
    if not url:
        return "PDF Document"
    try:
        name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        name = re.sub(r"\.pdf.*$", "", name, flags=re.IGNORECASE).replace("-", " ").replace("_", " ")
        return _clean(name) if name else "PDF Document"
    except Exception:
        return "PDF Document"


def title_for_chat(text: str) -> str:
    """First meaningful line, max ~60 chars."""
    for line in (text or "").splitlines():
        line = line.strip()
        if len(line) >= 8:  # skip tiny greetings
            return _clean(line, max_len=80)
    return _clean(text or "", max_len=80)


def auto_title(detected: str, url: Optional[str], text: Optional[str]) -> str:
    if detected == "youtube" and url:
        return title_for_youtube(url)
    if detected == "pdf":
        return title_for_pdf(url)
    if detected == "web" and url:
        return title_for_web(url)
    if detected == "chat":
        return title_for_chat(text or "")
    return "Untitled Resource"


# ---------- Routes ----------

@app.get("/")
def root():
    return {"service": "emberground", "step": 1, "routes": ["GET /health", "POST /ingest"]}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    raw_text = req.text or req.content or ""
    url = (req.url or "").strip()

    detected = detect_source_type(url, raw_text, req.source_hint)
    title = auto_title(detected, url or None, raw_text)

    # Step 2 will: fetch full content per type, chunk, write to Breeth.
    # Step 3 will: Groq/OpenAI synthesis + ElevenLabs. Intentionally not here yet.
    return IngestResponse(
        status="routed",
        detected_type=detected,
        title=title,
        source=url or "pasted-text",
        intent=req.intent,
        subject=req.subject,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
