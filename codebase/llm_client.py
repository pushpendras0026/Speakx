"""
LLM Client  –  Project Aurora (SpeakX)
========================================
Centralised Gemini API client configuration.

All pipeline stages that need an LLM should import from here:

    from llm_client import client, OLLAMA_MODEL

Google Gemini exposes an OpenAI-compatible REST API at
https://generativelanguage.googleapis.com/v1beta/openai/
We use the official `openai` SDK so that all downstream calling code
remains identical.

Override defaults via environment variables (or a .env file):
    GEMINI_API_KEY    (required) your Google AI Studio API key (AIza...)
    GEMINI_MODEL      default: gemini-2.5-flash
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Always load .env from the codebase/ directory (where this file lives)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Configuration ──────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL:   str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.  Add your Google AI Studio key to .env:\n"
        "  GEMINI_API_KEY=AIza..."
    )

# ── Single shared client ───────────────────────────────────────────────────────
client: OpenAI = OpenAI(
    base_url=GEMINI_BASE_URL,
    api_key=GEMINI_API_KEY,
)

# Backward-compatible alias so downstream files don't need any changes
OLLAMA_MODEL: str = GEMINI_MODEL

logger = logging.getLogger("aurora.llm_client")
logger.info("Gemini client ready  |  model=%s", GEMINI_MODEL)
