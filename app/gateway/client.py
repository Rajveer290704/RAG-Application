import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from groq import Groq

from app.config import settings


# Production gateway config:
#   - Fallback: primary @rag/GROQ_MODEL → @brag/GROQ_GUARD_MODEL on failure
#   - Cache: simple mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.GROQ_GUARD_MODEL}"}},
    ]
}

try:
    _raw_portkey_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=GATEWAY_CONFIG
    ) if settings.PORTKEY_API_KEY else None
except Exception as e:
    logfire.warning(f"Failed to initialize Portkey client: {e}")
    _raw_portkey_client = None

_groq_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None


class ResilientPortkeyClient:
    """
    Wrapper around Portkey client with seamless Groq fallback
    if Portkey encounters errors (e.g., inline config restrictions or downtime).
    """
    def __init__(self, portkey_client, groq_client):
        self._portkey = portkey_client
        self._groq = groq_client
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.completions = self.Completions(outer)

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, *args, **kwargs):
                if self.outer._portkey:
                    try:
                        return self.outer._portkey.chat.completions.create(*args, **kwargs)
                    except Exception as e:
                        logfire.warning(f"⚠️ Portkey Gateway call failed ({e}). Falling back directly to Groq.")
                
                # Direct Groq execution
                model = kwargs.pop("model", None) or settings.GROQ_MODEL
                return self.outer._groq.chat.completions.create(model=model, *args, **kwargs)


portkey_client = ResilientPortkeyClient(_raw_portkey_client, _groq_client)


def get_langchain_llm(feature: str = "rag"):
    """
    Returns an LLM for LangChain nodes:
    Attempts Portkey-backed ChatOpenAI first, with automatic fallback to ChatGroq.
    """
    target_model = settings.GROQ_GUARD_MODEL if feature == "planner" else settings.GROQ_MODEL
    fallback_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=target_model,
        temperature=0
    )

    if not settings.PORTKEY_API_KEY:
        return fallback_llm

    try:
        portkey_llm = ChatOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            model=f"@{settings.GROQ_SLUG}/{target_model}",
            temperature=0,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=GATEWAY_CONFIG,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production"
                }
            )
        )
        return portkey_llm.with_fallbacks([fallback_llm])
    except Exception as e:
        logfire.warning(f"Portkey LLM creation failed ({e}), using direct ChatGroq.")
        return fallback_llm


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"