import os
import time
import logfire
from flashrank import Ranker, RerankRequest

# ============================================================
# Configuration
# ============================================================

# Set FLASHRANK_ENABLED=false in Render temporarily
# to test the RAG pipeline without FlashRank.
FLASHRANK_ENABLED = os.getenv(
    "FLASHRANK_ENABLED",
    "true"
).lower() == "true"

FLASHRANK_CACHE_DIR = "/tmp/flashrank"

# Singleton Ranker instance
_ranker = None


# ============================================================
# FlashRank Initialization
# ============================================================
def initialize_ranker():
    """
    Initialize FlashRank once during application startup.
    """
    return _get_ranker()
def _get_ranker():
    global _ranker

    if not FLASHRANK_ENABLED:
        return None

    if _ranker is not None:
        return _ranker

    logfire.info("🧠 Initializing FlashRank Model...")

    try:
        os.makedirs(FLASHRANK_CACHE_DIR, exist_ok=True)

        _ranker = Ranker(
            cache_dir=FLASHRANK_CACHE_DIR,
            model_name="ms-marco-TinyBERT-L-2-v2"
        )

        logfire.info("✅ FlashRank Model initialized successfully")

        return _ranker

    except Exception as e:
        logfire.error(
            f"❌ FlashRank initialization failed: {e}"
        )

        _ranker = None
        return None


# ============================================================
# Reranking
# ============================================================

def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:

    if not documents:
        return []

    # --------------------------------------------------------
    # Temporary / production safety switch
    # --------------------------------------------------------

    if not FLASHRANK_ENABLED:
        logfire.warning(
            "⚠️ FlashRank disabled — using Qdrant ranking"
        )

        return documents[:top_n]

    start_time = time.time()

    logfire.info(
        f"📡 [Reranker] Sending "
        f"{len(documents)} docs to FlashRank..."
    )

    try:
        ranker = _get_ranker()

        # If model couldn't initialize,
        # gracefully fall back to Qdrant order.
        if ranker is None:
            logfire.warning(
                "⚠️ FlashRank unavailable — "
                "using original Qdrant ranking"
            )

            return documents[:top_n]

        passages = [
            {
                "id": i,
                "text": doc
            }
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(
            query=query,
            passages=passages
        )

        results = ranker.rerank(request)

        reranked_docs = [
            result["text"]
            for result in results[:top_n]
        ]

        duration = time.time() - start_time

        top_score = (
            results[0]["score"]
            if results
            else "N/A"
        )

        logfire.info(
            f"✅ [Reranker] Done in "
            f"{duration:.2f}s. "
            f"Top semantic score: {top_score}"
        )

        return reranked_docs

    except Exception as e:

        logfire.error(
            f"❌ [Reranker] Semantic Reranking Failed: {e}"
        )

        # Critical fallback:
        # RAG can continue without reranking.
        return documents[:top_n]
