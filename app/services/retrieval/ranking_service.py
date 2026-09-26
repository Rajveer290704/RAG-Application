import time
import logfire
from flashrank import Ranker, RerankRequest

_ranker = None


def initialize_ranker():
    """
    Initialize FlashRank once when the application starts.
    """
    global _ranker

    if _ranker is not None:
        return _ranker

    logfire.info("🧠 Initializing FlashRank Model...")

    try:
        _ranker = Ranker(
            cache_dir="/tmp/flashrank"
        )

        logfire.info("✅ FlashRank initialized successfully")

    except Exception as e:
        logfire.error(
            f"❌ FlashRank initialization failed: {e}"
        )
        raise

    return _ranker


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:

    if not documents:
        return []

    start_time = time.time()

    logfire.info(
        f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder..."
    )

    try:
        ranker = initialize_ranker()

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
            res["text"]
            for res in results[:top_n]
        ]

        duration = time.time() - start_time

        top_score = (
            results[0]["score"]
            if results
            else "N/A"
        )

        logfire.info(
            f"✅ [Reranker] Done in {duration:.2f}s. "
            f"Top semantic score: {top_score}"
        )

        return reranked_docs

    except Exception as e:

        logfire.error(
            f"❌ [Reranker] Semantic Reranking Failed: {e}"
        )

        return documents[:top_n]
