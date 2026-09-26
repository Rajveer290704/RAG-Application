import time
import logfire
from flashrank import Ranker, RerankRequest

_ranker = None


def _get_ranker() -> Ranker:
    global _ranker

    if _ranker is None:
        logfire.info("🧠 Loading FlashRank model...")

        try:
            _ranker = Ranker(
                model_name="ms-marco-MiniLM-L-12-v2",
                cache_dir="/tmp/flashrank"
            )

            logfire.info("✅ FlashRank model loaded successfully")

        except Exception as e:
            logfire.error(f"❌ FlashRank initialization failed: {repr(e)}")
            raise

    return _ranker


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:

    if not documents:
        logfire.warning("⚠️ Reranker received 0 documents")
        return []

    start_time = time.time()

    logfire.info(
        f"📡 Reranker received {len(documents)} documents"
    )

    logfire.info(
        f"🔎 Query: {query}"
    )

    try:
        ranker = _get_ranker()

        passages = [
            {
                "id": i,
                "text": str(doc)
            }
            for i, doc in enumerate(documents)
        ]

        logfire.info(
            f"📦 Sending {len(passages)} passages to FlashRank"
        )

        request = RerankRequest(
            query=query,
            passages=passages
        )

        results = ranker.rerank(request)

        logfire.info(
            f"📊 FlashRank returned {len(results)} results"
        )

        # VERY IMPORTANT: inspect the actual result
        for i, result in enumerate(results):
            logfire.info(
                f"🏆 Rank {i + 1}: "
                f"id={result.get('id')} "
                f"score={result.get('score')}"
            )

        reranked_docs = [
            result["text"]
            for result in results[:top_n]
        ]

        duration = time.time() - start_time

        logfire.info(
            f"✅ Reranking completed in {duration:.3f}s"
        )

        logfire.info(
            f"📤 Returning {len(reranked_docs)} reranked documents"
        )

        return reranked_docs

    except Exception as e:

        logfire.error(
            f"❌ RERANKING FAILED: {repr(e)}"
        )

        # During debugging, DO NOT silently hide the error.
        raise
