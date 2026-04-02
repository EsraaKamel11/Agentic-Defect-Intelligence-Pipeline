"""
RAG Indexer Agent — LangGraph node.

- Chunks DefectEvents using structure-aware strategy
- Embeds using all-mpnet-base-v2 (local)
- Content-hash deduplication
- Upserts to Qdrant (or in-memory fallback)
- Writes metadata to PostgreSQL
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from adip.agents.supervisor import AgentTimer, append_error
from adip.graph.state import ADIPState, DefectEvent, IndexingResult
from adip.rag.chunker import chunk_event
from adip.rag.embedder import embed_dense
from adip.rag.vector_store import VectorStore
from adip.persistence.defect_db import DefectDB

logger = logging.getLogger(__name__)

# Module-level singletons
_vector_store: VectorStore | None = None
_db: DefectDB | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


async def get_db() -> DefectDB:
    global _db
    if _db is None:
        _db = DefectDB()
        await _db.initialize()
    return _db


async def index_to_rag(state: ADIPState) -> Dict[str, Any]:
    """
    LangGraph node: chunk, embed, deduplicate, and index defect events.
    """
    with AgentTimer("rag_indexer"):
        try:
            events_data = state.get("defect_events", [])
            if not events_data:
                logger.info("No events to index")
                result = IndexingResult(indexed_count=0)
                return {
                    "indexing_result": result.model_dump(),
                    "indexed_count": 0,
                }

            store = get_vector_store()
            db = await get_db()

            # Chunk all events
            all_texts: List[str] = []
            all_ids: List[str] = []
            all_payloads: List[Dict[str, Any]] = []

            for event_dict in events_data:
                event = DefectEvent(**event_dict)
                # Choose content to chunk
                content = event.normalized_content or event.raw_content
                source_type = "stack_trace" if event.stack_trace else event.source

                chunks = chunk_event(content, source_type, event.id)
                for chunk in chunks:
                    chunk_id = f"{event.id}_{chunk.chunk_index}"
                    all_ids.append(chunk_id)
                    all_texts.append(chunk.text)
                    all_payloads.append({
                        "content": chunk.text,
                        "source_type": chunk.source_type,
                        "event_id": event.id,
                        "component": event.component,
                        "severity": event.severity,
                        "file_path": event.file_path or "",
                        "timestamp": event.timestamp.isoformat() if event.timestamp else "",
                    })

            if not all_texts:
                result = IndexingResult(indexed_count=0)
                return {"indexing_result": result.model_dump(), "indexed_count": 0}

            # Embed
            vectors = embed_dense(all_texts)

            # Upsert with deduplication
            inserted = store.upsert(all_ids, vectors, all_payloads, all_texts)
            skipped = len(all_texts) - inserted

            # Write metadata to DB
            await db.store_defect_events_batch(events_data)

            # Update embedding IDs in state
            for event_dict in events_data:
                event_dict["embedding_id"] = f"{event_dict['id']}_0"

            result = IndexingResult(
                indexed_count=inserted,
                skipped_duplicates=skipped,
            )

            logger.info(
                "Indexed %d chunks (%d skipped duplicates), total in store: %d",
                inserted, skipped, store.count(),
            )

            return {
                "indexing_result": result.model_dump(),
                "indexed_count": inserted,
                "defect_events": events_data,
            }

        except Exception as exc:
            logger.error("RAG indexing failed: %s", exc, exc_info=True)
            return {
                **append_error(state, f"rag_indexer: {exc}"),
                "indexing_result": IndexingResult(
                    indexed_count=0, errors=[str(exc)]
                ).model_dump(),
                "indexed_count": 0,
            }
