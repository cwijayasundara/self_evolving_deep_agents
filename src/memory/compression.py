"""Memory compression utilities.

Provides deduplication, consolidation, and token-budgeted context building
to prevent unbounded memory growth across evolution cycles.
"""

import logging
import uuid
from datetime import UTC, datetime

from src.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two strings based on word tokens."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def deduplicate_semantic(memory_store: MemoryStore, threshold: float = 0.7) -> int:
    """Remove near-duplicate semantic memories using Jaccard similarity.

    Args:
        memory_store: The memory store to deduplicate.
        threshold: Similarity threshold above which memories are considered duplicates.

    Returns:
        Count of memories removed.
    """
    memories = memory_store.list_all("semantic")
    if len(memories) < 2:
        return 0

    keys_to_remove: set[str] = set()
    for i in range(len(memories)):
        if memories[i]["_key"] in keys_to_remove:
            continue
        for j in range(i + 1, len(memories)):
            if memories[j]["_key"] in keys_to_remove:
                continue
            content_a = memories[i].get("content", str(memories[i]))
            content_b = memories[j].get("content", str(memories[j]))
            if _jaccard_similarity(content_a, content_b) > threshold:
                keys_to_remove.add(memories[j]["_key"])

    for key in keys_to_remove:
        memory_store.delete("semantic", key)

    if keys_to_remove:
        logger.info("Deduplicated %d semantic memories", len(keys_to_remove))
    return len(keys_to_remove)


def consolidate_episodic(
    memory_store: MemoryStore,
    current_cycle: int,
    max_recent: int = 10,
) -> int:
    """Consolidate older episodic memories, keeping the most recent intact.

    Keeps the *max_recent* most recent episodic memories as-is.
    Groups older memories by date bucket and creates a single summary per group.

    Args:
        memory_store: The memory store to consolidate.
        current_cycle: Current evolution cycle number (used in summary metadata).
        max_recent: Number of most recent memories to keep unchanged.

    Returns:
        Count of individual memories replaced by consolidated summaries.
    """
    memories = memory_store.list_all("episodic")
    if len(memories) <= max_recent:
        return 0

    # Sort by stored_at descending (most recent first)
    memories.sort(key=lambda m: m.get("_stored_at", ""), reverse=True)
    recent = memories[:max_recent]
    old = memories[max_recent:]

    if not old:
        return 0

    # Group old memories by date (YYYY-MM-DD from _stored_at)
    groups: dict[str, list[dict]] = {}
    for mem in old:
        stored_at = mem.get("_stored_at", "")
        date_key = stored_at[:10] if len(stored_at) >= 10 else "unknown"
        groups.setdefault(date_key, []).append(mem)

    consolidated_count = 0
    for date_key, group in groups.items():
        summaries = []
        for mem in group:
            summary = mem.get("summary", mem.get("task", str(mem.get("_key", ""))))
            summaries.append(summary)

        consolidated_summary = f"Consolidated {len(group)} episodes from {date_key}: " + "; ".join(summaries)

        # Delete individual old memories
        for mem in group:
            memory_store.delete("episodic", mem["_key"])
            consolidated_count += 1

        # Store consolidated memory
        memory_store.store("episodic", f"consolidated-{date_key}-{uuid.uuid4().hex[:8]}", {
            "summary": consolidated_summary,
            "type": "consolidated",
            "source_count": len(group),
            "date_bucket": date_key,
            "consolidated_at_cycle": current_cycle,
        })

    if consolidated_count:
        logger.info(
            "Consolidated %d old episodic memories into %d summaries",
            consolidated_count, len(groups),
        )
    return consolidated_count


def compress_memory_context(
    memory_store: MemoryStore,
    task: str,
    token_budget: int = 2000,
) -> str:
    """Build memory context within a token budget.

    Searches episodic and semantic memories for the task, then builds a
    context string that fits within *token_budget* tokens (estimated as
    1 token ~ 4 characters).

    Prioritises semantic matches (more reusable) before episodic matches.

    Args:
        memory_store: The memory store to search.
        task: The current task description to match memories against.
        token_budget: Maximum number of tokens for the output.

    Returns:
        Formatted memory context string, or empty string if no matches.
    """
    char_budget = token_budget * 4

    semantic = memory_store.search("semantic", task)
    episodic = memory_store.search("episodic", task)

    if not semantic and not episodic:
        return ""

    parts: list[str] = ["\n\n## Relevant Past Experience"]
    current_len = len(parts[0])

    # Semantic matches first (more reusable across tasks)
    if semantic:
        header = "### Learned Facts & Patterns"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in semantic:
                line = f"- {mem.get('content', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    # Episodic matches second
    if episodic and current_len < char_budget:
        header = "### Previous Runs"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in episodic:
                line = f"- {mem.get('summary', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    # If only the header was added and nothing else, return empty
    if len(parts) <= 1:
        return ""

    return "\n".join(parts)
