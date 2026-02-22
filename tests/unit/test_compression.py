"""Tests for memory compression utilities."""

import pytest

from src.memory.compression import (
    compress_memory_context,
    consolidate_episodic,
    deduplicate_semantic,
)


class TestDeduplicateSemantic:
    def test_removes_near_duplicates(self, memory_store):
        memory_store.store("semantic", "s1", {
            "content": "Tavily search works best with specific detailed queries",
        })
        memory_store.store("semantic", "s2", {
            "content": "Tavily search works best with specific and detailed queries",
        })
        memory_store.store("semantic", "s3", {
            "content": "Quantum computing uses qubits for parallel computation",
        })

        removed = deduplicate_semantic(memory_store, threshold=0.7)
        assert removed == 1
        assert memory_store.count("semantic") == 2

    def test_keeps_distinct_memories(self, memory_store):
        memory_store.store("semantic", "s1", {
            "content": "Tavily search works best with specific queries",
        })
        memory_store.store("semantic", "s2", {
            "content": "Quantum computing uses qubits for parallel computation",
        })
        memory_store.store("semantic", "s3", {
            "content": "Machine learning models require labeled training data",
        })

        removed = deduplicate_semantic(memory_store, threshold=0.7)
        assert removed == 0
        assert memory_store.count("semantic") == 3


class TestConsolidateEpisodic:
    def test_keeps_recent_memories(self, memory_store):
        for i in range(15):
            memory_store.store("episodic", f"r{i:03d}", {
                "summary": f"Completed task {i}",
                "task": f"Task {i}",
            })

        consolidated = consolidate_episodic(memory_store, current_cycle=3, max_recent=10)
        assert consolidated > 0
        # After consolidation we should have <= 10 recent + consolidated summaries
        remaining = memory_store.count("episodic")
        assert remaining <= 15  # fewer than original
        assert remaining < 15   # some were consolidated

    def test_no_op_when_under_limit(self, memory_store):
        for i in range(5):
            memory_store.store("episodic", f"r{i}", {
                "summary": f"Task {i}",
                "task": f"Task {i}",
            })

        consolidated = consolidate_episodic(memory_store, current_cycle=1, max_recent=10)
        assert consolidated == 0
        assert memory_store.count("episodic") == 5


class TestCompressMemoryContext:
    def test_respects_token_budget(self, memory_store):
        # Store many memories with long content
        for i in range(20):
            memory_store.store("semantic", f"s{i}", {
                "content": f"Fact number {i}: " + "word " * 50,
            })
        for i in range(20):
            memory_store.store("episodic", f"r{i}", {
                "summary": f"Episode {i}: " + "detail " * 50,
                "task": "test task word fact",
            })

        # Small token budget = ~400 chars
        result = compress_memory_context(memory_store, "test task word fact", token_budget=100)
        assert len(result) <= 100 * 4 + 200  # some slack for headers

    def test_empty_store_returns_empty(self, memory_store):
        result = compress_memory_context(memory_store, "any task")
        assert result == ""

    def test_prioritises_semantic_over_episodic(self, memory_store):
        memory_store.store("semantic", "s1", {
            "content": "Semantic fact about research methods",
        })
        memory_store.store("episodic", "r1", {
            "summary": "Episodic memory about research run",
            "task": "research methods",
        })

        result = compress_memory_context(memory_store, "research methods")
        # Semantic section should appear before episodic
        semantic_pos = result.find("Learned Facts")
        episodic_pos = result.find("Previous Runs")
        assert semantic_pos < episodic_pos
