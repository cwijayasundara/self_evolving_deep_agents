"""Tests for the memory store."""

import pytest


class TestMemoryStore:
    def test_store_and_retrieve_episodic(self, memory_store):
        data = {"task": "test task", "score": 0.8}
        memory_store.store("episodic", "run-001", data)
        retrieved = memory_store.retrieve("episodic", "run-001")
        assert retrieved is not None
        assert retrieved["task"] == "test task"
        assert retrieved["score"] == 0.8
        assert retrieved["_key"] == "run-001"

    def test_store_and_retrieve_semantic(self, memory_store):
        data = {"type": "fact", "content": "Tavily works better with specific queries"}
        memory_store.store("semantic", "fact-001", data)
        retrieved = memory_store.retrieve("semantic", "fact-001")
        assert retrieved is not None
        assert retrieved["content"] == "Tavily works better with specific queries"

    def test_retrieve_nonexistent(self, memory_store):
        assert memory_store.retrieve("episodic", "nope") is None

    def test_list_all(self, memory_store):
        memory_store.store("episodic", "r1", {"task": "t1"})
        memory_store.store("episodic", "r2", {"task": "t2"})
        all_mems = memory_store.list_all("episodic")
        assert len(all_mems) == 2

    def test_search_by_keyword(self, memory_store):
        memory_store.store("semantic", "f1", {"content": "quantum computing is advancing"})
        memory_store.store("semantic", "f2", {"content": "biology research methods"})
        memory_store.store("semantic", "f3", {"content": "quantum error correction"})
        results = memory_store.search("semantic", "quantum")
        assert len(results) == 2
        # Most relevant first
        assert results[0]["_key"] in ("f1", "f3")

    def test_search_no_results(self, memory_store):
        memory_store.store("semantic", "f1", {"content": "unrelated"})
        results = memory_store.search("semantic", "quantum computing")
        assert len(results) == 0

    def test_count(self, memory_store):
        assert memory_store.count("episodic") == 0
        memory_store.store("episodic", "r1", {"x": 1})
        assert memory_store.count("episodic") == 1

    def test_delete(self, memory_store):
        memory_store.store("episodic", "r1", {"x": 1})
        assert memory_store.delete("episodic", "r1") is True
        assert memory_store.retrieve("episodic", "r1") is None
        assert memory_store.count("episodic") == 0

    def test_delete_nonexistent(self, memory_store):
        assert memory_store.delete("episodic", "nope") is False

    def test_invalid_namespace(self, memory_store):
        with pytest.raises(ValueError, match="Unknown namespace"):
            memory_store.store("invalid", "k", {})

    def test_stored_at_timestamp(self, memory_store):
        memory_store.store("episodic", "r1", {"x": 1})
        retrieved = memory_store.retrieve("episodic", "r1")
        assert "_stored_at" in retrieved
