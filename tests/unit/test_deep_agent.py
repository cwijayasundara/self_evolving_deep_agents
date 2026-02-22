"""Tests for the deep agent factory."""

import pytest

from src.agent.deep_agent import _has_provider_prefix, build_memory_context


class TestHasProviderPrefix:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4o", False),                     # no colon
            ("qwen3.5:cloud", False),              # qwen3.5 is not a provider
            ("llama3:8b", False),                   # llama3 is not a provider
            ("ollama:qwen3.5:cloud", True),         # ollama IS a provider
            ("ollama:llama3", True),                # ollama IS a provider
            ("openai:gpt-4o", True),                # openai IS a provider
            ("anthropic:claude-sonnet-4-5-20250929", True),  # anthropic IS a provider
            ("groq:llama-3.3-70b-versatile", True), # groq IS a provider
        ],
    )
    def test_detects_provider_prefix(self, model, expected):
        assert _has_provider_prefix(model) is expected


class TestBuildMemoryContext:
    def test_empty_when_no_memories(self, memory_store):
        result = build_memory_context(memory_store, "some task")
        assert result == ""

    def test_includes_episodic_memories(self, memory_store):
        memory_store.store("episodic", "r1", {
            "summary": "Used iterative search strategy",
            "task": "Research quantum computing",
        })
        result = build_memory_context(memory_store, "quantum computing")
        assert "Previous Runs" in result
        assert "iterative search" in result

    def test_includes_semantic_memories(self, memory_store):
        memory_store.store("semantic", "f1", {
            "content": "Specific queries yield better results",
            "type": "fact",
        })
        result = build_memory_context(memory_store, "specific queries research")
        assert "Learned Facts" in result
        assert "Specific queries" in result

    def test_respects_token_budget(self, memory_store):
        for i in range(10):
            memory_store.store("episodic", f"r{i}", {
                "summary": f"Memory item {i} " + "detail " * 80,
                "task": "test task",
            })
        # Use a small token budget so not all memories fit
        result = build_memory_context(memory_store, "test task", token_budget=100)
        # 100 tokens ~ 400 chars; output should be bounded
        assert len(result) < 800
