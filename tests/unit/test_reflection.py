"""Tests for the reflection engine."""

import json

from src.memory.reflection import (
    _safe_json_parse,
    generate_reflection,
    reflect_and_store,
)


class TestSafeJsonParse:
    def test_parse_valid_json(self):
        result = _safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _safe_json_parse(text)
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        result = _safe_json_parse("not json at all")
        assert "strategy" in result


class TestGenerateReflection:
    def test_generate_reflection(self, mock_llm):
        mock_llm.invoke.return_value.content = json.dumps({
            "strategy": "Multi-step search",
            "worked_well": "Good source diversity",
            "improvements": "Could be more concise",
            "facts": ["Fact 1", "Fact 2"],
            "patterns": ["Pattern 1"],
        })
        result = generate_reflection(
            mock_llm,
            task="Test task",
            output="Test output",
            tool_calls=[{"name": "search", "args": {}}],
            grader_results={"score": 0.8},
        )
        assert result["strategy"] == "Multi-step search"
        assert len(result["facts"]) == 2


class TestReflectAndStore:
    def test_stores_episodic_and_semantic(self, mock_llm, memory_store):
        mock_llm.invoke.return_value.content = json.dumps({
            "strategy": "Iterative search",
            "worked_well": "Good coverage",
            "improvements": "Too many API calls",
            "facts": ["Important fact"],
            "patterns": ["Useful pattern"],
        })
        reflection = reflect_and_store(
            llm=mock_llm,
            memory_store=memory_store,
            run_id="test-run-001",
            task="Test task",
            output="Test output",
            tool_calls=[],
            grader_results={"average_score": 0.8},
        )
        assert reflection["strategy"] == "Iterative search"

        # Verify episodic memory stored
        episodic = memory_store.retrieve("episodic", "test-run-001")
        assert episodic is not None
        assert episodic["task"] == "Test task"

        # Verify semantic memories stored
        assert memory_store.count("semantic") == 2  # 1 fact + 1 pattern

    def test_handles_empty_facts_patterns(self, mock_llm, memory_store):
        mock_llm.invoke.return_value.content = json.dumps({
            "strategy": "Basic",
            "worked_well": "",
            "improvements": "",
            "facts": [],
            "patterns": [],
        })
        reflect_and_store(
            llm=mock_llm,
            memory_store=memory_store,
            run_id="test-run-002",
            task="Task",
            output="Output",
            tool_calls=[],
            grader_results={},
        )
        assert memory_store.count("episodic") == 1
        assert memory_store.count("semantic") == 0
