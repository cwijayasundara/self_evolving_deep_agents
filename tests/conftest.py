"""Shared test fixtures."""

import json
from unittest.mock import MagicMock

import pytest

from src.agent.prompt_store import PromptStore
from src.memory.store import MemoryStore
from src.tracing.trajectory import ToolCall, Trajectory, TrajectoryMetrics


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def prompt_store(tmp_path):
    """Provide a PromptStore backed by a temp directory."""
    return PromptStore(tmp_path / "prompts")


@pytest.fixture
def memory_store(tmp_path):
    """Provide a MemoryStore backed by a temp directory."""
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def skills_dir(tmp_path):
    """Provide a temp skills directory."""
    d = tmp_path / "skills"
    d.mkdir()
    return d


@pytest.fixture
def sample_trajectory():
    """Provide a sample trajectory for testing."""
    return Trajectory(
        run_id="test-run-001",
        task="Research quantum computing advances",
        output="# Quantum Computing Report\n\nKey findings...",
        tool_calls=[
            ToolCall(name="tavily_search", args={"query": "quantum computing 2024"}),
            ToolCall(name="tavily_search", args={"query": "quantum error correction"}),
        ],
        metrics=TrajectoryMetrics(
            total_tokens=5000,
            total_steps=3,
            latency_seconds=25.0,
            tool_call_count=2,
        ),
        status="completed",
    )


@pytest.fixture
def mock_llm():
    """Provide a mock LLM that returns configurable responses."""
    llm = MagicMock()
    response = MagicMock()
    response.content = json.dumps({
        "score": 0.85,
        "passed": True,
        "reasoning": "Good output",
    })
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def sample_skill_md(skills_dir):
    """Create a sample SKILL.md file."""
    skill_dir = skills_dir / "multi-source-research"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: Multi-Source Research\n"
        "description: Research using multiple diverse sources\n"
        "---\n\n"
        "## When to Use\n"
        "Use when the task requires comprehensive research.\n\n"
        "## Steps\n"
        "1. Search multiple databases\n"
        "2. Cross-reference findings\n"
        "3. Synthesize into a report\n"
    )
    return skill_file
