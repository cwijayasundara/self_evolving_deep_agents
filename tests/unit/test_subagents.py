"""Tests for sub-agent builder functions."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.subagents import build_research_subagent, build_synthesis_subagent
from src.config.settings import Settings


@pytest.fixture
def settings(tmp_path):
    """Settings with a temp skills path."""
    s = Settings(
        openai_api_key="test-key",
        tavily_api_key="test-tavily",
        skills_dir=str(tmp_path / "skills"),
    )
    # Create skills dir so it exists
    s.skills_path.mkdir(parents=True, exist_ok=True)
    return s


class TestBuildResearchSubagent:
    def test_has_expected_keys(self, settings):
        subagent = build_research_subagent(settings)
        assert subagent["name"] == "research-agent"
        assert "description" in subagent
        assert "system_prompt" in subagent
        assert subagent["tools"] is not None
        assert len(subagent["tools"]) == 1  # only search tool


class TestBuildSynthesisSubagent:
    def test_has_skills_no_tools(self, settings):
        subagent = build_synthesis_subagent(settings)
        assert subagent["name"] == "synthesis-agent"
        assert subagent["tools"] is None
        assert subagent["skills"] is not None
        assert str(settings.skills_path) in subagent["skills"]


class TestCreateAgentWithSubagents:
    @patch("src.agent.deep_agent.create_deep_agent")
    @patch("src.agent.deep_agent.create_llm")
    @patch("src.agent.deep_agent.create_search_tool")
    def test_passes_subagents_when_enabled(
        self, mock_search, mock_llm, mock_create, settings, tmp_path
    ):
        from src.agent.deep_agent import create_agent
        from src.agent.prompt_store import PromptStore
        from src.memory.store import MemoryStore

        settings.use_subagents = True
        prompt_store = PromptStore(tmp_path / "prompts")
        memory_store = MemoryStore(tmp_path / "memory")

        mock_llm.return_value = MagicMock()
        mock_search.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        create_agent(settings, prompt_store, memory_store, task="test")

        call_kwargs = mock_create.call_args
        assert "subagents" in call_kwargs.kwargs
        assert len(call_kwargs.kwargs["subagents"]) == 2

    @patch("src.agent.deep_agent.create_deep_agent")
    @patch("src.agent.deep_agent.create_llm")
    @patch("src.agent.deep_agent.create_search_tool")
    def test_no_subagents_when_disabled(
        self, mock_search, mock_llm, mock_create, settings, tmp_path
    ):
        from src.agent.deep_agent import create_agent
        from src.agent.prompt_store import PromptStore
        from src.memory.store import MemoryStore

        settings.use_subagents = False
        prompt_store = PromptStore(tmp_path / "prompts")
        memory_store = MemoryStore(tmp_path / "memory")

        mock_llm.return_value = MagicMock()
        mock_search.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        create_agent(settings, prompt_store, memory_store, task="test")

        call_kwargs = mock_create.call_args
        assert "subagents" not in call_kwargs.kwargs
