"""Tests for configuration settings."""

import os
from unittest.mock import patch

from src.config.settings import Settings, configure_logging, load_settings

# Base env for OpenAI provider tests - used to isolate from real .env
_BASE_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "TAVILY_API_KEY": "tvly-test",
    "LANGSMITH_API_KEY": "lsv2-test",
}


def _make_settings(extra_env: dict | None = None) -> Settings:
    """Create Settings from a clean environment (no .env file leakage)."""
    env = {**_BASE_ENV, **(extra_env or {})}
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)  # type: ignore[call-arg]


class TestSettings:
    def test_settings_loads_from_env(self):
        s = _make_settings()
        assert s.openai_api_key == "sk-test"
        assert s.tavily_api_key == "tvly-test"
        assert s.langsmith_api_key == "lsv2-test"

    def test_settings_defaults(self):
        s = _make_settings()
        assert s.model == "gpt-4o"
        assert s.model_provider == "openai"
        assert s.max_evolution_cycles == 5
        assert s.batch_size == 3
        assert s.langchain_tracing_v2 is True
        assert s.model_base_url == ""

    def test_settings_custom_values(self):
        s = _make_settings({"MODEL": "gpt-4o-mini", "MAX_EVOLUTION_CYCLES": "10"})
        assert s.model == "gpt-4o-mini"
        assert s.max_evolution_cycles == 10

    def test_settings_ollama_cloud_provider(self):
        """Ollama cloud provider works without OpenAI API key."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(
                _env_file=None,  # type: ignore[call-arg]
                model="qwen3.5:cloud",
                model_provider="ollama",
                ollama_api_key="test-key",
            )
        assert s.model == "qwen3.5:cloud"
        assert s.model_provider == "ollama"
        assert s.ollama_api_key == "test-key"
        assert s.openai_api_key == ""

    def test_settings_ollama_local_provider(self):
        """Ollama local provider with base_url."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(
                _env_file=None,  # type: ignore[call-arg]
                model="qwen3.5",
                model_provider="ollama",
                model_base_url="http://localhost:11434",
            )
        assert s.model == "qwen3.5"
        assert s.model_base_url == "http://localhost:11434"

    def test_settings_provider_prefix_model(self):
        """Model string with provider:model prefix (e.g. ollama:qwen3.5:cloud)."""
        s = _make_settings({"MODEL": "ollama:qwen3.5:cloud"})
        assert s.model == "ollama:qwen3.5:cloud"

    def test_settings_paths(self):
        s = _make_settings()
        assert s.skills_path.name == "skills"
        assert s.prompts_path.name == "prompts"
        assert s.memory_path.name == "memory"

    def test_load_settings_loads_with_defaults(self, tmp_path, monkeypatch):
        """All API keys are optional now, so load_settings succeeds with empty env."""
        monkeypatch.chdir(tmp_path)
        with patch.dict(os.environ, {}, clear=True):
            result = load_settings()
            assert result is not None

    def test_configure_logging(self):
        s = _make_settings()
        configure_logging(s)  # Should not raise

    def test_resolved_api_key_from_langsmith(self):
        s = _make_settings({"LANGSMITH_API_KEY": "lsv2-from-langsmith"})
        assert s.resolved_api_key == "lsv2-from-langsmith"

    def test_resolved_api_key_from_langchain(self):
        s = _make_settings({"LANGCHAIN_API_KEY": "lsv2-from-langchain"})
        assert s.resolved_api_key == "lsv2-from-langchain"

    def test_resolved_api_key_prefers_langchain(self):
        s = _make_settings({
            "LANGCHAIN_API_KEY": "lsv2-langchain",
            "LANGSMITH_API_KEY": "lsv2-langsmith",
        })
        assert s.resolved_api_key == "lsv2-langchain"
