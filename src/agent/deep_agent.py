"""Deep agent factory.

Creates a LangGraph-based deep agent using the deepagents library
with configurable tools, system prompt, and skills.

Supports multiple LLM providers via ``init_chat_model``:
  - OpenAI:    MODEL=gpt-4o              MODEL_PROVIDER=openai
  - Anthropic: MODEL=claude-sonnet-4-5-20250929    MODEL_PROVIDER=anthropic
  - Ollama:    MODEL=qwen3.5:cloud       MODEL_PROVIDER=ollama
  - Or use provider prefix: MODEL=ollama:qwen3.5:cloud
"""

import logging
from typing import Any

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from src.agent.prompt_store import PromptStore
from src.agent.prompts import DEFAULT_SYSTEM_PROMPT
from src.config.settings import Settings
from src.memory.store import MemoryStore
from src.tools.search import create_search_tool

logger = logging.getLogger(__name__)

AGENT_NAME = "self-evolving-agent"
MAX_AGENT_ITERATIONS = 20

# Providers recognised by init_chat_model — used to distinguish a provider
# prefix (e.g. "ollama:qwen3.5:cloud") from a model variant tag
# (e.g. "qwen3.5:cloud") which is part of the model name itself.
_KNOWN_PROVIDERS = frozenset({
    "openai", "anthropic", "ollama", "google_vertexai", "google_genai",
    "azure_openai", "bedrock", "groq", "mistralai", "cohere", "deepseek",
    "fireworks", "perplexity", "xai", "together", "huggingface", "nvidia",
    "ibm", "upstage", "azure_ai", "google_anthropic_vertex",
})


def _has_provider_prefix(model: str) -> bool:
    """Return True if *model* starts with a known ``provider:`` prefix."""
    if ":" not in model:
        return False
    prefix = model.split(":", maxsplit=1)[0]
    return prefix in _KNOWN_PROVIDERS


OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"


def _is_ollama_cloud(model_str: str, settings: Settings) -> bool:
    """Return True if the model should use Ollama cloud inference."""
    is_ollama = settings.model_provider == "ollama" or (
        _has_provider_prefix(model_str) and model_str.startswith("ollama:")
    )
    return is_ollama and (":cloud" in model_str or bool(settings.ollama_api_key))


def create_llm(settings: Settings) -> BaseChatModel:
    """Create the LLM instance via init_chat_model (supports all providers).

    Provider is resolved in order:
      1. Explicit ``provider:model`` prefix in settings.model
         e.g. "ollama:qwen3.5:cloud" → provider=ollama, model=qwen3.5:cloud
      2. settings.model_provider  (e.g. "openai", "ollama", "anthropic")
      3. Auto-inferred from model name  (e.g. "gpt-4o" → openai)

    For Ollama cloud models (":cloud" suffix or OLLAMA_API_KEY set), we use
    the OpenAI-compatible endpoint at https://ollama.com/v1 because the
    native Ollama API strips the ":cloud" tag and returns 404.
    """
    kwargs: dict[str, Any] = {}
    model_str = settings.model

    # Pass base_url for providers that need it (Ollama, custom endpoints)
    if settings.model_base_url:
        kwargs["base_url"] = settings.model_base_url

    # Ollama cloud: use the OpenAI-compatible endpoint instead of native API.
    # The native Ollama API strips ":cloud" from the model name, causing 404.
    # The /v1 endpoint works like any OpenAI-compatible API.
    if _is_ollama_cloud(model_str, settings) and "base_url" not in kwargs:
        # Strip "ollama:" provider prefix if present — pass bare model name
        bare_model = model_str.split(":", maxsplit=1)[1] if _has_provider_prefix(model_str) else model_str
        kwargs["base_url"] = OLLAMA_CLOUD_BASE_URL
        kwargs["api_key"] = settings.ollama_api_key
        logger.info(
            "Ollama cloud detected → using OpenAI-compat endpoint %s, model=%s",
            OLLAMA_CLOUD_BASE_URL, bare_model,
        )
        return init_chat_model(bare_model, model_provider="openai", **kwargs)

    # Pass API key when available (not needed for local providers like Ollama)
    if settings.openai_api_key and settings.model_provider == "openai":
        kwargs["api_key"] = settings.openai_api_key

    # "ollama:qwen3.5:cloud" → let init_chat_model split on first ":"
    # "qwen3.5:cloud"        → colon is part of the model name, pass model_provider
    if _has_provider_prefix(model_str):
        return init_chat_model(model_str, **kwargs)

    return init_chat_model(
        model_str,
        model_provider=settings.model_provider,
        **kwargs,
    )


def build_memory_context(memory_store: MemoryStore, task: str) -> str:
    """Retrieve relevant memories and format them for prompt injection."""
    episodic = memory_store.search("episodic", task)
    semantic = memory_store.search("semantic", task)

    if not episodic and not semantic:
        return ""

    parts = ["\n\n## Relevant Past Experience"]
    if episodic:
        parts.append("### Previous Runs")
        for mem in episodic[:3]:
            parts.append(f"- {mem.get('summary', str(mem))}")
    if semantic:
        parts.append("### Learned Facts & Patterns")
        for mem in semantic[:5]:
            parts.append(f"- {mem.get('content', str(mem))}")

    return "\n".join(parts)


def create_agent(
    settings: Settings,
    prompt_store: PromptStore,
    memory_store: MemoryStore,
    task: str = "",
    extra_tools: list[BaseTool] | None = None,
) -> CompiledStateGraph[Any, Any]:
    """Create the deep research agent with memory-augmented prompts.

    Assembles a LangGraph agent using create_deep_agent with:
    - OpenAI model as the orchestrator
    - Tavily search tool for web research
    - Memory-augmented system prompt
    - Skills from the project skills/ directory
    """
    llm = create_llm(settings)
    search_tool = create_search_tool(settings)
    tools: list[BaseTool] = [search_tool]
    if extra_tools:
        tools.extend(extra_tools)

    # Get the current best prompt or use default
    prompt = prompt_store.get_current_prompt() or DEFAULT_SYSTEM_PROMPT

    # Inject memory context if a task is provided
    memory_context = build_memory_context(memory_store, task) if task else ""
    prompt = prompt.format(memory_context=memory_context)

    # Discover learned skills and pass to deepagents for injection
    skills_sources: list[str] = []
    if settings.skills_path.exists():
        skills_sources = [str(settings.skills_path)]

    logger.info(
        "Creating agent '%s' with model=%s, tools=%d, skills=%d",
        AGENT_NAME,
        settings.model,
        len(tools),
        len(skills_sources),
    )

    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        name=AGENT_NAME,
        skills=skills_sources or None,
    )


def extract_output(result: dict[str, Any]) -> str:
    """Extract the final text output from an agent invoke result.

    create_deep_agent returns {"messages": [...]}.  The final answer
    is the `.content` of the last AIMessage in the list.
    """
    # Direct "output" key (future-proofing)
    if "output" in result and isinstance(result["output"], str):
        return result["output"]

    messages = result.get("messages", [])
    # Walk backwards to find the last AI message with text content
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            return content

    return str(result)
