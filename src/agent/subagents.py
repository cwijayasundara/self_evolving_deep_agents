"""Sub-agent builder functions for research and synthesis isolation.

Each sub-agent gets an isolated context with only the tools it needs,
preventing context pollution between research and synthesis phases.
"""

from typing import Any

from src.agent.prompts import RESEARCH_SUBAGENT_PROMPT, SYNTHESIS_SUBAGENT_PROMPT
from src.config.settings import Settings
from src.tools.search import create_search_tool

# SubAgent is a TypedDict from deepagents; we use plain dicts for compatibility.
SubAgent = dict[str, Any]


def build_research_subagent(settings: Settings) -> SubAgent:
    """Build a sub-agent focused on web search and data gathering.

    The research sub-agent has access only to the search tool and a
    focused system prompt that restricts it to information gathering.
    """
    search_tool = create_search_tool(settings)
    return {
        "name": "research-agent",
        "description": "Performs web searches and gathers raw information on a topic",
        "system_prompt": RESEARCH_SUBAGENT_PROMPT,
        "tools": [search_tool],
    }


def build_synthesis_subagent(settings: Settings) -> SubAgent:
    """Build a sub-agent focused on synthesizing information into reports.

    The synthesis sub-agent has no tools (works with information provided
    by the orchestrator) but has access to learned skills.
    """
    skills_sources = None
    if settings.skills_path.exists():
        skills_sources = [str(settings.skills_path)]

    return {
        "name": "synthesis-agent",
        "description": "Synthesizes gathered information into structured reports",
        "system_prompt": SYNTHESIS_SUBAGENT_PROMPT,
        "tools": None,
        "skills": skills_sources,
    }
