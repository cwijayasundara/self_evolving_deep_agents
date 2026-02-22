"""Tests for the skill extractor."""

import json

import pytest

from src.evolution.skill_extractor import (
    _is_duplicate_skill,
    extract_skill,
    extract_skills_from_batch,
)
from src.evolution.state import AnalysisResult, GraderResult


@pytest.fixture
def successful_analysis():
    return AnalysisResult(
        run_id="run-001",
        task="Research quantum computing",
        classification="successful",
        average_score=0.85,
        grader_results=[
            GraderResult(name="tc", score=0.9, passed=True, reasoning=""),
        ],
        output="# Quantum Computing Report\nDetailed findings...",
        tool_calls=[{"name": "search", "args": {"query": "quantum"}}],
    )


@pytest.fixture
def partial_analysis():
    return AnalysisResult(
        run_id="run-002",
        task="Research AI in healthcare",
        classification="partial",
        average_score=0.74,
        grader_results=[
            GraderResult(name="tc", score=0.8, passed=True, reasoning=""),
        ],
        output="# AI Healthcare Report\nSome findings...",
        tool_calls=[{"name": "search", "args": {"query": "AI healthcare"}}],
    )


@pytest.fixture
def failed_analysis():
    return AnalysisResult(
        run_id="run-003",
        task="Failed task",
        classification="failed",
        average_score=0.3,
        grader_results=[],
        output="Error",
        tool_calls=[],
    )


class TestIsDuplicateSkill:
    def test_exact_duplicate(self):
        existing = {"multi-source-research": {"name": "Multi Source"}}
        assert _is_duplicate_skill("multi-source-research", existing) is True

    def test_no_duplicate(self):
        existing = {"multi-source-research": {"name": "Multi Source"}}
        assert _is_duplicate_skill("data-analysis", existing) is False

    def test_similar_name(self):
        existing = {"multi-source-research": {"name": "Multi Source"}}
        assert _is_duplicate_skill("multi-source-deep-research", existing) is True


class TestExtractSkill:
    def test_extract_from_successful(self, mock_llm, successful_analysis, skills_dir):
        mock_llm.invoke.return_value.content = json.dumps({
            "name": "iterative-search",
            "description": "Use iterative search for comprehensive coverage",
            "content": "## Steps\n1. Search broadly\n2. Refine queries\n3. Synthesize",
        })
        path = extract_skill(mock_llm, successful_analysis, skills_dir)
        assert path is not None
        assert path.exists()
        assert "SKILL.md" in str(path)

    def test_extract_from_high_scoring_partial(self, mock_llm, partial_analysis, skills_dir):
        mock_llm.invoke.return_value.content = json.dumps({
            "name": "healthcare-research",
            "description": "Research AI healthcare topics",
            "content": "## Steps\n1. Search medical databases\n2. Synthesize",
        })
        path = extract_skill(mock_llm, partial_analysis, skills_dir)
        assert path is not None
        assert path.exists()

    def test_skip_low_score(self, mock_llm, failed_analysis, skills_dir):
        path = extract_skill(mock_llm, failed_analysis, skills_dir)
        assert path is None

    def test_skip_incomplete_response(self, mock_llm, successful_analysis, skills_dir):
        mock_llm.invoke.return_value.content = json.dumps({"name": ""})
        path = extract_skill(mock_llm, successful_analysis, skills_dir)
        assert path is None


class TestExtractSkillsFromBatch:
    def test_batch_extraction(
        self, mock_llm, successful_analysis, partial_analysis, failed_analysis, skills_dir
    ):
        # Each call returns a unique skill name to avoid dedup
        responses = [
            json.dumps({"name": f"skill-{i}", "description": "d", "content": "## c\nsteps"})
            for i in range(3)
        ]
        mock_llm.invoke.return_value.content = responses[0]
        mock_llm.invoke.side_effect = None  # Reset side_effect
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            resp = type("R", (), {"content": responses[min(call_count, len(responses) - 1)]})()
            call_count += 1
            return resp

        mock_llm.invoke.side_effect = _side_effect

        # successful (0.85) + partial (0.74) are eligible, failed (0.3) is not
        paths = extract_skills_from_batch(
            mock_llm, [successful_analysis, partial_analysis, failed_analysis], skills_dir
        )
        assert len(paths) == 2
