"""Tests for the prompt optimizer."""


import pytest

from src.evolution.prompt_optimizer import (
    analyze_failures,
    generate_improved_prompt,
    optimize_prompt,
)
from src.evolution.state import AnalysisResult, GraderResult


@pytest.fixture
def failed_analyses():
    return [
        AnalysisResult(
            run_id="r1",
            task="Task 1",
            classification="failed",
            average_score=0.3,
            grader_results=[
                GraderResult(
                    name="task_completion",
                    score=0.2,
                    passed=False,
                    reasoning="Did not address the core question",
                ),
                GraderResult(
                    name="quality",
                    score=0.4,
                    passed=False,
                    reasoning="Poor source citations",
                ),
            ],
            output="Bad output",
            tool_calls=[],
        ),
        AnalysisResult(
            run_id="r2",
            task="Task 2",
            classification="partial",
            average_score=0.55,
            grader_results=[
                GraderResult(
                    name="task_completion",
                    score=0.6,
                    passed=False,
                    reasoning="Partially addressed",
                ),
            ],
            output="Partial output",
            tool_calls=[],
        ),
    ]


class TestAnalyzeFailures:
    def test_aggregate_failures(self, failed_analyses):
        result = analyze_failures(failed_analyses)
        assert "2 failed/partial" in result["failure_analysis"]
        assert len(result["common_issues"]) > 0

    def test_no_failures(self):
        result = analyze_failures([])
        assert result["common_issues"] == []


class TestGenerateImprovedPrompt:
    def test_generates_prompt(self, mock_llm):
        mock_llm.invoke.return_value.content = (
            "You are an improved agent.\n{memory_context}"
        )
        result = generate_improved_prompt(
            mock_llm,
            current_prompt="You are an agent.",
            current_score=0.5,
            failure_info={
                "failure_analysis": "Issues found",
                "common_issues": ["Issue 1"],
            },
        )
        assert "improved agent" in result
        assert "{memory_context}" in result

    def test_adds_memory_placeholder_if_missing(self, mock_llm):
        mock_llm.invoke.return_value.content = "Prompt without placeholder"
        result = generate_improved_prompt(
            mock_llm, "old", 0.5, {"failure_analysis": "", "common_issues": []}
        )
        assert "{memory_context}" in result


class TestOptimizePrompt:
    def test_optimize_creates_new_version(
        self, mock_llm, prompt_store, failed_analyses
    ):
        prompt_store.add_version("Initial prompt\n{memory_context}", score=0.5)
        mock_llm.invoke.return_value.content = (
            "Better prompt\n{memory_context}"
        )
        version = optimize_prompt(mock_llm, prompt_store, failed_analyses)
        assert version == 2
        v2 = prompt_store.get_version(2)
        assert "Better prompt" in v2.prompt

    def test_skip_when_no_failures(self, mock_llm, prompt_store):
        prompt_store.add_version("Good prompt", score=0.9)
        successful = [
            AnalysisResult(
                run_id="r1",
                task="T",
                classification="successful",
                average_score=0.9,
                grader_results=[],
                output="Good",
                tool_calls=[],
            )
        ]
        version = optimize_prompt(mock_llm, prompt_store, successful)
        assert version == 1  # No new version created
