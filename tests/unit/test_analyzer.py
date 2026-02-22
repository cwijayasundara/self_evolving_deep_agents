"""Tests for the trajectory analyzer."""

import json
from unittest.mock import MagicMock

from src.evolution.analyzer import analyze_trajectory, classify_trajectory
from src.evolution.state import GraderResult


class TestClassifyTrajectory:
    def test_successful_classification(self):
        graders = [
            GraderResult(name="tc", score=0.9, passed=True, reasoning=""),
            GraderResult(name="eff", score=0.8, passed=True, reasoning=""),
            GraderResult(name="qual", score=0.85, passed=True, reasoning=""),
        ]
        cls, score = classify_trajectory(graders)
        assert cls == "successful"
        assert score > 0.75

    def test_partial_classification(self):
        graders = [
            GraderResult(name="tc", score=0.7, passed=False, reasoning=""),
            GraderResult(name="eff", score=0.6, passed=True, reasoning=""),
            GraderResult(name="qual", score=0.65, passed=False, reasoning=""),
        ]
        cls, _score = classify_trajectory(graders)
        assert cls == "partial"

    def test_failed_classification(self):
        graders = [
            GraderResult(name="tc", score=0.2, passed=False, reasoning=""),
            GraderResult(name="eff", score=0.3, passed=False, reasoning=""),
            GraderResult(name="qual", score=0.1, passed=False, reasoning=""),
        ]
        cls, score = classify_trajectory(graders)
        assert cls == "failed"
        assert score < 0.5

    def test_empty_graders(self):
        cls, score = classify_trajectory([])
        assert cls == "failed"
        assert score == 0.0


class TestAnalyzeTrajectory:
    def test_analyze_full_pipeline(self, mock_llm, sample_trajectory):
        # Mock different responses for task completion and quality graders
        responses = [
            json.dumps({"score": 0.85, "passed": True, "reasoning": "Good"}),
            json.dumps({
                "overall_score": 0.8, "reasoning": "High quality",
                "accuracy": 0.8, "depth": 0.8, "clarity": 0.8, "relevance": 0.8,
            }),
        ]
        mock_llm.invoke.side_effect = [
            MagicMock(content=r) for r in responses
        ]

        result = analyze_trajectory(mock_llm, sample_trajectory)
        assert result["run_id"] == "test-run-001"
        assert result["classification"] in ("successful", "partial", "failed")
        assert 0.0 <= result["average_score"] <= 1.0
        assert len(result["grader_results"]) == 3
