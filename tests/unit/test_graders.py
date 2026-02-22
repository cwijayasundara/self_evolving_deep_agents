"""Tests for the grading system."""

import json

from src.evolution.graders.efficiency import _score_metric, grade_efficiency
from src.evolution.graders.quality import grade_quality
from src.evolution.graders.task_completion import grade_task_completion
from src.tracing.trajectory import TrajectoryMetrics


class TestEfficiencyGrader:
    def test_score_metric_ideal(self):
        assert _score_metric(1000, 10000, 50000) == 1.0

    def test_score_metric_midrange(self):
        score = _score_metric(30000, 10000, 50000)
        assert 0.3 < score < 0.8

    def test_score_metric_beyond_acceptable(self):
        score = _score_metric(100000, 10000, 50000)
        assert score < 0.3

    def test_grade_ideal_metrics(self):
        metrics = TrajectoryMetrics(
            total_tokens=5000, total_steps=3, latency_seconds=15.0
        )
        result = grade_efficiency(metrics)
        assert result["score"] == 1.0
        assert result["passed"] is True

    def test_grade_poor_metrics(self):
        metrics = TrajectoryMetrics(
            total_tokens=80000, total_steps=20, latency_seconds=200.0
        )
        result = grade_efficiency(metrics)
        assert result["score"] < 0.5
        assert result["name"] == "efficiency"

    def test_grade_includes_reasoning(self):
        metrics = TrajectoryMetrics(total_tokens=5000)
        result = grade_efficiency(metrics)
        assert "Tokens" in result["reasoning"]


class TestTaskCompletionGrader:
    def test_grade_with_passing_response(self, mock_llm):
        mock_llm.invoke.return_value.content = json.dumps({
            "score": 0.9,
            "passed": True,
            "reasoning": "Thorough coverage",
        })
        result = grade_task_completion(mock_llm, "Test task", "Test output")
        assert result["score"] == 0.9
        assert result["passed"] is True
        assert result["name"] == "task_completion"

    def test_grade_with_failing_response(self, mock_llm):
        mock_llm.invoke.return_value.content = json.dumps({
            "score": 0.3,
            "passed": False,
            "reasoning": "Missing key info",
        })
        result = grade_task_completion(mock_llm, "Test task", "Bad output")
        assert result["score"] == 0.3
        assert result["passed"] is False


class TestQualityGrader:
    def test_grade_quality(self, mock_llm):
        mock_llm.invoke.return_value.content = json.dumps({
            "accuracy": 0.9,
            "depth": 0.8,
            "clarity": 0.85,
            "relevance": 0.9,
            "overall_score": 0.86,
            "reasoning": "High quality output",
        })
        result = grade_quality(mock_llm, "Test task", "Test output")
        assert result["score"] == 0.86
        assert result["name"] == "quality"
