"""Tests for the evolution orchestrator."""



from src.evolution.orchestrator import MIN_IMPROVEMENT, PLATEAU_CYCLES
from src.evolution.state import EvolutionMetrics


class TestPlateuDetection:
    """Test the plateau detection logic used in the orchestrator."""

    def test_detects_plateau(self):
        metrics = [
            EvolutionMetrics(
                cycle=0, avg_score=0.6, skills_learned=0,
                prompt_version=1, memories_stored=0, trajectories_analyzed=3,
            ),
            EvolutionMetrics(
                cycle=1, avg_score=0.62, skills_learned=1,
                prompt_version=2, memories_stored=5, trajectories_analyzed=3,
            ),
            EvolutionMetrics(
                cycle=2, avg_score=0.63, skills_learned=1,
                prompt_version=3, memories_stored=10, trajectories_analyzed=3,
            ),
        ]
        # Check last PLATEAU_CYCLES
        recent = metrics[-PLATEAU_CYCLES:]
        improvements = [
            recent[i]["avg_score"] - recent[i - 1]["avg_score"]
            for i in range(1, len(recent))
        ]
        is_plateau = all(imp < MIN_IMPROVEMENT for imp in improvements)
        assert is_plateau is True

    def test_no_plateau_when_improving(self):
        metrics = [
            EvolutionMetrics(
                cycle=0, avg_score=0.4, skills_learned=0,
                prompt_version=1, memories_stored=0, trajectories_analyzed=3,
            ),
            EvolutionMetrics(
                cycle=1, avg_score=0.5, skills_learned=1,
                prompt_version=2, memories_stored=5, trajectories_analyzed=3,
            ),
            EvolutionMetrics(
                cycle=2, avg_score=0.7, skills_learned=2,
                prompt_version=3, memories_stored=10, trajectories_analyzed=3,
            ),
        ]
        recent = metrics[-PLATEAU_CYCLES:]
        improvements = [
            recent[i]["avg_score"] - recent[i - 1]["avg_score"]
            for i in range(1, len(recent))
        ]
        is_plateau = all(imp < MIN_IMPROVEMENT for imp in improvements)
        assert is_plateau is False


class TestEvolutionMetrics:
    def test_metrics_structure(self):
        m = EvolutionMetrics(
            cycle=0,
            avg_score=0.75,
            skills_learned=2,
            prompt_version=3,
            memories_stored=15,
            trajectories_analyzed=5,
        )
        assert m["cycle"] == 0
        assert m["avg_score"] == 0.75
        assert m["skills_learned"] == 2
