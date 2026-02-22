"""Tests for the versioned prompt store."""


from src.agent.prompt_store import PromptVersion


class TestPromptVersion:
    def test_to_dict_roundtrip(self):
        pv = PromptVersion(version=1, prompt="You are a helpful agent.", score=0.8)
        data = pv.to_dict()
        restored = PromptVersion.from_dict(data)
        assert restored.version == 1
        assert restored.prompt == "You are a helpful agent."
        assert restored.score == 0.8

    def test_default_timestamp(self):
        pv = PromptVersion(version=1, prompt="test")
        assert pv.timestamp is not None


class TestPromptStore:
    def test_add_and_retrieve(self, prompt_store):
        prompt_store.add_version("prompt v1", score=0.7)
        v = prompt_store.get_version(1)
        assert v is not None
        assert v.prompt == "prompt v1"
        assert v.score == 0.7

    def test_get_current_returns_best_scored(self, prompt_store):
        prompt_store.add_version("prompt v1", score=0.6)
        prompt_store.add_version("prompt v2", score=0.9)
        prompt_store.add_version("prompt v3", score=0.7)
        assert prompt_store.get_current_prompt() == "prompt v2"

    def test_get_current_returns_latest_if_unscored(self, prompt_store):
        prompt_store.add_version("prompt v1")
        prompt_store.add_version("prompt v2")
        assert prompt_store.get_current_prompt() == "prompt v2"

    def test_get_current_empty_store(self, prompt_store):
        assert prompt_store.get_current_prompt() == ""

    def test_get_all_versions(self, prompt_store):
        prompt_store.add_version("p1", score=0.5)
        prompt_store.add_version("p2", score=0.8)
        versions = prompt_store.get_all_versions()
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    def test_version_number_increment(self, prompt_store):
        prompt_store.add_version("p1")
        prompt_store.add_version("p2")
        assert prompt_store.get_latest_version_number() == 2

    def test_update_score(self, prompt_store):
        prompt_store.add_version("p1", score=0.5)
        prompt_store.update_score(1, 0.9)
        v = prompt_store.get_version(1)
        assert v.score == 0.9

    def test_parent_version(self, prompt_store):
        prompt_store.add_version("p1")
        prompt_store.add_version("p2", parent_version=1)
        v = prompt_store.get_version(2)
        assert v.parent_version == 1

    def test_get_nonexistent_version(self, prompt_store):
        assert prompt_store.get_version(999) is None
