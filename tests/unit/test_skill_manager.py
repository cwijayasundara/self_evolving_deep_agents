"""Tests for the skill manager."""


from src.skills.manager import (
    create_skill,
    discover_skills,
    list_skills,
    load_skill,
    parse_skill_frontmatter,
    validate_skill,
)


class TestParseSkillFrontmatter:
    def test_parse_valid_frontmatter(self):
        content = "---\nname: Test Skill\ndescription: A test\n---\n\nBody content"
        result = parse_skill_frontmatter(content)
        assert result["name"] == "Test Skill"
        assert result["description"] == "A test"
        assert result["body"] == "Body content"

    def test_parse_no_frontmatter(self):
        content = "Just plain content"
        result = parse_skill_frontmatter(content)
        assert result["name"] == "unknown"
        assert result["body"] == "Just plain content"


class TestDiscoverSkills:
    def test_discover_existing_skills(self, sample_skill_md, skills_dir):
        skills = discover_skills(skills_dir)
        assert "multi-source-research" in skills
        assert skills["multi-source-research"]["name"] == "Multi-Source Research"

    def test_discover_empty_dir(self, skills_dir):
        skills = discover_skills(skills_dir)
        assert skills == {}

    def test_discover_nonexistent_dir(self, tmp_path):
        skills = discover_skills(tmp_path / "nonexistent")
        assert skills == {}


class TestCreateSkill:
    def test_create_and_validate(self, skills_dir):
        path = create_skill(
            skills_dir,
            name="test-skill",
            description="A test skill",
            content="## Usage\nUse this skill for testing.",
        )
        assert path.exists()
        is_valid, msg = validate_skill(path)
        assert is_valid, msg

    def test_create_with_spaces_in_name(self, skills_dir):
        path = create_skill(
            skills_dir,
            name="My Test Skill",
            description="Handles spaces",
            content="Content here",
        )
        assert "my-test-skill" in str(path)


class TestLoadSkill:
    def test_load_existing(self, sample_skill_md, skills_dir):
        skill = load_skill(skills_dir, "multi-source-research")
        assert skill is not None
        assert skill["name"] == "Multi-Source Research"
        assert "## When to Use" in skill["body"]

    def test_load_nonexistent(self, skills_dir):
        assert load_skill(skills_dir, "nope") is None


class TestValidateSkill:
    def test_valid_skill(self, sample_skill_md):
        is_valid, _msg = validate_skill(sample_skill_md)
        assert is_valid

    def test_missing_file(self, tmp_path):
        is_valid, msg = validate_skill(tmp_path / "missing.md")
        assert not is_valid
        assert "not found" in msg

    def test_missing_frontmatter(self, tmp_path):
        bad_file = tmp_path / "SKILL.md"
        bad_file.write_text("No frontmatter here")
        is_valid, _msg = validate_skill(bad_file)
        assert not is_valid


class TestListSkills:
    def test_list_with_skills(self, sample_skill_md, skills_dir):
        output = list_skills(skills_dir)
        assert "Multi-Source Research" in output
        assert "Total: 1" in output

    def test_list_empty(self, skills_dir):
        output = list_skills(skills_dir)
        assert "No skills found" in output
