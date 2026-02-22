"""SKILL.md CRUD management.

Handles discovery, creation, validation, and listing of skills
stored as SKILL.md files with YAML frontmatter.
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_skill_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns:
        Dict with 'name', 'description', and 'body' keys.
    """
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        return {"name": "unknown", "description": "No description", "body": content}

    frontmatter_text = match.group(1)
    body = match.group(2)

    metadata: dict[str, str] = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return {
        "name": metadata.get("name", "unknown"),
        "description": metadata.get("description", "No description"),
        "body": body.strip(),
    }


def discover_skills(skills_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover all available skills from the skills directory.

    Uses progressive disclosure - only loads name and description at startup.

    Returns:
        Dict mapping skill_id to {name, description, path}
    """
    skills: dict[str, dict[str, Any]] = {}

    if not skills_dir.exists():
        return skills

    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text()
                parsed = parse_skill_frontmatter(content)
                skill_id = skill_dir.name
                skills[skill_id] = {
                    "name": parsed["name"],
                    "description": parsed["description"],
                    "path": str(skill_file),
                }

    return skills


def load_skill(skills_dir: Path, skill_id: str) -> dict[str, Any] | None:
    """Load full skill content by ID."""
    skill_file = skills_dir / skill_id / "SKILL.md"
    if not skill_file.exists():
        return None
    content = skill_file.read_text()
    parsed = parse_skill_frontmatter(content)
    return {
        "id": skill_id,
        "name": parsed["name"],
        "description": parsed["description"],
        "body": parsed["body"],
    }


def create_skill(
    skills_dir: Path,
    name: str,
    description: str,
    content: str,
) -> Path:
    """Create a new skill as a SKILL.md file.

    Args:
        skills_dir: Root skills directory
        name: Skill name (kebab-case)
        description: One-line description
        content: Markdown body content

    Returns:
        Path to the created SKILL.md file
    """
    skill_id = name.lower().replace(" ", "-").replace("_", "-")
    skill_dir = skills_dir / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = (
        f"---\nname: {name}\ndescription: {description}\n---\n\n{content}\n"
    )

    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(skill_md)
    logger.info("Created skill '%s' at %s", name, skill_file)
    return skill_file


def validate_skill(skill_path: Path) -> tuple[bool, str]:
    """Validate a SKILL.md file structure.

    Returns:
        (is_valid, message)
    """
    if not skill_path.exists():
        return False, f"File not found: {skill_path}"

    content = skill_path.read_text()
    if not content.startswith("---"):
        return False, "Missing YAML frontmatter (must start with ---)"

    parsed = parse_skill_frontmatter(content)
    if parsed["name"] == "unknown":
        return False, "Missing 'name' in frontmatter"
    if parsed["description"] == "No description":
        return False, "Missing 'description' in frontmatter"
    if not parsed["body"].strip():
        return False, "Skill body is empty"

    return True, "Valid"


def list_skills(skills_dir: Path) -> str:
    """List all skills in a human-readable format."""
    skills = discover_skills(skills_dir)
    if not skills:
        return "No skills found."

    lines = ["Available Skills:", "=" * 50]
    for skill_id, skill in sorted(skills.items()):
        lines.append(f"\n  {skill['name']} ({skill_id})")
        lines.append(f"    {skill['description']}")

    lines.append(f"\n{'=' * 50}")
    lines.append(f"Total: {len(skills)} skills")
    return "\n".join(lines)
