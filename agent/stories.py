from __future__ import annotations
import json
from pathlib import Path

from agent.config import ProjectConfig
from agent.models import UserStory


class StoryFileReader:
    def read(self, project: ProjectConfig) -> list[UserStory]:
        if not project.stories_file:
            return []  # uploaded projects have no stories file
        path = Path(project.stories_file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Stories file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return [UserStory(**item) for item in data]
