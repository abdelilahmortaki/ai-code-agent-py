from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agent.codebase import CodebaseService
from agent.config import ProjectConfig
from agent.paths import resolve_within


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_source_hash(files: list[tuple[str, Path]]) -> str:
    """Hash relative path plus raw content per file into one deterministic digest.

    Each file contributes ``<relative_path>\0<raw bytes>\0``; entries are
    hashed in sorted relative-path order so the digest is reproducible.
    Files that cannot be read are skipped (their content is not part of
    the snapshot).
    """
    hasher = hashlib.sha256()
    for rel, path in sorted(files, key=lambda item: item[0]):
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    hasher.update(chunk)
        except Exception:
            continue
        hasher.update(b"\0")
    return hasher.hexdigest()


@dataclass
class GitMetadata:
    """Branch and commit SHA captured from a git repository."""
    branch: str = ""
    commit_sha: str = ""


def _git_output(repo_root: Path, *args: str) -> str | None:
    """Run a git command and return trimmed stdout, or None on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def capture_git_metadata(repo_root: str | Path) -> GitMetadata:
    """Capture branch and commit SHA when the path is a git repository."""
    root = Path(repo_root)
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    commit_sha = _git_output(root, "rev-parse", "HEAD")
    return GitMetadata(branch=branch or "", commit_sha=commit_sha or "")


class VersionIndexer:
    """Indexes project snapshots into PostgreSQL with version identity and hashes."""

    def __init__(self, codebase: CodebaseService, store: PgStore | None) -> None:
        self.codebase = codebase
        self.store = store

    def index_version(self, project: ProjectConfig) -> dict:
        """Create a new version snapshot for the project and index its files."""
        if self.store is None:
            raise RuntimeError("database is not configured")
        root = Path(project.repo_root).resolve()
        if not root.exists():
            raise ValueError(f"Repo root not found: {root}")

        files: list[tuple[str, Path]] = []
        for rel in self.codebase.scan_paths(project):
            try:
                target = resolve_within(root, rel)
                target.read_bytes()
            except Exception:
                continue
            files.append((rel, target))

        source_hash = compute_source_hash(files)
        git = capture_git_metadata(root)
        external_id = str(root)
        project_row = self.store.upsert_project(
            external_id=external_id,
            name=project.name,
            repo_root=external_id,
        )
        version = self.store.create_version(
            project_id=project_row["id"],
            version_number=self.store.next_version_number(project_row["id"]),
            source_hash=source_hash,
            branch=git.branch,
            commit_sha=git.commit_sha,
        )
        for rel, target in files:
            try:
                self.store.upsert_file(
                    project_version_id=version["id"],
                    path=rel,
                    language=target.suffix.lstrip(".").lower(),
                    file_hash=compute_file_hash(target),
                )
            except Exception:
                continue
        return {"version": version, "file_count": len(files)}

    def search(
        self,
        project_version_id: str,
        embedding: Sequence[float],
        top_k: int = 5,
        file_id: str | None = None,
    ) -> list[dict]:
        """Search symbols within a single project version."""
        if self.store is None:
            raise RuntimeError("database is not configured")
        return self.store.search_similar(
            embedding,
            top_k=top_k,
            file_id=file_id,
            project_version_id=project_version_id,
        )


# --------------------------------------------------------------------------- CLI

def _redact_secrets(text: str, dsn: str) -> str:
    """Remove the DSN and any password from a message before it is printed."""
    redacted = text.replace(dsn, "***")
    try:
        from psycopg.conninfo import conninfo_to_dict

        password = conninfo_to_dict(dsn).get("password")
    except Exception:
        password = None
    if password:
        redacted = redacted.replace(password, "***")
    return redacted


def _database_url(url_arg: str | None) -> str:
    if url_arg:
        return url_arg
    env_url = os.getenv("AGENT_DATABASE_URL", "").strip()
    if env_url:
        return env_url
    try:
        from agent.config import Settings

        return Settings.load("config.yml").database.url.get_secret_value().strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.versioning",
        description="Index a project snapshot with version identity and file hashes.",
    )
    parser.add_argument("--repo-root", required=True, help="Absolute path of the project to index")
    parser.add_argument("--name", default="", help="Project name (defaults to the directory name)")
    parser.add_argument(
        "--external-id",
        default="",
        help="Unique external id (defaults to the resolved repo root path)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides AGENT_DATABASE_URL and config.yml database.url)",
    )
    args = parser.parse_args(argv)

    dsn = _database_url(args.url)
    if not dsn:
        print(
            "No database URL configured: pass --url, set AGENT_DATABASE_URL, "
            "or set database.url in config.yml",
            file=sys.stderr,
        )
        return 1

    try:
        from agent.db.store import PgStore

        store = PgStore(dsn)
        root = Path(args.repo_root).resolve()
        name = args.name or root.name
        external_id = args.external_id or str(root)
        project = ProjectConfig(
            id=external_id,
            name=name,
            repo_root=str(root),
            stories_file="",
            index_file="",
        )
        result = VersionIndexer(CodebaseService(), store).index_version(project)
    except Exception as exc:
        print(f"Indexing failed: {_redact_secrets(str(exc), dsn)}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
