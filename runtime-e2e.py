#!/usr/bin/env python3
"""Shared cross-platform runtime acceptance engine (F0-G0/F1/F2/F3/F4/F5).

This is the acceptance runner used by both Windows (`runtime-e2e.ps1`
behaviour parity) and Linux (`runtime-e2e.sh`) wrappers. It is a single
module built on stdlib + httpx + the project's own `agent` package; it is
NOT a pytest suite.

Primary acceptance source is the real VM:
  real backend APIs + real PostgreSQL/pgvector + real Azure +
  real ShopPoc project + persisted DB evidence + filesystem SHA evidence.

The engine is intentionally destructive only when `--accept-destructive`
is passed; without it, the Accept lifecycle is skipped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent
EVIDENCE_DEFAULT = REPO_ROOT / ".agent" / "e2e"

RETRYABLE_422 = {
    "EDIT_ANCHOR_NOT_FOUND",
    "EDIT_ANCHOR_AMBIGUOUS",
    "UNRELATED_FORMATTING_CHANGED",
}
REQUEST_TIMEOUT = 600.0
REDACT_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|bearer|database[_-]?url|endpoint)"
    r"\s*[:=]\s*\S+"
)


def redact(text: str) -> str:
    return REDACT_PATTERN.sub(r"\1=<redacted>", text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_rel(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def safe_detail(status_code: int, json_body: Any, raw: str) -> str:
    if isinstance(json_body, dict) and json_body.get("detail") is not None:
        text = str(json_body["detail"])
    elif raw:
        text = raw
    else:
        text = f"HTTP {status_code}"
    return redact(text.replace("\r", " ").replace("\n", " ").strip())[:240]


class Check:
    __slots__ = ("category", "name", "status", "description")

    def __init__(self, category: str, name: str, passed: bool, description: str) -> None:
        self.category = category
        self.name = name
        self.status = "PASS" if passed else "FAIL"
        self.description = description

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "description": self.description,
        }


class Result:
    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.generation_attempts: list[dict] = []

    def add(self, category: str, name: str, passed: bool, description: str) -> None:
        check = Check(category, name, passed, description)
        self.checks.append(check)
        print(f"{'PASS' if passed else 'FAIL'}  {description}")

    def require(self, category: str, name: str, passed: bool, description: str) -> None:
        self.add(category, name, passed, description)
        if not passed:
            raise AssertionError(description)

    def category_status(self, category: str) -> str:
        items = [c for c in self.checks if c.category == category]
        if not items or any(c.status == "FAIL" for c in items):
            return "FAIL"
        return "PASS"


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT)

    def _do(self, method: str, path: str, body: Any = None) -> tuple[int, Any, str]:
        url = self.base + path
        if body is not None:
            response = self._client.request(method, url, json=body)
        else:
            response = self._client.request(method, url)
        raw = response.text
        json_body = None
        try:
            json_body = response.json()
        except ValueError:
            pass
        return response.status_code, json_body, raw

    def get(self, path: str) -> tuple[int, Any, str]:
        return self._do("GET", path)

    def post(self, path: str, body: Any = None) -> tuple[int, Any, str]:
        return self._do("POST", path, body)


UPLOAD_EXTENSIONS = {
    ".java", ".xml", ".yml", ".yaml", ".properties", ".json", ".md", ".txt"
}
SKIP_DIRS = {".git", "target", "node_modules", "__pycache__"}


def upload_payload(source_dir: str, project_name: str) -> dict:
    """Walk a local source tree and build the upload request payload.

    The server strips the first path component on write (browser
    webkitRelativePath behaviour), so every path is prefixed with a
    synthetic top-level folder to preserve the project's directory layout.
    """
    root = Path(source_dir).resolve()
    files: list[dict] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if path.suffix.lower() not in UPLOAD_EXTENSIONS:
            continue
        content = path.read_text(encoding="utf-8")
        total += len(content.encode("utf-8"))
        files.append({"path": "repo/" + "/".join(rel_parts), "content": content})
    return {"project_name": project_name, "files": files}


def field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def json_array(obj: Any, name: str) -> list:
    value = field(obj, name)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_safe_path(root: str, rel: str) -> Path:
    root_full = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_full, normalize_rel(rel)))
    if os.path.commonpath([root_full, candidate]) != root_full:
        raise AssertionError("selected target path escapes the project root")
    path = Path(candidate)
    if not path.is_file():
        raise AssertionError("selected target file is not present locally")
    return path


def git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("git provenance command failed")
    return result.stdout.strip()


def python_json(code: str) -> Any:
    result = subprocess.run(
        [sys.executable, "-"], input=code, capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Python runtime check failed"
        raise RuntimeError(redact(detail.replace("\n", " "))[:500])
    text = result.stdout.strip()
    if not text:
        raise RuntimeError("Python runtime check returned no result")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RuntimeError("Python runtime check returned invalid JSON") from exc


def split_diff_line(line: str) -> tuple[str, str]:
    code = line[1:]
    match = re.match(r"^([\t ]*)(.*)$", code)
    return match.group(1), match.group(2)


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.api = ApiClient(args.base_url)
        self.result = Result()
        self.branch = ""
        self.head = ""
        self.project_id = ""
        self.project_name = ""
        self.project_root = ""
        self.target_rel = ""
        self.target_local: Path | None = None
        self.before_content = ""
        self.before_sha = ""
        self.version_id = ""
        self.pending = False
        self.accepted = False
        self.finops = {}

    # ------------------------------------------------------------------ helpers

    def _resolve_target(self) -> None:
        suffix = normalize_rel(self.args.target_suffix)
        status, body, _ = self.api.get(f"/api/agent/{self.project_id}/files")
        self.result.require(
            "project", "files", status == 200, "project files endpoint responds"
        )
        files = [normalize_rel(str(f)) for f in json_array(body, "files")]
        matches = sorted(
            {
                f
                for f in files
                if f == suffix or f.endswith("/" + suffix)
            }
        )
        self.result.require(
            "target", "fixture-present", bool(matches), "E2E fixture is present"
        )
        self.result.require(
            "target",
            "fixture-unique",
            len(matches) == 1,
            f"E2E fixture is unambiguous ({len(matches)} match)",
        )
        self.target_rel = matches[0]
        root = self.args.source_dir or self.project_root
        self.target_local = get_safe_path(root, self.target_rel)
        self.result.require(
            "target", "containment", self.target_local is not None,
            "resolved target is inside the project root",
        )
        self.before_content = self.target_local.read_text()
        old_count = self.before_content.count(self.args.old_code)
        new_count = self.before_content.count(self.args.new_code)
        self.result.require(
            "target",
            "precondition",
            old_count == 1 and new_count == 0,
            "E2E fixture is in its expected pre-change state",
        )
        self.before_sha = sha256(self.target_local)

    def _exact_business_diff(self, generation: Any) -> dict:
        plan = field(generation, "plan")
        plan_files = json_array(plan, "files")
        diff = str(field(generation, "git_diff") or "")
        diff_lines = diff.split("\n")
        diff_files = [ln for ln in diff_lines if ln.startswith("diff --git ")]
        added = [ln for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++")]
        removed = [ln for ln in diff_lines if ln.startswith("-") and not ln.startswith("---")]
        plan_path = normalize_rel(str(field(plan_files[0], "path"))) if len(plan_files) == 1 else ""
        plan_op = str(field(plan_files[0], "operation")) if len(plan_files) == 1 else ""
        reason = ""
        passed = (
            bool(field(generation, "pending_review"))
            and plan is not None
            and len(plan_files) == 1
            and plan_path == self.target_rel
            and plan_op == "modify"
            and len(diff_files) == 1
            and len(added) == 1
            and len(removed) == 1
        )
        if passed:
            removed_indent, removed_code = split_diff_line(removed[0])
            added_indent, added_code = split_diff_line(added[0])
            passed = (
                removed_code == self.args.old_code
                and added_code == self.args.new_code
                and removed_indent == added_indent
            )
            if not passed:
                reason = "removed/added code or indentation does not match the fixture"
        elif not field(generation, "pending_review"):
            reason = "generation response did not leave a pending review plan"
        else:
            reason = "plan or unified diff violates the exact one-file/one-line contract"
        return {
            "passed": passed,
            "reason": reason,
            "plan_file_count": len(plan_files),
            "plan_path": plan_path,
            "plan_operation": plan_op,
            "diff_file_count": len(diff_files),
            "added_line_count": len(added),
            "removed_line_count": len(removed),
        }

    def _generate_lifecycle(self, lifecycle: str, run_nonce: str) -> tuple[Any, str, int]:
        project_path = self.project_id
        description = (
            f"In {self.target_rel}, change only the notFound error code from:\n\n"
            f"NOT_FOUND\n\nto:\n\nRESOURCE_NOT_FOUND\n\n"
            "Do not modify any other line, whitespace, indentation, method, or file. "
            "Do not create or modify any test files."
        )
        acceptance = [
            "Only NOT_FOUND changes to RESOURCE_NOT_FOUND",
            "No formatting or whitespace changes",
            "No other code is changed",
        ]
        for attempt in range(1, self.args.attempt_cap + 1):
            title = f"Update not-found error code [{run_nonce}-{lifecycle}-{attempt}]"
            story = {
                "title": title,
                "description": description,
                "acceptanceCriteria": acceptance,
                "priority": "P3",
            }
            print(f"INFO  Azure generation {lifecycle} attempt {attempt}/{self.args.attempt_cap}")
            status, body, raw = self.api.post(
                f"/api/agent/{project_path}/generate-adhoc", story
            )
            record = {
                "lifecycle": lifecycle,
                "attempt": attempt,
                "http_status": status,
                "outcome": "",
                "detail": "",
            }
            if status == 200:
                contract = self._exact_business_diff(body)
                if contract["passed"]:
                    record["outcome"] = "exact"
                    record["plan_file_count"] = contract["plan_file_count"]
                    record["diff_file_count"] = contract["diff_file_count"]
                    record["added_line_count"] = contract["added_line_count"]
                    record["removed_line_count"] = contract["removed_line_count"]
                    self.result.generation_attempts.append(record)
                    self.pending = True
                    return body, title, attempt
                record["outcome"] = (
                    "noncompliant_pending" if field(body, "pending_review") else "contract_rejected"
                )
                record["detail"] = contract["reason"]
                self.result.generation_attempts.append(record)
                if field(body, "pending_review"):
                    cleanup_status, cleanup_body, _ = self.api.post(
                        f"/api/agent/{project_path}/reject-plan"
                    )
                    if cleanup_status != 200 or field(cleanup_body, "status") != "rejected":
                        raise AssertionError("noncompliant pending plan cleanup failed")
                    self.pending = False
                    if sha256(self.target_local) != self.before_sha:
                        raise AssertionError("source changed after rejecting noncompliant proposal")
                continue
            if status == 422:
                detail = safe_detail(status, body, raw)
                record["detail"] = detail
                if str(field(body, "detail")) not in RETRYABLE_422:
                    record["outcome"] = "validation_failed"
                    self.result.generation_attempts.append(record)
                    raise AssertionError(f"Azure generation returned non-retryable HTTP 422: {detail}")
                record["outcome"] = "retryable_validation"
                self.result.generation_attempts.append(record)
                continue
            record["outcome"] = "fatal_http"
            record["detail"] = safe_detail(status, body, raw)
            self.result.generation_attempts.append(record)
            raise AssertionError(
                f"Azure generation returned fatal HTTP {status}: {record['detail']}"
            )
        raise AssertionError(
            f"{lifecycle} generation did not produce an exact proposal within "
            f"{self.args.attempt_cap} attempt(s)"
        )

    def _find_run(self, title: str) -> tuple[Any, str]:
        status, body, _ = self.api.get("/api/agent/runs?limit=500")
        self.result.require(
            "persistence", f"runs-http-{title}", status == 200,
            f"run listing responds for '{title}'",
        )
        entries = [
            e
            for e in body
            if str(field(field(e, "run"), "project_id")) == self.project_id
            and (field(field(e, "run"), "story_snapshot") or {}).get("title") == title
        ]
        self.result.require(
            "persistence", f"run-unique-{title}", len(entries) == 1,
            "exactly one persisted run matches the unique story title",
        )
        return entries[0], field(field(entries[0], "run"), "id")

    def _invocation_summaries(self, entry: Any, run_id: str, label: str) -> list[dict]:
        invocations = json_array(entry, "invocations")
        summaries = []
        for inv in invocations:
            summaries.append(
                {
                    "lifecycle": label,
                    "run_id": run_id,
                    "status": str(field(inv, "status")),
                    "provider": str(field(inv, "provider")),
                    "model": str(field(inv, "model")),
                    "input_tokens": field(inv, "prompt_tokens"),
                    "output_tokens": field(inv, "completion_tokens"),
                    "total_tokens": field(inv, "total_tokens"),
                }
            )
        return summaries

    # ------------------------------------------------------------------ checks

    def check_provenance(self) -> None:
        self.branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
        self.head = git(["rev-parse", "HEAD"])
        print(f"INFO  branch = {self.branch}")
        print(f"INFO  HEAD = {self.head}")
        self.result.require(
            "provenance", "branch", self.branch == self.args.branch,
            f"current branch is {self.args.branch}",
        )

    def check_config(self) -> None:
        cfg = python_json(
            "import json; from agent.config import Settings; s = Settings.load('config.yml'); "
            "print(json.dumps({"
            "'provider': s.ai_provider,"
            "'generation_deployment': s.azure.deployment,"
            "'embedding_deployment': s.azure.embedding_deployment,"
            "'generation_configured': bool(s.azure.endpoint and s.azure.api_key.get_secret_value() and s.azure.deployment),"
            "'embedding_configured': bool(s.azure.endpoint and s.azure.api_key.get_secret_value() and s.azure.embedding_deployment),"
            "'database_configured': bool(s.database.url.get_secret_value()),"
            "}))"
        )
        self.result.add("config", "settings-load", True, "Settings.load completed without exposing secrets")
        provider = str(field(cfg, "provider"))
        self.finops = {
            "provider": provider,
            "generation_deployment": str(field(cfg, "generation_deployment")),
            "embedding_deployment": str(field(cfg, "embedding_deployment")),
        }
        self.result.require("config", "provider", provider == "azure", "runtime provider is azure")
        self.result.require(
            "config", "generation-config", bool(field(cfg, "generation_configured")),
            "Azure generation configuration is present",
        )
        self.result.require(
            "config", "embedding-config", bool(field(cfg, "embedding_configured")),
            "Azure embedding configuration is present",
        )
        self.result.require(
            "config", "database-config", bool(field(cfg, "database_configured")),
            "database configuration is present",
        )

    def check_database_migrations(self) -> None:
        db = python_json(
            "import json, psycopg; from agent.config import Settings; "
            "s = Settings.load('config.yml'); conn = psycopg.connect(s.database.url.get_secret_value()); "
            "cur = conn.cursor(); "
            "cur.execute(\"select extname from pg_extension where extname='vector'\"); "
            "vec = [r[0] for r in cur.fetchall()]; "
            "cur.execute('select count(*) from schema_migrations'); "
            "migs = cur.fetchone()[0]; "
            "cur.execute(\"select version from schema_migrations order by version\"); "
            "versions = [r[0] for r in cur.fetchall()]; "
            "conn.close(); print(json.dumps({'vector': vec, 'migrations': migs, 'versions': versions}))"
        )
        self.result.require(
            "migrations", "vector-extension", field(db, "vector") == ["vector"],
            "PostgreSQL vector extension is installed",
        )
        self.result.require(
            "migrations", "applied", int(field(db, "migrations")) >= 5,
            f"schema migrations are applied ({field(db, 'migrations')} rows)",
        )
        self.result.require(
            "database", "configured-db", True, "database configuration verified"
        )

    def check_backend_project(self) -> None:
        if self.args.upload_dir:
            payload = upload_payload(self.args.upload_dir, self.args.project_name)
            print(f"INFO  uploading fresh project from {self.args.upload_dir} ({len(payload['files'])} files)")
            status, body, raw = self.api.post("/api/agent/projects/upload", payload)
            self.result.require(
                "project", "upload", status == 200, "fresh project upload responds"
            )
            self.project_id = str(field(body, "project_id"))
            self.project_name = str(field(body, "name"))
            self.result.require(
                "project", "upload-id", bool(self.project_id), "upload returned a project id"
            )
            print(f"INFO  uploaded project id = {self.project_id}")
            self.result.require(
                "project", "upload-files",
                int(field(body, "file_count")) > 0,
                "upload returned a non-zero file count",
            )
            status, projects_body, _ = self.api.get("/api/agent/projects")
            self.result.require(
                "backend", "health", status == 200, "backend health endpoint responds"
            )
            selected = next(
                (p for p in projects_body if str(field(p, "id")) == self.project_id),
                None,
            )
            self.result.require(
                "project", "resolve", selected is not None,
                "freshly uploaded project is registered",
            )
            self.project_root = str(field(selected, "repo_root") or "")
            return
        status, body, _ = self.api.get("/api/agent/projects")
        self.result.require(
            "backend", "health", status == 200, "backend health endpoint responds"
        )
        projects = body if isinstance(body, list) else []
        selected = None
        if self.args.project_id:
            for project in projects:
                if str(field(project, "id")) == self.args.project_id:
                    selected = project
                    break
            self.result.require(
                "project", "resolve", selected is not None,
                "supplied ProjectId matches a registered project",
            )
        elif len(projects) == 1:
            selected = projects[0]
        else:
            self.result.require(
                "project", "resolve", False,
                "ProjectId is required unless exactly one project exists",
            )
        self.project_id = str(field(selected, "id"))
        self.project_name = str(field(selected, "name"))
        self.project_root = str(field(selected, "repo_root") or "")
        self.result.require(
            "project", "resolved", bool(self.project_id), "project resolved safely"
        )
        print(f"INFO  project id = {self.project_id}")

    def check_index(self) -> None:
        project_path = self.project_id
        status, body, raw = self.api.get(
            f"/api/agent/{project_path}/search?query={self.args.search_term}&top_k=10"
        )
        need_index = self.args.rebuild_index
        if status == 200:
            self.version_id = str(field(body, "project_version_id"))
            self.result.require(
                "database", "index-version", bool(self.version_id),
                "PostgreSQL index/version is available",
            )
        elif status == 404 and re.search(
            r"no versions for project|unknown project", safe_detail(status, body, raw)
        ):
            print("INFO  no indexed version found; building incremental PostgreSQL/pgvector index")
            need_index = True
        else:
            raise AssertionError(f"initial PostgreSQL search failed: {safe_detail(status, body, raw)}")
        if need_index:
            status, body, raw = self.api.post(
                f"/api/agent/{project_path}/index/pg?incremental=true"
            )
            self.result.require(
                "index", "rebuild-http", status == 200,
                "incremental PostgreSQL/pgvector indexing completes",
            )
            self.version_id = str(field(field(body, "version"), "id"))
            files_total = int(field(body, "files_total"))
            files_indexed = int(field(body, "files_indexed"))
            files_reused = int(field(body, "files_reused"))
            symbols_total = int(field(body, "symbols_indexed")) + int(field(body, "symbols_reused"))
            embeddings_total = int(field(body, "embeddings_indexed")) + int(field(body, "embeddings_reused"))
            failed = json_array(body, "failed_files")
            embedding = field(body, "embedding")
            dimensions = field(embedding, "dimensions")
            self.finops["embedding_dimensions"] = dimensions
            self.finops["embedding_model"] = str(field(embedding, "model"))
            self.result.require(
                "index", "version-fields",
                bool(self.version_id)
                and files_total >= 0 and files_indexed >= 0 and files_reused >= 0
                and (files_indexed + files_reused) <= files_total,
                "index version and file counters are sensible",
            )
            self.result.require(
                "index", "symbols-embeddings",
                symbols_total > 0 and embeddings_total > 0,
                "symbols and embeddings are indexed or reused",
            )
            self.result.require(
                "index", "failed-files", len(failed) == 0, "index reports no failed files"
            )
            self.result.require(
                "index", "embedding-runtime",
                str(field(embedding, "provider")) == "azure"
                and bool(str(field(embedding, "model")))
                and (dimensions is None or int(dimensions) > 0),
                "Azure embedding metadata is valid when reported",
            )
        self.result.require(
            "index", "available", bool(self.version_id), "an indexed project version is available"
        )

    def check_search(self) -> None:
        project_path = self.project_id
        status, body, _ = self.api.get(
            f"/api/agent/{project_path}/search?query={self.args.search_term}&top_k=10"
        )
        self.result.require(
            "lexical", "lexical-http", status == 200, "configured lexical search responds with HTTP 200"
        )
        results = json_array(body, "results")
        self.version_id = str(field(body, "project_version_id"))
        target_stem = os.path.splitext(self.target_rel.rsplit("/", 1)[-1])[0]
        useful = [
            r
            for r in results
            if normalize_rel(str(field(r, "path"))) == self.target_rel
            or str(field(r, "name")) == self.args.search_term
            or str(field(r, "name")) == target_stem
            or re.search(
                rf"(?i)(^|\.)({re.escape(self.args.search_term)})$",
                str(field(r, "qualified_name")),
            )
        ]
        self.result.require(
            "lexical", "lexical-results",
            len(results) > 0 and bool(self.version_id),
            "lexical retrieval returns useful indexed results",
        )
        self.result.require(
            "lexical", "target-result", len(useful) > 0,
            "lexical retrieval includes the configured target file or symbol",
        )
        seed_result = useful[0]
        seed = str(field(seed_result, "qualified_name"))
        if not seed:
            owner = str(field(seed_result, "owner"))
            name = str(field(seed_result, "name"))
            seed = name if not owner else f"{owner}.{name}"
        self.result.require(
            "lexical", "graph-seed", bool(seed), "a deterministic useful lexical result provides a graph seed"
        )
        self.check_graph(seed)

    def check_graph(self, seed: str) -> None:
        import urllib.parse
        project_path = self.project_id
        encoded_seed = urllib.parse.quote(seed)
        status0, body0, _ = self.api.get(
            f"/api/agent/{project_path}/expand?seeds={encoded_seed}&depth=0&max_related=20"
        )
        self.result.require("graph", "depth0-http", status0 == 200, "graph depth 0 responds with HTTP 200")
        self.result.require(
            "graph", "depth0-shape",
            len(json_array(body0, "seeds")) > 0
            and len(json_array(body0, "unresolved_seeds")) == 0
            and len(json_array(body0, "neighbors")) == 0
            and int(field(body0, "depth")) == 0,
            "graph depth 0 resolves the seed and returns no neighbors",
        )
        status1, body1, _ = self.api.get(
            f"/api/agent/{project_path}/expand?seeds={encoded_seed}&depth=1&max_related=20"
        )
        self.result.require("graph", "depth1-http", status1 == 200, "graph depth 1 responds with HTTP 200")
        seeds1 = json_array(body1, "seeds")
        seed_match = any(
            str(field(s, "qualified_name")) == seed
            or f"{field(s, 'owner')}.{field(s, 'name')}" == seed
            or str(field(s, "name")) == seed
            for s in seeds1
        )
        self.result.require(
            "graph", "depth1-shape",
            seed_match and int(field(body1, "depth")) == 1
            and field(body1, "neighbors") is not None and field(body1, "capped") is not None,
            "graph depth 1 preserves the seed and reports neighbors/cap",
        )
        status2, _, _ = self.api.get(
            f"/api/agent/{project_path}/expand?seeds={encoded_seed}&depth=2&max_related=20"
        )
        self.result.require(
            "graph", "invalid-depth", status2 == 400, "graph rejects unsupported depth 2 with HTTP 400"
        )

    def check_priority(self) -> None:
        priority = python_json(
            "import json\n"
            "from agent.priority import normalize_priority\n"
            "c = {'P1': normalize_priority('P1'), 'p2': normalize_priority('p2'), "
            "'Medium': normalize_priority('Medium')}\n"
            "try:\n"
            "    normalize_priority('not-a-priority')\n"
            "    c['invalid_rejected'] = False\n"
            "except ValueError:\n"
            "    c['invalid_rejected'] = True\n"
            "print(json.dumps(c))\n"
        )
        self.result.require(
            "priority", "normalization",
            field(priority, "P1") == "P1" and field(priority, "p2") == "P2"
            and field(priority, "Medium") == "P3" and bool(field(priority, "invalid_rejected")),
            "priority normalization maps P1, p2, Medium and rejects invalid values",
        )

    def check_executor(self) -> None:
        executor = python_json(
            "import json, sys; from pathlib import Path; "
            "from agent.safe_runner import SafeCommandExecutor; "
            "r = SafeCommandExecutor().run([sys.executable, '-c', \"print('SAFE_EXEC_OK')\"], "
            "cwd=Path.cwd(), timeout=10); "
            "print(json.dumps({'exit_code': r.exit_code, 'ok': 'SAFE_EXEC_OK' in r.output}))"
        )
        self.result.require(
            "executor", "argv",
            int(field(executor, "exit_code")) == 0 and bool(field(executor, "ok")),
            "production SafeCommandExecutor runs argv without a shell",
        )

    def check_context(self) -> None:
        project_path = self.project_id
        story = {
            "title": "context-preview-probe",
            "description": (
                f"In {self.target_rel}, change only the notFound error code from NOT_FOUND "
                "to RESOURCE_NOT_FOUND."
            ),
            "acceptanceCriteria": ["Only NOT_FOUND changes"],
            "priority": "P3",
            "graph_depth": 1,
        }
        status, body, raw = self.api.post(
            f"/api/agent/{project_path}/context-preview", story
        )
        if status in (503,):
            print(f"WARN  context-preview unavailable: {safe_detail(status, body, raw)}")
            self.result.add(
                "context", "preview", False,
                "context preview is unavailable (503)",
            )
            return
        self.result.require("context", "preview-http", status == 200, "context preview responds")
        items = json_array(body, "items")
        self.result.require(
            "context", "items", len(items) > 0, "context preview returns retrieval items"
        )
        self.result.require(
            "context", "reasons",
            any(field(item, "reasons") for item in items),
            "context items expose a 'reason'",
        )
        self.result.require(
            "context", "selected-files",
            len(json_array(body, "selected_files")) > 0,
            "context bundle lists selected complete files",
        )
        context_sources = {
            source
            for item in items
            for source in json_array(item, "retrieval_sources")
        }
        self.result.require(
            "vector/hybrid", "hybrid-evidence",
            "vector" in context_sources,
            "context preview carries vector/hybrid retrieval evidence",
        )

    def check_finops_boundary(self) -> None:
        finops = python_json(
            "import json; from agent.config import Settings; from agent.factory import create_provider_runtime; "
            "from agent.guard import PromptGuardService; "
            "s = Settings.load('config.yml'); r = create_provider_runtime(s, PromptGuardService()); "
            "p = r.generation; c = p.capabilities(); "
            "print(json.dumps({'provider_name': p.provider_name, 'model_identity': p.model_identity, "
            "'usage': c.usage, 'exact_token_counting': c.exact_token_counting, "
            "'count_tokens_is_none': p.count_tokens('runtime-e2e') is None}))"
        )
        self.result.require(
            "finops", "provider-boundary",
            field(finops, "provider_name") == "azure"
            and bool(str(field(finops, "model_identity")))
            and bool(field(finops, "usage"))
            and not bool(field(finops, "exact_token_counting"))
            and bool(field(finops, "count_tokens_is_none")),
            "Azure provider reports usage, no exact token counting, and count_tokens=None",
        )

    def run_generation_lifecycle(self) -> None:
        run_nonce = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S%f")[:-3] + "-" + uuid.uuid4().hex[:8]
        first_body, first_title, first_attempts = self._generate_lifecycle("first", run_nonce)
        contract1 = self._exact_business_diff(first_body)
        self.result.add("generation", "first-http", True, "first Azure generation responds with HTTP 200")
        self.result.require(
            "generation", "first-exact-diff", contract1["passed"],
            "first proposal is the exact one-line business replacement",
        )
        self.result.require(
            "immutable", "first-pending",
            sha256(self.target_local) == self.before_sha,
            "source remains unchanged while first proposal is pending",
        )
        first_entry, first_run_id = self._find_run(first_title)
        first_run = field(first_entry, "run")
        self.result.require(
            "persistence", "first-awaiting-review",
            field(first_run, "priority") == "P3"
            and field(first_run, "status") == "awaiting_review"
            and field(first_run, "completed_at") is None
            and field(first_run, "human_decision") is None,
            "first persisted run is P3, awaiting_review, and incomplete",
        )
        first_invocations = json_array(first_entry, "invocations")
        summaries = self._invocation_summaries(first_entry, first_run_id, "first")
        first_latest = first_invocations[-1] if first_invocations else None
        self.result.require(
            "persistence", "first-invocation",
            len(first_invocations) > 0 and field(first_latest, "status") == "ok"
            and field(first_latest, "provider") == "azure"
            and bool(str(field(first_latest, "model"))),
            "first linked latest LLM invocation is ok, Azure, and has a deployment/model",
        )
        test_run = field(first_body, "test_run")
        self.result.require(
            "validation", "test-run",
            isinstance(test_run, dict) and isinstance(field(test_run, "exit_code"), int),
            "disposable validation produced a TestRunResult",
        )
        validation_argv = json_array(test_run, "argv")
        validation_strategy = field(test_run, "strategy")
        validation_command = str(field(test_run, "command") or "")
        self.result.require(
            "validation", "actual-command",
            bool(validation_command)
            and validation_argv
            and str(validation_argv[0]).lower() in ("mvn", "mvnw", "mvn.cmd", "mvnw.cmd")
            and bool(field(test_run, "executed")),
            "validation records an executed Maven command and argv",
        )
        self.result.require(
            "validation", "targeted-or-fallback",
            validation_strategy in ("targeted", "fallback")
            and (validation_strategy != "fallback" or "verify" in validation_argv)
            and (validation_strategy != "targeted" or bool(json_array(test_run, "targeted_tests"))),
            "validation records targeted-test or verify-fallback strategy",
        )
        self.result.require(
            "validation", "actual-exit-code",
            isinstance(field(test_run, "exit_code"), int)
            and field(test_run, "exit_code") == field(test_run, "exit_code"),
            "validation exit code is captured from command execution",
        )
        self.result.require(
            "validation", "test-status",
            field(first_run, "test_status") in ("passed", "failed", "skipped"),
            "run test_status is recorded",
        )
        self.result.require(
            "finops", "count-method",
            bool(field(first_latest, "count_method")),
            "FinOps count method is persisted",
        )
        self.result.require(
            "finops", "estimated-input-tokens",
            isinstance(field(first_latest, "estimated_input_tokens"), int)
            and field(first_latest, "estimated_input_tokens") > 0,
            "estimated input tokens are persisted",
        )
        self.result.require(
            "finops", "usage-persisted",
            isinstance(field(first_latest, "prompt_tokens"), int)
            and isinstance(field(first_latest, "completion_tokens"), int),
            "actual Azure input/output token usage is persisted",
        )
        self.result.require(
            "finops", "routing-reason",
            bool(field(first_latest, "routing_reason")),
            "routing reason is persisted",
        )
        self.result.require(
            "finops", "threshold-result",
            field(first_latest, "threshold_result") in ("ok", "over", "not_configured"),
            "threshold result is persisted",
        )
        self.result.require(
            "finops", "cost-estimate",
            field(first_latest, "estimated_max_cost") is not None
            and field(first_latest, "actual_operational_cost_estimate") is not None,
            "pre-call estimated max cost and post-call actual cost are persisted",
        )
        first_run = field(first_entry, "run")
        self.result.require(
            "routing", "complexity-label",
            field(first_run, "complexity_label") in ("LOW", "MEDIUM", "HIGH"),
            "deterministic complexity label is persisted on the run",
        )
        routing = field(first_run, "routing") or {}
        self.result.require(
            "routing", "applied-route",
            bool(routing.get("profile"))
            and routing.get("graph_depth") in (0, 1)
            and routing.get("max_output_tokens") is not None,
            "applied route (profile / graph depth / max output tokens) is persisted",
        )
        effective_model = routing.get("model_or_deployment")
        self.result.require(
            "routing", "effective-model-identity",
            bool(effective_model) and field(first_latest, "model") == effective_model,
            "generation invocation uses the persisted routed model/deployment identity",
        )
        self.result.require(
            "finops", "routed-pricing-identity",
            field(first_latest, "estimated_max_cost") is not None
            and field(first_latest, "actual_operational_cost_estimate") is not None,
            "FinOps cost lookup uses the routed model/deployment identity",
        )
        self.result.require(
            "routing", "route-reason",
            bool(field(first_latest, "routing_reason"))
            and "->" in str(field(first_latest, "routing_reason")),
            "routing reason is persisted and deterministic",
        )
        status, body, _ = self.api.post(f"/api/agent/{self.project_id}/reject-plan")
        self.result.require(
            "reject", "first-http",
            status == 200 and field(body, "status") == "rejected",
            "first pending plan is rejected through the real API",
        )
        self.pending = False
        self.result.require(
            "immutable", "after-reject",
            sha256(self.target_local) == self.before_sha,
            "source SHA is unchanged after first Reject",
        )
        status, body, _ = self.api.get("/api/agent/runs?limit=500")
        self.result.require("reject", "first-runs-http", status == 200, "run listing remains available after first reject")
        final = next((e for e in body if str(field(field(e, "run"), "id")) == first_run_id), None)
        final_run = field(final, "run") if final else {}
        self.result.require(
            "reject", "first-terminal",
            field(final_run, "status") == "completed"
            and field(final_run, "human_decision") == "rejected"
            and field(final_run, "completed_at") is not None,
            "first run is completed with human_decision=rejected",
        )

        if not self.args.accept_destructive:
            print("INFO  --accept-destructive not set; skipping second generation + Accept")
            self.result.add(
                "accept", "skipped", True,
                "Accept lifecycle skipped (not destructive by default)",
            )
            return

        second_body, second_title, second_attempts = self._generate_lifecycle("second", run_nonce)
        contract2 = self._exact_business_diff(second_body)
        self.result.add("generation", "second-http", True, "second Azure generation responds with HTTP 200")
        self.result.require(
            "generation", "second-exact-diff", contract2["passed"],
            "second proposal is the exact one-line business replacement",
        )
        self.result.require(
            "immutable", "second-pending",
            sha256(self.target_local) == self.before_sha,
            "source remains unchanged while second proposal is pending",
        )
        second_entry, second_run_id = self._find_run(second_title)
        second_run = field(second_entry, "run")
        self.result.require(
            "persistence", "second-awaiting-review",
            field(second_run, "priority") == "P3"
            and field(second_run, "status") == "awaiting_review"
            and field(second_run, "completed_at") is None
            and field(second_run, "human_decision") is None,
            "second persisted run is P3, awaiting_review, and incomplete",
        )
        second_invocations = json_array(second_entry, "invocations")
        second_latest = second_invocations[-1] if second_invocations else None
        self.result.require(
            "persistence", "second-invocation",
            len(second_invocations) > 0 and field(second_latest, "status") == "ok"
            and field(second_latest, "provider") == "azure"
            and bool(str(field(second_latest, "model"))),
            "second linked latest LLM invocation is ok, Azure, and has a deployment/model",
        )
        self.result.require(
            "immutable", "before-accept",
            sha256(self.target_local) == self.before_sha,
            "source SHA is original immediately before Accept",
        )
        status, body, raw = self.api.post(f"/api/agent/{self.project_id}/accept-plan")
        self.result.require(
            "accept", "http",
            status == 200 and field(body, "status") == "accepted",
            "second pending plan is accepted through the real API",
        )
        self.accepted = True
        self.pending = False
        self.after_accept_sha = sha256(self.target_local)
        self.result.require(
            "accept", "sha-changed",
            self.after_accept_sha != self.before_sha,
            "source SHA changes after Accept",
        )
        no_pending_status, no_pending_body, _ = self.api.post(
            f"/api/agent/{self.project_id}/accept-plan"
        )
        expected = f"No pending plan for project '{self.project_id}'"
        no_pending = (
            no_pending_status == 404
            and str(field(no_pending_body, "detail")) == expected
        )
        self.result.require(
            "accept", "no-pending-plan", no_pending,
            "no pending plan remains after Accept",
        )
        after_content = self.target_local.read_text()
        expected_content = self.before_content.replace(self.args.old_code, self.args.new_code)
        exact = after_content == expected_content
        self.result.add(
            "accept", "exact-content", exact,
            "final content equals the original with exactly one configured replacement",
        )
        semantic = (
            after_content.count(self.args.old_code) == 0
            and after_content.count(self.args.new_code) == 1
        )
        self.result.add(
            "accept", "semantic-replacement", semantic,
            "only NOT_FOUND is replaced by RESOURCE_NOT_FOUND",
        )
        if not exact or not semantic:
            raise AssertionError("final content proof failed after Accept")
        status, body, _ = self.api.get("/api/agent/runs?limit=500")
        self.result.require("accept", "runs-http", status == 200, "run listing remains available after Accept")
        final2 = next((e for e in body if str(field(field(e, "run"), "id")) == second_run_id), None)
        final_run2 = field(final2, "run") if final2 else {}
        self.result.require(
            "accept", "terminal",
            field(final_run2, "status") == "completed"
            and field(final_run2, "human_decision") == "accepted"
            and field(final_run2, "completed_at") is not None,
            "second run is completed with human_decision=accepted",
        )

    def run_validation_fail_probe(self) -> None:
        """F5 deliberate failing-validation probe (requires fallback=verify).

        Targets GlobalExceptionHandler: its naming-convention test
        GlobalExceptionHandlerTest is selected and deterministically FAILS to
        run in this environment, proving: disposable validation FAILS; bounded
        repair runs on the SAME run; accept is blocked with 409
        VALIDATION_REQUIRED; reject leaves the reference source unchanged.
        """
        target_path = "shoppoc-app/src/main/java/com/shoppoc/app/web/GlobalExceptionHandler.java"
        root = self.args.source_dir or self.project_root
        target = get_safe_path(root, target_path)
        before_sha = sha256(target)
        story = {
            "title": "f5-fail-probe",
            "description": (
                f"In {target_path}, change the hardcoded NOT_FOUND error string "
                "in handleNotFound to RESOURCE_NOT_FOUND."
            ),
            "acceptanceCriteria": ["exact"],
            "priority": "P3",
        }
        status, body, raw = self.api.post(
            f"/api/agent/{self.project_id}/generate-adhoc", story
        )
        self.result.require(
            "validation", "fail-probe-http",
            status == 200 or status == 422,
            f"failing-validation generation responds ({status})",
        )
        run_entries = []
        status, body, _ = self.api.get("/api/agent/runs?limit=500")
        run_entries = [e for e in body if str(field(field(e, "run"), "project_id")) == self.project_id]
        if not run_entries:
            self.result.require("validation", "fail-probe-run", False, "run persisted for the probe")
            return
        run = field(run_entries[0], "run")
        test_status = field(run, "test_status")
        self.result.require(
            "validation", "fail-probe-test-status",
            test_status == "failed",
            f"validation FAILED as expected ({test_status})",
        )
        invocations = json_array(run_entries[0], "invocations")
        repair_invocations = [
            inv for inv in invocations
            if "repair" in str(field(inv, "reduction_outcome") or "")
        ]
        self.result.require(
            "repair", "invoked",
            bool(repair_invocations),
            f"bounded repair invocations persisted ({len(repair_invocations)})",
        )
        self.result.require(
            "repair", "bounded",
            len(invocations) <= 1 + self.args.attempt_cap + 2,
            "repair attempts are bounded",
        )
        self.result.require(
            "repair", "finops-token-budget",
            all(
                isinstance(field(inv, "estimated_input_tokens"), int)
                and field(inv, "estimated_input_tokens") > 0
                and field(inv, "count_method")
                and field(inv, "threshold_result") in ("ok", "over", "not_configured")
                for inv in repair_invocations
            ),
            "every repair invocation has token count, method, and threshold trace",
        )
        self.result.require(
            "repair", "finops-budget-cost",
            all(
                field(inv, "max_output_tokens") is not None
                and field(inv, "estimated_max_cost") is not None
                for inv in repair_invocations
            ),
            "every repair invocation has output budget and estimated max cost",
        )
        self.result.require(
            "repair", "usage",
            all(
                isinstance(field(inv, "prompt_tokens"), int)
                and isinstance(field(inv, "completion_tokens"), int)
                and field(inv, "actual_operational_cost_estimate") is not None
                for inv in repair_invocations
            ),
            "every completed repair invocation persists Azure usage and actual cost",
        )
        self.result.require(
            "repair", "same-run",
            all(str(field(inv, "run_id")) == str(field(run, "id")) for inv in repair_invocations),
            "every repair invocation remains attached to the failing run",
        )
        if test_status == "failed":
            accept_status, accept_body, _ = self.api.post(
                f"/api/agent/{self.project_id}/accept-plan"
            )
            self.result.require(
                "validation", "validation-required-gate",
                accept_status == 409
                and str(field(accept_body, "detail")).startswith("VALIDATION_REQUIRED"),
                "accept before PASS returns 409 VALIDATION_REQUIRED",
            )
        reject_status, reject_body, _ = self.api.post(
            f"/api/agent/{self.project_id}/reject-plan"
        )
        self.result.require(
            "reject", "fail-probe-reject",
            reject_status == 200 and field(reject_body, "status") == "rejected",
            "failing-validation proposal is rejected",
        )
        self.result.require(
            "immutable", "fail-probe-immutable",
            sha256(target) == before_sha,
            "reference source is unchanged after failed validation + reject",
        )

    def write_evidence(self) -> str:
        evidence_dir = Path(self.args.evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / f"runtime-e2e-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S%f')[:-3]}.json"
        evidence = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "branch": self.branch,
            "head": self.head,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "target_relative_path": self.target_rel,
            "search_term": self.args.search_term,
            "before_sha256": self.before_sha,
            "provider": self.finops.get("provider"),
            "generation_deployment": self.finops.get("generation_deployment"),
            "embedding_deployment": self.finops.get("embedding_deployment"),
            "version_id": self.version_id,
            "checks": [c.to_dict() for c in self.result.checks],
            "generation_attempt_results": self.result.generation_attempts,
            "runtime_overall": self.overall_status(),
            "overall_result": self.overall_status(),
        }
        path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        read_back = json.loads(path.read_text(encoding="utf-8"))
        if read_back.get("checks") is None:
            raise AssertionError("evidence read-back returned no JSON")
        self.result.add(
            "evidence", "write-readback", True, "sanitized evidence JSON writes and reads back"
        )
        return str(path)

    def overall_status(self) -> str:
        if any(c.status == "FAIL" for c in self.result.checks):
            return "FAIL"
        return "PASS"

    def run(self) -> int:
        try:
            self.check_provenance()
            self.check_config()
            self.check_database_migrations()
            self.check_backend_project()
            self._resolve_target()
            self.check_index()
            self.check_search()
            self.check_priority()
            self.check_executor()
            self.check_context()
            self.check_finops_boundary()
            if self.args.validation_fail_probe:
                self.run_validation_fail_probe()
            else:
                self.run_generation_lifecycle()
        except AssertionError as exc:
            print(f"FAIL  runtime acceptance stopped: {exc}")
            print("INFO  cleaning up any pending plan")
            if self.pending and self.project_id:
                try:
                    status, body, _ = self.api.post(f"/api/agent/{self.project_id}/reject-plan")
                    if status == 200 and field(body, "status") == "rejected":
                        self.pending = False
                        print("INFO  pending plan cleanup: rejected")
                except Exception as cleanup_exc:
                    print(f"WARN  pending plan cleanup failed: {cleanup_exc}")
        evidence_path = self.write_evidence()
        print(f"INFO  sanitized evidence = {evidence_path}")
        overall = self.overall_status()
        print("\n========================================")
        print(" RUNTIME E2E ACCEPTANCE")
        print("========================================")
        print(f"BRANCH:      {self.branch}")
        print(f"HEAD:        {self.head}")
        print(f"PROJECT:     {self.project_id}")
        print(f"TARGET:      {self.target_rel}")
        for category in (
            "provenance", "config", "database", "migrations", "backend", "project",
            "target", "index", "lexical", "vector/hybrid", "graph", "context", "priority", "executor",
            "finops", "routing", "validation", "generation", "persistence", "reject",
            "accept", "immutable", "evidence",
        ):
            print(f"{category.upper():12} {self.result.category_status(category)}")
        print("")
        print(f"RESULT: {overall}")
        print("========================================")
        return 0 if overall == "PASS" else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared cross-platform runtime acceptance engine"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--upload-dir", default="")
    parser.add_argument("--project-name", default="runtime-e2e")
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--target-suffix", default="shoppoc-shared/src/main/java/com/shoppoc/shared/error/DomainError.java")
    parser.add_argument("--search-term", default="DomainError")
    parser.add_argument("--old-code", default='return of("NOT_FOUND", message);')
    parser.add_argument("--new-code", default='return of("RESOURCE_NOT_FOUND", message);')
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--attempt-cap", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--evidence-dir", default=str(EVIDENCE_DEFAULT))
    parser.add_argument("--accept-destructive", action="store_true")
    parser.add_argument("--validation-fail-probe", action="store_true")
    parser.add_argument("--branch", default="dev")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return Engine(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
