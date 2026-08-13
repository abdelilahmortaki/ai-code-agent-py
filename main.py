from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent.analysis import StaticAnalysisService
from agent.codebase import CodebaseService
from agent.config import Settings, ProjectConfig
from agent.factory import create_database_store, create_provider_runtime
from agent.git_service import GitDiffService
from agent.guard import PatchGuardService, PromptGuardService
from agent.index import SemanticIndexService
from agent.indexer import SymbolEmbeddingIndexer
from agent.models import PatchResult, ProjectOverview, UserStory
from agent.orchestrator import AgentOrchestrator
from agent.paths import resolve_within
from agent.patch import PatchApplierService
from agent.runner import TestRunner
from agent.search import LexicalSearchService
from agent.stories import StoryFileReader
from agent.versioning import VersionIndexer

# ---------------------------------------------------------------------------
# Project registry — persisted to .agent/projects.json
# ---------------------------------------------------------------------------

_REGISTRY_FILE = Path(".agent/projects.json")


def _load_registry() -> list[ProjectConfig]:
    """Load persisted projects from disk; skip any whose repo_root no longer exists."""
    if not _REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
        projects = []
        for item in data:
            try:
                p = ProjectConfig(**item)
                if Path(p.repo_root).exists():
                    projects.append(p)
            except Exception:
                pass
        return projects
    except Exception:
        return []


def _save_registry(projects: list[ProjectConfig]) -> None:
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_FILE.write_text(
        json.dumps([p.model_dump() for p in projects], indent=2),
        encoding="utf-8",
    )

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

settings = Settings.load("config.yml")

# Merge persisted projects into settings (in-memory list is the source of truth at runtime)
for _p in _load_registry():
    if not any(existing.id == _p.id for existing in settings.agent.projects):
        settings.agent.projects.append(_p)

codebase        = CodebaseService(max_file_chars=settings.agent.max_file_chars)
prompt_guard    = PromptGuardService()
patch_guard     = PatchGuardService()
provider_runtime = create_provider_runtime(settings, prompt_guard)
semantic_index  = SemanticIndexService(codebase, provider_runtime.embedding)
patch_applier   = PatchApplierService(patch_guard)
git_diff        = GitDiffService()
static_analysis = StaticAnalysisService(codebase)
test_runner     = TestRunner()
story_reader    = StoryFileReader()
db_store        = create_database_store(settings)
version_indexer = VersionIndexer(codebase, db_store)
symbol_embedding_indexer = SymbolEmbeddingIndexer(codebase, db_store, provider_runtime.embedding)

orchestrator = AgentOrchestrator(
    config=settings.agent,
    codebase=codebase,
    semantic_index=semantic_index,
    generation_provider=provider_runtime.generation,
    patch_applier=patch_applier,
    git_diff=git_diff,
    static_analysis=static_analysis,
    test_runner=test_runner,
    story_reader=story_reader,
    prompt_guard=prompt_guard,
    store=db_store,
)

# Active SSE log queues — keyed by client-generated run_id (UUID)
_run_queues: dict[str, asyncio.Queue] = {}

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="CardPro Code Agent", version="1.0.0")

_GENERATION_VALIDATION_ERRORS = {
    "UNRELATED_FORMATTING_CHANGED",
    "PROTECTED_CONTENT_CHANGED",
    "SOURCE_CONTEXT_MISSING",
    "SOURCE_CONTEXT_TRUNCATED",
    "EDIT_ANCHOR_NOT_FOUND",
    "EDIT_ANCHOR_AMBIGUOUS",
    "EDIT_ANCHOR_OVERLAP",
}


def _generation_http_exception(exc: ValueError) -> HTTPException:
    detail = str(exc)
    if detail in _GENERATION_VALIDATION_ERRORS:
        return HTTPException(status_code=422, detail=detail)
    if detail.startswith(("Project not found:", "Story not found:")):
        return HTTPException(status_code=404, detail=detail)
    return HTTPException(status_code=500, detail="Generation failed")


@app.get("/api/agent/projects", response_model=list[ProjectOverview])
def list_projects():
    return orchestrator.list_projects()


@app.post("/api/agent/{project_id}/index/rebuild")
def rebuild_index(project_id: str):
    try:
        msg = orchestrator.rebuild_index(project_id)
        return {"message": msg}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/agent/{project_id}/index/version")
def index_version(project_id: str):
    if db_store is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
    try:
        proj = orchestrator.config.project_by_id(project_id)
        return version_indexer.index_version(proj)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/agent/{project_id}/index/pg")
def index_symbols_pg(project_id: str, incremental: bool = Query(default=False)):
    if db_store is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
    try:
        proj = orchestrator.config.project_by_id(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if incremental:
        return symbol_embedding_indexer.index_incremental(proj)
    return symbol_embedding_indexer.index(proj)


@app.get("/api/agent/{project_id}/search")
def search_symbols(
    project_id: str,
    query: str = Query(...),
    top_k: int = Query(default=10),
    symbol_type: str = Query(default=None),
):
    if db_store is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
    try:
        orchestrator.config.project_by_id(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    filters = {"symbol_type": symbol_type} if symbol_type else None
    try:
        return LexicalSearchService(db_store).search_latest(
            project_id, query, top_k=top_k, filters=filters
        )
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith("unknown project") or detail == "no versions for project":
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=400, detail=detail)


@app.post("/api/agent/{project_id}/generate/{story_id}", response_model=PatchResult)
def generate_patch(project_id: str, story_id: str):
    try:
        return orchestrator.process_story(project_id, story_id)
    except ValueError as exc:
        raise _generation_http_exception(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Generation failed") from exc


# ---------------------------------------------------------------------------
# Local folder upload — browser sends files, server saves them as a project
# ---------------------------------------------------------------------------

class _FileEntry(BaseModel):
    path: str    # webkitRelativePath, e.g. "myapp/src/main/java/Foo.java"
    content: str

class _UploadRequest(BaseModel):
    project_name: str
    files: list[_FileEntry]


@app.post("/api/agent/projects/upload")
def upload_project(req: _UploadRequest):
    if not req.files:
        raise HTTPException(status_code=400, detail="Upload must contain at least one file")
    project_id = f"upload-{int(time.time())}"
    upload_root = Path(".agent/uploads") / project_id
    destinations: list[tuple[_FileEntry, Path]] = []
    seen_destinations: set[str] = set()
    total_bytes = 0

    for f in req.files:
        normalized_path = f.path.replace("\\", "/")
        try:
            resolve_within(upload_root, normalized_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid file path: {f.path}") from exc

        # Strip the top-level folder name: "myapp/src/..." -> "src/..."
        parts = Path(normalized_path).parts
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
        try:
            dest = resolve_within(upload_root, rel)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid file path: {f.path}") from exc

        if dest.suffix.lower() not in settings.agent.normalized_upload_allowed_extensions():
            raise HTTPException(status_code=400, detail=f"Invalid file extension: {f.path}")

        file_bytes = len(f.content.encode("utf-8"))
        if file_bytes > settings.agent.upload_max_file_bytes:
            raise HTTPException(status_code=413, detail=f"File is too large: {f.path}")
        total_bytes += file_bytes
        if total_bytes > settings.agent.upload_max_total_bytes:
            raise HTTPException(status_code=413, detail="Upload is too large")

        destination_key = dest.relative_to(upload_root.resolve()).as_posix().casefold()
        if destination_key in seen_destinations:
            raise HTTPException(status_code=400, detail=f"Duplicate file path: {f.path}")
        seen_destinations.add(destination_key)
        destinations.append((f, dest))

    for f, dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.content, encoding="utf-8")

    new_proj = ProjectConfig(
        id=project_id,
        name=req.project_name,
        repo_root=str(upload_root.resolve()),
        stories_file="",
        index_file=str(Path(".agent/index") / f"{project_id}-index.json"),
        test_command="mvn test",
        static_analysis_command="",
    )
    # Avoid duplicates (re-upload of same folder name creates a new id anyway)
    if not any(p.id == project_id for p in settings.agent.projects):
        settings.agent.projects.append(new_proj)
    _save_registry(settings.agent.projects)
    return {"project_id": project_id, "name": req.project_name, "file_count": len(req.files)}


# ---------------------------------------------------------------------------
# Ad-hoc story (submitted from the web UI)
# ---------------------------------------------------------------------------

class AdHocStoryRequest(BaseModel):
    title: str
    description: str
    acceptanceCriteria: list[str] = []
    priority: str = "Medium"


@app.get("/api/agent/{project_id}/run/{run_id}/log")
async def run_log_stream(project_id: str, run_id: str):
    """Server-Sent Events stream of real-time agent log messages for a single run."""
    q: asyncio.Queue = asyncio.Queue()
    _run_queues[run_id] = q

    async def generate():
        # Flush an initial comment immediately so the browser knows the stream
        # is alive and starts rendering the log panel without waiting for the
        # first real log event (which may be seconds away).
        yield ": stream-open\n\n"
        yield f"data: {json.dumps({'msg': 'Agent initializing…'})}\n\n"
        try:
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=300.0)
                if msg is None:  # sentinel — agent finished
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"data: {json.dumps({'msg': msg})}\n\n"
        except asyncio.TimeoutError:
            yield "event: done\ndata: {}\n\n"
        finally:
            _run_queues.pop(run_id, None)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent/{project_id}/generate-adhoc", response_model=PatchResult)
async def generate_adhoc(
    project_id: str,
    request: AdHocStoryRequest,
    run_id: str = Query(default=""),
):
    story = UserStory(
        id=f"adhoc-{int(time.time())}",
        title=request.title,
        description=request.description,
        acceptanceCriteria=request.acceptanceCriteria,
        priority=request.priority,
    )
    loop = asyncio.get_running_loop()
    q: asyncio.Queue | None = _run_queues.get(run_id) if run_id else None

    def log(msg: str) -> None:
        if q is not None:
            loop.call_soon_threadsafe(q.put_nowait, msg)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.process_adhoc_story(project_id, story, log_callback=log),
        )
        return result
    except ValueError as exc:
        raise _generation_http_exception(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Generation failed") from exc
    finally:
        if q is not None:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel → close SSE stream


@app.post("/api/agent/{project_id}/accept-plan")
def accept_plan(project_id: str):
    try:
        diff = orchestrator.accept_plan(project_id)
        return {"status": "accepted", "git_diff": diff}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/agent/{project_id}/reject-plan")
def reject_plan(project_id: str):
    orchestrator.reject_plan(project_id)
    return {"status": "rejected"}


# ---------------------------------------------------------------------------
# File browser (used by the web UI sidebar)
# ---------------------------------------------------------------------------

@app.get("/api/agent/{project_id}/files")
def list_files(project_id: str):
    try:
        proj = orchestrator.config.project_by_id(project_id)
        docs = codebase.scan(proj)
        return {"files": [path for path, _ in docs]}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/agent/{project_id}/file")
def get_file(project_id: str, path: str = Query(..., description="Repo-relative file path")):
    try:
        proj = orchestrator.config.project_by_id(project_id)
        content = codebase.read_file(proj, path)
        if content is None:
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        return {"path": path, "content": content}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/agent/{project_id}/stories")
def list_stories(project_id: str):
    try:
        proj = orchestrator.config.project_by_id(project_id)
        return story_reader.read(proj)
    except FileNotFoundError:
        return []
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Web UI — serves static/index.html
# ---------------------------------------------------------------------------

_UI = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(_UI.read_text(encoding="utf-8"))
