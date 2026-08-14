"""Shared fixtures for the F2 hybrid RAG acceptance suite.

The suite runs against a local PostgreSQL/pgvector database
(``F2_TEST_DATABASE_URL``, defaulting to the ``agentdb_f2test`` database
used by the F2 verification on the VM). No cloud credentials are needed:
embeddings come from ``tests.fake_embeddings``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEST_DATABASE_URL = os.environ.get(
    "F2_TEST_DATABASE_URL",
    "postgresql://postgres:test@localhost:5432/agentdb_f2test",
)


@pytest.fixture(scope="session")
def dsn() -> str:
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def store(dsn: str):
    from agent.db.migrate import run_migrations
    from agent.db.store import PgStore

    run_migrations(dsn)
    return PgStore(dsn)


@pytest.fixture(scope="session")
def fake_embedding():
    from tests.fake_embeddings import FakeEmbeddingProvider

    return FakeEmbeddingProvider()


@pytest.fixture
def sample_project():
    """A Java sample project config with a unique external id per test."""
    from agent.config import ProjectConfig

    root = str((REPO / "samples" / "java-sample").resolve())
    return ProjectConfig(
        id=f"f2test-{uuid.uuid4().hex[:8]}",
        name="java-sample",
        repo_root=root,
        stories_file="",
        index_file="",
    )


@pytest.fixture
def index_sample(store, fake_embedding):
    """Index the sample project (full or incremental) with fake embeddings."""

    def _index(project, incremental: bool = False):
        from agent.codebase import CodebaseService
        from agent.indexer import SymbolEmbeddingIndexer

        indexer = SymbolEmbeddingIndexer(CodebaseService(), store, fake_embedding)
        if incremental:
            return indexer.index_incremental(project)
        return indexer.index(project)

    return _index


@pytest.fixture
def retrieval_service(store, fake_embedding):
    from agent.retrieval import HybridRetrievalService

    return HybridRetrievalService(store, fake_embedding)


@pytest.fixture
def context_builder():
    from agent.codebase import CodebaseService
    from agent.context import ContextBuilder

    return ContextBuilder(CodebaseService())


@pytest.fixture
def codebase():
    from agent.codebase import CodebaseService

    return CodebaseService()
