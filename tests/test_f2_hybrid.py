"""F2 hybrid contextual RAG acceptance suite (#19/#20/#21).

Runs against a real local PostgreSQL/pgvector database with deterministic
fake embeddings (``tests.fake_embeddings``) — no cloud credentials needed.
Covers the acceptance criteria of:

- #19 deterministic hybrid ranking (lexical + vector + graph fusion);
- #20 explainable Context Builder (per-item reason/source/priority/token
  estimate/state, budget reduction, authoritative full files);
- #21 Why-this-context endpoint + preserved Accept/Reject surface.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fake_embeddings import FakeEmbeddingProvider

# Query texts used across the suite.
Q_EXACT = "validateContract"            # exact identifier -> lexical fast path
Q_CLASS = "CustomerService"             # exact class name -> lexical + vector + graph (TESTED_BY)
Q_SEMANTIC = "contract status validate"  # no exact identifier -> vector signal


def _item(retrieval: dict, qualified_name: str) -> dict | None:
    for item in retrieval["items"]:
        if item["qualified_name"] == qualified_name:
            return item
    return None


def _find(items: list, qualified_name: str):
    for item in items:
        value = item["qualified_name"] if isinstance(item, dict) else item.qualified_name
        if value == qualified_name:
            return item
    return None


class _StoryLexicalFixture:
    def __init__(self):
        self.queries = []
        self.rows = {
            "RecordError": [
                {
                    "id": "class-record-error",
                    "name": "RecordError",
                    "qualified_name": "example.errors.RecordError",
                    "symbol_type": "class",
                    "path": "src/RecordError.java",
                    "source": "class RecordError {}",
                    "lexical_score": 100.0,
                    "match_kind": "exact",
                    "matched_fields": ["name"],
                    "name_sim": 1.0,
                    "start_line": 1,
                },
                {
                    "id": "constructor-record-error",
                    "name": "constructor",
                    "qualified_name": "example.errors.RecordError.constructor",
                    "symbol_type": "constructor",
                    "path": "src/RecordError.java",
                    "source": "RecordError() {}",
                    "lexical_score": 20.0,
                    "match_kind": "contains",
                    "matched_fields": ["qualified_name"],
                    "name_sim": 0.0,
                    "start_line": 2,
                },
            ],
            "NOT_FOUND": [
                {
                    "id": "method-not-found",
                    "name": "notFound",
                    "qualified_name": "example.errors.RecordError.notFound",
                    "symbol_type": "method",
                    "path": "src/RecordError.java",
                    "source": "return new RecordError(NOT_FOUND);",
                    "lexical_score": 63.33,
                    "match_kind": "fuzzy",
                    "matched_fields": ["source"],
                    "name_sim": 0.5833,
                    "start_line": 10,
                }
            ],
            "notFound": [
                {
                    "id": "method-not-found",
                    "name": "notFound",
                    "qualified_name": "example.errors.RecordError.notFound",
                    "symbol_type": "method",
                    "path": "src/RecordError.java",
                    "source": "return new RecordError(NOT_FOUND);",
                    "lexical_score": 100.0,
                    "match_kind": "exact",
                    "matched_fields": ["name"],
                    "name_sim": 1.0,
                    "start_line": 10,
                }
            ],
        }

    def search(self, _version_id, query, top_k=10):
        self.queries.append(query)
        return [dict(row) for row in self.rows.get(query, [])[:top_k]]


class _StoryStoreFixture:
    def __init__(self):
        self.vector_calls = []
        self.rows = {
            "class-record-error": {
                "id": "class-record-error",
                "name": "RecordError",
                "qualified_name": "example.errors.RecordError",
                "symbol_type": "class",
                "path": "src/RecordError.java",
                "source": "class RecordError {}",
                "distance": 0.90,
                "start_line": 1,
            },
            "constructor-record-error": {
                "id": "constructor-record-error",
                "name": "constructor",
                "qualified_name": "example.errors.RecordError.constructor",
                "symbol_type": "constructor",
                "path": "src/RecordError.java",
                "source": "RecordError() {}",
                "distance": 0.35,
                "start_line": 2,
            },
            "method-not-found": {
                "id": "method-not-found",
                "name": "notFound",
                "qualified_name": "example.errors.RecordError.notFound",
                "symbol_type": "method",
                "path": "src/RecordError.java",
                "source": "return new RecordError(NOT_FOUND);",
                "distance": 0.70,
                "start_line": 10,
            },
        }

    def find_project_by_external_id(self, project_id):
        return {"id": "project-row", "external_id": project_id}

    def latest_version(self, _project_id):
        return {"id": "version-row", "version_number": 7}

    def search_similar(self, embedding, provider, deployment_or_model, top_k, project_version_id):
        self.vector_calls.append(
            (list(embedding), provider, deployment_or_model, top_k, project_version_id)
        )
        return [dict(self.rows[key], embedding=[0.1, 0.2]) for key in self.rows]

    def edges_among(self, _version_id, _seed_ids):
        return []

    def symbol_sources(self, _version_id, symbol_ids):
        return {symbol_id: self.rows[symbol_id]["source"] for symbol_id in symbol_ids}


class _StoryGraphFixture:
    def __init__(self, store):
        self.store = store

    def expand(self, _version_id, _seed_ids, depth, max_related):
        assert depth == 1
        return {
            "neighbors": [
                {
                    **self.store.rows["constructor-record-error"],
                    "relations": [{"relation_type": "CALLS", "direction": "incoming"}],
                }
            ][:max_related]
        }


def test_long_story_derives_lexical_terms_and_keeps_vector_query_unchanged():
    from agent.retrieval import HybridRetrievalService, derive_lexical_terms

    story = (
        "Update not-found error code\n"
        "Change the not-found domain error in RecordError from NOT_FOUND to "
        "RESOURCE_NOT_FOUND.\n"
        "Only modify the intended behavior; preserve the existing source structure."
    )
    terms = derive_lexical_terms(story)
    assert terms == [
        "not-found",
        "notFound",
        "RecordError",
        "NOT_FOUND",
        "RESOURCE_NOT_FOUND",
        "resourceNotFound",
    ]
    assert len(terms) <= 24 and all(len(term) <= 80 for term in terms)

    store = _StoryStoreFixture()
    lexical = _StoryLexicalFixture()
    retrieval_service = HybridRetrievalService(store, FakeEmbeddingProvider())
    retrieval_service.lexical = lexical
    retrieval_service.graph = _StoryGraphFixture(store)

    runs = [
        retrieval_service.retrieve("story-project", story, top_k=10, graph_depth=1)
        for _ in range(5)
    ]
    result = runs[0]
    target = _item(result, "example.errors.RecordError.notFound")
    constructor = _item(result, "example.errors.RecordError.constructor")
    assert target is not None and constructor is not None
    assert result["query"] == story
    assert all(call[0] == list(FakeEmbeddingProvider().embed_texts([story])[0]) for call in store.vector_calls)
    assert all(call[4] == "version-row" and call[1:3] == ("fake", "fake-model-v1") for call in store.vector_calls)
    assert lexical.queries == terms * 5
    assert story not in lexical.queries
    assert "lexical" in target["retrieval_sources"]
    assert {match["term"] for match in target["lexical"]["matches"]} == {"NOT_FOUND", "notFound"}
    assert all({"term", "score", "match_kind", "matched_fields", "reason"} <= set(match) for match in target["lexical"]["matches"])
    assert any("notFound" in reason for reason in target["reasons"])
    assert target["final_score"] > constructor["final_score"]
    assert constructor["scores"]["graph"] == 0.10
    assert "embedding" not in str(result).lower()

    depth0 = retrieval_service.retrieve("story-project", story, top_k=10, graph_depth=0)
    assert all(item["scores"]["graph"] == 0 for item in depth0["items"])
    assert [(item["symbol_id"], item["final_score"]) for item in result["items"]] == [
        (item["symbol_id"], item["final_score"]) for item in runs[1]["items"]
    ]


# ---------------------------------------------------------------------------
# #19 — deterministic hybrid ranking
# ---------------------------------------------------------------------------

class TestHybridRanking:
    def test_single_project_version_for_all_signals(
        self, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        result = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=1)
        version = retrieval_service.store.latest_version(
            retrieval_service.store.find_project_by_external_id(sample_project.id)["id"]
        )
        assert result["project_version_id"] == version["id"]
        for item in result["items"]:
            assert item["symbol_id"]

    def test_lexical_only_candidate_works(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        result = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=0)
        item = _item(result, "com.acme.service.ContractValidator.validateContract")
        assert item is not None, "lexical candidate missing"
        assert item["lexical"] is not None
        assert item["lexical"]["score"] > 0
        assert item["scores"]["lexical"] > 0
        assert "exact identifier match" in item["reasons"]
        assert "lexical" in item["retrieval_sources"]

    def test_vector_only_candidate_works(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        result = retrieval_service.retrieve(sample_project.id, Q_SEMANTIC, top_k=20, graph_depth=0)
        vector_only = [
            i for i in result["items"]
            if i["vector"] is not None and i["lexical"] is None
        ]
        assert vector_only, "no vector-only candidate found"
        assert all("semantic match" in i["reasons"] for i in vector_only)
        assert all("vector" in i["retrieval_sources"] for i in vector_only)
        assert all(i["scores"]["vector"] > 0 for i in vector_only)

    def test_combined_candidate_carries_both(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        result = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        item = _item(result, "com.acme.service.CustomerService")
        assert item is not None
        assert {"lexical", "vector"} <= set(item["retrieval_sources"])
        assert item["scores"]["lexical"] > 0 and item["scores"]["vector"] > 0

    def test_combined_candidate_carries_graph(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        result = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        item = _item(result, "com.acme.servicetest.CustomerServiceTest")
        assert item is not None, "associated test not retrieved via graph"
        assert set(item["retrieval_sources"]) == {"lexical", "vector", "graph"}
        assert item["scores"]["graph"] > 0
        rels = {r["relation_type"] for r in item["graph"]["relations"]}
        assert "TESTED_BY" in rels

    def test_graph_depth1_influences_ranking_depth0_does_not(
        self, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        r0 = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=0, max_related=8)
        r1 = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=1, max_related=8)
        assert all(i["scores"]["graph"] == 0 for i in r0["items"])
        boosted = [i for i in r1["items"] if i["scores"]["graph"] > 0]
        assert boosted, "depth 1 produced no graph evidence"
        # graph depth re-orders the ranking (influence) …
        assert [i["symbol_id"] for i in r1["items"]] != [i["symbol_id"] for i in r0["items"]]
        # … and every boost is exactly the configured graph weight.
        for item in boosted:
            assert item["scores"]["final"] == pytest.approx(
                item["scores"]["lexical"] + item["scores"]["vector"] + 0.10
            )

    def test_graph_depth1_surfaces_neighbors(
        self, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        r0 = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=10, graph_depth=0, max_related=8)
        r1 = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=10, graph_depth=1, max_related=8)
        new_ids = set(i["symbol_id"] for i in r1["items"]) - set(i["symbol_id"] for i in r0["items"])
        assert new_ids, "graph depth 1 must surface neighbors absent at depth 0"

    def test_deterministic_ordering_same_query(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        runs = [
            retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=1)
            for _ in range(5)
        ]
        first = [(i["symbol_id"], i["final_score"]) for i in runs[0]["items"]]
        for run in runs[1:]:
            assert [(i["symbol_id"], i["final_score"]) for i in run["items"]] == first

    def test_deterministic_ordering_semantic_query(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        runs = [
            retrieval_service.retrieve(sample_project.id, Q_SEMANTIC, top_k=10, graph_depth=1)
            for _ in range(5)
        ]
        first = [(i["symbol_id"], i["final_score"]) for i in runs[0]["items"]]
        for run in runs[1:]:
            assert [(i["symbol_id"], i["final_score"]) for i in run["items"]] == first

    def test_no_raw_vectors_returned(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        result = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=1)
        dumped = str(result)
        assert "embedding" not in dumped.lower()

    def test_version_isolation(self, store, retrieval_service, index_sample, sample_project):
        v1 = index_sample(sample_project)["version"]["id"]
        v2 = index_sample(sample_project, incremental=True)["version"]["id"]
        assert v1 != v2
        result = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=50, graph_depth=1)
        assert result["project_version_id"] == v2
        v2_ids = {row["id"] for row in store.list_symbols(v2)}
        v1_ids = {row["id"] for row in store.list_symbols(v1)}
        assert v2_ids.isdisjoint(v1_ids)
        for item in result["items"]:
            assert item["symbol_id"] in v2_ids, "retrieved a symbol from an old version"

    def test_provider_model_isolation(self, store, index_sample, sample_project):
        index_sample(sample_project)
        other = FakeEmbeddingProvider(provider_name="fake-other", model_identity="fake-other-v1")
        from agent.retrieval import HybridRetrievalService

        mismatched = HybridRetrievalService(store, other)
        result = mismatched.retrieve(sample_project.id, Q_EXACT, top_k=20, graph_depth=1)
        assert result["counts"]["vector"] == 0
        assert all(i["vector"] is None for i in result["items"])
        status = mismatched.hybrid_index_status(sample_project.id)
        assert not status["available"]
        assert "current provider/model" in status["reason"]

    def test_dimension_isolation(self, store, index_sample, sample_project):
        index_sample(sample_project)
        other_dim = FakeEmbeddingProvider(dimensions=32)
        from agent.retrieval import HybridRetrievalService

        mismatched = HybridRetrievalService(store, other_dim)
        result = mismatched.retrieve(sample_project.id, Q_EXACT, top_k=20, graph_depth=1)
        assert result["counts"]["vector"] == 0

    def test_complexity_interface_ready_for_f4(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        base = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=10, graph_depth=1)
        with_complexity = retrieval_service.retrieve(
            sample_project.id, Q_EXACT, top_k=10, graph_depth=1, complexity=3
        )
        assert with_complexity["complexity"] == 3
        assert [(i["symbol_id"], i["final_score"]) for i in base["items"]] == [
            (i["symbol_id"], i["final_score"]) for i in with_complexity["items"]
        ]
        with pytest.raises(ValueError):
            retrieval_service.retrieve(sample_project.id, Q_EXACT, complexity="high")

    def test_graph_depth_validation(self, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        with pytest.raises(ValueError):
            retrieval_service.retrieve(sample_project.id, Q_EXACT, graph_depth=2)


# ---------------------------------------------------------------------------
# #20 — explainable Context Builder
# ---------------------------------------------------------------------------

_PRIORITY_LEVEL = {"high": 0, "medium": 1, "low": 2}


class TestContextBuilder:
    def _build(self, context_builder, sample_project, retrieval, budget=None, max_files=6):
        return context_builder.build(
            project=sample_project,
            query=retrieval["query"],
            retrieval=retrieval,
            estimated_token_budget=budget,
            max_context_files=max_files,
        )

    def test_items_categorised_from_retrieval_and_graph(
        self, context_builder, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        sources = {item.source for item in bundle.items}
        assert "main_candidate" in sources
        assert "ticket_evidence" in sources
        # TESTED_BY neighbour is categorized as the associated test
        test_item = _find(bundle.items, "com.acme.servicetest.CustomerServiceTest")
        assert test_item is not None and test_item.source == "associated_test"

    def test_every_item_has_full_metadata(
        self, context_builder, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        assert bundle.items
        for item in bundle.items:
            assert item.reasons, f"{item.qualified_name} has no reason"
            assert item.retrieval_sources, f"{item.qualified_name} has no retrieval source"
            assert item.priority in ("high", "medium", "low")
            assert item.estimated_tokens >= 0
            assert item.included in (True, False)
            assert item.source in (
                "ticket_evidence", "main_candidate", "related_class",
                "dependency", "associated_test", "signature",
            )
            assert item.final_score >= 0
            assert item.file_path

    def test_token_estimate_is_ceil_chars_over_4(
        self, context_builder, retrieval_service, index_sample, sample_project, codebase
    ):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        import math

        for item in bundle.items:
            symbol_source = None
            for raw in retrieval["items"]:
                if str(raw["symbol_id"]) == item.symbol_id:
                    symbol_source = raw.get("source") or ""
                    break
            expected = max(0, math.ceil(len(symbol_source or "") / 4))
            assert item.estimated_tokens == expected
        for ft in bundle.file_tokens:
            assert ft["estimated_tokens"] == max(0, math.ceil(ft["chars"] / 4))

    def test_low_value_excluded_first_under_budget(
        self, context_builder, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=20, graph_depth=1)
        budget = 200
        bundle = self._build(context_builder, sample_project, retrieval, budget=budget)
        assert bundle.excluded_count > 0, "budget too generous — tighten it"
        assert bundle.included_count > 0

        def greedy_included(items, b):
            ordered = sorted(
                items,
                key=lambda it: (_PRIORITY_LEVEL[it["priority"]], -it["final_score"], it["qualified_name"]),
            )
            used, included = 0, set()
            for it in ordered:
                if it["estimated_tokens"] <= b - used:
                    included.add(it["symbol_id"])
                    used += it["estimated_tokens"]
            return included

        actual = {i.symbol_id for i in bundle.items if i.included}
        assert actual == greedy_included([i.model_dump() for i in bundle.items], budget)

    def test_high_value_evidence_retained(self, context_builder, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=20, graph_depth=1)
        budget = 200
        bundle = self._build(context_builder, sample_project, retrieval, budget=budget)
        main = _find(bundle.items, "com.acme.service.ContractValidator.validateContract")
        assert main is not None and main.included, "main candidate lost under budget"

    def test_useful_dependency_preserved(self, context_builder, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        dep = _find(bundle.items, "com.acme.repo.CustomerRepository.exists")
        assert dep is not None, "graph dependency missing"
        assert dep.source in ("dependency", "related_class")
        assert dep.graph_relations

    def test_associated_test_preserved(self, context_builder, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        test_item = _find(bundle.items, "com.acme.servicetest.CustomerServiceTest")
        assert test_item is not None and test_item.included
        assert any(r["relation_type"] == "TESTED_BY" for r in test_item.graph_relations)

    def test_generation_input_is_full_authoritative_file_content(
        self, context_builder, retrieval_service, index_sample, sample_project, codebase
    ):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        assert bundle.selected_files
        for path in bundle.selected_files:
            full = codebase.read_file(sample_project, path)
            assert full is not None
            assert bundle.file_contents[path] == full
            assert full.lstrip().startswith("package "), "file content is not the complete source file"
        # every selected file has an item pointing at it
        item_paths = {i.file_path for i in bundle.items if i.included}
        assert set(bundle.selected_files) <= item_paths

    def test_max_context_files_respected(
        self, context_builder, retrieval_service, index_sample, sample_project
    ):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_CLASS, top_k=50, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval, max_files=3)
        assert len(bundle.selected_files) <= 3

    def test_bundle_summary_counts(self, context_builder, retrieval_service, index_sample, sample_project):
        index_sample(sample_project)
        retrieval = retrieval_service.retrieve(sample_project.id, Q_EXACT, top_k=20, graph_depth=1)
        bundle = self._build(context_builder, sample_project, retrieval)
        assert bundle.included_count + bundle.excluded_count == len(bundle.items)
        assert bundle.estimated_tokens == sum(i.estimated_tokens for i in bundle.items if i.included)
        assert bundle.retrieval_mode == "hybrid"
        assert bundle.graph_depth == 1
        assert bundle.version_number is not None


# ---------------------------------------------------------------------------
# #21 — Why-this-context endpoint + preserved review surface
# ---------------------------------------------------------------------------

def _client_with_hybrid(monkeypatch, store, fake_embedding, sample_project):
    import main as main_module
    from agent.context import ContextBuilder
    from agent.retrieval import HybridRetrievalService

    # Register the test project in the runtime config.
    existing = [p for p in main_module.orchestrator.config.projects if p.id == sample_project.id]
    if not existing:
        main_module.orchestrator.config.projects.append(sample_project)
    main_module.hybrid_retrieval = HybridRetrievalService(store, fake_embedding)
    main_module.context_builder = ContextBuilder(main_module.codebase)
    return TestClient(main_module.app)


class TestWhyThisContextEndpoint:
    def test_context_preview_returns_structured_context(
        self, monkeypatch, store, fake_embedding, index_sample, sample_project
    ):
        index_sample(sample_project)
        client = _client_with_hybrid(monkeypatch, store, fake_embedding, sample_project)
        resp = client.post(
            f"/api/agent/{sample_project.id}/context-preview",
            json={
                "title": "Validate contract status",
                "description": "Ensure validateContract returns true for a valid stored contract.",
                "acceptanceCriteria": ["validateContract handles null"],
                "priority": "P2",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["retrieval_mode"] == "hybrid"
        assert data["items"]
        assert data["included_count"] > 0
        assert data["selected_files"]
        first = data["items"][0]
        assert first["qualified_name"]
        assert first["reasons"]
        assert first["retrieval_sources"]
        assert "file_contents" not in data, "full file contents must not leak into the API payload"
        assert data["graph_depth"] in (0, 1)

    def test_context_preview_rejects_when_hybrid_unavailable(
        self, monkeypatch, store, fake_embedding, sample_project
    ):
        client = _client_with_hybrid(monkeypatch, store, fake_embedding, sample_project)
        resp = client.post(
            f"/api/agent/{sample_project.id}/context-preview",
            json={"title": "x", "description": "y", "acceptanceCriteria": [], "priority": "P3"},
        )
        assert resp.status_code == 503
        assert "Hybrid index unavailable" in resp.json()["detail"]

    def test_context_preview_does_not_create_pending_patch_or_run(
        self, monkeypatch, store, fake_embedding, index_sample, sample_project
    ):
        index_sample(sample_project)
        client = _client_with_hybrid(monkeypatch, store, fake_embedding, sample_project)
        import main as main_module

        pending_before = dict(main_module.orchestrator._pending_plans)
        runs_before = store.list_runs() if store is not None else []
        resp = client.post(
            f"/api/agent/{sample_project.id}/context-preview",
            json={"title": "x", "description": "y", "acceptanceCriteria": [], "priority": "P3"},
        )
        assert resp.status_code == 200
        assert main_module.orchestrator._pending_plans == pending_before
        assert store.list_runs() == runs_before

    def test_review_routes_intact(self, monkeypatch, store, fake_embedding, sample_project):
        """Diff/Reject/Accept surface stays intact (#21, #3)."""
        client = _client_with_hybrid(monkeypatch, store, fake_embedding, sample_project)
        routes = {getattr(r, "path", "") for r in client.app.routes}
        assert "/api/agent/{project_id}/accept-plan" in routes
        assert "/api/agent/{project_id}/reject-plan" in routes
        assert "/api/agent/{project_id}/generate-adhoc" in routes

    def test_index_status_endpoint(self, monkeypatch, store, fake_embedding, index_sample, sample_project):
        index_sample(sample_project)
        client = _client_with_hybrid(monkeypatch, store, fake_embedding, sample_project)
        resp = client.get(f"/api/agent/{sample_project.id}/index/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "database_configured" in data
        assert data["hybrid"]["available"] is True
        assert data["hybrid"]["symbols"] > 0
        assert data["hybrid"]["embeddings"] > 0
        assert data["hybrid"]["version_number"] is not None


# ---------------------------------------------------------------------------
# Orchestrator integration — hybrid primary, legacy fallback, no provider fallback
# ---------------------------------------------------------------------------

class _StubSemanticIndex:
    def __init__(self, paths=()):
        self.paths = list(paths)

    def exists(self, project):
        return bool(self.paths)

    def rebuild(self, project):
        return []

    def search(self, project, query, top_k):
        return self.paths[:top_k]


class _FakeGenerationProvider:
    provider_name = "fake"
    model_identity = "fake-model"

    def generate_patch_plan(self, project, story, relevant_files, static_analysis, log_callback, validation_feedback=""):
        from agent.models import PatchProposalPlan

        return PatchProposalPlan(storyId=story.id, summary="no-op", files=[])

    def generate_fix_plan(self, ctx, log_callback):
        return self.generate_patch_plan(None, None, [], "", log_callback)

    def count_tokens(self, text):
        return None

    def normalize_usage(self, usage):
        return None

    def capabilities(self):
        from agent.providers import ProviderCapabilities

        return ProviderCapabilities()


class TestOrchestratorIntegration:
    def _orchestrator(self, config, codebase, store, hybrid_retrieval, context_builder):
        from agent.analysis import StaticAnalysisService
        from agent.models import PatchPlan
        from agent.orchestrator import AgentOrchestrator

        return AgentOrchestrator(
            config=config,
            codebase=codebase,
            semantic_index=_StubSemanticIndex(),
            generation_provider=_FakeGenerationProvider(),
            patch_applier=None,
            git_diff=None,
            static_analysis=StaticAnalysisService(codebase),
            test_runner=None,
            story_reader=None,
            prompt_guard=None,
            store=store,
            hybrid_retrieval=hybrid_retrieval,
            context_builder=context_builder,
        )

    def _story(self):
        from agent.models import UserStory

        return UserStory(
            id="adhoc-test",
            title="Validate contract status",
            description="Ensure validateContract returns true for a valid stored contract.",
            acceptanceCriteria=["validateContract handles null"],
            priority="P3",
        )

    def test_hybrid_primary_path(self, store, fake_embedding, index_sample, sample_project, context_builder, codebase):
        from agent.config import AgentConfig
        from agent.retrieval import HybridRetrievalService

        index_sample(sample_project)
        config = AgentConfig(max_context_files=4, projects=[sample_project])
        orch = self._orchestrator(
            config,
            codebase,
            store,
            HybridRetrievalService(store, fake_embedding),
            context_builder,
        )
        logs: list[str] = []
        result = orch.process_adhoc_story(sample_project.id, self._story(), log_callback=logs.append)
        assert "Using hybrid symbol context" in logs
        assert result.retrieval_mode == "hybrid"
        assert result.context is not None
        assert result.context.retrieval_mode == "hybrid"
        assert result.context.selected_files

    def test_legacy_fallback_when_hybrid_unavailable(self, store, fake_embedding, sample_project, context_builder, codebase):
        from agent.config import AgentConfig
        from agent.retrieval import HybridRetrievalService

        # hybrid services present but the project is NOT indexed in PostgreSQL
        config = AgentConfig(max_context_files=4, projects=[sample_project])
        orch = self._orchestrator(
            config,
            codebase,
            store,
            HybridRetrievalService(store, fake_embedding),
            context_builder,
        )
        logs: list[str] = []
        result = orch.process_adhoc_story(sample_project.id, self._story(), log_callback=logs.append)
        assert "Hybrid index unavailable — using legacy semantic context" in logs
        assert result.retrieval_mode == "legacy"

    def test_legacy_fallback_without_hybrid_services(self, store, sample_project, codebase):
        from agent.config import AgentConfig

        config = AgentConfig(max_context_files=4, projects=[sample_project])
        orch = self._orchestrator(config, codebase, store, None, None)
        logs: list[str] = []
        result = orch.process_adhoc_story(sample_project.id, self._story(), log_callback=logs.append)
        assert "Hybrid index unavailable — using legacy semantic context" in logs
        assert result.retrieval_mode == "legacy"
        assert result.context is not None
        assert result.context.retrieval_mode == "legacy"
