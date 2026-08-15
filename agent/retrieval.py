"""Deterministic hybrid symbol retrieval (F2.4).

Fuses three independent signals into one deterministic final ranking of
candidate symbols for a single project version:

- LEXICAL: exact/lexical symbol search (LexicalSearchService /
  ``PgStore.search_lexical``) on name, qualified name, signature,
  module, package, path and source;
- VECTOR: exact nearest-neighbor pgvector search
  (``PgStore.search_similar``) using the same provider / deployment-or-
  model / dimension triple that was used at index time — no cross
  provider or cross version mixing;
- GRAPH: bounded 0/1-hop expansion (GraphExpansionService /
  ``PgStore.expand_neighbors``) from the top initial candidates; depth 0
  produces no graph evidence, depth 1 boosts direct neighbors.

V1 deterministic scoring (no ML reranker):

    lexical = 0.55 * clamp(lexical_score / 100)
    vector  = 0.35 * clamp(1 - distance)
    graph   = 0.10 when direct graph evidence exists (depth 1 neighbor)
    final   = lexical + vector + graph

Deterministic tie-break:

    final DESC, lexical DESC, vector DESC,
    qualified_name ASC, start_line ASC, symbol_id ASC

The same inputs always produce the same ordering. Results carry symbol
metadata, lexical/vector/graph evidence, retrieval sources, human
readable reasons, score components and the final score — never raw
vectors. ``complexity`` is accepted (and validated) purely as a stable
interface for F4; it does not change ranking in F2.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import TYPE_CHECKING

from agent.graph import GraphExpansionService
from agent.search import LexicalSearchService
from agent.versioning import _database_url, _redact_secrets

if TYPE_CHECKING:
    from agent.db.store import PgStore
    from agent.providers import EmbeddingProvider

# V1 fusion weights.
_LEXICAL_WEIGHT = 0.55
_VECTOR_WEIGHT = 0.35
_GRAPH_WEIGHT = 0.10

_LEXICAL_REASONS = {
    "exact": "exact identifier match",
    "exact_ci": "case-insensitive exact identifier match",
    "qualified": "qualified name match",
    "prefix": "identifier prefix match",
    "fuzzy": "fuzzy identifier match",
    "contains": "contains match",
}

_MAX_POOL_SIZE = 50
_MAX_LEXICAL_TERMS = 24
_MAX_LEXICAL_TERM_CHARS = 80
_TECHNICAL_TOKEN = re.compile(
    r"[A-Za-z_$][A-Za-z0-9_$]*(?:(?:[.-])[A-Za-z0-9_$]+)*"
)
_UPPER_SNAKE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_PASCAL = re.compile(r"[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")
_CAMEL = re.compile(r"[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")


def _clamp01(value: float) -> float:
    """Clamp a value into the closed unit interval [0, 1]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _qualified_name(row: dict) -> str:
    if row.get("qualified_name"):
        return row["qualified_name"]
    return ".".join(
        part
        for part in (row.get("package_name"), row.get("owner"), row.get("name"))
        if part
    )


def _camel_case(parts: list[str]) -> str:
    return parts[0].lower() + "".join(part[:1].upper() + part[1:].lower() for part in parts[1:])


def derive_lexical_terms(query: str) -> list[str]:
    """Extract bounded, deterministic code-oriented probes from ticket text."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip()
        key = term.casefold()
        if (
            term
            and len(term) <= _MAX_LEXICAL_TERM_CHARS
            and key not in seen
            and len(terms) < _MAX_LEXICAL_TERMS
        ):
            seen.add(key)
            terms.append(term)

    for token in _TECHNICAL_TOKEN.findall(query):
        if "." in token:
            add(token)
        elif "-" in token:
            add(token)
            add(_camel_case(token.split("-")))
        elif _UPPER_SNAKE.fullmatch(token):
            add(token)
            add(_camel_case(token.split("_")))
        elif _PASCAL.fullmatch(token) or _CAMEL.fullmatch(token):
            add(token)
    return terms


def _lexical_reason(match_kind: str, matched_fields: list[str]) -> str:
    reason = _LEXICAL_REASONS.get(match_kind, "contains match")
    if match_kind == "contains" and matched_fields:
        reason += f" ({', '.join(matched_fields)})"
    return reason


def _lexical_match(row: dict, term: str) -> dict:
    matched_fields = list(row.get("matched_fields") or [])
    match_kind = row.get("match_kind", "contains")
    return {
        "term": term,
        "score": float(row.get("lexical_score", 0.0)),
        "match_kind": match_kind,
        "matched_fields": matched_fields,
        "name_sim": float(row.get("name_sim", 0.0)),
        "reason": _lexical_reason(match_kind, matched_fields),
    }


def _aggregate_lexical(matches: list[dict]) -> dict:
    ordered = sorted(
        matches,
        key=lambda match: (
            -float(match["score"]),
            match["term"].casefold(),
            match["match_kind"],
            tuple(match["matched_fields"]),
        ),
    )
    strongest = ordered[0]
    strongest_score = max(0.0, float(strongest["score"])) / 100.0
    secondary_score = min(
        1.0,
        sum(max(0.0, float(match["score"])) / 100.0 for match in ordered[1:]),
    )
    # ponytail: secondary lexical evidence is capped at one extra match;
    # calibrate per-term weights only if relevance data justifies it.
    aggregate_score = 100.0 * (strongest_score + 0.20 * secondary_score)
    return {
        "score": aggregate_score,
        "term": strongest["term"],
        "match_kind": strongest["match_kind"],
        "matched_fields": sorted(
            {field for match in ordered for field in match["matched_fields"]}
        ),
        "name_sim": max(float(match["name_sim"]) for match in ordered),
        "reason": f"{len(ordered)} direct lexical term match(es)",
        "matches": ordered,
    }


class HybridRetrievalService:
    """Lexical + vector + optional graph fusion over PostgreSQL symbols."""

    def __init__(
        self,
        store: PgStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.lexical = LexicalSearchService(store)
        self.graph = GraphExpansionService(store)

    # ------------------------------------------------------------------ public

    def hybrid_index_status(self, project_id: str) -> dict:
        """Report whether the PostgreSQL hybrid index is usable for a project.

        ``available`` is True only when all of the following hold: the
        database is configured, the project has a row, it has at least one
        indexed version, that version has symbols, and at least one embedding
        of the current provider/model/dimension triple exists (so the vector
        signal is not silently empty). ``reason`` explains the first unmet
        condition; ``project_version_id``/``version_number`` describe the
        version hybrid retrieval would run against.
        """
        if self.store is None:
            return {
                "available": False,
                "reason": "database not configured",
                "project_version_id": None,
                "version_number": None,
            }
        project = self.store.find_project_by_external_id(project_id)
        if project is None:
            return {
                "available": False,
                "reason": "project has no PostgreSQL index",
                "project_version_id": None,
                "version_number": None,
            }
        version = self.store.latest_version(project["id"])
        if version is None:
            return {
                "available": False,
                "reason": "project has no indexed version",
                "project_version_id": None,
                "version_number": None,
            }
        symbols = self.store.count_symbols(version["id"])
        if symbols <= 0:
            return {
                "available": False,
                "reason": "no symbols indexed for the latest version",
                "project_version_id": version["id"],
                "version_number": version["version_number"],
            }
        embeddings = 0
        if self.embedding_provider is not None:
            try:
                embeddings = self.store.count_embeddings(
                    version["id"],
                    provider=self.embedding_provider.provider_name,
                    deployment_or_model=self.embedding_provider.model_identity,
                    dimensions=self.embedding_provider.dimensions,
                )
            except Exception:
                embeddings = 0
        if embeddings <= 0:
            return {
                "available": False,
                "reason": (
                    "no embeddings for the current provider/model "
                    "(rebuild the hybrid index)"
                ),
                "project_version_id": version["id"],
                "version_number": version["version_number"],
            }
        return {
            "available": True,
            "reason": "ok",
            "project_version_id": version["id"],
            "version_number": version["version_number"],
            "symbols": symbols,
            "embeddings": embeddings,
        }

    def retrieve(
        self,
        project_id: str,
        query: str,
        top_k: int = 10,
        graph_depth: int = 1,
        max_related: int = 8,
        complexity: int | None = None,
    ) -> dict:
        """Rank candidate symbols for a project by hybrid fusion.

        All three signals run against ONE ``project_version_id`` (the latest
        indexed version of the project). Returns a dict with the version
        identity, per-source counts, and ranked ``items`` (see module
        docstring for the item shape). Raises ValueError for validation
        errors and unknown/no-version projects.
        """
        if self.store is None:
            raise RuntimeError("database is not configured")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if graph_depth not in (0, 1):
            raise ValueError("graph_depth must be 0 or 1")
        if not isinstance(max_related, int) or max_related < 0:
            raise ValueError("max_related must be a non-negative integer")
        if complexity is not None and not isinstance(complexity, int):
            raise ValueError("complexity must be an integer or None")

        project = self.store.find_project_by_external_id(project_id)
        if project is None:
            raise ValueError(f"unknown project: {project_id}")
        version = self.store.latest_version(project["id"])
        if version is None:
            raise ValueError("no versions for project")
        version_id = version["id"]

        pool_size = min(_MAX_POOL_SIZE, max(top_k, max_related, 10))

        candidates: dict[str, dict] = {}
        lexical_candidate_ids: set[str] = set()
        vector_rows: list[dict] = []

        # --- lexical ---------------------------------------------------------
        lexical_matches: dict[str, list[dict]] = {}
        for term in derive_lexical_terms(query):
            for row in self.lexical.search(version_id, term, top_k=pool_size):
                row.pop("embedding", None)
                lexical_candidate_ids.add(row["id"])
                matches = lexical_matches.setdefault(row["id"], [])
                matches.append(_lexical_match(row, term))
                lexical_evidence = _aggregate_lexical(matches)
                lex_score = _LEXICAL_WEIGHT * (
                    float(lexical_evidence["score"]) / 100.0
                )
                candidate = candidates.get(row["id"])
                if candidate is None:
                    candidate = {
                        "row": row,
                        "lexical_comp": lex_score,
                        "lexical": lexical_evidence,
                        "vector_comp": 0.0,
                        "vector": None,
                        "graph_comp": 0.0,
                        "graph": None,
                    }
                    candidates[row["id"]] = candidate
                else:
                    candidate["lexical_comp"] = lex_score
                    candidate["lexical"] = lexical_evidence

        # --- vector ----------------------------------------------------------
        if self.embedding_provider is not None:
            try:
                query_vector = self.embedding_provider.embed_texts([query])[0]
            except Exception:
                query_vector = []
            if query_vector:
                vector_rows = self.store.search_similar(
                    query_vector,
                    provider=self.embedding_provider.provider_name,
                    deployment_or_model=self.embedding_provider.model_identity,
                    top_k=pool_size,
                    project_version_id=version_id,
                )
        for row in vector_rows:
            row.pop("embedding", None)
            distance = float(row.get("distance", 2.0))
            similarity = _clamp01(1.0 - distance)
            vec_comp = _VECTOR_WEIGHT * similarity
            vector_evidence = {
                "distance": distance,
                "similarity": similarity,
            }
            candidate = candidates.get(row["id"])
            if candidate is None:
                candidate = {
                    "row": row,
                    "lexical_comp": 0.0,
                    "lexical": None,
                    "vector_comp": vec_comp,
                    "vector": vector_evidence,
                    "graph_comp": 0.0,
                    "graph": None,
                }
                candidates[row["id"]] = candidate
            else:
                candidate["vector_comp"] = vec_comp
                candidate["vector"] = vector_evidence

        # --- initial ranking (lexical + vector) ------------------------------
        def _initial_key(item: tuple[str, dict]) -> tuple:
            _, c = item
            row = c["row"]
            return (
                -(c["lexical_comp"] + c["vector_comp"]),
                _qualified_name(row),
                row.get("start_line", 0),
                row["id"],
            )

        ordered = sorted(candidates.items(), key=_initial_key)
        seeds = [candidate for _, candidate in ordered[:max_related]]

        # --- graph expansion --------------------------------------------------
        graph_evidence_count = 0
        if graph_depth == 1 and seeds:
            seed_ids = [candidate["row"]["id"] for candidate in seeds]
            expansion = self.graph.expand(
                version_id,
                seed_ids,
                depth=1,
                max_related=max_related,
            )
            for neighbor in expansion["neighbors"]:
                relations = neighbor.get("relations") or []
                if not relations:
                    continue
                graph_evidence = {
                    "relations": [
                        {
                            "relation_type": r.get("relation_type", ""),
                            "direction": r.get("direction", ""),
                        }
                        for r in relations
                    ]
                }
                graph_evidence_count += 1
                candidate = candidates.get(neighbor["id"])
                if candidate is None:
                    candidates[neighbor["id"]] = {
                        "row": neighbor,
                        "lexical_comp": 0.0,
                        "lexical": None,
                        "vector_comp": 0.0,
                        "vector": None,
                        "graph_comp": _GRAPH_WEIGHT,
                        "graph": graph_evidence,
                    }
                else:
                    candidate["graph_comp"] = _GRAPH_WEIGHT
                    candidate["graph"] = graph_evidence

            # Seeds that share a direct edge with another seed also carry
            # direct graph evidence (e.g. a test type TESTED_BY a candidate
            # class) even though the expansion excludes seeds themselves.
            relations_by_seed: dict[str, list[dict]] = {}
            for edge in self.store.edges_among(version_id, seed_ids):
                source = edge["source_symbol_id"]
                target = edge["target_symbol_id"]
                if source == target:
                    continue
                relation_type = edge["relation_type"]
                relations_by_seed.setdefault(source, []).append(
                    {"relation_type": relation_type, "direction": "outgoing"}
                )
                relations_by_seed.setdefault(target, []).append(
                    {"relation_type": relation_type, "direction": "incoming"}
                )
            for candidate in seeds:
                relations = relations_by_seed.get(candidate["row"]["id"])
                if not relations:
                    continue
                relations = sorted(
                    relations, key=lambda r: (r["relation_type"], r["direction"])
                )
                candidate["graph_comp"] = _GRAPH_WEIGHT
                candidate["graph"] = {"relations": relations}
                graph_evidence_count += 1

        # --- final ranking ----------------------------------------------------
        def _final_key(item: tuple[str, dict]) -> tuple:
            _, c = item
            row = c["row"]
            final = c["lexical_comp"] + c["vector_comp"] + c["graph_comp"]
            return (
                -final,
                -c["lexical_comp"],
                -c["vector_comp"],
                _qualified_name(row),
                row.get("start_line", 0),
                row["id"],
            )

        ranked = sorted(candidates.items(), key=_final_key)[:top_k]

        # Enrich graph-neighbor candidates with their source text so the
        # Context Builder can compute meaningful token estimates.
        missing_source_ids = [
            candidate["row"]["id"]
            for _, candidate in ranked
            if not candidate["row"].get("source")
        ]
        if missing_source_ids:
            sources = self.store.symbol_sources(version_id, missing_source_ids)
            for _, candidate in ranked:
                row = candidate["row"]
                if not row.get("source"):
                    row["source"] = sources.get(row["id"], "")

        items = [
            self._build_item(candidate)
            for _, candidate in ranked
        ]

        return {
            "project_id": project_id,
            "project_version_id": version_id,
            "version_number": version["version_number"],
            "query": query.strip(),
            "top_k": top_k,
            "graph_depth": graph_depth,
            "max_related": max_related,
            "complexity": complexity,
            "counts": {
                "lexical": len(lexical_candidate_ids),
                "vector": len(vector_rows),
                "graph_neighbors": graph_evidence_count,
                "candidates": len(candidates),
            },
            "items": items,
        }

    # ------------------------------------------------------------------ private

    def _build_item(self, candidate: dict) -> dict:
        row = candidate["row"]
        lexical = candidate["lexical"]
        vector = candidate["vector"]
        graph = candidate["graph"]

        sources: list[str] = []
        reasons: list[str] = []
        if lexical is not None:
            sources.append("lexical")
            matches = lexical.get("matches") or []
            if matches:
                if len(matches) == 1:
                    reasons.append(matches[0]["reason"])
                reasons.extend(
                    f"direct lexical match: {match['term']} ({match['reason']})"
                    for match in matches
                )
            else:
                match_kind = lexical.get("match_kind", "contains")
                reasons.append(
                    _lexical_reason(match_kind, lexical.get("matched_fields") or [])
                )
        if vector is not None:
            sources.append("vector")
            reasons.append("semantic match")
        if graph is not None:
            sources.append("graph")
            relations = graph.get("relations", [])
            for relation in relations[:3]:
                relation_type = relation.get("relation_type", "")
                direction = relation.get("direction", "")
                reasons.append(
                    f"related dependency ({relation_type} {direction})"
                )

        lexical_comp = float(candidate["lexical_comp"])
        vector_comp = float(candidate["vector_comp"])
        graph_comp = float(candidate["graph_comp"])
        final_score = lexical_comp + vector_comp + graph_comp

        return {
            "symbol_id": row["id"],
            "qualified_name": _qualified_name(row),
            "symbol_type": row.get("symbol_type", ""),
            "name": row.get("name", ""),
            "file_path": row.get("path", ""),
            "module": row.get("module", ""),
            "package_name": row.get("package_name", ""),
            "owner": row.get("owner", ""),
            "signature": row.get("signature", ""),
            "start_line": row.get("start_line", 0),
            "end_line": row.get("end_line", 0),
            "source": row.get("source", ""),
            "lexical": lexical,
            "vector": vector,
            "graph": graph,
            "retrieval_sources": sources,
            "reasons": reasons,
            "scores": {
                "lexical": round(lexical_comp, 6),
                "vector": round(vector_comp, 6),
                "graph": round(graph_comp, 6),
                "final": round(final_score, 6),
            },
            "final_score": round(final_score, 6),
        }


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.retrieval",
        description="Deterministic hybrid symbol retrieval (lexical + vector + "
        "graph) over PostgreSQL (F2.4).",
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Runtime project id (external_id, e.g. upload-123)",
    )
    parser.add_argument("--query", required=True, help="Search query / ticket text")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Maximum ranked symbols (1-50)"
    )
    parser.add_argument(
        "--graph-depth",
        type=int,
        default=1,
        help="Graph expansion depth: 0 (no graph evidence) or 1 (direct neighbors)",
    )
    parser.add_argument(
        "--max-related",
        type=int,
        default=8,
        help="Maximum related symbols for the graph expansion",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=None,
        help="Optional F4 complexity input (does not affect F2 ranking)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides DATABASE_URL)",
    )
    args = parser.parse_args(argv)

    dsn = _database_url(args.url)
    if not dsn:
        print(
            "No database URL configured: pass --url or set DATABASE_URL",
            file=sys.stderr,
        )
        return 1

    try:
        from agent.db.store import PgStore

        store = PgStore(dsn)
        service = HybridRetrievalService(store)
        result = service.retrieve(
            project_id=args.project_id,
            query=args.query,
            top_k=args.top_k,
            graph_depth=args.graph_depth,
            max_related=args.max_related,
            complexity=args.complexity,
        )
    except Exception as exc:
        print(
            f"Hybrid retrieval failed: {_redact_secrets(str(exc), dsn)}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
