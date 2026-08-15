#!/usr/bin/env python3
"""F2.7 benchmark — compare legacy chunk retrieval vs hybrid retrieval on
real ShopPoc cases (#22).

Runs against the real indexed project in PostgreSQL/pgvector with real
Azure embeddings. Legacy = SemanticIndexService (chunk-level, file paths
out); Hybrid = HybridRetrievalService (symbol-level, full files out).

Usage: python3 f2-benchmark.py [--project-id ...] [--out-dir ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from agent.codebase import CodebaseService  # noqa: E402
from agent.config import ProjectConfig, Settings  # noqa: E402
from agent.factory import create_database_store, create_provider_runtime  # noqa: E402
from agent.guard import PromptGuardService  # noqa: E402
from agent.index import SemanticIndexService  # noqa: E402
from agent.retrieval import HybridRetrievalService  # noqa: E402

CASES = [
    {
        "id": "S1",
        "dimension": "exact-class",
        "problem": "Admin product create endpoint returns 500; trace the controller handling POST /api/v1/admin/products",
        "expected_file": "shoppoc-catalog/src/main/java/com/shoppoc/catalog/infrastructure/web/AdminProductController.java",
        "expected_symbol": "com.shoppoc.catalog.infrastructure.web.AdminProductController",
    },
    {
        "id": "S2",
        "dimension": "exact-method",
        "problem": "createProduct silently skips the SKU uniqueness check and stores duplicates",
        "expected_file": "shoppoc-catalog/src/main/java/com/shoppoc/catalog/application/CatalogApplicationService.java",
        "expected_symbol": "com.shoppoc.catalog.application.CatalogApplicationService.createProduct",
    },
    {
        "id": "S3",
        "dimension": "error-code",
        "problem": "Change the NOT_FOUND error code constant to RESOURCE_NOT_FOUND in the shared domain error type",
        "expected_file": "shoppoc-shared/src/main/java/com/shoppoc/shared/error/DomainError.java",
        "expected_symbol": "com.shoppoc.shared.error.DomainError.notFound",
    },
    {
        "id": "S4",
        "dimension": "semantic-description",
        "problem": "Order listing should only return orders belonging to the signed-in user; implement the current-user query",
        "expected_file": "shoppoc-order/src/main/java/com/shoppoc/order/application/OrderApplicationService.java",
        "expected_symbol": "com.shoppoc.order.application.OrderApplicationService.listCurrentUserOrders",
    },
    {
        "id": "S5",
        "dimension": "controller-service",
        "problem": "Creating an order needs product prices at checkout time; the order service must ask the catalog",
        "expected_file": "shoppoc-order/src/main/java/com/shoppoc/order/application/OrderApplicationService.java",
        "expected_symbol": "com.shoppoc.order.application.OrderApplicationService.createOrder",
    },
    {
        "id": "S6",
        "dimension": "associated-test",
        "problem": "Fix the failing admin product controller test after the request validation change",
        "expected_file": "shoppoc-app/src/test/java/com/shoppoc/catalog/infrastructure/web/AdminProductControllerTest.java",
        "expected_symbol": "com.shoppoc.catalog.infrastructure.web.AdminProductController",
    },
    {
        "id": "S7",
        "dimension": "implements",
        "problem": "Implement the admin order listing use case on the order service alongside the existing use cases",
        "expected_file": "shoppoc-order/src/main/java/com/shoppoc/order/application/OrderApplicationService.java",
        "expected_symbol": "com.shoppoc.order.application.OrderApplicationService.listAllOrders",
    },
    {
        "id": "S8",
        "dimension": "repository",
        "problem": "Persist payment records through the repository adapter backed by Spring Data",
        "expected_file": "shoppoc-payment/src/main/java/com/shoppoc/payment/infrastructure/persistence/JpaPaymentRepositoryAdapter.java",
        "expected_symbol": "com.shoppoc.payment.infrastructure.persistence.JpaPaymentRepositoryAdapter",
    },
    {
        "id": "S9",
        "dimension": "multi-module",
        "problem": "Checkout needs product lookup; the order module must use the catalog API port",
        "expected_file": "shoppoc-order/src/main/java/com/shoppoc/order/application/OrderApplicationService.java",
        "expected_symbol": "com.shoppoc.order.application.OrderApplicationService.createOrder",
    },
    {
        "id": "S10",
        "dimension": "typo-fuzzy",
        "problem": "AdmnProductContoller broke after the package rename; find the controller class",
        "expected_file": "shoppoc-catalog/src/main/java/com/shoppoc/catalog/infrastructure/web/AdminProductController.java",
        "expected_symbol": "com.shoppoc.catalog.infrastructure.web.AdminProductController",
    },
    {
        "id": "S11",
        "dimension": "db-impact",
        "problem": "Add a DRAFT status so draft products are not queryable in the catalog listings",
        "expected_file": "shoppoc-catalog/src/main/java/com/shoppoc/catalog/domain/ProductStatus.java",
        "expected_symbol": "com.shoppoc.catalog.domain.ProductStatus",
    },
    {
        "id": "S12",
        "dimension": "api-impact",
        "problem": "Add pagination to GET /api/v1/orders in the order controller",
        "expected_file": "shoppoc-order/src/main/java/com/shoppoc/order/infrastructure/web/OrderController.java",
        "expected_symbol": "com.shoppoc.order.infrastructure.web.OrderController",
    },
    {
        "id": "S13",
        "dimension": "security-relation",
        "problem": "Admins must not see other users' order summaries; enforce in the admin order controller",
        "expected_file": "shoppoc-order/src/main/java/com/shoppoc/order/infrastructure/web/AdminOrderController.java",
        "expected_symbol": "com.shoppoc.order.infrastructure.web.AdminOrderController",
    },
]


def hybrid_rank(results: list, expected_file: str, expected_symbol: str) -> int | None:
    for index, item in enumerate(results, start=1):
        qualified = item.get("qualified_name") or ""
        path = item.get("file_path") or item.get("path") or ""
        if qualified == expected_symbol or path == expected_file:
            return index
        if path == expected_file and item.get("symbol_type") in ("class", "interface", "enum"):
            return index
    return None


def legacy_rank(paths: list[str], expected_file: str) -> int | None:
    for index, path in enumerate(paths, start=1):
        if path == expected_file:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="upload-1786756681")
    parser.add_argument("--out-dir", default=str(REPO / ".agent" / "evidence"))
    args = parser.parse_args()

    settings = Settings.load("config.yml")
    registry = REPO / ".agent" / "projects.json"
    if registry.exists():
        for item in json.loads(registry.read_text(encoding="utf-8")):
            if not any(p.id == item["id"] for p in settings.agent.projects):
                settings.agent.projects.append(ProjectConfig(**item))
    codebase = CodebaseService(max_file_chars=settings.agent.max_file_chars)
    runtime = create_provider_runtime(settings, PromptGuardService())
    store = create_database_store(settings)
    if store is None:
        print("FAIL database not configured")
        return 1
    project = settings.agent.project_by_id(args.project_id)
    semantic = SemanticIndexService(codebase, runtime.embedding)
    hybrid = HybridRetrievalService(store, runtime.embedding)

    if not semantic.exists(project):
        print(f"INFO building legacy chunk index for {project.id} …")
        chunk_ids = semantic.rebuild(project)
        print(f"INFO legacy index built: {len(chunk_ids)} chunks")

    rows = []
    for case in CASES:
        query = case["problem"]
        legacy_paths = semantic.search(project, query, top_k=10)
        hybrid_results = hybrid.retrieve(
            project.id, query, top_k=10, graph_depth=1, max_related=8
        ).get("items", [])
        legacy = legacy_rank(legacy_paths, case["expected_file"])
        hybrid_ = hybrid_rank(hybrid_results, case["expected_file"], case["expected_symbol"])
        passed = hybrid_ is not None
        rows.append(
            {
                "case_id": case["id"],
                "dimension": case["dimension"],
                "problem": case["problem"],
                "expected_file": case["expected_file"],
                "expected_symbol": case["expected_symbol"],
                "legacy_top_results": legacy_paths[:5],
                "legacy_expected_rank": legacy,
                "hybrid_top_symbols": [
                    {
                        "qualified_name": item.get("qualified_name"),
                        "symbol_type": item.get("symbol_type"),
                        "file_path": item.get("file_path"),
                    }
                    for item in hybrid_results[:5]
                ],
                "hybrid_expected_rank": hybrid_,
                "pass": passed,
                "explanation": (
                    f"hybrid found expected in top-10 at rank {hybrid_}"
                    if passed
                    else "hybrid did not surface expected symbol in top-10"
                ),
            }
        )

    passed = sum(1 for row in rows if row["pass"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "f2-retrieval-benchmark.json"
    json_path.write_text(
        json.dumps(
            {
                "project_id": args.project_id,
                "retrieval": {"top_k": 10, "graph_depth": 1},
                "cases_total": len(rows),
                "cases_passed": passed,
                "result": "PASS" if passed == len(rows) else "FAIL",
                "cases": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"INFO evidence: {json_path}")

    md = [
        "# F2.7 Retrieval Benchmark (#22) — real ShopPoc",
        "",
        f"Project: `{args.project_id}` — {passed}/{len(rows)} cases PASS",
        "",
        "| Case | Dimension | Expected file | Legacy rank | Hybrid rank | Result |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['case_id']} | {row['dimension']} | `{row['expected_file'].split('/')[-1]}` "
            f"| {row['legacy_expected_rank']} | {row['hybrid_expected_rank']} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    md_path = out_dir / "f2-retrieval-benchmark.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"INFO evidence: {md_path}")

    print(f"RESULT: {'PASS' if passed == len(rows) else 'FAIL'} ({passed}/{len(rows)})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
