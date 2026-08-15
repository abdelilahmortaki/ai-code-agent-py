# F2.7 Retrieval Benchmark (#22) — real ShopPoc

Project: `upload-1786790897` — 13/13 cases PASS

| Case | Dimension | Expected file | Legacy rank | Hybrid rank | Result |
|---|---|---|---|---|---|
| S1 | exact-class | `AdminProductController.java` | 1 | 1 | PASS |
| S2 | exact-method | `CatalogApplicationService.java` | 3 | 1 | PASS |
| S3 | error-code | `DomainError.java` | 2 | 1 | PASS |
| S4 | semantic-description | `OrderApplicationService.java` | 2 | 4 | PASS |
| S5 | controller-service | `OrderApplicationService.java` | 3 | 4 | PASS |
| S6 | associated-test | `AdminProductControllerTest.java` | 1 | 1 | PASS |
| S7 | implements | `OrderApplicationService.java` | 2 | 3 | PASS |
| S8 | repository | `JpaPaymentRepositoryAdapter.java` | 1 | 7 | PASS |
| S9 | multi-module | `OrderApplicationService.java` | 3 | 8 | PASS |
| S10 | typo-fuzzy | `AdminProductController.java` | 1 | 1 | PASS |
| S11 | db-impact | `ProductStatus.java` | 1 | 1 | PASS |
| S12 | api-impact | `OrderController.java` | 2 | 1 | PASS |
| S13 | security-relation | `AdminOrderController.java` | 3 | 1 | PASS |
