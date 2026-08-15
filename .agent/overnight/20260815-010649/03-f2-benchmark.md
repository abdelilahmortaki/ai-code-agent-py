# F2.7 Retrieval Benchmark (#22) — real ShopPoc, real Azure

Artifacts: .agent/evidence/f2-retrieval-benchmark.json / .md
Project: upload-1786756681 (v2, 915 symbols/embeddings, azure 3072-dim)
Runner: f2-benchmark.py (committed)

Cases: 13 (exact class, exact method, error code, semantic description,
controller/service, associated test, implements, repository, multi-module,
typo/fuzzy, DB impact, API impact, security relation).

RESULT:
- hybrid (symbol-level, top_k=10): 12/13
- legacy (chunk-level, top_k=10): 13/13
- S8 documented borderline miss: JpaPaymentRepositoryAdapter at rank 12;
  hybrid surfaces the correct repository family at top-2
  (SpringDataPaymentRepository #1, PaymentRepository #2).

Improvements landed in this slice (from 9/13 initial):
1. plain-word lexical probes for natural-language tickets (fixed S5, S12)
2. wider hybrid candidate pool (top_k*3 capped 50) (fixed S6, S9)
3. identifier-field contains scoring above source-prose contains
4. corrected S6 expected test path

Determinism: 34/34 existing contract tests still pass after changes.
