# F5 Controlled Validation + Repair + Human Gate Acceptance (#37-#44)

## #37 per-file reason
- PATCH_PLAN_SCHEMA files[] gains required `reason`; ProposalFile/PatchFile
  carry it; materializer propagates it.
- REAL generation: modify DomainError.java reason="Update domain error code for
  not found scenarios to RESOURCE_NOT_FOUND".

## #38 stale patch (regression) — already proven by base E2E
- base_file_hash recomputed at accept (patch.py) -> 409 STALE_PATCH on drift.

## #39 disposable validation workspace
- agent/validation.py ValidationWorkspace: temp copy (no .git/target/.agent),
  proposal applied there, cleanup on every path. Reference SHA never changes.
- Proven: workspace apply+cleanup tests; reference file untouched.

## #40 safe executor (regression) — proven by base E2E (argv, no shell)

## #41 targeted Maven selection
- select_targeted_tests: TESTED_BY edges (fixed edge direction) + naming
  convention with on-disk existence check (never runs a guessed test name);
  fallback none|verify. Shell-free mvn argv.
- Real targeted run: mvn -q -Dtest=<tests> test in disposable workspace.

## #43 bounded repair — REAL, same run, traced
- run 1d9c006c test_status=failed: 3 invocations on SAME run —
  initial generation (no_limits) + repair attempt 1 + repair attempt 2
  (bounded at max_repair_attempts=2). All traced with provider azure,
  model gpt-5-mini, routing_reason "P3 + HIGH -> strong profile".

## #44 READY_FOR_REVIEW + human gate — REAL
- Validation PASS path: E2E lifecycle with validation enabled -> accept HTTP
  200 -> exact NOT_FOUND->RESOURCE_NOT_FOUND mutation (accept-after-PASS).
- Validation FAIL path (fail probe): test_status=failed ->
  accept -> HTTP 409 VALIDATION_REQUIRED -> reject -> source SHA unchanged.
- No auto-accept; no reference apply before PASS.

## #42 validation UI
- Validation tab (command, exit code, output tail) + PASS/FAIL badge in
  results header. No headless browser on VM -> screenshot not captured
  (documented, not faked).

## E2E
- Standard lifecycle: 20 categories PASS (incl. VALIDATION) on real Azure.
- Fail probe: validation FAILED, 2 repair invocations, 409 gate, reject,
  immutable — all PASS.
- 53 tests pass (46 prior + 7 validation contract tests).
