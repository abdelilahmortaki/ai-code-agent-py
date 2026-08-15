# Final V1 acceptance

## Provenance

- Core hardening merged to `dev`: `86068bc6b362c8d1c9505f7a6f23ff561f6e6071`.
- Final byte-preserving upload fix merged to `dev`: `0139ddecd8a6eb4aa76d8cf43ca5e714c506370d`.
- Fresh ShopPoc source commit: `8f18e3b35076376c752e475d24287a0d2ba27463`.

## Acceptance

- #22 frozen benchmark: legacy `13/13`; hybrid `13/13`; S8 expected adapter rank `7`.
- #41 targeted command: `mvn.cmd -q -Dtest=GlobalExceptionHandlerTest -DfailIfNoTests=false test`, real exit code captured. Fallback command: `mvn.cmd -q verify`, real exit code `0` in canonical E2E. No synthetic PASS path remains.
- #43 merged-dev fail probe: two bounded repair invocations, each with token estimate/count method, budget/threshold, routed model, max cost; successful calls contain Azure usage/cost; failed provider calls contain status/error; same run; Accept gate `409 VALIDATION_REQUIRED`; Reject preserves source.
- F3/F4: module complexity is derived from indexed/Maven metadata; routed `gpt-5-mini` is used consistently for provider call, token/accounting identity, pricing, and persisted trace.
- Canonical merged-dev E2E: fresh project `upload-1786797499`, all happy-path categories PASS.
- Fail probe merged-dev: fresh project `upload-1786798106`, failure/repair/gate/immutability categories PASS.
- #30/#36/#42 screenshots are in `ui/` and were captured from the rendered application.
- #77 final-dev byte proof: `upload-1786798697`; BOM, CRLF, and plain UTF-8 SHA-256 values are stable; no CRCRLF.

## Local evidence

- `canonical-merged-dev-86068bc.json`
- `fail-probe-merged-dev-86068bc.json`
- `f2-retrieval-benchmark.json` and `.md`
- `bom-upload-0139dde.json`
- `ui/finops-routing.png`, `ui/validation.png`, `ui/acceptance-review.png`

Focused regression: `25 passed`; compileall and `git diff --check` passed.
