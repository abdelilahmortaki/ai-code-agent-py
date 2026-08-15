param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ProjectId = "",
    [string]$UploadDir = "",
    [string]$ProjectName = "runtime-e2e",
    [string]$SourceDir = "",
    [string]$TargetPathSuffix = "shoppoc-shared/src/main/java/com/shoppoc/shared/error/DomainError.java",
    [string]$SearchTerm = "DomainError",
    [string]$OldCode = 'return of("NOT_FOUND", message);',
    [string]$NewCode = 'return of("RESOURCE_NOT_FOUND", message);',
    [switch]$RefreshIndex,
    [ValidateRange(1, 5)]
    [int]$GenerationAttempts = 3,
    [switch]$ValidationFailProbe,
    [switch]$AcceptDestructive,
    [string]$EvidenceDir = ""
)

$ErrorActionPreference = "Stop"
$python = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
if ($null -eq $python) { throw "Python launcher not found" }

$argsList = @(
    (Join-Path $PSScriptRoot "runtime-e2e.py"),
    "--base-url", $BaseUrl,
    "--project-name", $ProjectName,
    "--target-suffix", $TargetPathSuffix,
    "--search-term", $SearchTerm,
    "--old-code", $OldCode,
    "--new-code", $NewCode,
    "--attempt-cap", [string]$GenerationAttempts
)
if ($ProjectId) { $argsList += @("--project-id", $ProjectId) }
if ($UploadDir) { $argsList += @("--upload-dir", $UploadDir) }
if ($SourceDir) { $argsList += @("--source-dir", $SourceDir) }
if ($RefreshIndex) { $argsList += "--rebuild-index" }
if ($ValidationFailProbe) { $argsList += "--validation-fail-probe" }
if ($AcceptDestructive) { $argsList += "--accept-destructive" }
if ($EvidenceDir) { $argsList += @("--evidence-dir", $EvidenceDir) }

& $python.Source @argsList
exit $LASTEXITCODE
