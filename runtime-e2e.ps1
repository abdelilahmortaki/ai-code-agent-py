param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ProjectId = "",
    [string]$UploadDir = "",
    [string]$ProjectName = "runtime-e2e",
    [string]$SourceDir = "",
    [string]$TargetPathSuffix = "shoppoc-shared/src/main/java/com/shoppoc/shared/error/DomainError.java",
    [string]$SearchTerm = "DomainError",
    [string]$OldCode = "",
    [string]$NewCode = "",
    [switch]$RefreshIndex,
    [ValidateRange(1, 5)]
    [int]$GenerationAttempts = 3,
    [switch]$ValidationFailProbe,
    [switch]$AcceptDestructive,
    [string]$EvidenceDir = "",
    [string]$Branch = "dev"
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
    "--attempt-cap", [string]$GenerationAttempts,
    "--branch", $Branch
)
if ($ProjectId) { $argsList += @("--project-id", $ProjectId) }
if ($UploadDir) { $argsList += @("--upload-dir", $UploadDir) }
if ($SourceDir) { $argsList += @("--source-dir", $SourceDir) }
if ($OldCode) { $argsList += @("--old-code", $OldCode) }
if ($NewCode) { $argsList += @("--new-code", $NewCode) }
if ($RefreshIndex) { $argsList += "--rebuild-index" }
if ($ValidationFailProbe) { $argsList += "--validation-fail-probe" }
if ($AcceptDestructive) { $argsList += "--accept-destructive" }
if ($EvidenceDir) { $argsList += @("--evidence-dir", $EvidenceDir) }

& $python.Source @argsList
exit $LASTEXITCODE
