param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ProjectId = "",
    [string]$TargetPathSuffix = "shoppoc-shared/src/main/java/com/shoppoc/shared/error/DomainError.java",
    [string]$SearchTerm = "DomainError",
    [string]$OldCode = 'return of("NOT_FOUND", message);',
    [string]$NewCode = 'return of("RESOURCE_NOT_FOUND", message);',
    [switch]$RefreshIndex,
    [ValidateRange(1, 5)]
    [int]$GenerationAttempts = 3
)

$ErrorActionPreference = "Stop"
$OriginalLocation = (Get-Location).Path
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
$Checks = New-Object 'System.Collections.Generic.List[object]'
$GenerationAttemptResults = New-Object 'System.Collections.Generic.List[object]'
$PendingPlan = $false
$Accepted = $false
$AcceptSucceeded = $false
$NoPendingAfterAccept = $false
$FinalContentExact = $false
$SelectedProjectId = ""
$SelectedProjectName = ""
$TargetRelativePath = ""
$TargetLocalPath = ""
$BeforeContent = ""
$ProviderName = ""
$GenerationDeployment = ""
$EmbeddingDeployment = ""
$EmbeddingDimensions = $null
$IndexRebuilt = $false
$EmbeddingMetadataObserved = $false
$VersionId = ""
$SearchResultCount = 0
$GraphNeighborCount = 0
$BeforeSha256 = ""
$PendingFirstSha256 = ""
$AfterRejectSha256 = ""
$PendingSecondSha256 = ""
$AfterAcceptSha256 = ""
$FirstGenerationAttempts = 0
$SecondGenerationAttempts = 0
$FirstRunId = ""
$SecondRunId = ""
$FirstFinalStatus = ""
$FirstHumanDecision = ""
$SecondFinalStatus = ""
$SecondHumanDecision = ""
$InvocationSummaries = New-Object 'System.Collections.Generic.List[object]'
$TokenUsage = New-Object 'System.Collections.Generic.List[object]'
$PythonCommand = $null
$Head = ""
$Branch = ""
$EvidencePath = ""
$ExitCode = 1

function Write-Pass([string]$Message) { Write-Host "PASS  $Message" -ForegroundColor Green }
function Write-Fail([string]$Message) { Write-Host "FAIL  $Message" -ForegroundColor Red }
function Write-Info([string]$Message) { Write-Host "INFO  $Message" -ForegroundColor Cyan }
function Write-Warn([string]$Message) { Write-Host "WARN  $Message" -ForegroundColor Yellow }

function Add-Check {
    param([string]$Category, [string]$Name, [bool]$Passed, [string]$Description)
    $status = if ($Passed) { "PASS" } else { "FAIL" }
    $Checks.Add([ordered]@{ category = $Category; name = $Name; status = $status; description = $Description })
    if ($Passed) { Write-Pass $Description } else { Write-Fail $Description }
}

function Require-Check {
    param([string]$Category, [string]$Name, [bool]$Passed, [string]$Description)
    Add-Check -Category $Category -Name $Name -Passed $Passed -Description $Description
    if (-not $Passed) { throw $Description }
}

function Get-Field {
    param([object]$Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-JsonArray {
    param([object]$Object, [string]$Name)
    $value = Get-Field $Object $Name
    if ($null -eq $value) { Write-Output -NoEnumerate @(); return }
    Write-Output -NoEnumerate @($value)
}

function Get-SafeDetail {
    param([object]$Response)
    $detail = Get-Field $Response.Json "detail"
    if ($null -eq $detail) { $detail = $Response.Body }
    $text = if ($null -eq $detail) { "HTTP $($Response.StatusCode)" } else { [string]$detail }
    $text = $text -replace '(?i)(api[_-]?key|secret|token|password|bearer|database[_-]?url|endpoint)\s*[:=]\s*\S+', '$1=<redacted>'
    $text = ($text -replace '\r?\n', ' ').Trim()
    return $text.Substring(0, [Math]::Min(240, $text.Length))
}

function Test-NoPendingPlanResponse {
    param([object]$Response, [string]$ProjectId)
    $detail = [string](Get-Field $Response.Json "detail")
    $expected = "No pending plan for project '$ProjectId'"
    return ([int]$Response.StatusCode -eq 404) -and ($detail -ceq $expected)
}

function Invoke-GitText {
    param([string[]]$Arguments)
    $output = & git @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) { throw "git provenance command failed" }
    return (($output -join [Environment]::NewLine).Trim())
}

function Find-PythonCommand {
    $candidates = @()
    foreach ($candidatePath in @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "venv\Scripts\python.exe")
    )) {
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) { $candidates += $candidatePath }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) { $candidates += $py.Source }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python -and $python.Source -notlike "*WindowsApps*") { $candidates += $python.Source }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        try {
            & $candidate -c "import agent.config" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    return $null
}

function Invoke-PythonJson {
    param([string]$Code)
    if ($null -eq $PythonCommand) { throw "usable Python interpreter not found" }
    $stderrFile = [IO.Path]::GetTempFileName()
    try {
        $output = $Code | & $PythonCommand - 2>$stderrFile
        $exitCode = $LASTEXITCODE
        $stderr = if (Test-Path $stderrFile) { Get-Content $stderrFile -Raw -ErrorAction SilentlyContinue } else { "" }
        if ($exitCode -ne 0) {
            $safeError = if ([string]::IsNullOrWhiteSpace($stderr)) { "Python runtime check failed" } else { ($stderr.Trim() -replace '\r?\n', ' ') }
            throw $safeError
        }
        $text = ($output -join [Environment]::NewLine).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) { throw "Python runtime check returned no result" }
        try { return ($text | ConvertFrom-Json) } catch { throw "Python runtime check returned invalid JSON" }
    } finally {
        Remove-Item $stderrFile -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-AgentApi {
    param([ValidateSet("GET", "POST")][string]$Method, [string]$Path, [object]$Body = $null)
    $request = @{
        Uri = ($BaseUrl.TrimEnd("/") + $Path)
        Method = $Method
        UseBasicParsing = $true
        ErrorAction = "Stop"
        TimeoutSec = 600
    }
    if ($Method -eq "POST") {
        $request.ContentType = "application/json"
        if ($null -ne $Body) { $request.Body = if ($Body -is [string]) { $Body } else { $Body | ConvertTo-Json -Depth 10 } }
    }
    try {
        $response = Invoke-WebRequest @request
        $raw = [string]$response.Content
        $json = $null
        if (-not [string]::IsNullOrWhiteSpace($raw)) { try { $json = $raw | ConvertFrom-Json } catch { } }
        return [pscustomobject]@{ StatusCode = [int]$response.StatusCode; Json = $json; Body = $raw }
    } catch {
        $response = $_.Exception.Response
        if ($null -eq $response) { return [pscustomobject]@{ StatusCode = 0; Json = $null; Body = "" } }
        $raw = [string]$_.ErrorDetails.Message
        try {
            if ([string]::IsNullOrWhiteSpace($raw) -and $response.PSObject.Methods.Name -contains "GetResponseStream") {
                $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
                $raw = $reader.ReadToEnd()
                $reader.Dispose()
            } elseif ([string]::IsNullOrWhiteSpace($raw) -and $null -ne $response.Content) { $raw = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() }
        } catch { }
        $json = $null
        if (-not [string]::IsNullOrWhiteSpace($raw)) { try { $json = $raw | ConvertFrom-Json } catch { } }
        return [pscustomobject]@{ StatusCode = [int]$response.StatusCode; Json = $json; Body = $raw }
    }
}

function Normalize-RelativePath([string]$Path) {
    $normalized = ($Path -replace "\\", "/").Trim()
    while ($normalized.StartsWith("./")) { $normalized = $normalized.Substring(2) }
    return $normalized.TrimStart("/")
}

function Resolve-SafeProjectFile([string]$Root, [string]$RelativePath) {
    if ([string]::IsNullOrWhiteSpace($Root)) { throw "project repo_root is missing" }
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd("\") + "\"
    $candidate = [IO.Path]::GetFullPath((Join-Path $Root ((Normalize-RelativePath $RelativePath) -replace "/", "\")))
    if (-not $candidate.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) { throw "selected target path escapes the project root" }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "selected target file is not present locally" }
    return $candidate
}

function Get-Sha256([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Read-Text([string]$Path) { return [IO.File]::ReadAllText($Path) }

function Get-CategoryStatus([string]$Category) {
    $items = @($Checks | Where-Object { $_.category -eq $Category })
    if ($items.Count -eq 0 -or @($items | Where-Object { $_.status -eq "FAIL" }).Count -gt 0) { return "FAIL" }
    return "PASS"
}

function Split-DiffCodeLine([string]$Line) {
    $code = $Line.Substring(1)
    $match = [regex]::Match($code, '^([\t ]*)(.*)$')
    return [pscustomobject]@{ Indent = $match.Groups[1].Value; Code = $match.Groups[2].Value }
}

function Test-ExactBusinessDiff([object]$GenerationJson) {
    $plan = Get-Field $GenerationJson "plan"
    $planFiles = Get-JsonArray $plan "files"
    $diff = [string](Get-Field $GenerationJson "git_diff")
    $diffLines = @($diff -split '\r?\n')
    $diffFiles = @($diffLines | Where-Object { $_ -match '^diff --git ' })
    $addedLines = @($diffLines | Where-Object { ($_ -match '^\+') -and ($_ -notmatch '^\+\+\+') })
    $removedLines = @($diffLines | Where-Object { ($_ -match '^-') -and ($_ -notmatch '^---') })
    $planPath = if ($planFiles.Count -eq 1) { Normalize-RelativePath ([string](Get-Field $planFiles[0] "path")) } else { "" }
    $planOperation = if ($planFiles.Count -eq 1) { [string](Get-Field $planFiles[0] "operation") } else { "" }
    $reason = ""
    $passed = [bool](Get-Field $GenerationJson "pending_review") -and
        ($null -ne $plan) -and ($planFiles.Count -eq 1) -and ($planPath -ceq $TargetRelativePath) -and
        ($planOperation -ceq "modify") -and ($diffFiles.Count -eq 1) -and
        ($addedLines.Count -eq 1) -and ($removedLines.Count -eq 1)
    if ($passed) {
        $removed = Split-DiffCodeLine $removedLines[0]
        $added = Split-DiffCodeLine $addedLines[0]
        $passed = ($removed.Code -ceq $OldCode) -and ($added.Code -ceq $NewCode) -and ($removed.Indent -ceq $added.Indent)
        if (-not $passed) { $reason = "removed/added code or indentation does not match the configured fixture" }
    } elseif (-not [bool](Get-Field $GenerationJson "pending_review")) {
        $reason = "generation response did not leave a pending review plan"
    } else {
        $reason = "plan or unified diff violates the exact one-file/one-line contract"
    }
    return [pscustomobject]@{
        Passed = $passed
        Reason = $reason
        PlanFileCount = $planFiles.Count
        PlanPath = $planPath
        PlanOperation = $planOperation
        DiffFileCount = $diffFiles.Count
        AddedLineCount = $addedLines.Count
        RemovedLineCount = $removedLines.Count
    }
}

function Invoke-GenerationLifecycle {
    param([string]$Lifecycle, [string]$RunNonce)
    $encodedProject = [Uri]::EscapeDataString($SelectedProjectId)
    $description = "In $TargetRelativePath, change only the notFound error code from:$([Environment]::NewLine)$([Environment]::NewLine)NOT_FOUND$([Environment]::NewLine)$([Environment]::NewLine)to:$([Environment]::NewLine)$([Environment]::NewLine)RESOURCE_NOT_FOUND$([Environment]::NewLine)$([Environment]::NewLine)Do not modify any other line, whitespace, indentation, method, or file."
    $acceptance = @("Only NOT_FOUND changes to RESOURCE_NOT_FOUND", "No formatting or whitespace changes", "No other code is changed")
    for ($attempt = 1; $attempt -le $GenerationAttempts; $attempt++) {
        $title = "Update not-found error code [$RunNonce-$Lifecycle-$attempt]"
        $storyBody = @{ title = $title; description = $description; acceptanceCriteria = $acceptance; priority = "P3" }
        Write-Info "Azure generation $Lifecycle attempt $attempt/$GenerationAttempts"
        $generation = Invoke-AgentApi -Method POST -Path "/api/agent/$encodedProject/generate-adhoc" -Body $storyBody
        $result = [ordered]@{ lifecycle = $Lifecycle; attempt = $attempt; http_status = $generation.StatusCode; outcome = ""; detail = "" }
        if ($generation.StatusCode -eq 200) {
            $contract = Test-ExactBusinessDiff $generation.Json
            $result.outcome = if ($contract.Passed) { "exact" } elseif ([bool](Get-Field $generation.Json "pending_review")) { "noncompliant_pending" } else { "contract_rejected" }
            $result.plan_file_count = $contract.PlanFileCount
            $result.diff_file_count = $contract.DiffFileCount
            $result.added_line_count = $contract.AddedLineCount
            $result.removed_line_count = $contract.RemovedLineCount
            if ($contract.Passed) {
                $script:PendingPlan = $true
                $GenerationAttemptResults.Add($result)
                return [pscustomobject]@{ Json = $generation.Json; Title = $title; Attempts = $attempt }
            }
            $result.detail = $contract.Reason
            $GenerationAttemptResults.Add($result)
            if ([bool](Get-Field $generation.Json "pending_review")) {
                $cleanup = Invoke-AgentApi -Method POST -Path "/api/agent/$encodedProject/reject-plan"
                if (($cleanup.StatusCode -ne 200) -or ((Get-Field $cleanup.Json "status") -ne "rejected")) { throw "noncompliant pending plan cleanup failed" }
                $script:PendingPlan = $false
                if ((Get-Sha256 $TargetLocalPath) -ne $BeforeSha256) { throw "source changed after rejecting noncompliant proposal" }
            }
            continue
        }
        if ($generation.StatusCode -eq 422) {
            $detail = Get-SafeDetail $generation
            $result.detail = $detail
            $retryable = @("EDIT_ANCHOR_NOT_FOUND", "EDIT_ANCHOR_AMBIGUOUS", "UNRELATED_FORMATTING_CHANGED")
            if ($retryable -notcontains ([string](Get-Field $generation.Json "detail"))) {
                $result.outcome = "validation_failed"
                $GenerationAttemptResults.Add($result)
                throw "Azure generation returned non-retryable HTTP 422: $detail"
            }
            $result.outcome = "retryable_validation"
            $GenerationAttemptResults.Add($result)
            continue
        }
        $result.outcome = "fatal_http"
        $result.detail = Get-SafeDetail $generation
        $GenerationAttemptResults.Add($result)
        throw "Azure generation returned fatal HTTP $($generation.StatusCode): $($result.detail)"
    }
    throw "$Lifecycle generation did not produce an exact proposal within $GenerationAttempts attempt(s)"
}

function Find-RunEntry([string]$Title) {
    $runsResponse = Invoke-AgentApi -Method GET -Path "/api/agent/runs?limit=500"
    Require-Check -Category "persistence" -Name "runs-http-$Title" -Passed ($runsResponse.StatusCode -eq 200) -Description "run listing responds for '$Title'"
    $entry = @($runsResponse.Json | Where-Object {
        $run = Get-Field $_ "run"
        $snapshot = Get-Field $run "story_snapshot"
        ([string](Get-Field $run "project_id") -ieq $SelectedProjectId) -and ([string](Get-Field $snapshot "title") -ceq $Title)
    })
    Require-Check -Category "persistence" -Name "run-unique-$Title" -Passed ($entry.Count -eq 1) -Description "exactly one persisted run matches the unique story title"
    return $entry[0]
}

function Get-InvocationSummaries([object]$Invocations, [string]$RunId, [string]$Label) {
    $summaries = @($Invocations | ForEach-Object {
        [ordered]@{
            lifecycle = $Label
            run_id = $RunId
            status = [string](Get-Field $_ "status")
            provider = [string](Get-Field $_ "provider")
            model = [string](Get-Field $_ "model")
            input_tokens = Get-Field $_ "prompt_tokens"
            output_tokens = Get-Field $_ "completion_tokens"
            total_tokens = Get-Field $_ "total_tokens"
        }
    })
    foreach ($summary in $summaries) {
        $InvocationSummaries.Add($summary)
        $TokenUsage.Add([ordered]@{ lifecycle = $Label; run_id = $RunId; input_tokens = $summary.input_tokens; output_tokens = $summary.output_tokens; total_tokens = $summary.total_tokens })
    }
    return $summaries
}

try {
    Set-Location $RepoRoot
    Write-Host "[1] Repo provenance"
    $Branch = Invoke-GitText @("rev-parse", "--abbrev-ref", "HEAD")
    $Head = Invoke-GitText @("rev-parse", "HEAD")
    Write-Info "branch = $Branch"
    Write-Info "HEAD = $Head"
    Require-Check -Category "provenance" -Name "branch" -Passed ($Branch -eq "dev") -Description "current branch is dev"
    $PythonCommand = Find-PythonCommand
    Require-Check -Category "config" -Name "python" -Passed ($null -ne $PythonCommand) -Description "Python with current project imports is available"

    Write-Host "[2] Backend + config"
    $config = Invoke-PythonJson @'
import json
from agent.config import Settings
s = Settings.load("config.yml")
print(json.dumps({
    "provider": s.ai_provider,
    "generation_deployment": s.azure.deployment,
    "embedding_deployment": s.azure.embedding_deployment,
    "generation_configured": bool(s.azure.endpoint and s.azure.api_key.get_secret_value() and s.azure.deployment),
    "embedding_configured": bool(s.azure.endpoint and s.azure.api_key.get_secret_value() and s.azure.embedding_deployment),
    "database_configured": bool(s.database.url.get_secret_value()),
}))
'@
    Add-Check -Category "config" -Name "settings-load" -Passed $true -Description "current Settings.load completed without exposing secrets"
    $ProviderName = [string](Get-Field $config "provider")
    $GenerationDeployment = [string](Get-Field $config "generation_deployment")
    $EmbeddingDeployment = [string](Get-Field $config "embedding_deployment")
    Require-Check -Category "config" -Name "provider" -Passed ($ProviderName -eq "azure") -Description "runtime provider is azure"
    Require-Check -Category "config" -Name "generation-config" -Passed ([bool](Get-Field $config "generation_configured")) -Description "Azure generation configuration is present"
    Require-Check -Category "config" -Name "embedding-config" -Passed ([bool](Get-Field $config "embedding_configured")) -Description "Azure embedding configuration is present"
    Require-Check -Category "database" -Name "configured" -Passed ([bool](Get-Field $config "database_configured")) -Description "database configuration is present"

    $projectsResponse = Invoke-AgentApi -Method GET -Path "/api/agent/projects"
    Require-Check -Category "backend" -Name "health" -Passed ($projectsResponse.StatusCode -eq 200) -Description "backend health endpoint responds"
    $projects = @($projectsResponse.Json)
    if ([string]::IsNullOrWhiteSpace($ProjectId)) {
        if ($projects.Count -eq 1) { $selectedProject = $projects[0] }
        else { Require-Check -Category "project" -Name "resolve" -Passed $false -Description "ProjectId is required unless exactly one project exists" }
    } else {
        $selectedProject = $projects | Where-Object { [string](Get-Field $_ "id") -ieq $ProjectId } | Select-Object -First 1
        if ($null -eq $selectedProject) { Require-Check -Category "project" -Name "resolve" -Passed $false -Description "supplied ProjectId matches a registered project" }
    }
    $SelectedProjectId = [string](Get-Field $selectedProject "id")
    $SelectedProjectName = [string](Get-Field $selectedProject "name")
    Require-Check -Category "project" -Name "resolved" -Passed (-not [string]::IsNullOrWhiteSpace($SelectedProjectId)) -Description "project resolved safely"
    Write-Info "project id = $SelectedProjectId"

    Write-Host "[3] Target + DB/index"
    $encodedProject = [Uri]::EscapeDataString($SelectedProjectId)
    $filesResponse = Invoke-AgentApi -Method GET -Path "/api/agent/$encodedProject/files"
    Require-Check -Category "project" -Name "files" -Passed ($filesResponse.StatusCode -eq 200) -Description "project files endpoint responds"
    $normalizedSuffix = Normalize-RelativePath $TargetPathSuffix
    $files = Get-JsonArray $filesResponse.Json "files"
    $matches = @($files | ForEach-Object { Normalize-RelativePath ([string]$_) } | Where-Object {
        $_ -ieq $normalizedSuffix -or $_.EndsWith("/$normalizedSuffix", [StringComparison]::OrdinalIgnoreCase)
    } | Sort-Object -Unique)
    if ($matches.Count -eq 0) { Require-Check -Category "target" -Name "fixture-present" -Passed $false -Description "E2E fixture is not present: no file matches TargetPathSuffix" }
    if ($matches.Count -gt 1) { Require-Check -Category "target" -Name "fixture-unique" -Passed $false -Description "E2E fixture is ambiguous: TargetPathSuffix matches $($matches.Count) files" }
    $TargetRelativePath = [string]$matches[0]
    $TargetLocalPath = Resolve-SafeProjectFile ([string](Get-Field $selectedProject "repo_root")) $TargetRelativePath
    Require-Check -Category "target" -Name "containment" -Passed (-not [string]::IsNullOrWhiteSpace($TargetLocalPath)) -Description "resolved target is inside the project root"
    $BeforeContent = Read-Text $TargetLocalPath
    $oldCount = ([regex]::Matches($BeforeContent, [regex]::Escape($OldCode))).Count
    $newCount = ([regex]::Matches($BeforeContent, [regex]::Escape($NewCode))).Count
    Require-Check -Category "target" -Name "precondition" -Passed (($oldCount -eq 1) -and ($newCount -eq 0)) -Description "E2E fixture is in its expected pre-change state"
    $BeforeSha256 = Get-Sha256 $TargetLocalPath
    Write-Info "target = $TargetRelativePath"

    $initialSearch = Invoke-AgentApi -Method GET -Path "/api/agent/$encodedProject/search?query=$([Uri]::EscapeDataString($SearchTerm))&top_k=10"
    $needIndex = [bool]$RefreshIndex
    if ($initialSearch.StatusCode -eq 200) {
        $VersionId = [string](Get-Field $initialSearch.Json "project_version_id")
        Require-Check -Category "database" -Name "index-version" -Passed (-not [string]::IsNullOrWhiteSpace($VersionId)) -Description "PostgreSQL index/version is available"
    } elseif ($initialSearch.StatusCode -eq 404 -and (Get-SafeDetail $initialSearch) -match "no versions for project|unknown project") {
        Write-Info "no indexed version found; building incremental PostgreSQL/pgvector index"
        $needIndex = $true
    } else {
        throw "initial PostgreSQL search failed: $(Get-SafeDetail $initialSearch)"
    }
    if ($needIndex) {
        $indexResponse = Invoke-AgentApi -Method POST -Path "/api/agent/$encodedProject/index/pg?incremental=true"
        Require-Check -Category "index" -Name "rebuild-http" -Passed ($indexResponse.StatusCode -eq 200) -Description "incremental PostgreSQL/pgvector indexing completes"
        $indexJson = $indexResponse.Json
        $version = Get-Field $indexJson "version"
        $VersionId = [string](Get-Field $version "id")
        $filesTotal = [int](Get-Field $indexJson "files_total")
        $filesIndexed = [int](Get-Field $indexJson "files_indexed")
        $filesReused = [int](Get-Field $indexJson "files_reused")
        $symbolsTotal = [int](Get-Field $indexJson "symbols_indexed") + [int](Get-Field $indexJson "symbols_reused")
        $embeddingsTotal = [int](Get-Field $indexJson "embeddings_indexed") + [int](Get-Field $indexJson "embeddings_reused")
        $failedFiles = Get-JsonArray $indexJson "failed_files"
        $embedding = Get-Field $indexJson "embedding"
        $EmbeddingDimensions = Get-Field $embedding "dimensions"
        $EmbeddingMetadataObserved = ($null -ne $EmbeddingDimensions)
        $IndexRebuilt = $true
        Require-Check -Category "index" -Name "version-fields" -Passed ((-not [string]::IsNullOrWhiteSpace($VersionId)) -and ($filesTotal -ge 0) -and ($filesIndexed -ge 0) -and ($filesReused -ge 0) -and (($filesIndexed + $filesReused) -le $filesTotal)) -Description "index version and file counters are sensible"
        Require-Check -Category "index" -Name "symbols-embeddings" -Passed (($symbolsTotal -gt 0) -and ($embeddingsTotal -gt 0)) -Description "symbols and embeddings are indexed or reused"
        Require-Check -Category "index" -Name "failed-files" -Passed ($failedFiles.Count -eq 0) -Description "index reports no failed files"
        Require-Check -Category "index" -Name "embedding-runtime" -Passed (((Get-Field $embedding "provider") -eq "azure") -and (-not [string]::IsNullOrWhiteSpace([string](Get-Field $embedding "model"))) -and (($null -eq $EmbeddingDimensions) -or ([int]$EmbeddingDimensions -gt 0))) -Description "Azure embedding metadata is valid when reported"
    }
    Require-Check -Category "index" -Name "available" -Passed (-not [string]::IsNullOrWhiteSpace($VersionId)) -Description "an indexed project version is available"
    if (-not $EmbeddingMetadataObserved) { Write-Info "embedding dimensions not observed in this run; existing index metadata was reused" }

    Write-Host "[4] Search / RAG"
    $searchResponse = Invoke-AgentApi -Method GET -Path "/api/agent/$encodedProject/search?query=$([Uri]::EscapeDataString($SearchTerm))&top_k=10"
    Require-Check -Category "search" -Name "lexical-http" -Passed ($searchResponse.StatusCode -eq 200) -Description "configured lexical search responds with HTTP 200"
    $results = Get-JsonArray $searchResponse.Json "results"
    $VersionId = [string](Get-Field $searchResponse.Json "project_version_id")
    $SearchResultCount = $results.Count
    $targetStem = [IO.Path]::GetFileNameWithoutExtension($TargetRelativePath)
    $usefulResults = @($results | Where-Object {
        $path = Normalize-RelativePath ([string](Get-Field $_ "path"))
        $name = [string](Get-Field $_ "name")
        $qualified = [string](Get-Field $_ "qualified_name")
        ($path -ieq $TargetRelativePath) -or ($name -ieq $SearchTerm) -or ($name -ieq $targetStem) -or ($qualified -match "(?i)(^|\.)$([regex]::Escape($SearchTerm))$")
    } | Sort-Object @{Expression={ [string](Get-Field $_ "qualified_name") }}, @{Expression={ [int](Get-Field $_ "start_line") }}, @{Expression={ [string](Get-Field $_ "path") }})
    Require-Check -Category "search" -Name "lexical-results" -Passed (($results.Count -gt 0) -and (-not [string]::IsNullOrWhiteSpace($VersionId))) -Description "lexical retrieval returns useful indexed results"
    Require-Check -Category "search" -Name "target-result" -Passed ($usefulResults.Count -gt 0) -Description "lexical retrieval includes the configured target file or symbol"
    $seedResult = $usefulResults[0]
    $seed = [string](Get-Field $seedResult "qualified_name")
    if ([string]::IsNullOrWhiteSpace($seed)) {
        $owner = [string](Get-Field $seedResult "owner")
        $name = [string](Get-Field $seedResult "name")
        $seed = if ([string]::IsNullOrWhiteSpace($owner)) { $name } else { "$owner.$name" }
    }
    Require-Check -Category "search" -Name "graph-seed" -Passed (-not [string]::IsNullOrWhiteSpace($seed)) -Description "a deterministic useful lexical result provides a graph seed"
    $encodedSeed = [Uri]::EscapeDataString($seed)

    Write-Host "[5] Graph"
    $depth0 = Invoke-AgentApi -Method GET -Path "/api/agent/$encodedProject/expand?seeds=$encodedSeed&depth=0&max_related=20"
    Require-Check -Category "graph" -Name "depth0-http" -Passed ($depth0.StatusCode -eq 200) -Description "graph depth 0 responds with HTTP 200"
    Require-Check -Category "graph" -Name "depth0-shape" -Passed (((Get-JsonArray $depth0.Json "seeds").Count -gt 0) -and ((Get-JsonArray $depth0.Json "unresolved_seeds").Count -eq 0) -and ((Get-JsonArray $depth0.Json "neighbors").Count -eq 0) -and ([int](Get-Field $depth0.Json "depth") -eq 0)) -Description "graph depth 0 resolves the seed and returns no neighbors"
    $depth1 = Invoke-AgentApi -Method GET -Path "/api/agent/$encodedProject/expand?seeds=$encodedSeed&depth=1&max_related=20"
    Require-Check -Category "graph" -Name "depth1-http" -Passed ($depth1.StatusCode -eq 200) -Description "graph depth 1 responds with HTTP 200"
    $depth1Seeds = Get-JsonArray $depth1.Json "seeds"
    $depth1Neighbors = Get-JsonArray $depth1.Json "neighbors"
    $GraphNeighborCount = $depth1Neighbors.Count
    $depth1SeedMatch = @($depth1Seeds | Where-Object { ([string](Get-Field $_ "qualified_name") -eq $seed) -or ("$((Get-Field $_ 'owner')).$((Get-Field $_ 'name'))" -eq $seed) -or ([string](Get-Field $_ "name") -eq $seed) }).Count -gt 0
    Require-Check -Category "graph" -Name "depth1-shape" -Passed ($depth1SeedMatch -and ([int](Get-Field $depth1.Json "depth") -eq 1) -and ($null -ne (Get-Field $depth1.Json "neighbors")) -and ($null -ne (Get-Field $depth1.Json "capped"))) -Description "graph depth 1 preserves the seed and reports neighbors/cap"
    $invalidDepth = Invoke-AgentApi -Method GET -Path "/api/agent/$encodedProject/expand?seeds=$encodedSeed&depth=2&max_related=20"
    Require-Check -Category "graph" -Name "invalid-depth" -Passed ($invalidDepth.StatusCode -eq 400) -Description "graph rejects unsupported depth 2 with HTTP 400"

    Write-Host "[6] Priority + executor"
    $priority = Invoke-PythonJson @'
import json
from agent.priority import normalize_priority
checks = {"P1": normalize_priority("P1"), "p2": normalize_priority("p2"), "Medium": normalize_priority("Medium")}
try:
    normalize_priority("not-a-priority")
    checks["invalid_rejected"] = False
except ValueError:
    checks["invalid_rejected"] = True
print(json.dumps(checks))
'@
    Require-Check -Category "priority" -Name "normalization" -Passed (((Get-Field $priority "P1") -eq "P1") -and ((Get-Field $priority "p2") -eq "P2") -and ((Get-Field $priority "Medium") -eq "P3") -and ([bool](Get-Field $priority "invalid_rejected"))) -Description "priority normalization maps P1, p2, Medium and rejects invalid values"
    $executor = Invoke-PythonJson @'
import json
import sys
from pathlib import Path
from agent.safe_runner import SafeCommandExecutor
r = SafeCommandExecutor().run([sys.executable, "-c", "print('SAFE_EXEC_OK')"], cwd=Path.cwd(), timeout=10)
print(json.dumps({"exit_code": r.exit_code, "ok": "SAFE_EXEC_OK" in r.output}))
'@
    Require-Check -Category "executor" -Name "argv" -Passed (([int](Get-Field $executor "exit_code") -eq 0) -and ([bool](Get-Field $executor "ok"))) -Description "production SafeCommandExecutor runs argv without a shell"

    Write-Host "[7] First generation / reject"
    $runNonce = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmssfff") + "-" + (([guid]::NewGuid().ToString("N")).Substring(0, 8))
    $first = Invoke-GenerationLifecycle -Lifecycle "first" -RunNonce $runNonce
    $FirstGenerationAttempts = $first.Attempts
    $firstContract = Test-ExactBusinessDiff $first.Json
    Require-Check -Category "generation" -Name "first-http" -Passed $true -Description "first Azure generation responds with HTTP 200"
    Require-Check -Category "generation" -Name "first-exact-diff" -Passed $firstContract.Passed -Description "first proposal is the exact one-line business replacement"
    $PendingFirstSha256 = Get-Sha256 $TargetLocalPath
    Require-Check -Category "immutable" -Name "first-pending" -Passed ($PendingFirstSha256 -eq $BeforeSha256) -Description "source remains unchanged while first proposal is pending"
    $firstEntry = Find-RunEntry $first.Title
    $firstRun = Get-Field $firstEntry "run"
    $FirstRunId = [string](Get-Field $firstRun "id")
    Require-Check -Category "persistence" -Name "first-awaiting-review" -Passed (((Get-Field $firstRun "priority") -eq "P3") -and ((Get-Field $firstRun "status") -eq "awaiting_review") -and ($null -eq (Get-Field $firstRun "completed_at")) -and ($null -eq (Get-Field $firstRun "human_decision"))) -Description "first persisted run is P3, awaiting_review, and incomplete"
    $firstInvocations = Get-JsonArray $firstEntry "invocations"
    $null = Get-InvocationSummaries $firstInvocations $FirstRunId "first"
    $firstLatest = if ($firstInvocations.Count -gt 0) { $firstInvocations[-1] } else { $null }
    Require-Check -Category "persistence" -Name "first-invocation" -Passed (($firstInvocations.Count -gt 0) -and ((Get-Field $firstLatest "status") -eq "ok") -and ((Get-Field $firstLatest "provider") -eq "azure") -and (-not [string]::IsNullOrWhiteSpace([string](Get-Field $firstLatest "model")))) -Description "first linked latest LLM invocation is ok, Azure, and has a deployment/model"
    $reject = Invoke-AgentApi -Method POST -Path "/api/agent/$encodedProject/reject-plan"
    Require-Check -Category "reject" -Name "first-http" -Passed (($reject.StatusCode -eq 200) -and ((Get-Field $reject.Json "status") -eq "rejected")) -Description "first pending plan is rejected through the real API"
    $PendingPlan = $false
    $AfterRejectSha256 = Get-Sha256 $TargetLocalPath
    Require-Check -Category "immutable" -Name "after-reject" -Passed ($AfterRejectSha256 -eq $BeforeSha256) -Description "source SHA is unchanged after first Reject"
    $runsAfterReject = Invoke-AgentApi -Method GET -Path "/api/agent/runs?limit=500"
    Require-Check -Category "reject" -Name "first-runs-http" -Passed ($runsAfterReject.StatusCode -eq 200) -Description "run listing remains available after first reject"
    $firstFinalEntry = @($runsAfterReject.Json | Where-Object { [string](Get-Field (Get-Field $_ "run") "id") -eq $FirstRunId }) | Select-Object -First 1
    $firstFinalRun = if ($null -ne $firstFinalEntry) { Get-Field $firstFinalEntry "run" } else { $null }
    $FirstFinalStatus = [string](Get-Field $firstFinalRun "status")
    $FirstHumanDecision = [string](Get-Field $firstFinalRun "human_decision")
    Require-Check -Category "reject" -Name "first-terminal" -Passed (($FirstFinalStatus -eq "completed") -and ($FirstHumanDecision -eq "rejected") -and ($null -ne (Get-Field $firstFinalRun "completed_at"))) -Description "first run is completed with human_decision=rejected"

    Write-Host "[8] Second generation / accept"
    $second = Invoke-GenerationLifecycle -Lifecycle "second" -RunNonce $runNonce
    $SecondGenerationAttempts = $second.Attempts
    $secondContract = Test-ExactBusinessDiff $second.Json
    Require-Check -Category "generation" -Name "second-http" -Passed $true -Description "second Azure generation responds with HTTP 200"
    Require-Check -Category "generation" -Name "second-exact-diff" -Passed $secondContract.Passed -Description "second proposal is the exact one-line business replacement"
    $PendingSecondSha256 = Get-Sha256 $TargetLocalPath
    Require-Check -Category "immutable" -Name "second-pending" -Passed ($PendingSecondSha256 -eq $BeforeSha256) -Description "source remains unchanged while second proposal is pending"
    $secondEntry = Find-RunEntry $second.Title
    $secondRun = Get-Field $secondEntry "run"
    $SecondRunId = [string](Get-Field $secondRun "id")
    Require-Check -Category "persistence" -Name "second-awaiting-review" -Passed (((Get-Field $secondRun "priority") -eq "P3") -and ((Get-Field $secondRun "status") -eq "awaiting_review") -and ($null -eq (Get-Field $secondRun "completed_at")) -and ($null -eq (Get-Field $secondRun "human_decision"))) -Description "second persisted run is P3, awaiting_review, and incomplete"
    $secondInvocations = Get-JsonArray $secondEntry "invocations"
    $null = Get-InvocationSummaries $secondInvocations $SecondRunId "second"
    $secondLatest = if ($secondInvocations.Count -gt 0) { $secondInvocations[-1] } else { $null }
    Require-Check -Category "persistence" -Name "second-invocation" -Passed (($secondInvocations.Count -gt 0) -and ((Get-Field $secondLatest "status") -eq "ok") -and ((Get-Field $secondLatest "provider") -eq "azure") -and (-not [string]::IsNullOrWhiteSpace([string](Get-Field $secondLatest "model")))) -Description "second linked latest LLM invocation is ok, Azure, and has a deployment/model"
    Require-Check -Category "immutable" -Name "before-accept" -Passed ((Get-Sha256 $TargetLocalPath) -eq $BeforeSha256) -Description "source SHA is original immediately before Accept"
    $accept = Invoke-AgentApi -Method POST -Path "/api/agent/$encodedProject/accept-plan"
    Require-Check -Category "accept" -Name "http" -Passed (($accept.StatusCode -eq 200) -and ((Get-Field $accept.Json "status") -eq "accepted")) -Description "second pending plan is accepted through the real API"
    $Accepted = $true
    $AcceptSucceeded = $true
    $PendingPlan = $false
    $AfterAcceptSha256 = Get-Sha256 $TargetLocalPath
    Require-Check -Category "accept" -Name "sha-changed" -Passed ($AfterAcceptSha256 -ne $BeforeSha256) -Description "source SHA changes after Accept"
    $noPending = Invoke-AgentApi -Method POST -Path "/api/agent/$encodedProject/accept-plan"
    $NoPendingAfterAccept = Test-NoPendingPlanResponse $noPending $SelectedProjectId
    Require-Check -Category "accept" -Name "no-pending-plan" -Passed $NoPendingAfterAccept -Description "no pending plan remains after Accept"
    $AfterContent = Read-Text $TargetLocalPath
    $ExpectedContent = $BeforeContent.Replace($OldCode, $NewCode)
    $finalContentExact = $AfterContent -ceq $ExpectedContent
    $FinalContentExact = $finalContentExact
    Add-Check -Category "accept" -Name "exact-content" -Passed $finalContentExact -Description "final content equals the original with exactly one configured replacement"
    $semanticReplacement = (([regex]::Matches($AfterContent, [regex]::Escape($OldCode))).Count -eq 0) -and (([regex]::Matches($AfterContent, [regex]::Escape($NewCode))).Count -eq 1)
    Add-Check -Category "accept" -Name "semantic-replacement" -Passed $semanticReplacement -Description "only NOT_FOUND is replaced by RESOURCE_NOT_FOUND"
    $runsAfterAccept = Invoke-AgentApi -Method GET -Path "/api/agent/runs?limit=500"
    Require-Check -Category "accept" -Name "runs-http" -Passed ($runsAfterAccept.StatusCode -eq 200) -Description "run listing remains available after Accept"
    $secondFinalEntry = @($runsAfterAccept.Json | Where-Object { [string](Get-Field (Get-Field $_ "run") "id") -eq $SecondRunId }) | Select-Object -First 1
    $secondFinalRun = if ($null -ne $secondFinalEntry) { Get-Field $secondFinalEntry "run" } else { $null }
    $SecondFinalStatus = [string](Get-Field $secondFinalRun "status")
    $SecondHumanDecision = [string](Get-Field $secondFinalRun "human_decision")
    Require-Check -Category "accept" -Name "terminal" -Passed (($SecondFinalStatus -eq "completed") -and ($SecondHumanDecision -eq "accepted") -and ($null -ne (Get-Field $secondFinalRun "completed_at"))) -Description "second run is completed with human_decision=accepted"
    $secondFinalInvocations = Get-JsonArray $secondFinalEntry "invocations"
    $null = Get-InvocationSummaries $secondFinalInvocations $SecondRunId "second-final"

    Write-Host "[9] FinOps/provider boundary"
    $finops = Invoke-PythonJson @'
import json
from agent.config import Settings
from agent.factory import create_provider_runtime
from agent.guard import PromptGuardService
s = Settings.load("config.yml")
runtime = create_provider_runtime(s, PromptGuardService())
p = runtime.generation
c = p.capabilities()
print(json.dumps({"provider_name": p.provider_name, "model_identity": p.model_identity, "usage": c.usage, "exact_token_counting": c.exact_token_counting, "count_tokens_is_none": p.count_tokens("runtime-e2e") is None}))
'@
    Require-Check -Category "finops" -Name "provider-boundary" -Passed (((Get-Field $finops "provider_name") -eq "azure") -and (-not [string]::IsNullOrWhiteSpace([string](Get-Field $finops "model_identity"))) -and ([bool](Get-Field $finops "usage")) -and (-not [bool](Get-Field $finops "exact_token_counting")) -and ([bool](Get-Field $finops "count_tokens_is_none"))) -Description "Azure provider reports usage, no exact token counting, and count_tokens=None"
    $persistedUsageAvailable = @($InvocationSummaries | Where-Object { $null -ne $_.input_tokens -or $null -ne $_.output_tokens -or $null -ne $_.total_tokens }).Count -gt 0
    Add-Check -Category "finops" -Name "persisted-usage" -Passed $true -Description "persisted invocation token usage is reported when available (available=$persistedUsageAvailable)"
    if (-not $finalContentExact -or -not $semanticReplacement) { throw "final content proof failed after Accept" }
    $ExitCode = 0
} catch {
    Write-Fail "runtime acceptance stopped: $($_.Exception.Message)"
} finally {
    if ($PendingPlan -and -not [string]::IsNullOrWhiteSpace($SelectedProjectId)) {
        try {
            $cleanup = Invoke-AgentApi -Method POST -Path "/api/agent/$([Uri]::EscapeDataString($SelectedProjectId))/reject-plan"
            if (($cleanup.StatusCode -eq 200) -and ((Get-Field $cleanup.Json "status") -eq "rejected")) {
                Write-Info "pending plan cleanup: rejected"
                $PendingPlan = $false
                if (-not $Accepted -and -not [string]::IsNullOrWhiteSpace($TargetLocalPath)) {
                    $cleanupSha = Get-Sha256 $TargetLocalPath
                    Add-Check -Category "cleanup" -Name "source-immutable" -Passed ($cleanupSha -eq $BeforeSha256) -Description "cleanup leaves source at the original SHA before Accept"
                    if ($cleanupSha -ne $BeforeSha256) { $ExitCode = 1 }
                }
            } else { Write-Fail "pending plan cleanup failed"; $ExitCode = 1 }
        } catch { Write-Fail "pending plan cleanup failed: $($_.Exception.Message)"; $ExitCode = 1 }
    }
    $runtimeOverall = if ($ExitCode -eq 0 -and @($Checks | Where-Object { $_.status -eq "FAIL" }).Count -eq 0) { "PASS" } else { "FAIL" }
    $evidencePass = $false
    $evidenceDir = Join-Path $RepoRoot ".agent\e2e"
    try {
        New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
        $EvidencePath = Join-Path $evidenceDir ("runtime-e2e-{0}.json" -f (Get-Date).ToString("yyyyMMdd-HHmmssfff"))
        $evidence = [ordered]@{
            timestamp = (Get-Date).ToUniversalTime().ToString("o")
            branch = $Branch
            head = $Head
            project_id = $SelectedProjectId
            project_name = $SelectedProjectName
            target_relative_path = $TargetRelativePath
            search_term = $SearchTerm
            before_sha256 = $BeforeSha256
            pending_first_sha256 = $PendingFirstSha256
            after_reject_sha256 = $AfterRejectSha256
            pending_second_sha256 = $PendingSecondSha256
            after_accept_sha256 = $AfterAcceptSha256
            accept_success = $AcceptSucceeded
            no_pending_after_accept = $NoPendingAfterAccept
            final_content_exact = $FinalContentExact
            first_generation_attempts = $FirstGenerationAttempts
            second_generation_attempts = $SecondGenerationAttempts
            first_run_id = $FirstRunId
            first_final_status = $FirstFinalStatus
            first_human_decision = $FirstHumanDecision
            second_run_id = $SecondRunId
            second_final_status = $SecondFinalStatus
            second_human_decision = $SecondHumanDecision
            provider = $ProviderName
            generation_deployment = $GenerationDeployment
            embedding_deployment = $EmbeddingDeployment
            version_id = $VersionId
            index_available = (-not [string]::IsNullOrWhiteSpace($VersionId))
            index_rebuilt = $IndexRebuilt
            embedding_metadata_observed = $EmbeddingMetadataObserved
            embedding_dimensions = $EmbeddingDimensions
            search_result_count = $SearchResultCount
            graph_neighbor_count = $GraphNeighborCount
            invocation_summaries = $InvocationSummaries.ToArray()
            token_usage = $TokenUsage.ToArray()
            generation_attempt_results = $GenerationAttemptResults.ToArray()
            checks = $Checks.ToArray()
            pre_accept_immutable = ((Get-CategoryStatus 'immutable') -eq 'PASS')
            finops_result = Get-CategoryStatus 'finops'
            runtime_overall = $runtimeOverall
            overall_result = $runtimeOverall
        }
        $utf8 = New-Object System.Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($EvidencePath, ($evidence | ConvertTo-Json -Depth 12), $utf8)
        $readBack = [IO.File]::ReadAllText($EvidencePath) | ConvertFrom-Json
        if ($null -eq $readBack) { throw "evidence read-back returned no JSON" }
        $evidencePass = $true
        Write-Info "sanitized evidence = $EvidencePath"
        Write-Pass "sanitized evidence JSON writes and reads back"
    } catch { Write-Fail "sanitized evidence failed: $($_.Exception.Message)" }
    $overall = if ($runtimeOverall -eq "PASS" -and $evidencePass) { "PASS" } else { "FAIL" }
    $ExitCode = if ($overall -eq "PASS") { 0 } else { 1 }
    Write-Host ""
    Write-Host "========================================"
    Write-Host " RUNTIME E2E ACCEPTANCE"
    Write-Host "========================================"
    Write-Host "BRANCH:      $Branch"
    Write-Host "HEAD:        $Head"
    Write-Host "PROJECT:     $SelectedProjectId"
    Write-Host "TARGET:      $TargetRelativePath"
    Write-Host "DATABASE:    $(Get-CategoryStatus 'database')"
    Write-Host "INDEX/RAG:   $(Get-CategoryStatus 'index')"
    Write-Host "SEARCH:      $(Get-CategoryStatus 'search')"
    Write-Host "GRAPH:       $(Get-CategoryStatus 'graph')"
    Write-Host "PRIORITY:    $(Get-CategoryStatus 'priority')"
    Write-Host "EXECUTOR:    $(Get-CategoryStatus 'executor')"
    Write-Host "GENERATION:  $(Get-CategoryStatus 'generation')"
    Write-Host "RUN TRACE:   $(Get-CategoryStatus 'persistence')"
    Write-Host "REJECT:      $(Get-CategoryStatus 'reject')"
    Write-Host "ACCEPT:      $(Get-CategoryStatus 'accept')"
    Write-Host "IMMUTABLE:   $(Get-CategoryStatus 'immutable')"
    Write-Host "FINOPS:      $(Get-CategoryStatus 'finops')"
    Write-Host "EVIDENCE:    $(if ($evidencePass) { 'PASS' } else { 'FAIL' })"
    Write-Host ""
    Write-Host "RESULT: $overall" -ForegroundColor $(if ($overall -eq "PASS") { "Green" } else { "Red" })
    Write-Host "========================================"
    Set-Location $OriginalLocation
}
exit $ExitCode
