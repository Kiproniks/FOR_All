param(
    [switch]$Force,
    [switch]$Resume,
    [switch]$NoResume,
    [switch]$StrictPairwise,
    [string]$Model = "qwen2.5:1.5b",
    [switch]$StopOnFail,
    [switch]$DryPlan,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "Usage:"
    Write-Host "  .\run_all_thought_chain_books.ps1"
    Write-Host "  .\run_all_thought_chain_books.ps1 -DryPlan"
    Write-Host "  .\run_all_thought_chain_books.ps1 -Force"
    Write-Host "  .\run_all_thought_chain_books.ps1 -Resume"
    Write-Host "  .\run_all_thought_chain_books.ps1 -NoResume"
    Write-Host "  .\run_all_thought_chain_books.ps1 -StrictPairwise"
    Write-Host "  .\run_all_thought_chain_books.ps1 -Model qwen2.5:1.5b"
    Write-Host "  .\run_all_thought_chain_books.ps1 -StopOnFail"
    Write-Host ""
    Write-Host "Default mode is greedy: --mode greedy --merge-same-title-blocks --full --resume."
    Write-Host "Use -StrictPairwise for strict full pairwise LLM comparisons."
    Write-Host "DryPlan only prints books and commands. It does not run analysis and does not write DB data."
}

function Convert-ToSafeName {
    param([string]$Name)
    $safe = [System.IO.Path]::GetFileNameWithoutExtension($Name)
    foreach ($ch in [System.IO.Path]::GetInvalidFileNameChars()) {
        $safe = $safe.Replace([string]$ch, "_")
    }
    $safe = [regex]::Replace($safe, "\s+", "_")
    $safe = [regex]::Replace($safe, "[^\p{L}\p{Nd}_\.-]+", "_")
    $safe = $safe.Trim("._-")
    if ([string]::IsNullOrWhiteSpace($safe)) { $safe = "book" }
    if ($safe.Length -gt 90) { $safe = $safe.Substring(0, 90).Trim("._-") }
    return $safe
}

function ConvertTo-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    $Value | ConvertTo-Json -Depth 30 | Set-Content -Path $Path -Encoding UTF8
}

function Test-ReportQualityGate {
    param([string]$ReportPath)
    if (-not (Test-Path $ReportPath)) { return $false }
    try {
        $report = Get-Content -Raw -Path $ReportPath -Encoding UTF8 | ConvertFrom-Json
        $requiredZeroFields = @(
            "invalid_json_count",
            "timeout_count",
            "bad_thoughts_after_repair",
            "english_thoughts_after_repair",
            "mixed_language_tokens_after_repair",
            "weird_tokens_after_repair",
            "ungrounded_thoughts_after_repair",
            "bad_group_summaries_after_repair",
            "english_explanations_remaining",
            "relation_explanation_contradictions"
        )
        if ($report.quality_gate_passed -ne $true) { return $false }
        foreach ($field in $requiredZeroFields) {
            if (($report.$field | ForEach-Object { [int]$_ }) -ne 0) { return $false }
        }
        return $true
    }
    catch { return $false }
}

function Write-QualityReports {
    param(
        [Parameter(Mandatory = $true)][string]$BookName,
        [Parameter(Mandatory = $true)][string]$OutputDir
    )
    $reportPath = Join-Path $OutputDir "thought_chain_analysis_report.json"
    $qualityJson = Join-Path $OutputDir "quality_report.json"
    $qualityMd = Join-Path $OutputDir "quality_report.md"
    if (-not (Test-Path $reportPath)) {
        $quality = [ordered]@{
            book = $BookName
            report_found = $false
            quality_gate_passed = $false
            error = "thought_chain_analysis_report.json not found"
        }
        ConvertTo-JsonFile -Value $quality -Path $qualityJson
        "# Quality report`n`n- book: $BookName`n- report_found: false`n- quality_gate_passed: false`n" | Set-Content -Path $qualityMd -Encoding UTF8
        return $quality
    }

    $report = Get-Content -Raw -Path $reportPath -Encoding UTF8 | ConvertFrom-Json
    $quality = [ordered]@{
        book = $BookName
        report_found = $true
        quality_gate_passed = $report.quality_gate_passed
        quality_gate_status = $report.quality_gate_status
        quality_gate_blockers = $report.quality_gate_blockers
        status = $report.status
        analysis_mode = $report.analysis_mode
        block_generation_mode = $report.block_generation_mode
        model = $report.model
        total_sentences = $report.total_sentences
        thoughts_created = $report.thoughts_created
        sequential_groups_created = $report.sequential_groups_created
        pairwise_comparisons_total = $report.pairwise_comparisons_total
        pairwise_comparisons_done = $report.pairwise_comparisons_done
        pairwise_llm_calls = $report.pairwise_llm_calls
        greedy_comparisons_done = $report.greedy_comparisons_done
        greedy_seed_blocks = $report.greedy_seed_blocks
        merged_blocks = $report.merged_blocks
        merge_memberships_moved = $report.merge_memberships_moved
        fallback_count = $report.fallback_count
        invalid_json_count = $report.invalid_json_count
        timeout_count = $report.timeout_count
        report_json = $reportPath
        report_md = (Join-Path $OutputDir "thought_chain_analysis_report.md")
    }
    ConvertTo-JsonFile -Value $quality -Path $qualityJson

    $lines = @(
        "# Quality report",
        "",
        "- book: $BookName",
        "- report_found: true",
        "- status: $($quality.status)",
        "- analysis_mode: $($quality.analysis_mode)",
        "- block_generation_mode: $($quality.block_generation_mode)",
        "- model: $($quality.model)",
        "- quality_gate_passed: $($quality.quality_gate_passed)",
        "- quality_gate_status: $($quality.quality_gate_status)",
        "- quality_gate_blockers: $($quality.quality_gate_blockers -join ', ')",
        "- total_sentences: $($quality.total_sentences)",
        "- thoughts_created: $($quality.thoughts_created)",
        "- sequential_groups_created: $($quality.sequential_groups_created)",
        "- pairwise_comparisons_total: $($quality.pairwise_comparisons_total)",
        "- pairwise_comparisons_done: $($quality.pairwise_comparisons_done)",
        "- pairwise_llm_calls: $($quality.pairwise_llm_calls)",
        "- greedy_comparisons_done: $($quality.greedy_comparisons_done)",
        "- greedy_seed_blocks: $($quality.greedy_seed_blocks)",
        "- merged_blocks: $($quality.merged_blocks)",
        "- merge_memberships_moved: $($quality.merge_memberships_moved)",
        "- fallback_count: $($quality.fallback_count)",
        "- invalid_json_count: $($quality.invalid_json_count)",
        "- timeout_count: $($quality.timeout_count)"
    )
    $lines -join "`n" | Set-Content -Path $qualityMd -Encoding UTF8
    return $quality
}

function Join-CommandLine {
    param([string]$Exe, [string[]]$CommandArgs)
    $quoted = @("`"$Exe`"")
    foreach ($arg in $CommandArgs) {
        $escaped = $arg.Replace('"', '\"')
        $quoted += "`"$escaped`""
    }
    return ($quoted -join " ")
}

function New-ThoughtChainArgs {
    param(
        [Parameter(Mandatory = $true)][string]$BookPath,
        [Parameter(Mandatory = $true)][string]$OutputDir
    )
    $args = @(
        "manage.py",
        "run_thought_chain_analysis",
        "--file", $BookPath,
        "--full",
        "--strict",
        "--mode", $AnalysisMode,
        "--model", $Model,
        "--output-dir", $OutputDir
    )
    if ($StrictPairwise) {
        $args += "--strict-pairwise-llm"
    }
    else {
        $args += "--merge-same-title-blocks"
    }
    if ($Force) {
        $args += "--force-refresh"
    }
    elseif ($ResumeEnabled) {
        $args += "--resume"
    }
    return $args
}

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$StdoutLog,
        [string]$StderrLog,
        [string]$ProgressPath,
        [string]$BookName
    )
    $argumentLine = ($Arguments | ForEach-Object { "`"$($_.Replace('"', '\"'))`"" }) -join " "
    if (Test-Path $StdoutLog) { Remove-Item -LiteralPath $StdoutLog -Force }
    if (Test-Path $StderrLog) { Remove-Item -LiteralPath $StderrLog -Force }
    $process = Start-Process -FilePath $FilePath -ArgumentList $argumentLine -WorkingDirectory $WorkingDirectory -NoNewWindow -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
    $lastLine = ""
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 5
        if (Test-Path $StdoutLog) {
            $tail = Get-Content -Path $StdoutLog -Tail 1 -ErrorAction SilentlyContinue
            if ($tail -and $tail -ne $lastLine) {
                $lastLine = $tail
                Write-Host $lastLine
            }
        }
        ConvertTo-JsonFile -Value ([ordered]@{
            book = $BookName
            status = "running"
            pid = $process.Id
            last_output_line = $lastLine
            updated_at = (Get-Date).ToString("o")
        }) -Path $ProgressPath
    }
    $process.WaitForExit()
    return $process.ExitCode
}

if ($Help) {
    Show-Usage
    exit 0
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"
$BooksDir = Join-Path $ProjectRoot "test_books\new"
$RunsRoot = Join-Path $ProjectRoot "test_runs\thought_chain\full_auto"
$PythonExe = Join-Path $BackendDir "venv\Scripts\python.exe"
$ResumeEnabled = -not [bool]$NoResume
$AnalysisMode = if ($StrictPairwise) { "strict" } else { "greedy" }

if (-not (Test-Path $BooksDir)) { throw "Books directory not found: $BooksDir" }
$books = @(Get-ChildItem -Path $BooksDir -Filter "*.fb2" -File | Sort-Object Name)
if ($books.Count -eq 0) { throw "No .fb2 books found in $BooksDir" }

if ($DryPlan) {
    Write-Host "DRY PLAN: no analysis, no DB writes."
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Books found: $($books.Count)"
    Write-Host "Default analysis mode: $AnalysisMode"
    for ($i = 0; $i -lt $books.Count; $i++) {
        $book = $books[$i]
        $safeName = Convert-ToSafeName $book.Name
        $outputDir = Join-Path $RunsRoot $safeName
        $cmdArgs = New-ThoughtChainArgs -BookPath $book.FullName -OutputDir $outputDir
        Write-Host ""
        Write-Host ("Book {0}/{1}: {2}" -f ($i + 1), $books.Count, $book.Name)
        Write-Host "Output: $outputDir"
        Write-Host "Command:"
        Write-Host ("  " + (Join-CommandLine -Exe $PythonExe -CommandArgs $cmdArgs))
    }
    exit 0
}

if (-not (Test-Path $BackendDir)) { throw "Backend directory not found: $BackendDir" }
if (-not (Test-Path $PythonExe)) { throw "Python venv not found. Expected: $PythonExe" }

Write-Host "Checking Ollama API..."
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 8 | Out-Null }
catch { throw "Ollama is not running. Start ollama serve and retry." }

Write-Host "Checking Ollama model: $Model"
$ollamaList = & ollama list 2>$null
if ($LASTEXITCODE -ne 0) { throw "Cannot run 'ollama list'. Check Ollama PATH." }
if (-not ($ollamaList -match [regex]::Escape($Model))) { throw "Model '$Model' not found. Run: ollama pull $Model" }

Write-Host "Running Django system check..."
Push-Location $BackendDir
try {
    & $PythonExe manage.py check
    if ($LASTEXITCODE -ne 0) { throw "python manage.py check failed." }
}
finally { Pop-Location }

New-Item -ItemType Directory -Path $RunsRoot -Force | Out-Null
$summaryJson = Join-Path $RunsRoot "full_auto_summary.json"
$summaryMd = Join-Path $RunsRoot "full_auto_summary.md"
$started = Get-Date
$summary = [ordered]@{
    started_at = $started.ToString("o")
    finished_at = $null
    duration_seconds = $null
    books_found = $books.Count
    model = $Model
    analysis_mode = $AnalysisMode
    strict_pairwise = [bool]$StrictPairwise
    force = [bool]$Force
    resume = [bool]$ResumeEnabled
    stop_on_fail = [bool]$StopOnFail
    results = @()
}

for ($i = 0; $i -lt $books.Count; $i++) {
    $book = $books[$i]
    $safeName = Convert-ToSafeName $book.Name
    $outputDir = Join-Path $RunsRoot $safeName
    $outLog = Join-Path $outputDir "run.out.log"
    $errLog = Join-Path $outputDir "run.err.log"
    $progressPath = Join-Path $outputDir "progress.json"
    $reportPath = Join-Path $outputDir "thought_chain_analysis_report.json"
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

    Write-Host ""
    Write-Host ("Book {0}/{1}: {2}" -f ($i + 1), $books.Count, $book.Name)
    if (-not $Force -and (Test-ReportQualityGate -ReportPath $reportPath)) {
        Write-Host "SKIP: book already processed, quality_gate_passed=true"
        $summary.results += [ordered]@{ book = $book.Name; status = "skipped"; reason = "quality_gate_passed=true"; output_dir = $outputDir; report_json = $reportPath }
        ConvertTo-JsonFile -Value $summary -Path $summaryJson
        continue
    }

    $cmdArgs = New-ThoughtChainArgs -BookPath $book.FullName -OutputDir $outputDir
    ConvertTo-JsonFile -Value ([ordered]@{
        book = $book.Name
        status = "starting"
        analysis_mode = $AnalysisMode
        command = Join-CommandLine -Exe $PythonExe -CommandArgs $cmdArgs
        output_dir = $outputDir
        started_at = (Get-Date).ToString("o")
    }) -Path $progressPath

    Write-Host "Output: $outputDir"
    Write-Host "Logs:"
    Write-Host "  $outLog"
    Write-Host "  $errLog"
    Write-Host "Starting analysis..."

    $bookStart = Get-Date
    $exitCode = Invoke-LoggedProcess -FilePath $PythonExe -Arguments $cmdArgs -WorkingDirectory $BackendDir -StdoutLog $outLog -StderrLog $errLog -ProgressPath $progressPath -BookName $book.Name
    $bookEnd = Get-Date
    $quality = Write-QualityReports -BookName $book.Name -OutputDir $outputDir
    $bookStatus = if ($exitCode -eq 0 -and $quality.quality_gate_passed -eq $true) { "success" } else { "failed" }
    ConvertTo-JsonFile -Value ([ordered]@{
        book = $book.Name
        status = $bookStatus
        analysis_mode = $AnalysisMode
        exit_code = $exitCode
        quality_gate_passed = $quality.quality_gate_passed
        started_at = $bookStart.ToString("o")
        finished_at = $bookEnd.ToString("o")
        duration_seconds = [int]($bookEnd - $bookStart).TotalSeconds
        output_dir = $outputDir
        report_json = $reportPath
    }) -Path $progressPath

    $summary.results += [ordered]@{
        book = $book.Name
        status = $bookStatus
        analysis_mode = $AnalysisMode
        exit_code = $exitCode
        quality_gate_passed = $quality.quality_gate_passed
        output_dir = $outputDir
        report_json = $reportPath
        report_md = (Join-Path $outputDir "thought_chain_analysis_report.md")
        quality_report_json = (Join-Path $outputDir "quality_report.json")
        quality_report_md = (Join-Path $outputDir "quality_report.md")
        duration_seconds = [int]($bookEnd - $bookStart).TotalSeconds
    }
    ConvertTo-JsonFile -Value $summary -Path $summaryJson
    if ($bookStatus -ne "success" -and $StopOnFail) { break }
}

$finished = Get-Date
$summary.finished_at = $finished.ToString("o")
$summary.duration_seconds = [int]($finished - $started).TotalSeconds
ConvertTo-JsonFile -Value $summary -Path $summaryJson

$successCount = @($summary.results | Where-Object { $_.status -eq "success" }).Count
$failedCount = @($summary.results | Where-Object { $_.status -eq "failed" }).Count
$skippedCount = @($summary.results | Where-Object { $_.status -eq "skipped" }).Count
$md = @(
    "# Full auto thought-chain summary",
    "",
    "- started_at: $($summary.started_at)",
    "- finished_at: $($summary.finished_at)",
    "- duration_seconds: $($summary.duration_seconds)",
    "- books_found: $($summary.books_found)",
    "- success: $successCount",
    "- failed: $failedCount",
    "- skipped: $skippedCount",
    "- model: $Model",
    "- analysis_mode: $AnalysisMode",
    "- strict_pairwise: $([bool]$StrictPairwise)",
    "",
    "## Books"
)
foreach ($result in $summary.results) {
    $md += ""
    $md += "### $($result.book)"
    $md += "- status: $($result.status)"
    $md += "- analysis_mode: $($result.analysis_mode)"
    $md += "- quality_gate_passed: $($result.quality_gate_passed)"
    $md += "- output_dir: $($result.output_dir)"
    $md += "- report_json: $($result.report_json)"
}
$md -join "`n" | Set-Content -Path $summaryMd -Encoding UTF8

Write-Host ""
Write-Host "All done."
Write-Host "Summary JSON: $summaryJson"
Write-Host "Summary MD: $summaryMd"
