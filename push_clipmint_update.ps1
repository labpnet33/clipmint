# ClipMint: verified one-command Git update.
# Save this file inside D:\clipmint\clipmint and run it from Windows PowerShell.
# Only pushes explicitly allowed Ollama/SEO files to the development branch.
param(
    [string]$Message = "Update ClipMint Ollama integration and SEO",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$expectedBranch = "feature/ollama-startup-integration"
$expectedRemote = "fork"
$allowedRemoteUrls = @(
    "https://github.com/labpnet33/clipmint.git",
    "https://github.com/labpnet33/clipmint",
    "git@github.com:labpnet33/clipmint.git",
    "git@github.com:labpnet33/clipmint"
)
$files = @(
    "shorts_generator/config.py",
    "shorts_generator/user_config.py",
    "shorts_generator/local/llm.py",
    "shorts_generator/local/ollama_api.py",
    "shorts_generator/seo.py",
    "webapp/server.py",
    "webapp/jobs.py",
    "webapp/static/index.html",
    "webapp/static/app.js",
    "tests/test_ollama_integration.py",
    "tests/test_seo_quality.py",
    "push_clipmint_update.ps1"
)

function Stop-IfFailed([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed (exit code $LASTEXITCODE). No push was attempted."
    }
}

try {
    Set-Location -LiteralPath $PSScriptRoot
    Write-Host "ClipMint automated Git update" -ForegroundColor Cyan
    Write-Host "Location: $PSScriptRoot"

    $gitRoot = (& git rev-parse --show-toplevel 2>$null)
    Stop-IfFailed "Finding Git repository"
    if ([System.IO.Path]::GetFullPath($gitRoot.Trim()) -ne [System.IO.Path]::GetFullPath($PSScriptRoot)) {
        throw "Save and run the script from the ClipMint repository root. Found: $gitRoot"
    }

    $currentBranch = (& git branch --show-current).Trim()
    Stop-IfFailed "Checking Git branch"
    if ($currentBranch -cne $expectedBranch) {
        throw "You're on '$currentBranch'. Expected '$expectedBranch'. Switch to the development branch first; main is protected."
    }

    $remoteUrl = (& git remote get-url $expectedRemote).Trim()
    Stop-IfFailed "Checking Git remote"
    if ($remoteUrl -notin $allowedRemoteUrls) {
        throw "The '$expectedRemote' remote is '$remoteUrl', not labpnet33/clipmint. Nothing was pushed."
    }

    $alreadyStaged = @(git diff --cached --name-only)
    Stop-IfFailed "Checking staged files"
    if ($alreadyStaged.Count -gt 0) {
        throw "There are already staged files. Review or unstage them before running this script: $($alreadyStaged -join ', ')"
    }

    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Missing .venv\Scripts\python.exe. Set up the ClipMint virtual environment first."
    }

    Write-Host "`nRunning local regression tests ..." -ForegroundColor Yellow
    $testPatterns = @("test_ollama_integration.py", "test_seo_quality.py")
    foreach ($pattern in $testPatterns) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "tests\$pattern"))) {
            throw "Required test is missing: tests\$pattern. No push attempted."
        }
        Write-Host "Test: $pattern"
        & $python -m unittest discover -s tests -p $pattern -v
        Stop-IfFailed "Regression test $pattern"
    }

    $changed = @(git status --porcelain -- $files)
    Stop-IfFailed "Checking changes"
    if ($changed.Count -eq 0) {
        Write-Host "`nNo ClipMint Ollama/SEO changes to commit." -ForegroundColor Green
        exit 0
    }

    Write-Host "`nAllowed changes:" -ForegroundColor Cyan
    $changed | ForEach-Object { Write-Host $_ }
    if ($DryRun) {
        Write-Host "`nDry run complete. No files staged, committed or pushed." -ForegroundColor Green
        exit 0
    }

    foreach ($file in $files) {
        if (Test-Path -LiteralPath $file) {
            & git add -- $file
            Stop-IfFailed "Staging $file"
        }
    }

    $staged = @(git diff --cached --name-only)
    Stop-IfFailed "Listing staged files"
    if ($staged.Count -eq 0) {
        Write-Host "Nothing was staged. No push needed." -ForegroundColor Green
        exit 0
    }
    foreach ($path in $staged) {
        if ($path -notin $files) {
            throw "Unexpected staged path '$path'. Refusing to commit or push."
        }
    }

    & git diff --cached --check
    Stop-IfFailed "Checking staged diff"
    Write-Host "`nStaged summary:" -ForegroundColor Cyan
    & git diff --cached --stat
    Stop-IfFailed "Reviewing staged changes"

    & git commit -m $Message
    Stop-IfFailed "Git commit"

    Write-Host "`nPushing to $expectedRemote/$expectedBranch ..." -ForegroundColor Yellow
    & git push -u $expectedRemote "HEAD:$expectedBranch"
    Stop-IfFailed "Git push"

    Write-Host "`nSUCCESS: Update pushed to the ClipMint development branch." -ForegroundColor Green
    Write-Host "https://github.com/labpnet33/clipmint/tree/$expectedBranch"
    & git status --short
    exit 0
}
catch {
    Write-Host "`nSTOPPED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "No force push was used. Inspect 'git status' before retrying."
    exit 1
}
