<#
.SYNOPSIS
  Upload a built kernel to the Hub (Windows) and report the outcome to a PR.

.DESCRIPTION
  PowerShell counterpart of .github/scripts/upload_and_report.py. Kernels whose
  build.toml repo-id lives outside `kernels-community` are uploaded through a
  pull request (--create-pr) to their real repo-id, and the resulting Hub PR
  links are posted back on the originating GitHub PR. Direct kernels-community
  uploads keep the prior <repo_prefix>/<kernel> behaviour and never comment.

  Windows currently only builds kernels-community kernels (dispatch allowlist),
  so the create-pr path is here for correctness if that ever changes.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$KernelBuilder,
  [Parameter(Mandatory = $true)][string]$KernelSource,
  [string]$RepoPrefix = "kernels-community",
  [string]$Branch = "",
  [string]$CommentPrNumber = "",
  [string]$Repo = $env:GITHUB_REPOSITORY,
  [string]$Label = "Build (Windows)",
  [string]$RunUrl = ""
)

$ErrorActionPreference = "Stop"
# Do not let a non-zero uploader exit throw before we can capture $LASTEXITCODE
# and post a failure comment; we handle the exit code explicitly below.
$PSNativeCommandUseErrorActionPreference = $false
$CommunityOrg = "kernels-community"

# Resolve the upload target from build.toml. External-org kernels upload to
# their real repo-id via a pull request; everything else uploads directly.
$uploadRepoId = "$RepoPrefix/$KernelSource"
$createPr = $false
$buildTomlPath = Join-Path $KernelSource "build.toml"
if (Test-Path $buildTomlPath) {
  $match = Select-String -Path $buildTomlPath -Pattern '^\s*repo-id\s*=\s*"([^"]+)"' |
    Select-Object -First 1
  if ($match) {
    $tomlRepoId = $match.Matches[0].Groups[1].Value
    $org = $tomlRepoId.Split('/')[0]
    if ($org -ne $CommunityOrg) {
      $uploadRepoId = $tomlRepoId
      $createPr = $true
    }
  }
}

if ($createPr) {
  Write-Host "Kernel '$KernelSource' repo-id '$uploadRepoId' is outside '$CommunityOrg'; uploading via pull request."
} else {
  Write-Host "Kernel '$KernelSource' uploads directly to '$uploadRepoId'."
}

$uploadArgs = @("upload", (Join-Path $KernelSource "build"), "--repo-type", "kernel", "--repo-id", $uploadRepoId)
if ($Branch -ne "") { $uploadArgs += @("--branch", $Branch) }
if ($createPr) { $uploadArgs += "--create-pr" }

Write-Host "+ $KernelBuilder $($uploadArgs -join ' ')"
# Capture combined output so PR links can be parsed, while still echoing it.
$output = & $KernelBuilder @uploadArgs 2>&1
$exitCode = $LASTEXITCODE
$output | ForEach-Object { Write-Host $_ }

# Direct uploads keep prior behaviour: propagate the exit code, no comment.
if (-not $createPr) { exit $exitCode }

if ($CommentPrNumber -eq "" -or $Repo -eq "" -or -not $env:GITHUB_TOKEN) {
  if (-not $env:GITHUB_TOKEN) { Write-Host "No GITHUB_TOKEN available; skipping PR comment." }
  exit $exitCode
}

if ($exitCode -eq 0) {
  $prUrls = @()
  foreach ($line in $output) {
    $m = [regex]::Match([string]$line, '^Pull request created:\s*(\S+)\s*$')
    if ($m.Success) { $prUrls += $m.Groups[1].Value }
  }
  if ($prUrls.Count -gt 0) {
    $links = ($prUrls | ForEach-Object { "- $_" }) -join "`n"
    $message = "### Kernel upload -> pull request opened`n`nKernel ``$KernelSource`` ($Label) targets ``$uploadRepoId``, which is outside ``$CommunityOrg``, so the build was uploaded via pull request:`n`n$links"
  } else {
    $message = "### Kernel upload`n`nKernel ``$KernelSource`` ($Label) -> ``$uploadRepoId``: upload completed, but no pull request was opened (no changes to upload)."
  }
} else {
  $details = if ($RunUrl -ne "") { " See the [workflow run]($RunUrl) for details." } else { "" }
  $message = "### Kernel upload failed`n`nUploading kernel ``$KernelSource`` ($Label) to ``$uploadRepoId`` via pull request failed (exit code $exitCode).$details"
}

try {
  $headers = @{
    Authorization           = "Bearer $env:GITHUB_TOKEN"
    Accept                  = "application/vnd.github+json"
    "X-GitHub-Api-Version"  = "2022-11-28"
  }
  $bodyJson = @{ body = $message } | ConvertTo-Json
  Invoke-RestMethod -Method Post -ContentType "application/json" `
    -Uri "https://api.github.com/repos/$Repo/issues/$CommentPrNumber/comments" `
    -Headers $headers -Body $bodyJson | Out-Null
} catch {
  Write-Host "Failed to post PR comment: $_"
}

exit $exitCode
