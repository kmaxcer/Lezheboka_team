param(
  [Parameter(Mandatory=$true)][string]$Train,
  [Parameter(Mandatory=$true)][string]$Private,
  [Parameter(Mandatory=$true)][string]$Candidate,
  [string]$GroundTruth = "",
  [string]$Manifest = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
if ([string]::IsNullOrWhiteSpace($Manifest)) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $Manifest = Join-Path $root "reports\reproducibility_manifest_$stamp.json"
}
if (Test-Path -LiteralPath $Manifest) {
  throw "Refusing to overwrite existing manifest: $Manifest"
}
$args = @("$root\scripts\reproducibility_audit.py", "--train", $Train,
  "--private", $Private, "--candidate", $Candidate, "--manifest", $Manifest)
if (-not [string]::IsNullOrWhiteSpace($GroundTruth)) {
  $args += @("--ground-truth", $GroundTruth)
}
& $python @args
if ($LASTEXITCODE -ne 0) { throw "Reproducibility audit failed with exit code $LASTEXITCODE" }
Write-Host "Manifest: $Manifest"
