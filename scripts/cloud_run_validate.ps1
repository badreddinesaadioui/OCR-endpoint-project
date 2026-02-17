param(
  [Parameter(Mandatory = $true)]
  [string]$ServiceUrl,

  [Parameter(Mandatory = $true)]
  [string]$SampleFilePath,

  [Parameter(Mandatory = $false)]
  [string]$ApiAuthToken = "",

  [Parameter(Mandatory = $false)]
  [int]$MaxWaitSeconds = 120
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SampleFilePath)) {
  throw "Sample file not found: $SampleFilePath"
}

if ([string]::IsNullOrWhiteSpace($ServiceUrl) -or $ServiceUrl -match "YOUR_SERVICE_URL") {
  throw "Invalid ServiceUrl. Provide the real deployed Cloud Run URL."
}

if ($ServiceUrl -notmatch "^https?://") {
  throw "ServiceUrl must start with http:// or https://"
}

function New-Headers() {
  $h = @{}
  if ($ApiAuthToken -and $ApiAuthToken.Trim()) {
    $h["Authorization"] = "Bearer $ApiAuthToken"
  }
  return $h
}

$base = $ServiceUrl.TrimEnd("/")
$headers = New-Headers

Write-Host "Checking /health..." -ForegroundColor Cyan
$health = Invoke-RestMethod -Method Get -Uri "$base/health" -Headers $headers
Write-Host "Health OK: $($health.status)" -ForegroundColor Green

Write-Host "Creating async job..." -ForegroundColor Cyan
$form = @{
  file = Get-Item $SampleFilePath
}
$jobResp = Invoke-RestMethod -Method Post -Uri "$base/v1/jobs" -Headers $headers -Form $form
$jobId = $jobResp.job_id
if (-not $jobId) {
  throw "No job_id returned by API."
}

Write-Host "Job created: $jobId" -ForegroundColor Green
Write-Host "Polling status..." -ForegroundColor Cyan

$deadline = (Get-Date).AddSeconds($MaxWaitSeconds)
$final = $null
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 3
  $current = Invoke-RestMethod -Method Get -Uri "$base/v1/jobs/$jobId" -Headers $headers
  $status = $current.status
  Write-Host "Status: $status"
  if ($status -eq "succeeded" -or $status -eq "failed") {
    $final = $current
    break
  }
}

if ($null -eq $final) {
  throw "Validation timeout after $MaxWaitSeconds seconds."
}

if ($final.status -eq "succeeded") {
  Write-Host "Validation succeeded." -ForegroundColor Green
  Write-Host "Models: OCR=$($final.models.ocr), LLM=$($final.models.llm_parser)"
  Write-Host "Schema valid: $($final.quality.schema_valid)"
} else {
  Write-Host "Validation failed." -ForegroundColor Red
  if ($final.error) {
    Write-Host "Error code: $($final.error.code)"
    Write-Host "Error message: $($final.error.message)"
  }
}
