$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$tesseractCandidates = @(
    (Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR\tesseract.exe")
)
$tesseractExe = $tesseractCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $tesseractExe -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    winget install --id UB-Mannheim.TesseractOCR --exact --accept-source-agreements --accept-package-agreements
    $tesseractExe = $tesseractCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $tesseractExe) {
    throw "Tesseract installation was not found. Install it with winget, then run this script again."
}

$sourceTessdata = Join-Path (Split-Path -Parent $tesseractExe) "tessdata"
$targetTessdata = Join-Path $projectRoot "storage\tessdata"
New-Item -ItemType Directory -Path $targetTessdata -Force | Out-Null

$englishModel = Join-Path $sourceTessdata "eng.traineddata"
if (-not (Test-Path -LiteralPath $englishModel)) {
    throw "The English Tesseract model is missing from $sourceTessdata."
}
Copy-Item -LiteralPath $englishModel -Destination (Join-Path $targetTessdata "eng.traineddata") -Force

$arabicModel = Join-Path $targetTessdata "ara.traineddata"
if (-not (Test-Path -LiteralPath $arabicModel)) {
    Invoke-WebRequest `
        -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/ara.traineddata" `
        -OutFile $arabicModel
}

Write-Host "OCR is ready. Tesseract: $tesseractExe"
Write-Host "Language data: $targetTessdata (eng + ara)"
