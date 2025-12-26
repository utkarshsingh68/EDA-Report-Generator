param(
  [ValidateSet('red','white')] [string]$Variant = 'red'
)

$root = Split-Path -Parent $PSScriptRoot
$input = "C:\Users\utkarsh\Downloads\wine+quality (1)\winequality-$Variant.csv"
$output = Join-Path $root "reports\winequality-$Variant-report.md"

Write-Host "Input:  $input"
Write-Host "Output: $output"

& "$root\.venv\Scripts\python.exe" -m pip install -e "$root"
& "$root\.venv\Scripts\python.exe" -m eda_report --input "$input" --output "$output"
