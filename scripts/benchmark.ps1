param(
  [string]$Exe      = "build\Release\nbody.exe",
  [string[]]$Sizes  = @("1000", "2000", "4000", "8000"),
  [int]$Steps       = 20,
  [int]$Repeats     = 3,
  [string]$Init     = "plummer",
  [double]$Dt       = 0.001,
  [string]$OutDir   = "results"
)

$ErrorActionPreference = "Stop"

# Aceita tanto "-Sizes 1000,2000" (PowerShell) quanto
# "-Sizes 1000,2000" vindo como uma unica string (powershell -File).
# Usa uma variavel nova porque o parametro tipado [string[]] converteria os
# valores de volta para texto.
$sizeValues = @()
foreach ($item in $Sizes) {
  foreach ($part in ("$item" -split ',')) {
    $trimmed = $part.Trim()
    if ($trimmed -eq "") { continue }
    $value = 0
    if (-not [int]::TryParse($trimmed, [ref]$value) -or $value -lt 2) {
      Write-Error "tamanho invalido em -Sizes: '$trimmed'"
    }
    $sizeValues += $value
  }
}

if (-not (Test-Path $Exe)) {
  $alt = "build\nbody.exe"
  if (Test-Path $alt) {
    $Exe = $alt
  } else {
    Write-Error "executavel nao encontrado ($Exe). Rode scripts\build.ps1 primeiro."
  }
}

if (-not (Test-Path $OutDir)) {
  New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$csv = Join-Path $OutDir "benchmark.csv"
if (-not (Test-Path $csv)) {
  "n,steps,repeats,ms_per_step,ns_per_body_step" | Set-Content $csv
}

Write-Host "executavel : $Exe"
Write-Host "init       : $Init"
Write-Host "tamanhos   : $($sizeValues -join ', ')"
Write-Host "passos     : $Steps (repeticoes: $Repeats)"
Write-Host "csv        : $csv"
Write-Host ""

foreach ($size in $sizeValues) {
  $best = [double]::MaxValue

  for ($r = 1; $r -le $Repeats; $r++) {
    # --check-every Steps: calcula diagnostico so no ultimo passo, para nao
    # contaminar a medicao com o custo O(N^2) da energia potencial.
    $output = & $Exe --init $Init --n $size --steps $Steps --dt $Dt --check-every $Steps 2>&1
    $match = $output | Select-String -Pattern '\(([0-9.]+) ms/passo\)' | Select-Object -First 1
    if ($match -and $match.Matches[0].Groups[1].Value) {
      $msPerStep = [double]$match.Matches[0].Groups[1].Value
      if ($msPerStep -lt $best) { $best = $msPerStep }
    } else {
      Write-Warning "nao consegui ler o tempo para n=$size"
      $output | Write-Host
    }
  }

  if ($best -ne [double]::MaxValue) {
    $nsPerBodyStep = ($best * 1e6) / ($size * $Steps)
    # CSV sempre com ponto decimal: a cultura pt-BR usa virgula e quebraria
    # as colunas.
    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    Write-Host ("n={0,7}  {1,10:N3} ms/passo  {2,10:N2} ns/corpo-passo" -f $size, $best, $nsPerBodyStep)
    "{0},{1},{2},{3},{4}" -f $size, $Steps, $Repeats, $best.ToString("F4", $invariant), $nsPerBodyStep.ToString("F3", $invariant) |
      Add-Content $csv
  }
}

Write-Host ""
Write-Host "resultados anexados em $csv"
Write-Host "esperado no baseline: ms/passo ~ O(N^2) (dobrar N quadruplica o tempo)"
