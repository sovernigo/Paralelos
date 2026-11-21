param(
  [string]$BuildDir = "build",
  [string]$Config   = "Release",
  [switch]$Tests
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
  Write-Error "cmake nao encontrado no PATH. Instale com: winget install Kitware.CMake"
}

$configureArgs = @("-S", ".", "-B", $BuildDir, "-DCMAKE_BUILD_TYPE=$Config")
if ($Tests) {
  $configureArgs += @("-DNBODY_BUILD_TESTS=ON")
}

Write-Host "== configurando =="
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== compilando =="
& cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$candidates = @(
  (Join-Path $BuildDir "nbody.exe"),
  (Join-Path $BuildDir (Join-Path $Config "nbody.exe")),
  (Join-Path $BuildDir "nbody")
)
$exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($exe) {
  Write-Host "== executavel: $exe =="
} else {
  Write-Warning "executavel nao encontrado; verifique a pasta $BuildDir"
}

if ($Tests) {
  Write-Host "== testes =="
  & ctest --test-dir $BuildDir -C $Config --output-on-failure
}
