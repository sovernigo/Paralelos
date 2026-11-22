#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"
RUN_TESTS=0
CLEAN=0
USE_NATIVE=1
ENABLE_CUDA=1
USE_FLOAT=1
CUDA_ARCH="75"

usage() {
  cat <<USAGE
Uso: $0 [opcoes]
  --build-dir DIR    diretorio de build (padrao: build)
  --config TYPE      Release | Debug (padrao: Release)
  --tests            compila e executa ctest
  --cuda             habilita backend CUDA (padrao)
  --no-cuda          compila somente CPU (sem nvcc)
  --float            usa FP32 em CPU e GPU (padrao)
  --double           usa FP64 em CPU e GPU
  --arch SM          arquitetura CUDA (padrao: 75 para RTX 2060 SUPER)
  --no-native        desativa -march=native em Release
  --clean            remove o diretorio de build antes de configurar
  -h, --help         mostra esta ajuda
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --tests)     RUN_TESTS=1; shift ;;
    --cuda)      ENABLE_CUDA=1; shift ;;
    --no-cuda)   ENABLE_CUDA=0; shift ;;
    --float)     USE_FLOAT=1; shift ;;
    --double)    USE_FLOAT=0; shift ;;
    --arch)      CUDA_ARCH="$2"; shift 2 ;;
    --no-native) USE_NATIVE=0; shift ;;
    --clean)     CLEAN=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Erro: opcao desconhecida '$1'" >&2; usage; exit 2 ;;
  esac
done

if ! command -v cmake >/dev/null 2>&1; then
  echo "Erro: cmake nao encontrado no PATH." >&2
  exit 1
fi

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$BUILD_DIR"
fi

CUDA_OPT=$([[ "$ENABLE_CUDA" -eq 1 ]] && echo "ON" || echo "OFF")
FLOAT_OPT=$([[ "$USE_FLOAT" -eq 1 ]] && echo "ON" || echo "OFF")

CMAKE_ARGS=(
  -S .
  -B "$BUILD_DIR"
  "-DCMAKE_BUILD_TYPE=${CONFIG}"
  "-DNBODY_ENABLE_CUDA=${CUDA_OPT}"
  "-DNBODY_USE_FLOAT=${FLOAT_OPT}"
)

if [[ "$ENABLE_CUDA" -eq 1 ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
fi

if command -v ninja >/dev/null 2>&1; then
  CMAKE_ARGS+=(-G Ninja)
fi

if [[ "$RUN_TESTS" -eq 1 ]]; then
  CMAKE_ARGS+=(-DNBODY_BUILD_TESTS=ON)
fi

if [[ "$USE_NATIVE" -eq 1 && "$CONFIG" == "Release" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native")
fi

echo "== configurando (${CONFIG} | CUDA=${CUDA_OPT} | FLOAT=${FLOAT_OPT}) =="
cmake "${CMAKE_ARGS[@]}"

echo "== compilando =="
cmake --build "$BUILD_DIR" -j "$(nproc)"

if [[ -x "${BUILD_DIR}/nbody" ]]; then
  echo "== executavel: ${BUILD_DIR}/nbody =="
else
  echo "Aviso: executavel nao encontrado em ${BUILD_DIR}/nbody" >&2
fi

if [[ "$RUN_TESTS" -eq 1 ]]; then
  echo "== testes =="
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi