#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

EXE="build/nbody"
SIZES="1000,2000,4000,8000"
STEPS=20
REPEATS=3
WARMUP=1
INIT="plummer"
DT="0.001"
SOFTENING="0.05"
GRAVITY="1.0"
SOLVER="brute"
THETA="0.5"
SEED=42
OUT_DIR="results"
OUT_FILE=""

usage() {
  cat <<USAGE
Uso: $0 [opcoes]
  --exe PATH         executavel (padrao: build/nbody)
  --sizes LIST       lista de N separados por virgula (padrao: 1000,2000,4000,8000)
  --steps S          passos de integracao (padrao: 20)
  --repeats R        repeticoes medidas (padrao: 3)
  --warmup W         execucoes de aquecimento descartadas (padrao: 1)
  --init NAME        uniforme | plummer | disco (padrao: plummer)
  --dt D             passo de tempo (padrao: 0.001)
  --softening E      suavizacao de Plummer (padrao: 0.05)
  --gravity G        constante gravitacional (padrao: 1.0)
  --solver NAME      brute | barnes-hut | barnes-hut-cuda (padrao: brute)
  --theta T          criterio theta (padrao: 0.5)
  --seed S           semente aleatoria (padrao: 42)
  --out-dir DIR      diretorio de saida (padrao: results)
  --out-file PATH    arquivo CSV completo (padrao: <out-dir>/benchmark.csv)
  -h, --help         mostra esta ajuda
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exe)       EXE="$2"; shift 2 ;;
    --sizes)     SIZES="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --repeats)   REPEATS="$2"; shift 2 ;;
    --warmup)    WARMUP="$2"; shift 2 ;;
    --init)      INIT="$2"; shift 2 ;;
    --dt)        DT="$2"; shift 2 ;;
    --softening) SOFTENING="$2"; shift 2 ;;
    --gravity)   GRAVITY="$2"; shift 2 ;;
    --solver)    SOLVER="$2"; shift 2 ;;
    --theta)     THETA="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
    --out-file)  OUT_FILE="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Erro: opcao desconhecida '$1'" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -x "$EXE" ]]; then
  echo "Erro: executavel nao encontrado ($EXE). Rode ./scripts/build.sh primeiro." >&2
  exit 1
fi

if [[ -z "$OUT_FILE" ]]; then
  mkdir -p "$OUT_DIR"
  OUT_FILE="${OUT_DIR}/benchmark.csv"
else
  mkdir -p "$(dirname "$OUT_FILE")"
fi

if [[ ! -f "$OUT_FILE" ]]; then
  echo "solver,init,n,steps,theta,seed,repeats,best_ms_per_step,median_ms_per_step,ns_per_body_step" > "$OUT_FILE"
fi

IFS=',' read -r -a SIZE_ARRAY <<< "$SIZES"

echo "executavel : ${EXE}"
echo "solver     : ${SOLVER} (theta=${THETA})"
echo "init       : ${INIT} (seed=${SEED})"
echo "tamanhos   : ${SIZE_ARRAY[*]}"
echo "passos     : ${STEPS} (warmup: ${WARMUP}, repeticoes: ${REPEATS})"
echo "csv        : ${OUT_FILE}"
echo ""

for raw_size in "${SIZE_ARRAY[@]}"; do
  size="$(echo "$raw_size" | tr -d '[:space:]')"
  [[ -z "$size" ]] && continue

  if ! [[ "$size" =~ ^[0-9]+$ ]] || (( size < 2 )); then
    echo "Erro: tamanho invalido em --sizes: '$size'" >&2
    exit 2
  fi

  CMD=(
    "$EXE"
    --init "$INIT"
    --n "$size"
    --steps "$STEPS"
    --dt "$DT"
    --softening "$SOFTENING"
    --gravity "$GRAVITY"
    --theta "$THETA"
    --solver "$SOLVER"
    --seed "$SEED"
    --no-diagnostics
  )

  for ((w = 1; w <= WARMUP; ++w)); do
    "${CMD[@]}" >/dev/null 2>&1
  done

  samples=()
  for ((r = 1; r <= REPEATS; ++r)); do
    output="$("${CMD[@]}" 2>&1)"
    ms_per_step="$(printf '%s\n' "$output" | sed -nE 's/.*\(([0-9]+(\.[0-9]+)?) ms\/passo\).*/\1/p' | head -n 1)"
    if [[ -n "$ms_per_step" ]]; then
      samples+=("$ms_per_step")
    else
      echo "Aviso: nao consegui ler o tempo para n=${size} (repeticao ${r})" >&2
      printf '%s\n' "$output" >&2
    fi
  done

  if (( ${#samples[@]} > 0 )); then
    stats="$(
      printf '%s\n' "${samples[@]}" | sort -n | awk -v n="$size" '
        { a[NR] = $1 }
        END {
          best = a[1];
          median = (NR % 2 == 1) ? a[(NR + 1) / 2] : 0.5 * (a[NR / 2] + a[NR / 2 + 1]);
          ns_per_body_step = (best * 1e6) / n;
          printf "%.4f %.4f %.3f", best, median, ns_per_body_step;
        }'
    )"
    read -r best_ms median_ms ns_body <<< "$stats"
    printf "n=%7d  best=%10.3f ms/passo  median=%10.3f ms/passo  %10.2f ns/corpo-passo\n" \
      "$size" "$best_ms" "$median_ms" "$ns_body"
    printf "%s,%s,%d,%d,%s,%d,%d,%s,%s,%s\n" \
      "$SOLVER" "$INIT" "$size" "$STEPS" "$THETA" "$SEED" "$REPEATS" \
      "$best_ms" "$median_ms" "$ns_body" >> "$OUT_FILE"
  fi
done

echo ""
echo "resultados anexados em ${OUT_FILE}"