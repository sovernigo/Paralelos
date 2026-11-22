# Simulação N-Corpos — Força Bruta, Barnes-Hut CPU e Barnes-Hut CUDA

Projeto de programação paralela: simulação gravitacional de N corpos em C++17 e
CUDA com suavização de Plummer e integrador Velocity Verlet (*kick-drift-kick*).
Inclui referência exata por **força bruta $\mathcal{O}(N^2)$**, baseline
sequencial **Barnes-Hut $\mathcal{O}(N \log N)$** com Octree linearizada e
implementação paralela **100% em GPU (CUDA)** baseada na arquitetura de
Burtscher & Pingali.

## O que está implementado

| Parte | Status |
|---|---|
| Força gravitacional com softening de Plummer e 3ª lei de Newton, $\mathcal{O}(N^2)$ | pronto |
| Integrador Velocity Verlet (*kick-drift-kick*) | pronto |
| Condições iniciais: esfera uniforme, Plummer, disco | pronto |
| Diagnósticos: energia cinética/potencial/total, momento, centro de massa | pronto |
| Barnes-Hut sequencial em CPU (Octree linearizada em arrays, iterativo) | pronto |
| Barnes-Hut em CUDA 100% na VRAM (atômicas, Morton/CUB, 6 estágios) | pronto |
| Precisão configurável unificada (`--float` / `--double`) entre CPU e GPU | pronto |
| CLI com `--no-diagnostics`, `--verify-error`, `--profile-kernels` e CSV | pronto |
| Testes automatizados CPU + CUDA (`CTest`) e script de gráficos | pronto |

## Estrutura

```text
Paralelos/
├── CMakeLists.txt              build C++17 + CUDA (sm_75, FP32/FP64)
├── Dockerfile                  container Linux com CUDA Toolkit e Nsight Compute
├── README.md
├── .gitignore / .clang-format
├── include/nbody/
│   ├── types.hpp               Real = float (FP32) ou double (FP64)
│   ├── vec3.hpp                vetor 3D inline
│   ├── body_system.hpp         posições/velocidades/massas em SoA
│   ├── params.hpp              SimulationParams + enum Solver
│   ├── accelerations.hpp       força bruta, Barnes-Hut CPU e dispatch
│   ├── cuda_backend.hpp        interface C++ pura do backend GPU + timings
│   ├── simulation.hpp          integrador Velocity Verlet + delegação GPU
│   ├── diagnostics.hpp         energia, momento, centro de massa
│   ├── initial_conditions.hpp  distribuições iniciais
│   ├── io.hpp                  CSV de métricas e de snapshots
│   └── timer.hpp               cronômetro e estatísticas
├── src/
│   ├── main.cpp                CLI, --verify-error e --profile-kernels
│   ├── body_system.cpp
│   ├── solver.cpp              parse/nome do solver + dispatch
│   ├── brute_force.cpp         força bruta O(N²) (referência exata)
│   ├── barnes_hut.cpp          Octree linearizada iterativa em CPU
│   ├── simulation.cpp          integrador CPU e integração com CudaBackend
│   ├── diagnostics.cpp         energia/momento
│   ├── initial_conditions.cpp  amostragem das distribuições
│   ├── io.cpp
│   └── cuda/
│       ├── barnes_hut_cuda.cu  6 kernels CUDA (Burtscher & Pingali + Morton)
│       └── cuda_stub.cpp       fallback para builds sem CUDA (--no-cuda)
├── tests/test_all.cpp          testes unitários CPU e CUDA (CTest)
├── scripts/
│   ├── build.sh                configure + build + ctest (--cuda/--float/--double)
│   ├── benchmark.sh            varre N com warmup/mediana e grava CSV
│   └── plot_results.py         gera gráficos a partir de results/*.csv
├── docs/
│   ├── roteiro.md              checklist do projeto
│   └── relatorio.md            relatório técnico
├── data/                       entradas grandes (fora do git)
└── results/                    CSVs de benchmark e gráficos
```

## Requisitos

- Linux ou container Docker com GPU NVIDIA (`--gpus all`)
- CMake ≥ 3.18 e compilador C++17 (GCC/Clang)
- CUDA Toolkit (`nvcc`) e NVIDIA Nsight Compute (`ncu`) para GPU/Roofline
- Python 3 + `matplotlib` (opcional, para `scripts/plot_results.py`)

## Ambiente Docker

```sh
docker build -t nbody-cuda .
docker run --rm -it --gpus all -v "$PWD:/workspace" nbody-cuda
```

## Build e testes

```sh
# Build padrão: Release, CUDA sm_75, FP32 unificado (CPU e GPU) + testes
./scripts/build.sh --clean --cuda --float --tests
./build/nbody_tests

# Alternativas úteis
./scripts/build.sh --clean --cuda --double --tests   # validação em FP64
./scripts/build.sh --clean --no-cuda --tests         # somente CPU
```

## Uso

```sh
./build/nbody --list

# Verificação de erro relativo da aceleração vs. força bruta (t = 0)
./build/nbody --init plummer --n 4000 --solver barnes-hut-cuda --theta 0.5 --verify-error

# Simulação em GPU com decomposição de tempo por kernel
./build/nbody --init plummer --n 100000 --steps 20 --dt 0.001 \
  --solver barnes-hut-cuda --theta 0.5 --no-diagnostics --profile-kernels

# Conservação de energia (CSV de métricas)
./build/nbody --init plummer --n 2000 --steps 200 --dt 0.001 \
  --solver barnes-hut-cuda --theta 0.5 --metrics results/energia_cuda.csv
```

Opções principais:

| Opção | Padrão | Descrição |
|---|---|---|
| `--init` | `plummer` | `uniforme`, `plummer` ou `disco` |
| `--n` | `2000` | número de corpos |
| `--steps` | `100` | passos de integração |
| `--dt` | `0.002` | passo de tempo |
| `--softening` | `0.05` | $\varepsilon$ da força de Plummer |
| `--gravity` | `1.0` | constante gravitacional $G$ |
| `--theta` | `0.5` | critério $s/d < \theta$ do Barnes-Hut |
| `--solver` | `brute` | `brute`, `barnes-hut` ou `barnes-hut-cuda` |
| `--seed` | `42` | semente determinística |
| `--metrics` | — | CSV: passo, tempo, energias, momento |
| `--snapshot` | — | CSV: posições/velocidades/massas |
| `--snapshot-every` | `0` | grava quadro a cada K passos (`0` = só o final) |
| `--check-every` | `10` | frequência do diagnóstico $\mathcal{O}(N^2)$ |
| `--no-diagnostics` | desligado | desativa diagnósticos $\mathcal{O}(N^2)$ em benchmarks |
| `--profile-kernels` | desligado | mede tempo por kernel na GPU (eventos CUDA) |
| `--verify-error` | desligado | compara aceleração em $t=0$ contra `brute` e encerra |

## Benchmarks, gráficos e Roofline

```sh
# 1. Benchmarks CPU e GPU na mesma precisão (FP32)
./scripts/benchmark.sh --solver brute --sizes 1000,2000,4000,8000 --steps 20 --repeats 3
./scripts/benchmark.sh --solver barnes-hut --theta 0.5 \
  --sizes 1000,2000,4000,8000,16000,50000 --steps 20 --repeats 3
./scripts/benchmark.sh --solver barnes-hut-cuda --theta 0.5 \
  --sizes 1000,2000,4000,8000,16000,50000,100000,250000,500000 --steps 20 --repeats 3

# 2. Varredura de theta (precisão x tempo)
for t in 0.2 0.5 0.8; do
  ./build/nbody --init plummer --n 4000 --solver barnes-hut-cuda --theta "$t" --verify-error
  ./scripts/benchmark.sh --solver barnes-hut --theta "$t" --sizes 16000 --steps 20 --repeats 3
  ./scripts/benchmark.sh --solver barnes-hut-cuda --theta "$t" --sizes 16000 --steps 20 --repeats 3
done

# 3. Gráficos em results/plots/
python3 scripts/plot_results.py --benchmark results/benchmark.csv --theta 0.5 \
  --energy "Brute=results/energia_brute.csv" "BH CPU=results/energia_bh.csv" "BH CUDA=results/energia_cuda.csv"

# 4. Modelo Roofline no Nsight Compute (ncu)
ncu --set roofline --kernel-name k_compute_forces --launch-count 1 \
  ./build/nbody --init plummer --n 100000 --steps 1 --dt 0.001 \
  --solver barnes-hut-cuda --theta 0.5 --no-diagnostics
```

Consulte `docs/roteiro.md` para o checklist do projeto e `docs/relatorio.md`
para o relatório completo.