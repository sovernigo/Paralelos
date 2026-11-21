# Simulação N-Corpos — base sequencial (C++17)

Base do trabalho de programação paralela: simulação gravitacional de N corpos
com força de Plummer, integrador de velocity Verlet e solver de **força bruta
O(N²)**. É o baseline sequencial que depois será comparado com o Barnes-Hut
(O(N log N)) e com a implementação em CUDA.

O projeto já traz o que é chato de refazer: condições iniciais físicas,
diagnósticos de conservação, testes, medição de tempo, saída CSV e o ponto de
extensão onde o Barnes-Hut entra.

## O que já está implementado

| Parte | Status |
|---|---|
| Força gravitacional com softening (Plummer), O(N²) | pronto |
| Integrador velocity Verlet (kick-drift-kick) | pronto |
| Condições iniciais: esfera uniforme, Plummer, disco | pronto |
| Diagnósticos: energia cinética/potencial/total, momento, centro de massa | pronto |
| CLI com medição, progresso e CSV (métricas + snapshots) | pronto |
| Testes de corretude (órbita de 2 corpos, conservação, forças) | pronto |
| Barnes-Hut sequencial | esqueleto em `src/barnes_hut.cpp` (lança erro) |
| Barnes-Hut / N-corpos em CUDA | próxima etapa |

## Estrutura

```
Paralelos/
├── CMakeLists.txt              build C++17, sem dependências externas
├── README.md
├── .gitignore / .clang-format
├── include/nbody/
│   ├── types.hpp               Real = double (troque para float na GPU)
│   ├── vec3.hpp                vetor 3D inline
│   ├── body_system.hpp         posições/velocidades/massas em SoA
│   ├── params.hpp              SimulationParams + enum Solver
│   ├── accelerations.hpp       força bruta, Barnes-Hut, dispatch
│   ├── simulation.hpp          integrador velocity Verlet
│   ├── diagnostics.hpp         energia, momento, centro de massa
│   ├── initial_conditions.hpp  distribuições iniciais
│   ├── io.hpp                  CSV de métricas e de snapshots
│   └── timer.hpp               cronômetro e estatísticas
├── src/
│   ├── main.cpp                CLI
│   ├── body_system.cpp
│   ├── solver.cpp              parse/nome do solver + dispatch
│   ├── brute_force.cpp         força bruta O(N²) (baseline)
│   ├── barnes_hut.cpp          TODO: árvore + critério theta
│   ├── simulation.cpp          integrador
│   ├── diagnostics.cpp         energia/momento
│   ├── initial_conditions.cpp  amostragem das distribuições
│   └── io.cpp
├── tests/test_all.cpp          testes sem framework externo (CTest)
├── scripts/
│   ├── build.ps1               configure + build + ctest
│   └── benchmark.ps1           varre N e grava ms/passo em CSV
├── docs/
│   ├── roteiro.md              checklist: baseline → Barnes-Hut → CUDA
│   └── relatorio.md            template do relatório
├── data/                       entradas grandes (fora do git)
└── results/                    CSVs de benchmark
```

## Física e integração

Força sobre o corpo *i* (suavização de Plummer, evita singularidade em
encontros próximos):

```
a_i = Σ_{j≠i} G·m_j·(r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)
```

Integrador velocity Verlet, um passo:

```
v ← v + a·Δt/2      (kick)
x ← x + v·Δt        (drift)
a ← F(x)            (recalcula)
v ← v + a·Δt/2      (kick)
```

É simpletico: o erro de energia oscila limitado em vez de crescer. O teste
automatizado exige que a órbita de dois corpos mantenha a separação e a
energia dentro de 0,5% em 1000 passos.

## Requisitos

- CMake ≥ 3.16
- Compilador C++17 (MSVC 2019+, GCC 9+, Clang 10+)

> **Nesta máquina** não há `cmake`, `g++`/`cl` no PATH. Instale antes de
> compilar:
>
> ```powershell
> winget install Kitware.CMake
> winget install Microsoft.VisualStudio.2022.BuildTools
> ```
>
> Alternativa: `winget install MSYS2.MSYS2` e então
> `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja`.

## Build

```powershell
.\scripts\build.ps1 -Tests

# ou manualmente
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

O executável fica em `build/nbody` ou `build/Release/nbody.exe`, dependendo do
gerador.

## Uso

```powershell
.\build\Release\nbody.exe --list

# 2000 corpos, 100 passos, energia/momento em CSV
.\build\Release\nbody.exe --init plummer --n 2000 --steps 100 `
    --metrics results\energia.csv --snapshot results\final.csv

# disco com quadros a cada 5 passos (animação)
.\build\Release\nbody.exe --init disco --n 5000 --steps 200 --dt 0.01 `
    --snapshot results\disco.csv --snapshot-every 5 --check-every 20
```

Opções principais:

| Opção | Padrão | Descrição |
|---|---|---|
| `--init` | `plummer` | `uniforme`, `plummer` ou `disco` |
| `--n` | 2000 | número de corpos |
| `--steps` | 100 | passos de integração |
| `--dt` | 0.002 | passo de tempo |
| `--softening` | 0.05 | ε da força de Plummer |
| `--gravity` | 1.0 | G |
| `--theta` | 0.5 | critério s/d do Barnes-Hut |
| `--solver` | `brute` | `brute` (O(N²)) ou `barnes-hut` (a implementar) |
| `--metrics` | — | CSV: passo, tempo, energias, momento |
| `--snapshot` | — | CSV: posições/velocidades/massas |
| `--snapshot-every` | 0 | grava quadro a cada K passos (0 = só o final) |
| `--check-every` | 10 | frequência do diagnóstico O(N²) |

Saída típica:

```
distribuicao : plummer
corpos       : 2000
solver       : brute
passos       : 100 | dt = 0.002 | softening = 0.05

  passo 100/100 (100%)
  tempo total : ... s (... ms/passo)
  energia     : ... -> ... (deriva relativa ...)
  momento     : (...)
  centro mass : (...)
```

## Como saber se a física está certa

1. **Momento** deve ficar ~1e-15 (forças internas se cancelam).
2. **Centro de massa** deve ficar parado.
3. **Energia** deve oscilar em torno de um valor fixo; deriva grande indica
   `dt` grande demais (teste `--dt 0.001` e compare).
4. **Órbita de 2 corpos** (teste automatizado) mantém separação e energia.

## Medindo o baseline

```powershell
.\scripts\benchmark.ps1 -Sizes 1000,2000,4000,8000 -Steps 20 -Repeats 3
```

Gera `results/benchmark.csv` com `ms/passo` e `ns/corpo-passo`. No solver de
força bruta, dobrar `N` deve **quadruplicar** o tempo; essa curva é a
referência para o speedup do Barnes-Hut e da GPU.

## Decisões de projeto (vale citar no relatório)

- **SoA em vez de AoS**: arrays contíguos por grandeza → laços de força
  sequenciais, bons para cache/vetorização e transferência direta para a GPU.
- **Terceira lei de Newton**: cada par é calculado uma vez (`j > i`), metade
  do trabalho do laço completo.
- **Softening**: mantém a força finita e a energia bem definida em encontros
  próximos; é o mesmo ε no cálculo de força e de potencial.
- **Velocity Verlet**: conservação de energia sem custo extra de memória
  (não precisa guardar aceleração anterior).
- **Diagnóstico desacoplado**: energia não entra no laço de integração e pode
  ser amostrada a cada K passos.
- **`Real` centralizado**: trocar `double` por `float` para a GPU é uma linha.

## Próximos passos

1. **Barnes-Hut sequencial** — implementar `src/barnes_hut.cpp`:
   caixa envolvente → octree (idealmente linearizada em arrays) → massa e
   centro de massa por nó → travessia com `s/d < θ`. Compare tempo e erro de
   força contra o brute force no mesmo `N`.
2. **CUDA** — portar a travessia da árvore para a GPU (um thread por corpo),
   mantendo a árvore em buffers lineares; comparar com o baseline em CPU.
3. **Análise** — speedup × N, speedup × θ, erro médio de aceleração × θ,
   energia × tempo, Roofline com dados do Nsight Compute.

O roteiro detalhado está em `docs/roteiro.md`; o template do relatório em
`docs/relatorio.md`.
