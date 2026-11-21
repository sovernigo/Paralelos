# Simulação N-Corpos — Base Sequencial (C++17)

Base do trabalho de programação paralela: simulação gravitacional de N corpos
com suavização de Plummer, integrador de velocity Verlet e solver de **força
bruta O(N²)**. Serve como referência física e ponto de partida para o
**Barnes-Hut sequencial (O(N log N))** e sua **implementação paralela em CUDA**.

O projeto já inclui condições iniciais físicas, diagnósticos de conservação,
testes automatizados, medição de tempo, saída CSV e o ponto de extensão onde o
Barnes-Hut entra.

## O que já está implementado

| Parte | Status |
|---|---|
| Força gravitacional com softening (Plummer), O(N²) | pronto |
| Integrador velocity Verlet (kick-drift-kick) | pronto |
| Condições iniciais: esfera uniforme, Plummer, disco | pronto |
| Diagnósticos: energia cinética/potencial/total, momento, centro de massa | pronto |
| CLI com medição, `--no-diagnostics` e saída CSV (métricas + snapshots) | pronto |
| Testes de corretude (órbita de 2 corpos, conservação, forças) | pronto |
| Barnes-Hut sequencial | esqueleto em `src/barnes_hut.cpp` (lança erro) |
| Barnes-Hut em CUDA | próxima etapa |

## Estrutura

```text
Paralelos/
├── CMakeLists.txt              build C++17 (Linux), sem dependências externas
├── README.md
├── .gitignore / .clang-format
├── include/nbody/
│   ├── types.hpp               Real = double (ajustável para float na GPU)
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
│   ├── barnes_hut.cpp          TODO: Octree linearizada + critério theta
│   ├── simulation.cpp          integrador
│   ├── diagnostics.cpp         energia/momento
│   ├── initial_conditions.cpp  amostragem das distribuições
│   └── io.cpp
├── tests/test_all.cpp          testes sem framework externo (CTest)
├── scripts/
│   ├── build.sh                configure + build + ctest
│   └── benchmark.sh            varre N com warmup/mediana e grava CSV
├── docs/
│   ├── roteiro.md              checklist: baseline → Barnes-Hut → CUDA
│   └── relatorio.md            template do relatório
├── data/                       entradas grandes (fora do git)
└── results/                    CSVs de benchmark
```

## Física e integração

Força sobre o corpo *i* (suavização de Plummer, evita singularidade em
encontros próximos):

```text
a_i = Σ_{j≠i} G·m_j·(r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)
```

Integrador velocity Verlet, um passo:

```text
v ← v + a·Δt/2      (kick)
x ← x + v·Δt        (drift)
a ← F(x)            (recalcula)
v ← v + a·Δt/2      (kick)
```

É simplético: o erro de energia oscila limitado em vez de crescer. O teste
automatizado exige que a órbita de dois corpos mantenha a separação e a energia
dentro de 0,5% em 1000 passos.

## Requisitos

- Linux (ou container Docker com acesso à GPU)
- CMake ≥ 3.16
- Compilador C++17 (GCC 9+ ou Clang 10+)
- CUDA Toolkit + NVIDIA Nsight Compute (`ncu`) para a etapa paralela em GPU

## Build

```sh
./scripts/build.sh --clean --tests

# ou manualmente
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNBODY_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native"
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure
```

O executável é gerado em `build/nbody`.

## Uso

```sh
./build/nbody --list

# 2000 corpos, 100 passos, energia/momento em CSV
./build/nbody --init plummer --n 2000 --steps 100 \
  --metrics results/energia.csv --snapshot results/final.csv

# disco com quadros a cada 5 passos (animação)
./build/nbody --init disco --n 5000 --steps 200 --dt 0.01 \
  --snapshot results/disco.csv --snapshot-every 5 --check-every 20

# execução focada em desempenho (sem diagnósticos O(N²))
./build/nbody --init plummer --n 4000 --steps 20 --no-diagnostics
```

Opções principais:

| Opção | Padrão | Descrição |
|---|---|---|
| `--init` | `plummer` | `uniforme`, `plummer` ou `disco` |
| `--n` | `2000` | número de corpos |
| `--steps` | `100` | passos de integração |
| `--dt` | `0.002` | passo de tempo |
| `--softening` | `0.05` | ε da força de Plummer |
| `--gravity` | `1.0` | G |
| `--theta` | `0.5` | critério `s/d < θ` do Barnes-Hut |
| `--solver` | `brute` | `brute` (O(N²)) ou `barnes-hut` (a implementar) |
| `--seed` | `42` | semente determinística |
| `--metrics` | — | CSV: passo, tempo, energias, momento |
| `--snapshot` | — | CSV: posições/velocidades/massas |
| `--snapshot-every` | `0` | grava quadro a cada K passos (`0` = só o final) |
| `--check-every` | `10` | frequência do diagnóstico O(N²) |
| `--no-diagnostics` | desligado | desativa diagnósticos O(N²) em benchmarks |

Saída típica:

```text
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

```sh
./scripts/benchmark.sh --solver brute --sizes 1000,2000,4000,8000 --steps 20 --repeats 3
```

Gera `results/benchmark.csv` com `best_ms_per_step`, `median_ms_per_step` e
`ns_per_body_step`. No solver de força bruta, dobrar `N` deve aproximadamente
**quadruplicar** o tempo por passo.

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
  ser desativada com `--no-diagnostics` em benchmarks de larga escala.
- **`Real` centralizado**: trocar `double` por `float` de forma consistente
  entre CPU e GPU é feito em um único ponto (`include/nbody/types.hpp`).

## Próximos passos

1. **Barnes-Hut sequencial** — implementar `src/barnes_hut.cpp`:
   caixa envolvente → Octree linearizada em arrays → massa e centro de massa em
   pós-ordem → travessia com `s/d < θ`. Comparar tempo e erro de aceleração
   contra `brute` no mesmo `N`.
2. **Barnes-Hut em CUDA** — implementar a construção sincronizada via atômicas,
   travessia e integração 100% na GPU (Burtscher & Pingali), evitando tráfego
   PCIe por passo.
3. **Análise** — speedup × N (Barnes-Hut CPU vs. CUDA), erro médio × θ (vs.
   `brute`), conservação de energia e Modelo Roofline com NVIDIA Nsight Compute.

O roteiro detalhado está em `docs/roteiro.md`; o template do relatório em
`docs/relatorio.md`.