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