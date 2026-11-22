# Relatório — Simulação N-Corpos com Barnes-Hut

**Título:** Paralelização da Simulação de N-Body: O Algoritmo Barnes-Hut em CUDA
**Aluno:** Felipe Sovernigo
**Disciplina:** Programação Paralela
**Data:** Novembro de 2026

## 1. Objetivo

_Descreva o problema (simulação gravitacional de N corpos), a motivação do
Barnes-Hut (reduzir a complexidade de O(N²) para O(N log N)) e a pergunta que o
trabalho responde (ex.: "qual o ganho de desempenho do Barnes-Hut em CUDA em
relação ao baseline sequencial em CPU, variando N e θ, e quais gargalos de
memória e computação dominam a execução na RTX 2060 SUPER?")._

## 2. Metodologia

### 2.1 Hardware

| Item | Valor |
|---|---|
| CPU | _(preencher com `lscpu`)_ |
| RAM | _(preencher com `free -h`)_ |
| GPU | NVIDIA GeForce RTX 2060 SUPER (Turing, SM 7.5, 34 SMs, 2176 núcleos CUDA, 8 GB GDDR6) |
| SO / Ambiente | Ubuntu 24.04 (container Docker sobre WSL2 com acesso à GPU) |

### 2.2 Software

| Item | Valor |
|---|---|
| Compilador C++ | GCC 13.2.0 (`g++`) |
| Flags C++ | `-O3 -DNDEBUG -march=native` |
| Compilador CUDA | `nvcc` — CUDA Toolkit 12.6 (V12.6.77), driver NVIDIA 621.24 (KMD) |
| Flags CUDA | `-arch=sm_75 --use_fast_math -lineinfo` |
| Precisão numérica | FP32 (`float`, `-DNBODY_USE_FLOAT=ON`) unificada entre CPU e GPU |
| CMake | 3.28.3 |
| Nsight Compute | 2024.3.2.0 |

### 2.3 Modelo físico e integrador

- Força gravitacional com suavização de Plummer ($\varepsilon = 0{,}05$)
- Integrador: Velocity Verlet (*kick-drift-kick*), $\Delta t = 0{,}001$ (benchmarks) / $0{,}002$ (validação)
- Condição inicial: esfera de Plummer em equilíbrio (`--init plummer`, `--seed 42`)
- Unidades normalizadas: $G = 1$, massa total $M = 1$
- Precisão: `float` (FP32) consistente entre os solvers CPU e GPU (configurável em `include/nbody/types.hpp`)

### 2.4 Procedimento experimental

- Tamanhos avaliados: $N \in \{10^3, 2\cdot 10^3, 4\cdot 10^3, 8\cdot 10^3, 1{,}6\cdot 10^4, 5\cdot 10^4, 10^5, 2{,}5\cdot 10^5, 5\cdot 10^5\}$ _(ajustar conforme os testes finais)_
- Passos medidos por execução: 20 passos (com 1 execução de aquecimento descartada)
- Repetições: 3 repetições por ponto, reportando melhor tempo (`best_ms_per_step`) e mediana (`median_ms_per_step`)
- Diagnósticos desativados durante medição de tempo (`--no-diagnostics`) para eliminar o custo O(N²) da energia potencial
- Decomposição do tempo por *kernel* na GPU via eventos CUDA (`--profile-kernels`)
- Critério de precisão: erro relativo médio da aceleração contra o solver exato de força bruta (`brute`)
- Validação física: deriva relativa da energia total, conservação do momento linear e estabilidade do centro de massa

## 3. Implementações

### 3.1 Baseline sequencial (força bruta)

Os dados dos corpos são organizados em *Structure of Arrays* (SoA) em `BodySystem` (`x, y, z, vx, vy, vz, ax, ay, az, mass`), favorecendo acesso sequencial em cache e transferência direta para a GPU. O cálculo de acelerações em `src/brute_force.cpp` aplica a suavização de Plummer e explora a 3ª Lei de Newton avaliando cada par uma única vez (`j > i`), reduzindo pela metade o total de interações do algoritmo $\mathcal{O}(N^2)$. A integração temporal usa Velocity Verlet (*kick-drift-kick*).

### 3.2 Barnes-Hut sequencial

Implementado em `src/barnes_hut.cpp` sem recursão e sem ponteiros, usando uma Octree 3D linearizada em vetores contíguos (`children[8]`, `body`, `cx, cy, cz, half_size`, `mass, cmx, cmy, cmz`) reaproveitados entre passos via `thread_local`:

1. **Caixa envolvente:** calcula o cubo mínimo contendo todos os corpos.
2. **Construção iterativa:** insere os corpos subdividindo células ocupadas; corpos coincidentes ou abaixo da profundidade limite (`kMaxDepth = 32`) são encadeados por índice em `next_body`.
3. **Centro de massa em pós-ordem:** como filhos sempre recebem índice estritamente maior que o pai, um único laço reverso (`num_nodes - 1` até `0`) acumula massa total e centro de massa.
4. **Travessia iterativa:** usa pilha explícita e avalia o critério ao quadrado $s^2 < \theta^2 d^2$ (evitando `sqrt` na decisão de poda), garantindo que células contendo o próprio corpo nunca sejam aproximadas.

### 3.3 Barnes-Hut em CUDA

Implementado em `src/cuda/barnes_hut_cuda.cu` seguindo a arquitetura de Burtscher & Pingali com execução **100% residente na GPU**: os vetores SoA são enviados uma única vez para a VRAM na criação de `Simulation` e baixados para o *host* apenas sob demanda (diagnósticos ou *snapshots*), eliminando tráfego PCIe a cada passo. Os buffers da Octree são pré-alocados e cada $\Delta t$ executa 6 estágios:

1. **`k_kick_drift`:** primeiro meio-*kick* e *drift* do Velocity Verlet direto na VRAM.
2. **`k_bounding_box` + `k_init_root`:** redução paralela em memória compartilhada com `atomicMin`/`atomicMax` para definir o cubo raiz.
3. **Ordenação espacial (`k_morton_codes` + `cub::DeviceRadixSort`):** ordena índices de corpos por código de Morton 3D (curva Z) para agrupar corpos vizinhos nos mesmos *warps* e reduzir divergência na travessia.
4. **`k_build_tree`:** inserção paralela dos $N$ corpos na Octree linearizada, sincronizada por *locks* leves via `atomicCAS` e tratamento de corpos coincidentes em `next_body`.
5. **`k_count_children` + `k_summarize_tree`:** cálculo de massa e centro de massa *bottom-up* usando contador atômico de filhos pendentes (`atomicSub`), sem *spin-wait*.
6. **`k_compute_forces` + `k_kick`:** 1 thread por corpo (em ordem de Morton) percorre a Octree com pilha local aplicando $s^2 < \theta^2 d^2$, seguido pelo segundo meio-*kick*.

## 4. Resultados

### 4.1 Desempenho geral (FP32)

| Implementação | N | θ | Tempo (ms/passo) | Speedup vs. BH-CPU | Speedup vs. Brute | Erro rel. da força |
|---|---|---|---|---|---|---|
| Brute force (CPU) | | — | | — | 1,00× | 0 |
| Barnes-Hut (CPU) | | 0,5 | | 1,00× | | |
| Barnes-Hut (CUDA) | | 0,5 | | | | |

### 4.2 Precisão × θ

| θ | Erro médio da aceleração | Tempo CPU (ms/passo) | Tempo GPU (ms/passo) | Speedup GPU/CPU |
|---|---|---|---|---|
| 0,2 | | | | |
| 0,5 | | | | |
| 0,8 | | | | |

### 4.3 Decomposição por kernel na GPU (`--profile-kernels`)

| N | `bbox` (ms) | `sort` (ms) | `build` (ms) | `sum` (ms) | `force` (ms) | `verlet` (ms) | Total (ms/passo) |
|---|---|---|---|---|---|---|---|
| 16.000 | | | | | | | |
| 100.000 | 0,013 | 0,112 | 1,858 | 0,064 | 32,897 | 0,078 | 35,297 |
| 500.000 | | | | | | | |

### 4.4 Conservação de energia

_Inclua o gráfico energia total × passo para o baseline (`brute`), Barnes-Hut CPU e Barnes-Hut CUDA. Discuta a deriva relativa, o impacto da precisão FP32 e o efeito de $\theta$._

### 4.5 Escalabilidade e Modelo Roofline

_Inclua o gráfico tempo × N em escala log-log (apontando o ponto de cruzamento entre $\mathcal{O}(N^2)$ e $\mathcal{O}(N \log N)$ e o comportamento até centenas de milhares de corpos na GPU) e o posicionamento dos kernels principais no Modelo Roofline da RTX 2060 SUPER obtido com o NVIDIA Nsight Compute (`ncu`)._

## 5. Análise

- _O speedup acompanha a redução de complexidade e o paralelismo da GPU? A partir de qual $N$ a GPU compensa o overhead de lançamento e construção da árvore?_
- _Na tabela 4.3, qual kernel domina o tempo por passo (construção com atômicas, ordenação de Morton ou travessia de forças)?_
- _Como $\theta$ afeta o compromisso precisão × desempenho em CPU e GPU?_
- _Qual o impacto da ordenação de Morton na redução da divergência de warps em `k_compute_forces`?_
- _O que o Modelo Roofline / Nsight Compute indica (intensidade aritmética, limites de largura de banda GDDR6 vs. FP32, eficiência de cache L1/L2 e ocupação)?_

## 6. Conclusão

_Retome a pergunta do objetivo e responda com os números coletados. Aponte limitações (ex.: precisão FP32, contenção atômica em distribuições altamente concentradas, capacidade `max_cells` em VRAM) e possíveis melhorias futuras._

## 7. Referências

- BURTSCHER, M.; PINGALI, K. An Efficient CUDA Implementation of the Tree-Based Barnes-Hut n-Body Algorithm. GPU Computing Gems Emerald Edition, Morgan Kaufmann, p. 75-92, 2011.
- NYLAND, L.; HARRIS, M.; PRINS, J. Fast N-Body Simulation with CUDA. In: NGUYEN, H. (Ed.). GPU Gems 3, Addison-Wesley, cap. 31, p. 677-695, 2007.
- WILLIAMS, S. et al. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, v. 52, n. 4, p. 65-76, 2009.
- NVIDIA. CUDA C++ Programming Guide.

## Apêndice A — Como reproduzir

```sh
# 1. Build limpo em Release (FP32 unificado + CUDA sm_75) e testes unitarios
./scripts/build.sh --clean --cuda --float --tests
./build/nbody_tests

# 2. Benchmarks em CSV (sem diagnosticos O(N^2))
./scripts/benchmark.sh --solver brute --sizes 1000,2000,4000,8000 --steps 20 --repeats 3
./scripts/benchmark.sh --solver barnes-hut --theta 0.5 \
  --sizes 1000,2000,4000,8000,16000,50000 --steps 20 --repeats 3
./scripts/benchmark.sh --solver barnes-hut-cuda --theta 0.5 \
  --sizes 1000,2000,4000,8000,16000,50000,100000,250000,500000 --steps 20 --repeats 3

# 3. Decomposicao por kernel na GPU
for n in 16000 100000 500000; do
  ./build/nbody --init plummer --n "$n" --steps 20 --dt 0.001 \
    --solver barnes-hut-cuda --theta 0.5 --no-diagnostics --profile-kernels
done

# 4. Perfilamento Roofline com NVIDIA Nsight Compute (ncu)
ncu --set roofline --kernel-name k_compute_forces --launch-count 1 \
  ./build/nbody --init plummer --n 100000 --steps 1 --dt 0.001 \
  --solver barnes-hut-cuda --theta 0.5 --no-diagnostics

# 5. Sensibilidade a theta e erro relativo (Tabela 4.2)
for t in 0.2 0.5 0.8; do
  ./build/nbody --init plummer --n 4000 --solver barnes-hut-cuda --theta "$t" --verify-error
  ./scripts/benchmark.sh --solver barnes-hut --theta "$t" --sizes 16000 --steps 20 --repeats 3
  ./scripts/benchmark.sh --solver barnes-hut-cuda --theta "$t" --sizes 16000 --steps 20 --repeats 3
done

# 6. Conservacao de energia e graficos (Secoes 4.4 e 4.5)
./build/nbody --init plummer --n 2000 --steps 200 --dt 0.001 --check-every 5 \
  --solver brute --metrics results/energia_brute.csv
./build/nbody --init plummer --n 2000 --steps 200 --dt 0.001 --check-every 5 \
  --solver barnes-hut --theta 0.5 --metrics results/energia_bh.csv
./build/nbody --init plummer --n 2000 --steps 200 --dt 0.001 --check-every 5 \
  --solver barnes-hut-cuda --theta 0.5 --metrics results/energia_cuda.csv

python3 scripts/plot_results.py --benchmark results/benchmark.csv --theta 0.5 --theta-n 16000 \
  --energy "Brute=results/energia_brute.csv" "BH CPU=results/energia_bh.csv" "BH CUDA=results/energia_cuda.csv"
```