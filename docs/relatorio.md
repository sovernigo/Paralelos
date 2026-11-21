# Relatório — Simulação N-Corpos com Barnes-Hut

**Título:** Paralelização da Simulação de N-Body: O Algoritmo Barnes-Hut em CUDA
**Aluno:** Felipe Sovernigo 
**Disciplina:** Programação paralela

## 1. Objetivo

_Descreva o problema (simulação gravitacional de N corpos), a motivação do
Barnes-Hut (reduzir O(N²) para O(N log N)) e a pergunta que o trabalho
responde (ex.: "qual o ganho de desempenho do Barnes-Hut em CUDA em relação ao
baseline sequencial em CPU, variando N e θ?")._

## 2. Metodologia

### 2.1 Hardware

| Item | Valor |
|---|---|
| CPU | _(preencher com `lscpu`)_ |
| RAM | _(preencher com `free -h`)_ |
| GPU | NVIDIA GeForce RTX 2060 SUPER (Turing, SM 7.5, 34 SMs, 2176 núcleos CUDA, 8 GB GDDR6) |
| SO / Ambiente | Ubuntu 24.04 (container Docker com acesso à GPU) |

### 2.2 Software

| Item | Valor |
|---|---|
| Compilador | GCC 13.2.0 (`g++`) |
| Flags | `-O3 -DNDEBUG -march=native` |
| CMake | 3.28.3 |
| CUDA Toolkit | 12.6 (V12.6.77), driver 621.14 |
| Nsight Compute | 2024.3.2.0 |

### 2.3 Modelo físico e integrador

- Força com softening de Plummer (ε = ___)
- Integrador: velocity Verlet, Δt = ___
- Condição inicial: _(Plummer / disco / esfera uniforme)_, N = ___, seed = ___
- Unidades: G = 1, massa total = 1

### 2.4 Procedimento experimental

- Tamanhos: _(ex.: 10³, 10⁴, 10⁵, 10⁶)_
- Passos medidos: ___ (aquecimento descartado)
- Repetições: ___ (melhor tempo e mediana)
- Diagnósticos desativados durante medição de tempo (`--no-diagnostics`)
- Critério de precisão: erro relativo médio da aceleração vs. força bruta
- Validação física: deriva de energia e momento ao longo dos passos

## 3. Implementações

### 3.1 Baseline sequencial (força bruta)

_Estrutura SoA, laço com terceira lei de Newton, custo O(N²)._

### 3.2 Barnes-Hut sequencial

_Construção da Octree linearizada em arrays (índices em vez de ponteiros),
cálculo de massa/centro de massa em pós-ordem, critério s/d < θ e complexidade
O(N log N)._

### 3.3 Barnes-Hut em CUDA

_Abordagem baseada em Burtscher & Pingali: Octree linearizada em vetores
pré-alocados na VRAM, construção da árvore sincronizada via operações atômicas
(locks leves), travessia com um thread por corpo e execução 100% na GPU em
múltiplos kernels para evitar tráfego pelo barramento PCIe a cada passo._

## 4. Resultados

### 4.1 Desempenho

| Implementação | N | θ | Tempo (ms/passo) | Speedup | Erro rel. da força |
|---|---|---|---|---|---|
| Brute force (CPU) | | — | | — | 0 |
| Barnes-Hut (CPU) | | 0,5 | | 1,00 | |
| Barnes-Hut (GPU) | | 0,5 | | | |

### 4.2 Precisão × θ

| θ | Erro médio da aceleração | Tempo (ms/passo) | Speedup |
|---|---|---|---|
| 0,2 | | | |
| 0,5 | | | |
| 0,8 | | | |

### 4.3 Conservação de energia

_Inclua o gráfico energia total × passo para o baseline e para o Barnes-Hut.
Discuta a deriva e o efeito de θ._

### 4.4 Escalabilidade

_Gráfico tempo × N em escala log-log: a inclinação deve se aproximar de 2 no
brute force e de 1 no Barnes-Hut. Aponte o ponto de cruzamento e o comportamento
até milhões de corpos na GPU._

## 5. Análise

- _O speedup acompanha a redução de complexidade e o paralelismo da GPU? Onde o Barnes-Hut satura?_
- _Qual o custo da construção da árvore em relação ao cálculo das forças?_
- _Como θ afeta o compromisso precisão × desempenho?_
- _Na GPU: qual kernel domina o tempo de execução?_
- _O que o Modelo Roofline / Nsight Compute mostra (intensidade aritmética, limites de banda vs. computação, coalescência e divergência de warps)?_
- _Quais efeitos de memória (latência da travessia irregular, cache) aparecem?_

## 6. Conclusão

_Retome a pergunta do objetivo e responda com os números. Aponte limitações e
o que faria diferente._

## 7. Referências

- BURTSCHER, M.; PINGALI, K. An Efficient CUDA Implementation of the Tree-Based Barnes-Hut n-Body Algorithm. GPU Computing Gems Emerald Edition, Morgan Kaufmann, p. 75-92, 2011.
- NYLAND, L.; HARRIS, M.; PRINS, J. Fast N-Body Simulation with CUDA. In: NGUYEN, H. (Ed.). GPU Gems 3, Addison-Wesley, cap. 31, p. 677-695, 2007.
- WILLIAMS, S. et al. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, v. 52, n. 4, p. 65-76, 2009.
- NVIDIA. CUDA C++ Programming Guide.

## Apêndice A — Como reproduzir

```sh
./scripts/build.sh --clean --tests
./scripts/benchmark.sh --solver brute --sizes 1000,2000,4000,8000 --steps 20 --repeats 3