# Relatório — Simulação N-Corpos com Barnes-Hut

**Título:** _(preencher)_
**Aluno:** _(nome, matrícula)_
**Disciplina / Professor:** _(preencher)_
**Data:** _(preencher)_

## 1. Objetivo

_Descreva o problema (simulação gravitacional de N corpos), a motivação do
Barnes-Hut (reduzir O(N²) para O(N log N)) e a pergunta que o trabalho
responde (ex.: "qual o ganho de desempenho do Barnes-Hut em CUDA em relação ao
baseline sequencial, para cada N e para cada θ?")._

## 2. Metodologia

### 2.1 Hardware

| Item | Valor |
|---|---|
| CPU | _(modelo, núcleos físicos/lógicos, frequência)_ |
| RAM | _(GB)_ |
| GPU | _(modelo, memória, compute capability, SMs)_ |
| SO | _(versão)_ |

### 2.2 Software

| Item | Valor |
|---|---|
| Compilador | _(nome e versão)_ |
| Flags | _(ex.: -O3 -march=native)_ |
| CMake | _(versão)_ |
| CUDA Toolkit | _(versão)_ |
| Nsight Compute | _(versão, se usado)_ |

### 2.3 Modelo físico e integrador

- Força com softening de Plummer (ε = ___)
- Integrador: velocity Verlet, Δt = ___
- Condição inicial: _(Plummer / disco / esfera uniforme)_, N = ___, seed = ___
- Unidades: G = 1, massa total = 1

### 2.4 Procedimento experimental

- Tamanhos: _(ex.: 10³, 10⁴, 10⁵, 10⁶)_
- Passos medidos: ___ (aquecimento descartado)
- Repetições: ___ (melhor tempo e mediana)
- Critério de precisão: erro relativo médio da aceleração vs. força bruta
- Validação física: deriva de energia e momento ao longo dos passos

## 3. Implementações

### 3.1 Baseline sequencial (força bruta)

_Estrutura SoA, laço com terceira lei de Newton, custo O(N²)._


### 3.2 Barnes-Hut sequencial

_Construção do octree, cálculo de massa/centro de massa, critério s/d < θ,
complexidade e estruturas de dados (arrays lineares)._


### 3.3 Barnes-Hut em CUDA

_Um thread por corpo, árvore em buffers lineares, uso de memória
compartilhada, construção da árvore, sincronização entre kernels._


## 4. Resultados

### 4.1 Desempenho

| Implementação | N | θ | Tempo (ms/passo) | Speedup | Erro rel. da força |
|---|---|---|---|---|---|
| Brute force (CPU) | | — | | 1,00 | 0 |
| Barnes-Hut (CPU) | | 0,5 | | | |
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
brute force e de 1 no Barnes-Hut. Aponte o ponto de cruzamento._

## 5. Análise

- _O speedup acompanha a redução de complexidade? Onde o Barnes-Hut satura?_
- _Qual o custo da construção da árvore em relação ao cálculo das forças?_
- _Como θ afeta o compromisso precisão × desempenho?_
- _Na GPU: o gargalo é o kernel, a construção da árvore ou a transferência?_
- _O que o Roofline/Nsight mostra (intensidade aritmética, coalescência)?_
- _Quais efeitos de memória (latência da travessia irregular, cache) aparecem?_

## 6. Conclusão

_Retome a pergunta do objetivo e responda com os números. Aponte limitações e
o que faria diferente._

## 7. Referências

- _Material da disciplina_
- BURTSCHER, M.; PINGALI, K. An Efficient CUDA Implementation of the Tree-Based Barnes-Hut n-Body Algorithm. GPU Computing Gems, 2011.
- NYLAND, L.; HARRIS, M.; PRINS, J. Fast N-Body Simulation with CUDA. GPU Gems 3, cap. 31.
- WILLIAMS, S. et al. Roofline: an insightful visual performance model for multicore architectures. CACM, 2009.
- NVIDIA. CUDA C++ Programming Guide.

## Apêndice A — Como reproduzir

```powershell
.\scripts\build.ps1 -Tests
.\scripts\benchmark.ps1 -Sizes 1000,2000,4000,8000
```

_Descreva qualquer ajuste necessário (threads, GPU, flags)._
