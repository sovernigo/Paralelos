# Roteiro do trabalho

Checklist para sair do baseline sequencial até a implementação CUDA do
Barnes-Hut. Ajuste os prazos ao calendário da disciplina.

## 1. Baseline sequencial (esta base)

- [x] Estrutura SoA e vetor 3D
- [x] Força bruta O(N²) com softening e 3ª lei de Newton
- [x] Velocity Verlet
- [x] Condições iniciais (Plummer, esfera uniforme, disco)
- [x] Diagnósticos de energia/momento/centro de massa
- [x] Testes de corretude e medição em CSV
- [x] Flag `--no-diagnostics` e scripts Linux (`build.sh`, `benchmark.sh`)
- [ ] Rodar `./scripts/build.sh --clean --tests` no container Linux
- [ ] Medir o baseline com `./scripts/benchmark.sh` e guardar o CSV
- [ ] Confirmar empiricamente a curva O(N²) (dobrar N → ~4× tempo)

## 2. Barnes-Hut sequencial

- [ ] Definir a caixa envolvente (bounding box) de todos os corpos a cada passo
- [ ] Construir a Octree linearizada:
  - [ ] nós em arrays (`children[8]`, `mass`, `cm`, `half_size`, `body`), sem ponteiros
  - [ ] subdividir nós quando ocupados
  - [ ] tratar corpos coincidentes com limite de profundidade
- [ ] Calcular massa total e centro de massa em pós-ordem
- [ ] Travessia iterativa por corpo:
  - [ ] `s/d < theta` → usa centro de massa como corpo equivalente
  - [ ] caso contrário, abre o nó e visita os filhos
  - [ ] nó folha → soma as forças dos corpos individuais
- [ ] Preencher `ax/ay/az` no mesmo layout SoA
- [ ] Validar contra o brute force:
  - [ ] exatidão com `theta = 0`
  - [ ] erro relativo médio/máximo da aceleração para `theta = 0.2, 0.5, 0.8`
  - [ ] energia conservada ao longo dos passos
  - [ ] ponto de cruzamento e speedup para N = 10³…10⁵
- [ ] Otimizações: reaproveitar buffers entre passos e avaliar ordenação espacial (Morton)

## 3. Barnes-Hut em CUDA (Burtscher & Pingali)

- [ ] Garantir precisão consistente (`float` ou `double`) entre baseline CPU e GPU
- [ ] Octree linearizada em vetores pré-alocados na VRAM (índices em vez de ponteiros)
- [ ] Construção da árvore na GPU sincronizada via operações atômicas (locks leves)
- [ ] Um thread por corpo na travessia da árvore
- [ ] Execução 100% na GPU em múltiplos kernels (incluindo kick-drift-kick), evitando tráfego pelo barramento PCIe a cada passo
- [ ] Medir o tempo de cada kernel separadamente (bounding box, construção, centro de massa, força, integração)
- [ ] Ajustar block size e avaliar uso de memória compartilhada
- [ ] Coletar métricas com NVIDIA Nsight Compute (`ncu`) para o Modelo Roofline da RTX 2060 SUPER

## 4. Análise e entrega

- [ ] Tabela speedup × N para `brute` CPU, Barnes-Hut CPU e Barnes-Hut CUDA
- [ ] Tabela erro × `theta` (precisão contra o brute force)
- [ ] Gráficos: tempo × N (escala log-log), speedup × N, energia × passos, Roofline
- [ ] Discutir escalabilidade, divergência de warps, custo da construção da árvore e limites de memória
- [ ] Relatório preenchido (`docs/relatorio.md`)
- [ ] Código formatado (`clang-format`), sem warnings, build limpo do zero
- [ ] Pacote final (`.zip` ou `.tar.gz` via `git archive`) com código sequencial **e** paralelo + instruções

## Dicas de medição

- Sempre em build `Release` (`./scripts/build.sh` já usa `-O3 -DNDEBUG -march=native`).
- Use `--no-diagnostics` durante benchmarks de desempenho: o diagnóstico de energia potencial custa O(N²).
- Use o mesmo `--seed` ao comparar solvers (mesma condição inicial).
- Rode com aquecimento (`--warmup 1`) e 3+ repetições (`--repeats 3`), reportando melhor tempo e mediana.
- Se a energia derivar muito, reduza `dt` antes de culpar o solver.
- Em N pequeno o Barnes-Hut perde para o brute force porque o custo da árvore domina; encontre e relate o ponto de cruzamento.