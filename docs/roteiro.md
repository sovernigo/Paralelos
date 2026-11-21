# Roteiro do trabalho

Checklist para sair do baseline sequencial até a implementação CUDA do
Barnes-Hut. Ajuste prazos ao calendário da disciplina.

## 1. Baseline sequencial (esta base)

- [x] Estrutura SoA e vetor 3D
- [x] Força bruta O(N²) com softening
- [x] Velocity Verlet
- [x] Condições iniciais (Plummer, esfera uniforme, disco)
- [x] Diagnósticos de energia/momento/centro de massa
- [x] Testes de corretude e medição em CSV
- [ ] Rodar `build.ps1 -Tests` na máquina de desenvolvimento
- [ ] Medir o baseline com `benchmark.ps1` e guardar o CSV
- [ ] Confirmar empiricamente a curva O(N²) (dobrar N → ~4× tempo)

## 2. Barnes-Hut sequencial

- [ ] Definir a caixa envolvente (bounding box) de todos os corpos a cada passo
- [ ] Construir o octree:
  - [ ] nós em arrays (`children[8]`, `mass`, `cm`, `size`, `body_index`), sem ponteiros
  - [ ] subdividir quando o nó passa da capacidade máxima
  - [ ] tratar corpos coincidentes (limite de profundidade)
- [ ] Calcular massa total e centro de massa em pós-ordem
- [ ] Travessia por corpo:
  - [ ] `s/d < theta` → usa centro de massa como corpo equivalente
  - [ ] caso contrário, desce nos filhos
  - [ ] nó folha → soma as forças dos corpos individuais
- [ ] Preencher `ax/ay/az` no mesmo layout SoA
- [ ] Validar contra o brute force:
  - [ ] erro relativo médio/máximo da aceleração para θ = 0.3, 0.5, 0.7
  - [ ] energia conservada ao longo dos passos
  - [ ] speedup para N = 10³…10⁶
- [ ] Otimizações: ordem dos corpos (Morton), evitar recomputar caixa,
      reaproveitar buffers entre passos, percorrer filhos na ordem do octante
      do corpo

## 3. CUDA

- [ ] Portar as estruturas para `float` (ou manter `double` se a GPU suportar bem)
- [ ] Um thread por corpo, travessia da árvore na GPU
- [ ] Manter a árvore em buffers lineares (índices) — requisito para desempenho
- [ ] Construção da árvore: atômica/paralela ou sequencial no host (medir o custo)
- [ ] Minimizar transferências host↔device (dados ficam na GPU entre passos)
- [ ] Comparar: kernel × transferência × construção da árvore
- [ ] Ajustar block size e uso de memória compartilhada
- [ ] Coletar métricas com Nsight Compute (Roofline, ocupação, coalescência)

## 4. Análise e entrega

- [ ] Tabela speedup × N para brute CPU, Barnes-Hut CPU e GPU
- [ ] Tabela erro × θ (precisão contra o brute force)
- [ ] Gráficos: tempo × N (escala log), speedup × N, energia × passos
- [ ] Discutir lei de Amdahl, overhead de construção da árvore e banda de memória
- [ ] Relatório preenchido (`docs/relatorio.md`)
- [ ] Código formatado, sem warnings, build limpo do zero
- [ ] Zip/tar.gz com código sequencial **e** paralelo + resultados + instruções

## Dicas de medição

- Sempre em build `Release`; nunca compare tempos de `Debug`.
- `--check-every` no máximo de passos possível durante benchmarks: o
  diagnóstico de energia também é O(N²) e distorce a medição.
- Use o mesmo `--seed` ao comparar solvers (mesma condição inicial).
- Rode 3+ vezes e use o melhor tempo/mediana; anote o hardware no relatório.
- Se a energia derivar muito, reduza `dt` antes de culpar o solver.
- Em N pequeno o Barnes-Hut perde para o brute force: o custo da árvore domina.
  Encontre e relate o ponto de cruzamento.
