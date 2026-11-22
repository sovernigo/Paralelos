# Roteiro do trabalho

Checklist do projeto N-Corpos: baseline sequencial, Barnes-Hut CPU,
implementação CUDA (Burtscher & Pingali) e análise experimental.

## 1. Baseline sequencial (força bruta)

- [x] Estrutura SoA e vetor 3D
- [x] Força bruta O(N²) com softening e 3ª lei de Newton
- [x] Velocity Verlet (*kick-drift-kick*)
- [x] Condições iniciais (Plummer, esfera uniforme, disco)
- [x] Diagnósticos de energia/momento/centro de massa
- [x] Testes de corretude e medição em CSV
- [x] Flag `--no-diagnostics` e scripts Linux (`build.sh`, `benchmark.sh`)
- [x] Build limpo e testes validados no container Linux
- [x] Curva O(N²) confirmada no benchmark sequencial

## 2. Barnes-Hut sequencial (CPU)

- [x] Caixa envolvente cúbica (*bounding box*) a cada passo
- [x] Octree linearizada em arrays (`children[8]`, `body`, `cx/cy/cz/half_size`, `mass/cm`)
- [x] Construção iterativa com tratamento de corpos coincidentes (`kMaxDepth = 32`)
- [x] Massa total e centro de massa em pós-ordem sem recursão
- [x] Travessia iterativa com pilha explícita e critério $s^2 < \theta^2 d^2$
- [x] Reaproveitamento de buffers entre passos (`thread_local`)
- [x] Validação unitária ($\theta = 0$, monotonicidade $\theta = 0.2, 0.5, 0.8$, coincidentes, conservação de energia)
- [x] Ponto de cruzamento com `brute` verificado (entre N = 1.000 e N = 2.000)

## 3. Barnes-Hut em CUDA (Burtscher & Pingali)

- [x] Precisão configurável (`--float` / `--double`) unificada entre CPU e GPU
- [x] Octree linearizada em vetores pré-alocados na VRAM
- [x] Construção paralela na GPU sincronizada via operações atômicas (`atomicCAS`)
- [x] Ordenação espacial por código de Morton 3D (`cub::DeviceRadixSort`) para coerência de warps
- [x] Sumarização *bottom-up* sem *spin-wait* (`atomicSub`)
- [x] Travessia com 1 thread por corpo e execução 100% na GPU (incluindo Velocity Verlet)
- [x] Perfilamento por kernel via eventos CUDA (`--profile-kernels`)
- [x] Flag `--verify-error` para medir erro relativo em qualquer N e $\theta$
- [x] Validar `./build/nbody_tests` completo na RTX 2060 SUPER
- [ ] Coletar decomposição por kernel (`--profile-kernels`) em N = 16k, 100k e 500k
- [ ] Coletar métricas do Modelo Roofline com NVIDIA Nsight Compute (`ncu`)

## 4. Análise e entrega

- [x] Script de geração de gráficos (`scripts/plot_results.py`)
- [ ] Rodar benchmarks finais em FP32 unificado (`brute`, `barnes-hut`, `barnes-hut-cuda`)
- [ ] Medir erro relativo × $\theta$ (0.2, 0.5, 0.8) via `--verify-error`
- [ ] Coletar CSVs de conservação de energia (`--metrics`) para os 3 solvers
- [ ] Gerar gráficos finais (`tempo_vs_n`, `speedup_vs_n`, `tempo_vs_theta`, `energia_vs_passos`)
- [ ] Preencher `docs/relatorio.md` (hardware CPU/RAM, tabelas, gráficos, análise Roofline e conclusão)
- [ ] Formatar código (`clang-format`), conferir `git status` e gerar pacote final (`git archive`)