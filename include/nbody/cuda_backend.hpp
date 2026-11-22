#pragma once

#include <memory>

#include "nbody/body_system.hpp"
#include "nbody/params.hpp"

namespace nbody {

// Tempos acumulados por kernel (ms) para analise de gargalos no relatorio.
struct CudaKernelTimings {
  double integrate_ms = 0.0;
  double bbox_ms = 0.0;
  double sort_ms = 0.0;
  double build_tree_ms = 0.0;
  double summarize_ms = 0.0;
  double force_ms = 0.0;
};

class CudaBackend {
 public:
  virtual ~CudaBackend() = default;

  // Executa um passo completo de velocity Verlet 100% na GPU.
  virtual void step() = 0;

  // Copia o estado atual (x, y, z, vx, vy, vz, ax, ay, az) da VRAM para o host.
  virtual void download(BodySystem& out) const = 0;

  // Garante conclusao de todos os kernels antes de encerrar a medicao de tempo.
  virtual void synchronize() const = 0;

  virtual CudaKernelTimings timings() const = 0;
};

bool cuda_backend_available();

std::unique_ptr<CudaBackend> make_cuda_barnes_hut_backend(const BodySystem& initial,
                                                          const SimulationParams& params);

// Auxiliar para testes unitarios: avalia apenas a aceleracao em t=0 na GPU.
void compute_accelerations_barnes_hut_cuda(BodySystem& system, const SimulationParams& params);

}  // namespace nbody