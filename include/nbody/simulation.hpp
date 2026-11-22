#pragma once

#include <cstddef>
#include <memory>

#include "nbody/body_system.hpp"
#include "nbody/cuda_backend.hpp"
#include "nbody/params.hpp"
#include "nbody/types.hpp"

namespace nbody {

class Simulation {
 public:
  Simulation(const BodySystem& initial, const SimulationParams& params);

  void step();

  // No backend CUDA, sincroniza a GPU antes de encerrar cronometros externos.
  void synchronize() const;

  std::size_t step_count() const { return steps_; }
  Real time() const { return static_cast<Real>(steps_) * params_.dt; }

  // Em modo CUDA, baixa os dados da VRAM apenas quando houve passo pendente.
  const BodySystem& bodies() const;
  const SimulationParams& params() const { return params_; }

  bool uses_cuda() const { return cuda_ != nullptr; }
  CudaKernelTimings cuda_timings() const;

 private:
  mutable BodySystem bodies_;
  SimulationParams params_;
  std::size_t steps_ = 0;
  std::unique_ptr<CudaBackend> cuda_;
  mutable bool host_dirty_ = false;
};

}  // namespace nbody