#include "nbody/simulation.hpp"

#include <stdexcept>

#include "nbody/accelerations.hpp"

namespace nbody {
namespace {

void kick(BodySystem& system, Real dt) {
  const std::size_t n = system.size();
  for (std::size_t i = 0; i < n; ++i) {
    system.vx[i] += dt * system.ax[i];
    system.vy[i] += dt * system.ay[i];
    system.vz[i] += dt * system.az[i];
  }
}

void drift(BodySystem& system, Real dt) {
  const std::size_t n = system.size();
  for (std::size_t i = 0; i < n; ++i) {
    system.x[i] += dt * system.vx[i];
    system.y[i] += dt * system.vy[i];
    system.z[i] += dt * system.vz[i];
  }
}

}  // namespace

Simulation::Simulation(const BodySystem& initial, const SimulationParams& params)
    : bodies_(initial), params_(params) {
  if (bodies_.size() < 2) {
    throw std::invalid_argument("a simulacao precisa de pelo menos 2 corpos");
  }
  if (params_.dt <= Real(0.0)) {
    throw std::invalid_argument("dt deve ser positivo");
  }
  if (params_.softening < Real(0.0)) {
    throw std::invalid_argument("o softening nao pode ser negativo");
  }
  if (params_.gravity < Real(0.0)) {
    throw std::invalid_argument("a constante gravitacional nao pode ser negativa");
  }

  if (params_.solver == Solver::BarnesHutCuda) {
    cuda_ = make_cuda_barnes_hut_backend(bodies_, params_);
    host_dirty_ = true;
    return;
  }

  // Aceleracao em t = 0: o velocity Verlet precisa dela para o primeiro kick.
  compute_accelerations(bodies_, params_);
}

void Simulation::step() {
  if (cuda_) {
    cuda_->step();
    host_dirty_ = true;
    ++steps_;
    return;
  }

  const Real half_dt = Real(0.5) * params_.dt;
  kick(bodies_, half_dt);
  drift(bodies_, params_.dt);
  compute_accelerations(bodies_, params_);
  kick(bodies_, half_dt);

  ++steps_;
}

void Simulation::synchronize() const {
  if (cuda_) {
    cuda_->synchronize();
  }
}

const BodySystem& Simulation::bodies() const {
  if (cuda_ && host_dirty_) {
    cuda_->download(bodies_);
    host_dirty_ = false;
  }
  return bodies_;
}

CudaKernelTimings Simulation::cuda_timings() const {
  return cuda_ ? cuda_->timings() : CudaKernelTimings{};
}

}  // namespace nbody