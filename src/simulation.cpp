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
  if (params_.dt <= 0.0) {
    throw std::invalid_argument("dt deve ser positivo");
  }
  if (params_.softening < 0.0) {
    throw std::invalid_argument("o softening nao pode ser negativo");
  }
  if (params_.gravity < 0.0) {
    throw std::invalid_argument("a constante gravitacional nao pode ser negativa");
  }

  // Aceleracao em t = 0: o velocity Verlet precisa dela para o primeiro kick.
  compute_accelerations(bodies_, params_);
}

void Simulation::step() {
  const Real half_dt = 0.5 * params_.dt;

  kick(bodies_, half_dt);   // v(t + dt/2)
  drift(bodies_, params_.dt);  // x(t + dt)
  compute_accelerations(bodies_, params_);  // a(t + dt)
  kick(bodies_, half_dt);   // v(t + dt)

  ++steps_;
}

}  // namespace nbody
