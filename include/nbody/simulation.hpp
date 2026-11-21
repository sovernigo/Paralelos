#pragma once

#include <cstddef>

#include "nbody/body_system.hpp"
#include "nbody/params.hpp"
#include "nbody/types.hpp"

namespace nbody {

// Integrador de velocity Verlet (kick-drift-kick).
//
// E um integrador simpletico: o erro de energia oscila em torno de um valor
// constante em vez de crescer com o tempo, o que e essencial em simulacoes
// N-corpos de muitos passos (Euler explicito "explode").
class Simulation {
 public:
  // Copia os corpos e calcula as aceleracoes iniciais.
  // Lanca std::invalid_argument para entrada invalida.
  Simulation(const BodySystem& initial, const SimulationParams& params);

  // Avanca um passo: v += a*dt/2; x += v*dt; recalcula a; v += a*dt/2.
  void step();

  std::size_t step_count() const { return steps_; }
  Real time() const { return static_cast<Real>(steps_) * params_.dt; }

  const BodySystem& bodies() const { return bodies_; }
  const SimulationParams& params() const { return params_; }

 private:
  BodySystem bodies_;
  SimulationParams params_;
  std::size_t steps_ = 0;
};

}  // namespace nbody
