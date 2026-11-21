#pragma once

#include "nbody/body_system.hpp"
#include "nbody/types.hpp"
#include "nbody/vec3.hpp"

namespace nbody {

// Grandezas globais usadas para validar a simulacao (nao influenciam a fisica).
struct Diagnostics {
  Real kinetic = 0.0;
  Real potential = 0.0;
  Real total_energy = 0.0;
  Vec3 momentum;         // soma m*v
  Vec3 center_of_mass;   // soma m*r / M
};

// Custo O(N^2) por causa da energia potencial. Chame a cada K passos em vez
// de todo passo quando N for grande.
Diagnostics compute_diagnostics(const BodySystem& system, Real gravity, Real softening);

}  // namespace nbody
