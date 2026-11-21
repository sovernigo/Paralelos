#pragma once

#include "nbody/body_system.hpp"
#include "nbody/params.hpp"
#include "nbody/types.hpp"

namespace nbody {

// Aceleracao gravitacional com suavizacao de Plummer:
//
//   a_i = soma_{j != i} G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
//
// O resultado e escrito em ax/ay/az do proprio sistema (in-place), que por
// isso e passado por referencia nao-const.
void compute_accelerations_bruteforce(BodySystem& system, Real gravity, Real softening);

// Barnes-Hut sequencial: esqueleto reservado para a proxima etapa
// (src/barnes_hut.cpp). Hoje lanca std::runtime_error.
void compute_accelerations_barnes_hut(BodySystem& system, const SimulationParams& params);

// Escolhe o solver a partir de params.solver.
void compute_accelerations(BodySystem& system, const SimulationParams& params);

}  // namespace nbody
