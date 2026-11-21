#pragma once

#include <cstddef>
#include <string>

#include "nbody/types.hpp"

namespace nbody {

// Como as aceleracoes sao calculadas a cada passo.
enum class Solver {
  BruteForce,  // O(N^2): avalia todos os pares (baseline sequencial)
  BarnesHut,   // O(N log N): arvore + criterio theta (proxima etapa)
};

struct SimulationParams {
  Real gravity = 1.0;    // G
  Real softening = 0.05; // epsilon: evita aceleracao infinita em encontros proximos
  Real dt = 0.002;       // passo de integracao
  Real theta = 0.5;      // criterio s/d < theta do Barnes-Hut
  Solver solver = Solver::BruteForce;
};

bool parse_solver(const std::string& text, Solver& out);
const char* solver_name(Solver solver);

}  // namespace nbody
