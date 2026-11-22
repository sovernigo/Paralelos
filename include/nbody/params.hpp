#pragma once

#include <cstddef>
#include <string>

#include "nbody/types.hpp"

namespace nbody {

enum class Solver {
  BruteForce,     // O(N^2) sequencial em CPU (referencia exata)
  BarnesHut,      // O(N log N) sequencial em CPU (baseline para speedup)
  BarnesHutCuda,  // O(N log N) paralelo 100% em GPU (CUDA)
};

struct SimulationParams {
  Real gravity = Real(1.0);
  Real softening = Real(0.05);
  Real dt = Real(0.002);
  Real theta = Real(0.5);
  Solver solver = Solver::BruteForce;
  bool profile_kernels = false;
};

bool parse_solver(const std::string& text, Solver& out);
const char* solver_name(Solver solver);

}  // namespace nbody