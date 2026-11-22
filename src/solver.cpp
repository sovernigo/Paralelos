#include <stdexcept>
#include <string>

#include "nbody/accelerations.hpp"
#include "nbody/cuda_backend.hpp"
#include "nbody/params.hpp"

namespace nbody {

bool parse_solver(const std::string& text, Solver& out) {
  if (text == "brute" || text == "brute-force") {
    out = Solver::BruteForce;
    return true;
  }
  if (text == "barnes-hut" || text == "bh") {
    out = Solver::BarnesHut;
    return true;
  }
  if (text == "barnes-hut-cuda" || text == "bh-cuda" || text == "cuda") {
    out = Solver::BarnesHutCuda;
    return true;
  }
  return false;
}

const char* solver_name(Solver solver) {
  switch (solver) {
    case Solver::BruteForce:
      return "brute";
    case Solver::BarnesHut:
      return "barnes-hut";
    case Solver::BarnesHutCuda:
      return "barnes-hut-cuda";
  }
  return "desconhecido";
}

void compute_accelerations(BodySystem& system, const SimulationParams& params) {
  switch (params.solver) {
    case Solver::BruteForce:
      compute_accelerations_bruteforce(system, params.gravity, params.softening);
      return;
    case Solver::BarnesHut:
      compute_accelerations_barnes_hut(system, params);
      return;
    case Solver::BarnesHutCuda:
      compute_accelerations_barnes_hut_cuda(system, params);
      return;
  }
  throw std::invalid_argument("solver desconhecido");
}

}  // namespace nbody