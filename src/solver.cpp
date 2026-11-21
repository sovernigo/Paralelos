#include "nbody/accelerations.hpp"
#include "nbody/params.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace nbody {
namespace {

std::string to_lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

}  // namespace

bool parse_solver(const std::string& text, Solver& out) {
  const std::string key = to_lower(text);
  if (key == "brute" || key == "brute-force" || key == "bruteforce") {
    out = Solver::BruteForce;
    return true;
  }
  if (key == "barnes-hut" || key == "barneshut" || key == "barnes") {
    out = Solver::BarnesHut;
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
  }

  throw std::runtime_error("solver desconhecido");
}

}  // namespace nbody
