#include "nbody/accelerations.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace nbody {

// Baseline sequencial: O(N^2) por passo.
//
// Otimizacao importante: o laco interno comeca em j = i + 1 e usa a terceira
// lei de Newton (a forca que j exerce em i e o negativo da que i exerce em j).
// Cada par e avaliado uma unica vez e o custo cai pela metade em relacao ao
// laco completo, sem alterar o resultado.
void compute_accelerations_bruteforce(BodySystem& system, Real gravity, Real softening) {
  const std::size_t n = system.size();

  std::fill(system.ax.begin(), system.ax.end(), 0.0);
  std::fill(system.ay.begin(), system.ay.end(), 0.0);
  std::fill(system.az.begin(), system.az.end(), 0.0);

  const Real eps2 = softening * softening;

  for (std::size_t i = 0; i < n; ++i) {
    // Copia escalares para o laco mais interno: evita recarregar o vetor a
    // cada iteracao e deixa o compilador vetorizar com mais facilidade.
    const Real xi = system.x[i];
    const Real yi = system.y[i];
    const Real zi = system.z[i];
    const Real mi = system.mass[i];

    for (std::size_t j = i + 1; j < n; ++j) {
      const Real dx = system.x[j] - xi;
      const Real dy = system.y[j] - yi;
      const Real dz = system.z[j] - zi;

      const Real r2 = dx * dx + dy * dy + dz * dz + eps2;
      const Real inv_r = 1.0 / std::sqrt(r2);
      const Real factor = gravity * inv_r / r2;  // G / r^3

      const Real fx = factor * dx;
      const Real fy = factor * dy;
      const Real fz = factor * dz;

      const Real mj = system.mass[j];

      system.ax[i] += mj * fx;
      system.ay[i] += mj * fy;
      system.az[i] += mj * fz;

      system.ax[j] -= mi * fx;
      system.ay[j] -= mi * fy;
      system.az[j] -= mi * fz;
    }
  }
}

}  // namespace nbody
