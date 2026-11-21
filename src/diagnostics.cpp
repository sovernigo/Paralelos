#include "nbody/diagnostics.hpp"

#include <cmath>
#include <cstddef>

namespace nbody {

Diagnostics compute_diagnostics(const BodySystem& bodies, Real gravity, Real softening) {
  Diagnostics diag;
  const std::size_t n = bodies.size();
  const Real eps2 = softening * softening;

  Real total_mass = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const Real m = bodies.mass[i];
    const Real vx = bodies.vx[i];
    const Real vy = bodies.vy[i];
    const Real vz = bodies.vz[i];

    diag.kinetic += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    diag.momentum += Vec3(m * vx, m * vy, m * vz);
    diag.center_of_mass += Vec3(m * bodies.x[i], m * bodies.y[i], m * bodies.z[i]);
    total_mass += m;
  }

  // Energia potencial com o mesmo softening usado na forca, de modo que a
  // energia total seja (aproximadamente) conservada pelo integrador.
  for (std::size_t i = 0; i < n; ++i) {
    const Real xi = bodies.x[i];
    const Real yi = bodies.y[i];
    const Real zi = bodies.z[i];
    const Real mi = bodies.mass[i];

    for (std::size_t j = i + 1; j < n; ++j) {
      const Real dx = bodies.x[j] - xi;
      const Real dy = bodies.y[j] - yi;
      const Real dz = bodies.z[j] - zi;
      const Real r2 = dx * dx + dy * dy + dz * dz + eps2;
      diag.potential -= gravity * mi * bodies.mass[j] / std::sqrt(r2);
    }
  }

  if (total_mass > 0.0) {
    diag.center_of_mass *= (1.0 / total_mass);
  }

  diag.total_energy = diag.kinetic + diag.potential;
  return diag;
}

}  // namespace nbody
