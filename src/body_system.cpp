#include "nbody/body_system.hpp"

namespace nbody {

void BodySystem::resize(std::size_t n) {
  x.assign(n, 0.0);
  y.assign(n, 0.0);
  z.assign(n, 0.0);
  vx.assign(n, 0.0);
  vy.assign(n, 0.0);
  vz.assign(n, 0.0);
  ax.assign(n, 0.0);
  ay.assign(n, 0.0);
  az.assign(n, 0.0);
  mass.assign(n, 0.0);
}

void BodySystem::reserve(std::size_t n) {
  x.reserve(n);
  y.reserve(n);
  z.reserve(n);
  vx.reserve(n);
  vy.reserve(n);
  vz.reserve(n);
  ax.reserve(n);
  ay.reserve(n);
  az.reserve(n);
  mass.reserve(n);
}

void BodySystem::add_body(Real px, Real py, Real pz, Real pvx, Real pvy, Real pvz, Real pmass) {
  x.push_back(px);
  y.push_back(py);
  z.push_back(pz);
  vx.push_back(pvx);
  vy.push_back(pvy);
  vz.push_back(pvz);
  ax.push_back(0.0);
  ay.push_back(0.0);
  az.push_back(0.0);
  mass.push_back(pmass);
}

}  // namespace nbody
