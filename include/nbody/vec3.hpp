#pragma once

#include <cmath>

#include "nbody/types.hpp"

namespace nbody {

// Vetor 3D minimo. As operacoes sao inline para nao pagarem custo de chamada
// nos lacos internos (o lacos de forca e o gargalo O(N^2)).
struct Vec3 {
  Real x = 0.0;
  Real y = 0.0;
  Real z = 0.0;

  Vec3() = default;
  Vec3(Real px, Real py, Real pz) : x(px), y(py), z(pz) {}

  Vec3& operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  Vec3& operator-=(const Vec3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  Vec3& operator*=(Real scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
inline Vec3 operator*(Vec3 a, Real scalar) { return a *= scalar; }
inline Vec3 operator*(Real scalar, Vec3 a) { return a *= scalar; }

inline Real dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Real norm2(const Vec3& a) { return dot(a, a); }

inline Real norm(const Vec3& a) { return std::sqrt(norm2(a)); }

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline Vec3 normalized(const Vec3& a) {
  const Real length = norm(a);
  if (length == 0.0) {
    return Vec3{};
  }
  return a * (1.0 / length);
}

}  // namespace nbody
