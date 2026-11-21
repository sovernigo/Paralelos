#include "nbody/initial_conditions.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#include "nbody/vec3.hpp"

namespace nbody {
namespace {

constexpr Real kPi = 3.14159265358979323846;
constexpr Real kTwoPi = 2.0 * kPi;

std::string to_lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

Vec3 random_direction(std::mt19937_64& rng) {
  std::normal_distribution<Real> gauss(0.0, 1.0);
  Vec3 direction;
  do {
    direction = Vec3(gauss(rng), gauss(rng), gauss(rng));
  } while (norm2(direction) < 1e-12);
  return normalized(direction);
}

// Raio de Plummer: inverte a distribuicao acumulada M(<r)/M = r^3/(r^2+a^2)^(3/2).
Real sample_plummer_radius(Real scale_radius, std::mt19937_64& rng) {
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  const Real x = std::max<Real>(uniform(rng), 1e-12);
  return scale_radius / std::sqrt(std::pow(x, -2.0 / 3.0) - 1.0);
}

// Fator q da velocidade: g(q) = q^2 (1 - q^2)^(7/2) amostrada por rejeicao.
// O maximo de g em [0, 1) e ~0,092, entao 0,1 e uma cota superior segura.
Real sample_plummer_speed_factor(std::mt19937_64& rng) {
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  constexpr Real kMaxG = 0.1;
  for (;;) {
    const Real q = uniform(rng);
    const Real y = uniform(rng) * kMaxG;
    const Real g = q * q * std::pow(1.0 - q * q, 3.5);
    if (y <= g) {
      return q;
    }
  }
}

}  // namespace

bool parse_distribution(const std::string& text, Distribution& out) {
  const std::string key = to_lower(text);
  if (key == "uniforme" || key == "uniform" || key == "uniform-sphere") {
    out = Distribution::UniformSphere;
    return true;
  }
  if (key == "plummer") {
    out = Distribution::Plummer;
    return true;
  }
  if (key == "disco" || key == "disk") {
    out = Distribution::Disk;
    return true;
  }
  return false;
}

const char* distribution_name(Distribution distribution) {
  switch (distribution) {
    case Distribution::UniformSphere:
      return "uniforme";
    case Distribution::Plummer:
      return "plummer";
    case Distribution::Disk:
      return "disco";
  }
  return "desconhecida";
}

BodySystem make_uniform_sphere(std::size_t n, Real total_mass, Real radius, unsigned int seed) {
  if (n == 0) {
    throw std::invalid_argument("n deve ser maior que zero");
  }
  if (radius <= 0.0) {
    throw std::invalid_argument("o raio deve ser positivo");
  }

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  const Real body_mass = total_mass / static_cast<Real>(n);

  BodySystem bodies;
  bodies.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const Vec3 direction = random_direction(rng);
    // raio com densidade uniforme: cbrt(U) distribui o volume por igual.
    const Real r = radius * std::cbrt(uniform(rng));
    bodies.add_body(direction.x * r, direction.y * r, direction.z * r, 0.0, 0.0, 0.0, body_mass);
  }

  return bodies;
}

BodySystem make_plummer_sphere(std::size_t n, Real total_mass, Real scale_radius, Real gravity,
                               unsigned int seed) {
  if (n == 0) {
    throw std::invalid_argument("n deve ser maior que zero");
  }
  if (scale_radius <= 0.0 || total_mass <= 0.0 || gravity <= 0.0) {
    throw std::invalid_argument("massa, raio de escala e gravidade devem ser positivos");
  }

  std::mt19937_64 rng(seed);
  const Real body_mass = total_mass / static_cast<Real>(n);
  const Real velocity_scale = std::sqrt(2.0 * gravity * total_mass);
  const Real a2 = scale_radius * scale_radius;

  BodySystem bodies;
  bodies.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const Real r = sample_plummer_radius(scale_radius, rng);
    const Vec3 position_dir = random_direction(rng);
    const Vec3 velocity_dir = random_direction(rng);

    // v = q * v_esc(r), com v_esc = sqrt(2 G M / sqrt(r^2 + a^2)).
    const Real q = sample_plummer_speed_factor(rng);
    const Real speed = q * velocity_scale * std::pow(r * r + a2, -0.25);

    bodies.add_body(position_dir.x * r, position_dir.y * r, position_dir.z * r,
                    velocity_dir.x * speed, velocity_dir.y * speed, velocity_dir.z * speed,
                    body_mass);
  }

  return bodies;
}

BodySystem make_disk(std::size_t n, Real total_mass, Real radius, Real scale_height, Real gravity,
                     unsigned int seed) {
  if (n == 0) {
    throw std::invalid_argument("n deve ser maior que zero");
  }
  if (radius <= 0.0 || scale_height <= 0.0 || total_mass <= 0.0 || gravity <= 0.0) {
    throw std::invalid_argument("raio, altura, massa e gravidade devem ser positivos");
  }

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  std::uniform_real_distribution<Real> angle(0.0, kTwoPi);
  std::normal_distribution<Real> vertical(0.0, scale_height);

  const Real body_mass = total_mass / static_cast<Real>(n);

  BodySystem bodies;
  bodies.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    // Densidade superficial uniforme: r = R * sqrt(U).
    const Real r = radius * std::sqrt(uniform(rng));
    const Real phi = angle(rng);

    const Real px = r * std::cos(phi);
    const Real py = r * std::sin(phi);
    const Real pz = vertical(rng);

    // Rotacao circular: v^2 = G M_enc(r) / r, com M_enc = M (r/R)^2.
    const Real speed = std::sqrt(gravity * total_mass * r) / radius;
    const Real vx = -speed * std::sin(phi);
    const Real vy = speed * std::cos(phi);

    bodies.add_body(px, py, pz, vx, vy, 0.0, body_mass);
  }

  return bodies;
}

}  // namespace nbody
