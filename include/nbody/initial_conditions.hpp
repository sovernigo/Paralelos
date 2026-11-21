#pragma once

#include <cstddef>
#include <string>

#include "nbody/body_system.hpp"
#include "nbody/types.hpp"

namespace nbody {

// Distribuicoes iniciais de corpos (condicoes de contorno do experimento).
enum class Distribution {
  UniformSphere,  // esfera homogenea em repouso: colapso gravitacional
  Plummer,        // esfera de Plummer em equilibrio (padrao de teste)
  Disk,           // disco em rotacao: visualmente interessante
};

bool parse_distribution(const std::string& text, Distribution& out);
const char* distribution_name(Distribution distribution);

// Massa total distribuida uniformemente; velocidades nulas.
BodySystem make_uniform_sphere(std::size_t n, Real total_mass, Real radius, unsigned int seed);

// Posicoes amostradas da distribuicao de Plummer e velocidades da
// distribuicao de velocidades correspondente (aproximadamente em equilibrio,
// com razao virial Q = 2T/|U| ~ 0.5).
BodySystem make_plummer_sphere(std::size_t n, Real total_mass, Real scale_radius, Real gravity,
                               unsigned int seed);

// Disco de densidade superficial uniforme com rotacao circular em torno do
// eixo z. scale_height e o desvio padrao da altura z.
BodySystem make_disk(std::size_t n, Real total_mass, Real radius, Real scale_height, Real gravity,
                     unsigned int seed);

}  // namespace nbody
