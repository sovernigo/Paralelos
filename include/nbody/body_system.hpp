#pragma once

#include <cstddef>
#include <vector>

#include "nbody/types.hpp"

namespace nbody {

// Estrutura de arrays (SoA - Structure of Arrays).
//
// Em vez de um vetor de structs (AoS), cada grandeza fica em um array
// contiguo. Os lacos de forca leem posicoes de forma sequencial (bom para
// cache e para vetorizacao) e, na versao CUDA, o mesmo layout permite enviar
// cada array como uma unica transferencia host->device.
struct BodySystem {
  std::vector<Real> x;   // posicao
  std::vector<Real> y;
  std::vector<Real> z;
  std::vector<Real> vx;  // velocidade
  std::vector<Real> vy;
  std::vector<Real> vz;
  std::vector<Real> ax;  // aceleracao (preenchida pelo solver)
  std::vector<Real> ay;
  std::vector<Real> az;
  std::vector<Real> mass;

  std::size_t size() const { return mass.size(); }
  bool empty() const { return mass.empty(); }

  // Redimensiona todos os arrays e zera o conteudo.
  void resize(std::size_t n);

  void reserve(std::size_t n);

  void add_body(Real px, Real py, Real pz, Real pvx, Real pvy, Real pvz, Real pmass);
};

}  // namespace nbody
