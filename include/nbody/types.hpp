#pragma once

namespace nbody {

// Precisao dos calculos. double na versao sequencial; para a versao CUDA
// pode valer a pena trocar para float (mais banda de memoria por SM), com o
// custo de menor precisao na soma de forcas.
using Real = double;

}  // namespace nbody
