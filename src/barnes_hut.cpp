// Barnes-Hut sequencial: ponto de partida da proxima etapa.
//
// Roteiro de implementacao (detalhado em docs/roteiro.md):
//
//   1. Delimitar a caixa envolvente de todos os corpos.
//   2. Construir a arvore (quadtree em 2D, octree em 3D), subdividindo um no
//      quando ele passa da capacidade maxima. O octree linearizado em arrays
//      (indices em vez de ponteiros) e o caminho indicado para a GPU:
//      acesso previsivel, sem saltos de ponteiro, cabendo em um unico buffer.
//   3. Em pos-ordem, calcular massa total e centro de massa de cada no.
//   4. Para cada corpo, percorrer a arvore:
//        - no distante (s / d < theta): usa o centro de massa como um unico
//          corpo equivalente;
//        - no proximo: abre e visita os filhos.
//   5. Escrever o resultado em ax/ay/az (mesmo layout SoA do solver atual).
//
// A interface ja esta integrada: basta preencher compute_accelerations_barnes_hut
// e nada mais no projeto precisa mudar.
#include <stdexcept>

#include "nbody/accelerations.hpp"

namespace nbody {

void compute_accelerations_barnes_hut(BodySystem& /*system*/, const SimulationParams& /*params*/) {
  throw std::runtime_error(
      "solver Barnes-Hut ainda nao implementado: use --solver brute ou implemente "
      "compute_accelerations_barnes_hut em src/barnes_hut.cpp");
}

}  // namespace nbody
