#include "nbody/cuda_backend.hpp"

#include <stdexcept>

namespace nbody {

bool cuda_backend_available() { return false; }

std::unique_ptr<CudaBackend> make_cuda_barnes_hut_backend(const BodySystem&,
                                                          const SimulationParams&) {
  throw std::runtime_error("backend CUDA desabilitado neste build (compile com --cuda)");
}

void compute_accelerations_barnes_hut_cuda(BodySystem&, const SimulationParams&) {
  throw std::runtime_error("backend CUDA desabilitado neste build (compile com --cuda)");
}

}  // namespace nbody