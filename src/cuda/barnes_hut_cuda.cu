#include "nbody/cuda_backend.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nbody {
namespace {

constexpr int kEmpty = -1;
constexpr int kLocked = -2;
constexpr int kMaxDepth = 32;
constexpr int kBlockSize = 256;

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "erro CUDA em " << file << ":" << line << " (" << expr
        << "): " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

#define NBODY_CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

__device__ inline float rsqrt_real(float x) { return rsqrtf(x); }
__device__ inline double rsqrt_real(double x) { return rsqrt(x); }

__device__ inline float fabs_real(float x) { return fabsf(x); }
__device__ inline double fabs_real(double x) { return fabs(x); }

__device__ inline void atomic_min_real(float* addr, float val) {
  unsigned int* uaddr = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = *uaddr;
  while (val < __uint_as_float(old)) {
    const unsigned int assumed = old;
    old = atomicCAS(uaddr, assumed, __float_as_uint(val));
    if (old == assumed) {
      break;
    }
  }
}

__device__ inline void atomic_max_real(float* addr, float val) {
  unsigned int* uaddr = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = *uaddr;
  while (val > __uint_as_float(old)) {
    const unsigned int assumed = old;
    old = atomicCAS(uaddr, assumed, __float_as_uint(val));
    if (old == assumed) {
      break;
    }
  }
}

__device__ inline void atomic_min_real(double* addr, double val) {
  unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *uaddr;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(uaddr, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
    if (old == assumed) {
      break;
    }
  }
}

__device__ inline void atomic_max_real(double* addr, double val) {
  unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *uaddr;
  while (val > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(uaddr, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
    if (old == assumed) {
      break;
    }
  }
}

__device__ inline int octant_of(Real x, Real y, Real z, Real cx, Real cy, Real cz) {
  int oct = 0;
  if (x >= cx) oct |= 1;
  if (y >= cy) oct |= 2;
  if (z >= cz) oct |= 4;
  return oct;
}

__device__ inline uint32_t expand_bits_10(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ inline uint32_t morton3d(Real nx, Real ny, Real nz) {
  nx = nx < Real(0.0) ? Real(0.0) : (nx > Real(1.0) ? Real(1.0) : nx);
  ny = ny < Real(0.0) ? Real(0.0) : (ny > Real(1.0) ? Real(1.0) : ny);
  nz = nz < Real(0.0) ? Real(0.0) : (nz > Real(1.0) ? Real(1.0) : nz);
  const uint32_t xx = static_cast<uint32_t>(nx * Real(1023.0));
  const uint32_t yy = static_cast<uint32_t>(ny * Real(1023.0));
  const uint32_t zz = static_cast<uint32_t>(nz * Real(1023.0));
  return (expand_bits_10(xx) << 2) | (expand_bits_10(yy) << 1) | expand_bits_10(zz);
}

// ---------------------------------------------------------------------------
// Kernel 1: Velocity Verlet (kick + drift e kick final)
// ---------------------------------------------------------------------------
__global__ void k_kick_drift(int n, Real half_dt, Real dt, Real* __restrict__ x,
                             Real* __restrict__ y, Real* __restrict__ z, Real* __restrict__ vx,
                             Real* __restrict__ vy, Real* __restrict__ vz,
                             const Real* __restrict__ ax, const Real* __restrict__ ay,
                             const Real* __restrict__ az) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Real nvx = vx[i] + half_dt * ax[i];
  const Real nvy = vy[i] + half_dt * ay[i];
  const Real nvz = vz[i] + half_dt * az[i];

  vx[i] = nvx;
  vy[i] = nvy;
  vz[i] = nvz;

  x[i] += dt * nvx;
  y[i] += dt * nvy;
  z[i] += dt * nvz;
}

__global__ void k_kick(int n, Real half_dt, Real* __restrict__ vx, Real* __restrict__ vy,
                       Real* __restrict__ vz, const Real* __restrict__ ax,
                       const Real* __restrict__ ay, const Real* __restrict__ az) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  vx[i] += half_dt * ax[i];
  vy[i] += half_dt * ay[i];
  vz[i] += half_dt * az[i];
}

// ---------------------------------------------------------------------------
// Kernel 2: Caixa envolvente (bounding box) em memoria compartilhada
// ---------------------------------------------------------------------------
__global__ void k_init_bbox(Real big, Real* __restrict__ bbox) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    bbox[0] = big;
    bbox[1] = big;
    bbox[2] = big;
    bbox[3] = -big;
    bbox[4] = -big;
    bbox[5] = -big;
  }
}

__global__ void k_bounding_box(int n, Real big, const Real* __restrict__ x,
                               const Real* __restrict__ y, const Real* __restrict__ z,
                               Real* __restrict__ bbox) {
  __shared__ Real s_min_x[kBlockSize];
  __shared__ Real s_min_y[kBlockSize];
  __shared__ Real s_min_z[kBlockSize];
  __shared__ Real s_max_x[kBlockSize];
  __shared__ Real s_max_y[kBlockSize];
  __shared__ Real s_max_z[kBlockSize];

  Real min_x = big, min_y = big, min_z = big;
  Real max_x = -big, max_y = -big, max_z = -big;

  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const Real xi = x[i];
    const Real yi = y[i];
    const Real zi = z[i];
    if (xi < min_x) min_x = xi;
    if (yi < min_y) min_y = yi;
    if (zi < min_z) min_z = zi;
    if (xi > max_x) max_x = xi;
    if (yi > max_y) max_y = yi;
    if (zi > max_z) max_z = zi;
  }

  const int tid = threadIdx.x;
  s_min_x[tid] = min_x;
  s_min_y[tid] = min_y;
  s_min_z[tid] = min_z;
  s_max_x[tid] = max_x;
  s_max_y[tid] = max_y;
  s_max_z[tid] = max_z;
  __syncthreads();

  for (int offset = kBlockSize / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_min_x[tid + offset] < s_min_x[tid]) s_min_x[tid] = s_min_x[tid + offset];
      if (s_min_y[tid + offset] < s_min_y[tid]) s_min_y[tid] = s_min_y[tid + offset];
      if (s_min_z[tid + offset] < s_min_z[tid]) s_min_z[tid] = s_min_z[tid + offset];
      if (s_max_x[tid + offset] > s_max_x[tid]) s_max_x[tid] = s_max_x[tid + offset];
      if (s_max_y[tid + offset] > s_max_y[tid]) s_max_y[tid] = s_max_y[tid + offset];
      if (s_max_z[tid + offset] > s_max_z[tid]) s_max_z[tid] = s_max_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomic_min_real(&bbox[0], s_min_x[0]);
    atomic_min_real(&bbox[1], s_min_y[0]);
    atomic_min_real(&bbox[2], s_min_z[0]);
    atomic_max_real(&bbox[3], s_max_x[0]);
    atomic_max_real(&bbox[4], s_max_y[0]);
    atomic_max_real(&bbox[5], s_max_z[0]);
  }
}

__global__ void k_init_root(const Real* __restrict__ bbox, Real* __restrict__ cell_cx,
                            Real* __restrict__ cell_cy, Real* __restrict__ cell_cz,
                            Real* __restrict__ cell_hs, int* __restrict__ cell_parent,
                            int* __restrict__ children, int* __restrict__ next_cell,
                            int* __restrict__ overflow_flag) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const Real min_x = bbox[0], min_y = bbox[1], min_z = bbox[2];
    const Real max_x = bbox[3], max_y = bbox[4], max_z = bbox[5];

    const Real sx = max_x - min_x;
    const Real sy = max_y - min_y;
    const Real sz = max_z - min_z;
    const Real span = (sx > sy) ? ((sx > sz) ? sx : sz) : ((sy > sz) ? sy : sz);
    const Real root_hs =
        (span > Real(0.0)) ? (Real(0.5) * span * Real(1.00001) + Real(1e-6)) : Real(1.0);

    cell_cx[0] = Real(0.5) * (min_x + max_x);
    cell_cy[0] = Real(0.5) * (min_y + max_y);
    cell_cz[0] = Real(0.5) * (min_z + max_z);
    cell_hs[0] = root_hs;
    cell_parent[0] = -1;

    for (int k = 0; k < 8; ++k) {
      children[k] = kEmpty;
    }

    *next_cell = 1;
    *overflow_flag = 0;
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: Codigos de Morton 3D para ordenacao espacial (coerencia de warp)
// ---------------------------------------------------------------------------
__global__ void k_morton_codes(int n, const Real* __restrict__ x, const Real* __restrict__ y,
                               const Real* __restrict__ z, const Real* __restrict__ cell_cx,
                               const Real* __restrict__ cell_cy, const Real* __restrict__ cell_cz,
                               const Real* __restrict__ cell_hs, uint32_t* __restrict__ morton_out,
                               int* __restrict__ idx_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const Real hs = cell_hs[0];
  const Real inv_span = Real(1.0) / (Real(2.0) * hs);
  const Real nx = (x[i] - (cell_cx[0] - hs)) * inv_span;
  const Real ny = (y[i] - (cell_cy[0] - hs)) * inv_span;
  const Real nz = (z[i] - (cell_cz[0] - hs)) * inv_span;

  morton_out[i] = morton3d(nx, ny, nz);
  idx_out[i] = i;
}

// ---------------------------------------------------------------------------
// Kernel 4: Construcao paralela da Octree linearizada via atomicCAS
// ---------------------------------------------------------------------------
__global__ void k_build_tree(int n, int max_cells, const Real* __restrict__ x,
                             const Real* __restrict__ y, const Real* __restrict__ z,
                             Real* __restrict__ cell_cx, Real* __restrict__ cell_cy,
                             Real* __restrict__ cell_cz, Real* __restrict__ cell_hs,
                             int* __restrict__ cell_parent, int* __restrict__ children,
                             int* __restrict__ next_body, int* __restrict__ next_cell,
                             int* __restrict__ overflow_flag) {
  const int stride = blockDim.x * gridDim.x;
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n; b += stride) {
    const Real bx = x[b];
    const Real by = y[b];
    const Real bz = z[b];

    int cell = 0;
    Real cx = cell_cx[0];
    Real cy = cell_cy[0];
    Real cz = cell_cz[0];
    Real hs = cell_hs[0];
    int depth = 0;

    for (;;) {
      const int oct = octant_of(bx, by, bz, cx, cy, cz);
      const int slot = cell * 8 + oct;

      const int child = atomicAdd(&children[slot], 0);
      if (child == kLocked) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(32);
#endif
        continue;
      }

      if (child >= n) {
        cell = child - n;
        const Real chs = hs * Real(0.5);
        cx += (oct & 1) ? chs : -chs;
        cy += (oct & 2) ? chs : -chs;
        cz += (oct & 4) ? chs : -chs;
        hs = chs;
        ++depth;
        continue;
      }

      if (child == kEmpty) {
        if (atomicCAS(&children[slot], kEmpty, b) == kEmpty) {
          break;
        }
        continue;
      }

      // Folha ocupada por corpo(s): adquire lock leve no slot.
      if (atomicCAS(&children[slot], child, kLocked) != child) {
        continue;
      }

      const Real chs = hs * Real(0.5);
      if (depth >= kMaxDepth || !(chs > Real(0.0))) {
        next_body[b] = child;
        __threadfence();
        atomicExch(&children[slot], b);
        break;
      }

      const int existing = child;
      const Real ex = x[existing];
      const Real ey = y[existing];
      const Real ez = z[existing];

      int cur_parent = cell;
      Real cur_cx = cx + ((oct & 1) ? chs : -chs);
      Real cur_cy = cy + ((oct & 2) ? chs : -chs);
      Real cur_cz = cz + ((oct & 4) ? chs : -chs);
      Real cur_hs = chs;
      int cur_depth = depth + 1;

      int first_cell = -1;
      int prev_cell = -1;
      int prev_oct = -1;
      bool overflowed = false;

      for (;;) {
        const int new_cell = atomicAdd(next_cell, 1);
        if (new_cell >= max_cells) {
          atomicExch(overflow_flag, 1);
          overflowed = true;
          break;
        }

        for (int k = 0; k < 8; ++k) {
          children[new_cell * 8 + k] = kEmpty;
        }
        cell_cx[new_cell] = cur_cx;
        cell_cy[new_cell] = cur_cy;
        cell_cz[new_cell] = cur_cz;
        cell_hs[new_cell] = cur_hs;
        cell_parent[new_cell] = cur_parent;

        if (first_cell == -1) {
          first_cell = new_cell;
        } else {
          children[prev_cell * 8 + prev_oct] = n + new_cell;
        }

        const int oct_e = octant_of(ex, ey, ez, cur_cx, cur_cy, cur_cz);
        const int oct_b = octant_of(bx, by, bz, cur_cx, cur_cy, cur_cz);

        if (oct_e != oct_b) {
          children[new_cell * 8 + oct_e] = existing;
          children[new_cell * 8 + oct_b] = b;
          break;
        }

        const Real next_hs = cur_hs * Real(0.5);
        if (cur_depth >= kMaxDepth || !(next_hs > Real(0.0))) {
          next_body[b] = existing;
          children[new_cell * 8 + oct_b] = b;
          break;
        }

        cur_parent = new_cell;
        prev_cell = new_cell;
        prev_oct = oct_b;
        cur_cx += (oct_b & 1) ? next_hs : -next_hs;
        cur_cy += (oct_b & 2) ? next_hs : -next_hs;
        cur_cz += (oct_b & 4) ? next_hs : -next_hs;
        cur_hs = next_hs;
        ++cur_depth;
      }

      __threadfence();
      if (overflowed) {
        atomicExch(&children[slot], existing);
        return;
      }
      atomicExch(&children[slot], n + first_cell);
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 5: Massa total e centro de massa bottom-up (sem spin-wait)
// ---------------------------------------------------------------------------
__global__ void k_count_children(int n, int max_cells, const int* __restrict__ next_cell,
                                 const Real* __restrict__ x, const Real* __restrict__ y,
                                 const Real* __restrict__ z, const Real* __restrict__ mass,
                                 const int* __restrict__ children,
                                 const int* __restrict__ next_body, Real* __restrict__ cell_mass,
                                 Real* __restrict__ cell_cmx, Real* __restrict__ cell_cmy,
                                 Real* __restrict__ cell_cmz, int* __restrict__ cell_pending) {
  const int raw_cells = *next_cell;
  const int num_cells = (raw_cells < max_cells) ? raw_cells : max_cells;
  const int stride = blockDim.x * gridDim.x;

  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < num_cells; c += stride) {
    int subcells = 0;
    Real m = Real(0.0);
    Real mx = Real(0.0);
    Real my = Real(0.0);
    Real mz = Real(0.0);

    const int base = c * 8;
    for (int oct = 0; oct < 8; ++oct) {
      const int ch = children[base + oct];
      if (ch == kEmpty) continue;
      if (ch >= n) {
        ++subcells;
      } else {
        for (int b = ch; b != -1; b = next_body[b]) {
          const Real bm = mass[b];
          m += bm;
          mx += bm * x[b];
          my += bm * y[b];
          mz += bm * z[b];
        }
      }
    }

    cell_mass[c] = m;
    cell_cmx[c] = mx;
    cell_cmy[c] = my;
    cell_cmz[c] = mz;
    // -1 identifica celulas sem filhos internos (pontos de partida do bottom-up).
    cell_pending[c] = (subcells == 0) ? -1 : subcells;
  }
}

__global__ void k_summarize_tree(int max_cells, const int* __restrict__ next_cell,
                                 const Real* __restrict__ cell_cx,
                                 const Real* __restrict__ cell_cy,
                                 const Real* __restrict__ cell_cz,
                                 const int* __restrict__ cell_parent,
                                 Real* __restrict__ cell_mass, Real* __restrict__ cell_cmx,
                                 Real* __restrict__ cell_cmy, Real* __restrict__ cell_cmz,
                                 int* __restrict__ cell_pending) {
  const int raw_cells = *next_cell;
  const int num_cells = (raw_cells < max_cells) ? raw_cells : max_cells;
  const int stride = blockDim.x * gridDim.x;

  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < num_cells; c += stride) {
    if (cell_pending[c] != -1) {
      continue;
    }

    int cur = c;
    for (;;) {
      const Real m = __ldcg(&cell_mass[cur]);
      const Real mx = __ldcg(&cell_cmx[cur]);
      const Real my = __ldcg(&cell_cmy[cur]);
      const Real mz = __ldcg(&cell_cmz[cur]);

      if (m > Real(0.0)) {
        const Real inv_m = Real(1.0) / m;
        cell_cmx[cur] = mx * inv_m;
        cell_cmy[cur] = my * inv_m;
        cell_cmz[cur] = mz * inv_m;
      } else {
        cell_cmx[cur] = cell_cx[cur];
        cell_cmy[cur] = cell_cy[cur];
        cell_cmz[cur] = cell_cz[cur];
      }

      const int p = cell_parent[cur];
      if (p < 0) {
        break;
      }

      atomicAdd(&cell_mass[p], m);
      atomicAdd(&cell_cmx[p], mx);
      atomicAdd(&cell_cmy[p], my);
      atomicAdd(&cell_cmz[p], mz);
      __threadfence();

      // Apenas o ultimo filho a concluir o pai continua subindo a arvore.
      if (atomicSub(&cell_pending[p], 1) == 1) {
        cur = p;
      } else {
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 6: Travessia iterativa da Octree (1 thread por corpo ordenado)
// ---------------------------------------------------------------------------
__global__ void k_compute_forces(int n, Real gravity, Real eps2, Real theta2,
                                 const int* __restrict__ sorted_idx, const Real* __restrict__ x,
                                 const Real* __restrict__ y, const Real* __restrict__ z,
                                 const Real* __restrict__ mass, const Real* __restrict__ cell_cx,
                                 const Real* __restrict__ cell_cy,
                                 const Real* __restrict__ cell_cz,
                                 const Real* __restrict__ cell_hs,
                                 const Real* __restrict__ cell_mass,
                                 const Real* __restrict__ cell_cmx,
                                 const Real* __restrict__ cell_cmy,
                                 const Real* __restrict__ cell_cmz,
                                 const int* __restrict__ children,
                                 const int* __restrict__ next_body, Real* __restrict__ ax,
                                 Real* __restrict__ ay, Real* __restrict__ az) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  const int i = sorted_idx[tid];
  const Real xi = x[i];
  const Real yi = y[i];
  const Real zi = z[i];

  Real axi = Real(0.0);
  Real ayi = Real(0.0);
  Real azi = Real(0.0);

  // Pilha contem somente celulas internas a abrir (folhas sao avaliadas inline).
  int stack[256];
  int top = 0;
  stack[top++] = 0;

  while (top > 0) {
    const int c = stack[--top];
    const int base = c * 8;

    for (int oct = 0; oct < 8; ++oct) {
      const int ch = children[base + oct];
      if (ch == kEmpty) continue;

      if (ch < n) {
        for (int b = ch; b != -1; b = next_body[b]) {
          if (b == i) continue;
          const Real dx = x[b] - xi;
          const Real dy = y[b] - yi;
          const Real dz = z[b] - zi;
          const Real r2 = dx * dx + dy * dy + dz * dz + eps2;
          const Real inv_r = rsqrt_real(r2);
          const Real factor = gravity * mass[b] * inv_r / r2;
          axi += factor * dx;
          ayi += factor * dy;
          azi += factor * dz;
        }
      } else {
        const int cc = ch - n;
        const Real dx = cell_cmx[cc] - xi;
        const Real dy = cell_cmy[cc] - yi;
        const Real dz = cell_cmz[cc] - zi;
        const Real d2 = dx * dx + dy * dy + dz * dz;

        const Real hs = cell_hs[cc];
        const Real s = Real(2.0) * hs;
        if (s * s < theta2 * d2 &&
            !(fabs_real(xi - cell_cx[cc]) <= hs && fabs_real(yi - cell_cy[cc]) <= hs &&
              fabs_real(zi - cell_cz[cc]) <= hs)) {
          const Real r2 = d2 + eps2;
          const Real inv_r = rsqrt_real(r2);
          const Real factor = gravity * cell_mass[cc] * inv_r / r2;
          axi += factor * dx;
          ayi += factor * dy;
          azi += factor * dz;
        } else if (top < 256) {
          stack[top++] = cc;
        }
      }
    }
  }

  ax[i] = axi;
  ay[i] = ayi;
  az[i] = azi;
}

// ---------------------------------------------------------------------------
// Backend persistente na VRAM
// ---------------------------------------------------------------------------
class CudaBarnesHutBackend final : public CudaBackend {
 public:
  CudaBarnesHutBackend(const BodySystem& initial, const SimulationParams& params)
      : n_(static_cast<int>(initial.size())), params_(params) {
    if (params_.theta < Real(0.0)) {
      throw std::invalid_argument("theta nao pode ser negativo");
    }
    if (initial.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 8)) {
      throw std::overflow_error("numero de corpos excede o limite de indexacao CUDA");
    }

    int device = 0;
    NBODY_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    NBODY_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    sm_count_ = prop.multiProcessorCount;

    max_cells_ = std::max(1024, 3 * n_ + 1024);

    const std::size_t body_bytes = static_cast<std::size_t>(n_) * sizeof(Real);
    const std::size_t cell_real_bytes = static_cast<std::size_t>(max_cells_) * sizeof(Real);
    const std::size_t cell_int_bytes = static_cast<std::size_t>(max_cells_) * sizeof(int);

    NBODY_CUDA_CHECK(cudaMalloc(&d_x_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_y_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_z_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_vx_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_vy_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_vz_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_ax_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_ay_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_az_, body_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_mass_, body_bytes));

    NBODY_CUDA_CHECK(cudaMalloc(&d_bbox_, 6 * sizeof(Real)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_cx_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_cy_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_cz_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_hs_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_mass_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_cmx_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_cmy_, cell_real_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_cmz_, cell_real_bytes));

    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_parent_, cell_int_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_cell_pending_, cell_int_bytes));
    NBODY_CUDA_CHECK(cudaMalloc(&d_children_, static_cast<std::size_t>(max_cells_) * 8 * sizeof(int)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_next_body_, static_cast<std::size_t>(n_) * sizeof(int)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_next_cell_, sizeof(int)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_overflow_, sizeof(int)));

    NBODY_CUDA_CHECK(cudaMalloc(&d_morton_in_, static_cast<std::size_t>(n_) * sizeof(uint32_t)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_morton_out_, static_cast<std::size_t>(n_) * sizeof(uint32_t)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_idx_in_, static_cast<std::size_t>(n_) * sizeof(int)));
    NBODY_CUDA_CHECK(cudaMalloc(&d_idx_out_, static_cast<std::size_t>(n_) * sizeof(int)));

    NBODY_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes_, d_morton_in_, d_morton_out_, d_idx_in_, d_idx_out_, n_));
    NBODY_CUDA_CHECK(cudaMalloc(&d_sort_temp_, sort_temp_bytes_));

    if (params_.profile_kernels) {
      for (auto& ev : events_) {
        NBODY_CUDA_CHECK(cudaEventCreate(&ev));
      }
    }

    // Upload unico dos corpos para a VRAM.
    NBODY_CUDA_CHECK(cudaMemcpy(d_x_, initial.x.data(), body_bytes, cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(d_y_, initial.y.data(), body_bytes, cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(d_z_, initial.z.data(), body_bytes, cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(d_vx_, initial.vx.data(), body_bytes, cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(d_vy_, initial.vy.data(), body_bytes, cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(d_vz_, initial.vz.data(), body_bytes, cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(d_mass_, initial.mass.data(), body_bytes, cudaMemcpyHostToDevice));

    // Aceleracao em t = 0 na GPU (fora das estatisticas de passo).
    compute_forces_gpu(false);
    synchronize();
  }

  ~CudaBarnesHutBackend() override {
    if (params_.profile_kernels) {
      for (auto& ev : events_) {
        if (ev) cudaEventDestroy(ev);
      }
    }
    cudaFree(d_sort_temp_);
    cudaFree(d_idx_out_);
    cudaFree(d_idx_in_);
    cudaFree(d_morton_out_);
    cudaFree(d_morton_in_);
    cudaFree(d_overflow_);
    cudaFree(d_next_cell_);
    cudaFree(d_next_body_);
    cudaFree(d_children_);
    cudaFree(d_cell_pending_);
    cudaFree(d_cell_parent_);
    cudaFree(d_cell_cmz_);
    cudaFree(d_cell_cmy_);
    cudaFree(d_cell_cmx_);
    cudaFree(d_cell_mass_);
    cudaFree(d_cell_hs_);
    cudaFree(d_cell_cz_);
    cudaFree(d_cell_cy_);
    cudaFree(d_cell_cx_);
    cudaFree(d_bbox_);
    cudaFree(d_mass_);
    cudaFree(d_az_);
    cudaFree(d_ay_);
    cudaFree(d_ax_);
    cudaFree(d_vz_);
    cudaFree(d_vy_);
    cudaFree(d_vx_);
    cudaFree(d_z_);
    cudaFree(d_y_);
    cudaFree(d_x_);
  }

  CudaBarnesHutBackend(const CudaBarnesHutBackend&) = delete;
  CudaBarnesHutBackend& operator=(const CudaBarnesHutBackend&) = delete;

  void step() override {
    const int body_blocks = (n_ + kBlockSize - 1) / kBlockSize;
    const Real half_dt = Real(0.5) * params_.dt;
    const bool prof = params_.profile_kernels;

    if (prof) NBODY_CUDA_CHECK(cudaEventRecord(events_[0]));
    k_kick_drift<<<body_blocks, kBlockSize>>>(n_, half_dt, params_.dt, d_x_, d_y_, d_z_, d_vx_,
                                              d_vy_, d_vz_, d_ax_, d_ay_, d_az_);
    if (prof) NBODY_CUDA_CHECK(cudaEventRecord(events_[1]));

    compute_forces_gpu(prof);

    if (prof) NBODY_CUDA_CHECK(cudaEventRecord(events_[6]));
    k_kick<<<body_blocks, kBlockSize>>>(n_, half_dt, d_vx_, d_vy_, d_vz_, d_ax_, d_ay_, d_az_);
    if (prof) {
      NBODY_CUDA_CHECK(cudaEventRecord(events_[7]));
      NBODY_CUDA_CHECK(cudaEventSynchronize(events_[7]));
      float dt_kd = 0.0f, dt_bbox = 0.0f, dt_sort = 0.0f, dt_build = 0.0f, dt_sum = 0.0f,
            dt_force = 0.0f, dt_k = 0.0f;
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_kd, events_[0], events_[1]));
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_bbox, events_[1], events_[2]));
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_sort, events_[2], events_[3]));
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_build, events_[3], events_[4]));
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_sum, events_[4], events_[5]));
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_force, events_[5], events_[6]));
      NBODY_CUDA_CHECK(cudaEventElapsedTime(&dt_k, events_[6], events_[7]));

      timings_.integrate_ms += static_cast<double>(dt_kd + dt_k);
      timings_.bbox_ms += static_cast<double>(dt_bbox);
      timings_.sort_ms += static_cast<double>(dt_sort);
      timings_.build_tree_ms += static_cast<double>(dt_build);
      timings_.summarize_ms += static_cast<double>(dt_sum);
      timings_.force_ms += static_cast<double>(dt_force);
    }
  }

  void synchronize() const override {
    NBODY_CUDA_CHECK(cudaDeviceSynchronize());
    int overflow = 0;
    NBODY_CUDA_CHECK(cudaMemcpy(&overflow, d_overflow_, sizeof(int), cudaMemcpyDeviceToHost));
    if (overflow != 0) {
      throw std::runtime_error(
          "overflow na capacidade da Octree em VRAM (aumente max_cells em barnes_hut_cuda.cu)");
    }
  }

  void download(BodySystem& out) const override {
    synchronize();
    const std::size_t body_bytes = static_cast<std::size_t>(n_) * sizeof(Real);
    out.resize(static_cast<std::size_t>(n_));
    NBODY_CUDA_CHECK(cudaMemcpy(out.x.data(), d_x_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.y.data(), d_y_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.z.data(), d_z_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.vx.data(), d_vx_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.vy.data(), d_vy_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.vz.data(), d_vz_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.ax.data(), d_ax_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.ay.data(), d_ay_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.az.data(), d_az_, body_bytes, cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(out.mass.data(), d_mass_, body_bytes, cudaMemcpyDeviceToHost));
  }

  CudaKernelTimings timings() const override { return timings_; }

 private:
  void compute_forces_gpu(bool record_events) {
    const int body_blocks = (n_ + kBlockSize - 1) / kBlockSize;
    const int persistent_blocks = std::min(body_blocks, sm_count_ * 8);
    const int tree_blocks = std::max(1, sm_count_ * 8);
    const Real big = std::numeric_limits<Real>::max();

    // 1. Bounding box + raiz
    k_init_bbox<<<1, 1>>>(big, d_bbox_);
    k_bounding_box<<<persistent_blocks, kBlockSize>>>(n_, big, d_x_, d_y_, d_z_, d_bbox_);
    k_init_root<<<1, 1>>>(d_bbox_, d_cell_cx_, d_cell_cy_, d_cell_cz_, d_cell_hs_, d_cell_parent_,
                          d_children_, d_next_cell_, d_overflow_);
    if (record_events) NBODY_CUDA_CHECK(cudaEventRecord(events_[2]));

    // 2. Ordenacao espacial (Morton + CUB RadixSort sem alocacao por passo)
    k_morton_codes<<<body_blocks, kBlockSize>>>(n_, d_x_, d_y_, d_z_, d_cell_cx_, d_cell_cy_,
                                                d_cell_cz_, d_cell_hs_, d_morton_in_, d_idx_in_);
    NBODY_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_sort_temp_, sort_temp_bytes_, d_morton_in_, d_morton_out_, d_idx_in_, d_idx_out_, n_));
    if (record_events) NBODY_CUDA_CHECK(cudaEventRecord(events_[3]));

    // 3. Construcao paralela da Octree
    NBODY_CUDA_CHECK(cudaMemsetAsync(d_next_body_, 0xFF, static_cast<std::size_t>(n_) * sizeof(int)));
    k_build_tree<<<persistent_blocks, kBlockSize>>>(n_, max_cells_, d_x_, d_y_, d_z_, d_cell_cx_,
                                                    d_cell_cy_, d_cell_cz_, d_cell_hs_,
                                                    d_cell_parent_, d_children_, d_next_body_,
                                                    d_next_cell_, d_overflow_);
    if (record_events) NBODY_CUDA_CHECK(cudaEventRecord(events_[4]));

    // 4. Massa total e centro de massa bottom-up
    k_count_children<<<tree_blocks, kBlockSize>>>(n_, max_cells_, d_next_cell_, d_x_, d_y_, d_z_,
                                                  d_mass_, d_children_, d_next_body_, d_cell_mass_,
                                                  d_cell_cmx_, d_cell_cmy_, d_cell_cmz_,
                                                  d_cell_pending_);
    k_summarize_tree<<<tree_blocks, kBlockSize>>>(max_cells_, d_next_cell_, d_cell_cx_, d_cell_cy_,
                                                  d_cell_cz_, d_cell_parent_, d_cell_mass_,
                                                  d_cell_cmx_, d_cell_cmy_, d_cell_cmz_,
                                                  d_cell_pending_);
    if (record_events) NBODY_CUDA_CHECK(cudaEventRecord(events_[5]));

    // 5. Travessia da Octree e calculo de aceleracoes
    const Real eps2 = params_.softening * params_.softening;
    const Real theta2 = params_.theta * params_.theta;
    k_compute_forces<<<body_blocks, kBlockSize>>>(
        n_, params_.gravity, eps2, theta2, d_idx_out_, d_x_, d_y_, d_z_, d_mass_, d_cell_cx_,
        d_cell_cy_, d_cell_cz_, d_cell_hs_, d_cell_mass_, d_cell_cmx_, d_cell_cmy_, d_cell_cmz_,
        d_children_, d_next_body_, d_ax_, d_ay_, d_az_);
  }

  int n_ = 0;
  int max_cells_ = 0;
  int sm_count_ = 1;
  SimulationParams params_;
  CudaKernelTimings timings_;
  cudaEvent_t events_[8]{};

  Real* d_x_ = nullptr;
  Real* d_y_ = nullptr;
  Real* d_z_ = nullptr;
  Real* d_vx_ = nullptr;
  Real* d_vy_ = nullptr;
  Real* d_vz_ = nullptr;
  Real* d_ax_ = nullptr;
  Real* d_ay_ = nullptr;
  Real* d_az_ = nullptr;
  Real* d_mass_ = nullptr;

  Real* d_bbox_ = nullptr;
  Real* d_cell_cx_ = nullptr;
  Real* d_cell_cy_ = nullptr;
  Real* d_cell_cz_ = nullptr;
  Real* d_cell_hs_ = nullptr;
  Real* d_cell_mass_ = nullptr;
  Real* d_cell_cmx_ = nullptr;
  Real* d_cell_cmy_ = nullptr;
  Real* d_cell_cmz_ = nullptr;

  int* d_cell_parent_ = nullptr;
  int* d_cell_pending_ = nullptr;
  int* d_children_ = nullptr;
  int* d_next_body_ = nullptr;
  int* d_next_cell_ = nullptr;
  int* d_overflow_ = nullptr;

  uint32_t* d_morton_in_ = nullptr;
  uint32_t* d_morton_out_ = nullptr;
  int* d_idx_in_ = nullptr;
  int* d_idx_out_ = nullptr;
  void* d_sort_temp_ = nullptr;
  std::size_t sort_temp_bytes_ = 0;
};

}  // namespace

bool cuda_backend_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<CudaBackend> make_cuda_barnes_hut_backend(const BodySystem& initial,
                                                          const SimulationParams& params) {
  return std::make_unique<CudaBarnesHutBackend>(initial, params);
}

void compute_accelerations_barnes_hut_cuda(BodySystem& system, const SimulationParams& params) {
  if (params.theta < Real(0.0)) {
    throw std::invalid_argument("theta nao pode ser negativo");
  }
  if (system.size() < 2) {
    std::fill(system.ax.begin(), system.ax.end(), Real(0.0));
    std::fill(system.ay.begin(), system.ay.end(), Real(0.0));
    std::fill(system.az.begin(), system.az.end(), Real(0.0));
    return;
  }

  SimulationParams eval_params = params;
  eval_params.profile_kernels = false;
  CudaBarnesHutBackend backend(system, eval_params);
  backend.download(system);
}

}  // namespace nbody