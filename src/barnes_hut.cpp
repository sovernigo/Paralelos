#include "nbody/accelerations.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nbody {
namespace {

constexpr int kEmptyNode = -1;
constexpr int kInternalNode = -2;
constexpr int kMaxDepth = 32;

// Buffers lineares da Octree reaproveitados entre passos de integracao.
struct OctreeWorkspace {
  // 8 filhos por no: children[node * 8 + octant], ou -1 se vazio.
  std::vector<int> children;
  // -1 = vazio, -2 = no interno, >= 0 = folha (indice do corpo cabeca).
  std::vector<int> body;
  // Lista encadeada por indice para corpos coincidentes que atingem kMaxDepth.
  std::vector<int> next_body;

  // Geometria da celula (construcao e criterio s/d < theta).
  std::vector<Real> cx;
  std::vector<Real> cy;
  std::vector<Real> cz;
  std::vector<Real> half_size;

  // Massa total e centro de massa (preenchidos em pos-ordem).
  std::vector<Real> mass;
  std::vector<Real> cmx;
  std::vector<Real> cmy;
  std::vector<Real> cmz;

  // Pilha reutilizavel para travessia iterativa (DFS).
  std::vector<int> stack;

  void reset(std::size_t n) {
    children.clear();
    body.clear();
    cx.clear();
    cy.clear();
    cz.clear();
    half_size.clear();
    mass.clear();
    cmx.clear();
    cmy.clear();
    cmz.clear();

    const std::size_t est_nodes = std::max<std::size_t>(16, n * 2);
    if (body.capacity() < est_nodes) {
      children.reserve(est_nodes * 8);
      body.reserve(est_nodes);
      cx.reserve(est_nodes);
      cy.reserve(est_nodes);
      cz.reserve(est_nodes);
      half_size.reserve(est_nodes);
      mass.reserve(est_nodes);
      cmx.reserve(est_nodes);
      cmy.reserve(est_nodes);
      cmz.reserve(est_nodes);
    }

    next_body.assign(n, -1);
    if (stack.capacity() < static_cast<std::size_t>(kMaxDepth) * 8 + 16) {
      stack.reserve(static_cast<std::size_t>(kMaxDepth) * 8 + 16);
    }
  }

  int allocate_node(Real ncx, Real ncy, Real ncz, Real nhs) {
    const int idx = static_cast<int>(body.size());
    children.insert(children.end(), 8, kEmptyNode);
    body.push_back(kEmptyNode);
    cx.push_back(ncx);
    cy.push_back(ncy);
    cz.push_back(ncz);
    half_size.push_back(nhs);
    mass.push_back(Real(0.0));
    cmx.push_back(Real(0.0));
    cmy.push_back(Real(0.0));
    cmz.push_back(Real(0.0));
    return idx;
  }
};

inline int octant_of(Real x, Real y, Real z, Real cx, Real cy, Real cz) {
  int oct = 0;
  if (x >= cx) {
    oct |= 1;
  }
  if (y >= cy) {
    oct |= 2;
  }
  if (z >= cz) {
    oct |= 4;
  }
  return oct;
}

}  // namespace

void compute_accelerations_barnes_hut(BodySystem& system, const SimulationParams& params) {
  if (params.theta < Real(0.0)) {
    throw std::invalid_argument("theta nao pode ser negativo");
  }

  const std::size_t n = system.size();
  std::fill(system.ax.begin(), system.ax.end(), Real(0.0));
  std::fill(system.ay.begin(), system.ay.end(), Real(0.0));
  std::fill(system.az.begin(), system.az.end(), Real(0.0));

  if (n < 2) {
    return;
  }

  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max() / 4)) {
    throw std::overflow_error("numero de corpos excede o limite de indexacao da Octree");
  }

  // 1. Caixa envolvente cubica (bounding box).
  Real min_x = system.x[0], max_x = system.x[0];
  Real min_y = system.y[0], max_y = system.y[0];
  Real min_z = system.z[0], max_z = system.z[0];

  for (std::size_t i = 1; i < n; ++i) {
    min_x = std::min(min_x, system.x[i]);
    max_x = std::max(max_x, system.x[i]);
    min_y = std::min(min_y, system.y[i]);
    max_y = std::max(max_y, system.y[i]);
    min_z = std::min(min_z, system.z[i]);
    max_z = std::max(max_z, system.z[i]);
  }

  const Real root_cx = Real(0.5) * (min_x + max_x);
  const Real root_cy = Real(0.5) * (min_y + max_y);
  const Real root_cz = Real(0.5) * (min_z + max_z);
  const Real span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
  const Real root_hs =
      (span > Real(0.0)) ? (Real(0.5) * span * Real(1.000001) + Real(1e-9)) : Real(1.0);

  thread_local OctreeWorkspace ws;
  ws.reset(n);
  ws.allocate_node(root_cx, root_cy, root_cz, root_hs);

  // 2. Construcao iterativa da Octree linearizada.
  const int n_int = static_cast<int>(n);
  for (int b = 0; b < n_int; ++b) {
    const std::size_t bu = static_cast<std::size_t>(b);
    const Real bx = system.x[bu];
    const Real by = system.y[bu];
    const Real bz = system.z[bu];

    int node = 0;
    int depth = 0;

    for (;;) {
      const std::size_t nu = static_cast<std::size_t>(node);
      const int state = ws.body[nu];

      if (state == kEmptyNode) {
        ws.body[nu] = b;
        break;
      }

      if (state >= 0) {
        const Real chs = ws.half_size[nu] * Real(0.5);
        if (depth >= kMaxDepth || !(chs > Real(0.0))) {
          // Corpos coincidentes ou abaixo da resolucao limite: encadeia na folha.
          ws.next_body[bu] = state;
          ws.body[nu] = b;
          break;
        }

        // Converte folha em no interno e move o corpo existente para seu octante.
        const int existing = state;
        const std::size_t eu = static_cast<std::size_t>(existing);
        ws.body[nu] = kInternalNode;

        const Real ncx = ws.cx[nu];
        const Real ncy = ws.cy[nu];
        const Real ncz = ws.cz[nu];
        const int old_oct = octant_of(system.x[eu], system.y[eu], system.z[eu], ncx, ncy, ncz);

        const int old_child = ws.allocate_node(
            ncx + ((old_oct & 1) ? chs : -chs),
            ncy + ((old_oct & 2) ? chs : -chs),
            ncz + ((old_oct & 4) ? chs : -chs),
            chs);
        ws.children[nu * 8 + static_cast<std::size_t>(old_oct)] = old_child;
        ws.body[static_cast<std::size_t>(old_child)] = existing;
      }

      // No interno: desce para o octante do novo corpo.
      const Real ncx = ws.cx[nu];
      const Real ncy = ws.cy[nu];
      const Real ncz = ws.cz[nu];
      const Real nhs = ws.half_size[nu];
      const int oct = octant_of(bx, by, bz, ncx, ncy, ncz);
      const std::size_t slot = nu * 8 + static_cast<std::size_t>(oct);

      int child = ws.children[slot];
      if (child == kEmptyNode) {
        const Real chs = nhs * Real(0.5);
        child = ws.allocate_node(
            ncx + ((oct & 1) ? chs : -chs),
            ncy + ((oct & 2) ? chs : -chs),
            ncz + ((oct & 4) ? chs : -chs),
            chs);
        ws.children[slot] = child;
        ws.body[static_cast<std::size_t>(child)] = b;
        break;
      }

      node = child;
      ++depth;
    }
  }

  // 3. Pos-ordem sem recursao: filhos sempre tem indice > pai.
  const int num_nodes = static_cast<int>(ws.body.size());
  for (int node = num_nodes - 1; node >= 0; --node) {
    const std::size_t nu = static_cast<std::size_t>(node);
    const int state = ws.body[nu];

    if (state >= 0) {
      const std::size_t su = static_cast<std::size_t>(state);
      if (ws.next_body[su] == -1) {
        ws.mass[nu] = system.mass[su];
        ws.cmx[nu] = system.x[su];
        ws.cmy[nu] = system.y[su];
        ws.cmz[nu] = system.z[su];
      } else {
        Real m = Real(0.0);
        Real mx = Real(0.0);
        Real my = Real(0.0);
        Real mz = Real(0.0);
        for (int b = state; b != -1; b = ws.next_body[static_cast<std::size_t>(b)]) {
          const std::size_t bu = static_cast<std::size_t>(b);
          const Real bm = system.mass[bu];
          m += bm;
          mx += bm * system.x[bu];
          my += bm * system.y[bu];
          mz += bm * system.z[bu];
        }
        ws.mass[nu] = m;
        if (m > Real(0.0)) {
          const Real inv_m = Real(1.0) / m;
          ws.cmx[nu] = mx * inv_m;
          ws.cmy[nu] = my * inv_m;
          ws.cmz[nu] = mz * inv_m;
        }
      }
    } else if (state == kInternalNode) {
      Real m = Real(0.0);
      Real mx = Real(0.0);
      Real my = Real(0.0);
      Real mz = Real(0.0);
      const std::size_t base = nu * 8;
      for (std::size_t oct = 0; oct < 8; ++oct) {
        const int child = ws.children[base + oct];
        if (child != kEmptyNode) {
          const std::size_t cu = static_cast<std::size_t>(child);
          const Real cm = ws.mass[cu];
          m += cm;
          mx += cm * ws.cmx[cu];
          my += cm * ws.cmy[cu];
          mz += cm * ws.cmz[cu];
        }
      }
      ws.mass[nu] = m;
      if (m > Real(0.0)) {
        const Real inv_m = Real(1.0) / m;
        ws.cmx[nu] = mx * inv_m;
        ws.cmy[nu] = my * inv_m;
        ws.cmz[nu] = mz * inv_m;
      }
    }
  }

  // 4. Travessia iterativa da arvore para cada corpo.
  const Real eps2 = params.softening * params.softening;
  const Real gravity = params.gravity;
  const Real theta2 = params.theta * params.theta;

  for (int i = 0; i < n_int; ++i) {
    const std::size_t iu = static_cast<std::size_t>(i);
    const Real xi = system.x[iu];
    const Real yi = system.y[iu];
    const Real zi = system.z[iu];

    Real axi = Real(0.0);
    Real ayi = Real(0.0);
    Real azi = Real(0.0);

    ws.stack.clear();
    ws.stack.push_back(0);

    while (!ws.stack.empty()) {
      const int node = ws.stack.back();
      ws.stack.pop_back();
      const std::size_t nu = static_cast<std::size_t>(node);
      const int state = ws.body[nu];

      if (state >= 0) {
        const std::size_t su = static_cast<std::size_t>(state);
        if (ws.next_body[su] == -1) {
          if (state == i) {
            continue;
          }
          const Real dx = system.x[su] - xi;
          const Real dy = system.y[su] - yi;
          const Real dz = system.z[su] - zi;
          const Real r2 = dx * dx + dy * dy + dz * dz + eps2;
          const Real inv_r = Real(1.0) / std::sqrt(r2);
          const Real factor = gravity * system.mass[su] * inv_r / r2;
          axi += factor * dx;
          ayi += factor * dy;
          azi += factor * dz;
        } else {
          for (int b = state; b != -1; b = ws.next_body[static_cast<std::size_t>(b)]) {
            if (b == i) {
              continue;
            }
            const std::size_t bu = static_cast<std::size_t>(b);
            const Real dx = system.x[bu] - xi;
            const Real dy = system.y[bu] - yi;
            const Real dz = system.z[bu] - zi;
            const Real r2 = dx * dx + dy * dy + dz * dz + eps2;
            const Real inv_r = Real(1.0) / std::sqrt(r2);
            const Real factor = gravity * system.mass[bu] * inv_r / r2;
            axi += factor * dx;
            ayi += factor * dy;
            azi += factor * dz;
          }
        }
        continue;
      }

      if (state != kInternalNode) {
        continue;
      }

      const Real dx = ws.cmx[nu] - xi;
      const Real dy = ws.cmy[nu] - yi;
      const Real dz = ws.cmz[nu] - zi;
      const Real d2 = dx * dx + dy * dy + dz * dz;

      // Criterio s / d < theta (s^2 < theta^2 * d^2), garantindo abrir
      // qualquer celula que contenha o proprio corpo i.
      const Real hs = ws.half_size[nu];
      const Real s = Real(2.0) * hs;
      if (s * s < theta2 * d2 &&
          !(std::fabs(xi - ws.cx[nu]) <= hs && std::fabs(yi - ws.cy[nu]) <= hs &&
            std::fabs(zi - ws.cz[nu]) <= hs)) {
        const Real r2 = d2 + eps2;
        const Real inv_r = Real(1.0) / std::sqrt(r2);
        const Real factor = gravity * ws.mass[nu] * inv_r / r2;
        axi += factor * dx;
        ayi += factor * dy;
        azi += factor * dz;
      } else {
        const std::size_t base = nu * 8;
        for (std::size_t oct = 0; oct < 8; ++oct) {
          const int child = ws.children[base + oct];
          if (child != kEmptyNode) {
            ws.stack.push_back(child);
          }
        }
      }
    }

    system.ax[iu] = axi;
    system.ay[iu] = ayi;
    system.az[iu] = azi;
  }
}

}  // namespace nbody