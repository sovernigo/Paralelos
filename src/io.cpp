#include "nbody/io.hpp"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace nbody {

void write_metrics_header(std::ostream& out) {
  out << "step,time,kinetic,potential,total_energy,momentum_x,momentum_y,momentum_z,elapsed_ms\n";
}

void write_metrics_row(std::ostream& out, std::size_t step, Real time, const Diagnostics& diag,
                       double elapsed_ms) {
  out << step << ',' << std::setprecision(10) << time << ',' << diag.kinetic << ','
      << diag.potential << ',' << diag.total_energy << ',' << diag.momentum.x << ','
      << diag.momentum.y << ',' << diag.momentum.z << ',' << std::setprecision(6) << elapsed_ms
      << '\n';
}

void write_snapshot_csv(const std::string& path, const BodySystem& bodies, std::size_t step,
                        Real time, bool append) {
  std::ofstream out(path, append ? std::ios::app : std::ios::trunc);
  if (!out) {
    throw std::runtime_error("nao foi possivel escrever em '" + path + "'");
  }

  if (!append) {
    out << "step,time,index,x,y,z,vx,vy,vz,mass\n";
  }

  out << std::setprecision(12);
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    out << step << ',' << time << ',' << i << ',' << bodies.x[i] << ',' << bodies.y[i] << ','
        << bodies.z[i] << ',' << bodies.vx[i] << ',' << bodies.vy[i] << ',' << bodies.vz[i] << ','
        << bodies.mass[i] << '\n';
  }
}

}  // namespace nbody
