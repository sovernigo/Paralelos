// Testes de corretude da base sequencial e CUDA, sem dependencia externa:
// o executavel retorna 0 em caso de sucesso e integra direto com o CTest.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>

#include "nbody/accelerations.hpp"
#include "nbody/body_system.hpp"
#include "nbody/cuda_backend.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/initial_conditions.hpp"
#include "nbody/params.hpp"
#include "nbody/simulation.hpp"
#include "nbody/types.hpp"
#include "nbody/vec3.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& name) {
  ++g_checks;
  if (ok) {
    std::cout << "[ok]     " << name << "\n";
    return;
  }
  ++g_failures;
  std::cout << "[FALHOU] " << name << "\n";
}

void check_close(double got, double expected, double tol, const std::string& name) {
  const bool ok = std::fabs(got - expected) <= tol;
  ++g_checks;
  if (ok) {
    std::cout << "[ok]     " << name << "\n";
    return;
  }
  ++g_failures;
  std::cout << "[FALHOU] " << name << " (obtido " << got << ", esperado " << expected << ")\n";
}

bool all_finite(const nbody::BodySystem& bodies) {
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    if (!std::isfinite(bodies.x[i]) || !std::isfinite(bodies.y[i]) ||
        !std::isfinite(bodies.z[i]) || !std::isfinite(bodies.vx[i]) ||
        !std::isfinite(bodies.vy[i]) || !std::isfinite(bodies.vz[i])) {
      return false;
    }
  }
  return true;
}

double mean_relative_accel_error(const nbody::BodySystem& ref, const nbody::BodySystem& approx) {
  double sum_rel = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double rx = static_cast<double>(ref.ax[i]);
    const double ry = static_cast<double>(ref.ay[i]);
    const double rz = static_cast<double>(ref.az[i]);
    const double dx = static_cast<double>(approx.ax[i]) - rx;
    const double dy = static_cast<double>(approx.ay[i]) - ry;
    const double dz = static_cast<double>(approx.az[i]) - rz;
    const double ref_norm = std::sqrt(rx * rx + ry * ry + rz * rz);
    const double err_norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    sum_rel += err_norm / std::max(ref_norm, 1e-12);
  }
  return sum_rel / static_cast<double>(ref.size());
}

}  // namespace

int main() {
  const double tol_tight = nbody::kUsingFloat ? 1e-5 : 1e-12;
  const double tol_mass = nbody::kUsingFloat ? 1e-4 : 1e-12;
  const double tol_momentum = nbody::kUsingFloat ? 1e-4 : 1e-9;
  const double tol_exact_bh = nbody::kUsingFloat ? 1e-4 : 1e-12;

  std::cout << "[info]   precisao: " << (nbody::kUsingFloat ? "float (FP32)" : "double (FP64)")
            << "\n";

  // --- vec3 ---------------------------------------------------------------
  const nbody::Vec3 a(nbody::Real(1.0), nbody::Real(2.0), nbody::Real(3.0));
  const nbody::Vec3 b(nbody::Real(4.0), nbody::Real(5.0), nbody::Real(6.0));

  check_close(nbody::dot(a, b), 32.0, tol_tight, "vec3: produto escalar");
  check_close(nbody::norm(nbody::Vec3(nbody::Real(3.0), nbody::Real(4.0), nbody::Real(0.0))), 5.0,
              tol_tight, "vec3: norma");
  check_close(nbody::dot(a - b, nbody::Vec3(nbody::Real(-3.0), nbody::Real(-3.0),
                                            nbody::Real(-3.0))),
              27.0, tol_tight, "vec3: subtracao");
  check_close(nbody::norm(nbody::normalized(a)), 1.0, tol_tight, "vec3: normalizacao");

  // --- layout SoA ---------------------------------------------------------
  nbody::BodySystem soa;
  soa.add_body(nbody::Real(1.0), nbody::Real(2.0), nbody::Real(3.0), nbody::Real(0.0),
               nbody::Real(0.0), nbody::Real(0.0), nbody::Real(5.0));
  check(soa.size() == 1, "soa: tamanho apos add_body");
  check_close(soa.ax[0], 0.0, 0.0, "soa: aceleracao inicializada em zero");
  soa.resize(4);
  check(soa.size() == 4 && soa.mass[3] == nbody::Real(0.0) && soa.x[0] == nbody::Real(0.0),
        "soa: resize zera arrays");

  // --- forca de dois corpos -----------------------------------------------
  nbody::BodySystem pair;
  pair.add_body(nbody::Real(-0.5), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0),
                nbody::Real(0.0), nbody::Real(0.0), nbody::Real(1.0));
  pair.add_body(nbody::Real(0.5), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0),
                nbody::Real(0.0), nbody::Real(0.0), nbody::Real(1.0));
  nbody::compute_accelerations_bruteforce(pair, nbody::Real(1.0), nbody::Real(0.0));

  check_close(pair.ax[0], 1.0, tol_tight, "forca: corpo 1 atraido para +x");
  check_close(pair.ax[1], -1.0, tol_tight, "forca: corpo 2 atraido para -x (3a lei de Newton)");
  check_close(pair.ay[0], 0.0, tol_tight, "forca: componente y nula");

  nbody::BodySystem softened = pair;
  softened.ax.assign(2, nbody::Real(0.0));
  softened.ay.assign(2, nbody::Real(0.0));
  softened.az.assign(2, nbody::Real(0.0));
  nbody::compute_accelerations_bruteforce(softened, nbody::Real(1.0), nbody::Real(1.0));
  check(std::fabs(softened.ax[0]) < std::fabs(pair.ax[0]), "forca: softening reduz a aceleracao");

  // --- diagnosticos -------------------------------------------------------
  const nbody::Diagnostics diag_pair =
      nbody::compute_diagnostics(pair, nbody::Real(1.0), nbody::Real(0.0));
  check_close(diag_pair.potential, -1.0, tol_tight, "diagnostico: potencial de dois corpos");
  check_close(diag_pair.kinetic, 0.0, tol_tight, "diagnostico: energia cinetica nula");
  check_close(nbody::norm(diag_pair.momentum), 0.0, tol_tight, "diagnostico: momento nulo");
  check_close(diag_pair.total_energy, -1.0, tol_tight, "diagnostico: energia total");

  // --- orbita circular de dois corpos -------------------------------------
  nbody::BodySystem orbit;
  const nbody::Real v = static_cast<nbody::Real>(std::sqrt(0.5));
  orbit.add_body(nbody::Real(-0.5), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0), v,
                 nbody::Real(0.0), nbody::Real(1.0));
  orbit.add_body(nbody::Real(0.5), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0), -v,
                 nbody::Real(0.0), nbody::Real(1.0));

  nbody::SimulationParams params;
  params.gravity = nbody::Real(1.0);
  params.softening = nbody::Real(0.0);
  params.dt = nbody::Real(0.001);

  nbody::Simulation sim(orbit, params);
  const nbody::Diagnostics energy_before =
      nbody::compute_diagnostics(sim.bodies(), nbody::Real(1.0), nbody::Real(0.0));

  for (int i = 0; i < 1000; ++i) {
    sim.step();
  }

  const nbody::BodySystem& end = sim.bodies();
  const double dx = static_cast<double>(end.x[1] - end.x[0]);
  const double dy = static_cast<double>(end.y[1] - end.y[0]);
  const double dz = static_cast<double>(end.z[1] - end.z[0]);
  check_close(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0, 0.01,
              "orbita: separacao preservada (1000 passos)");

  const nbody::Diagnostics energy_after =
      nbody::compute_diagnostics(end, nbody::Real(1.0), nbody::Real(0.0));
  const double drift =
      std::fabs(static_cast<double>(energy_after.total_energy - energy_before.total_energy)) /
      std::fabs(static_cast<double>(energy_before.total_energy));
  check(drift < 5e-3, "orbita: energia conservada (drift < 0.5%)");
  check(nbody::norm(energy_after.momentum) < tol_momentum, "orbita: momento conservado");
  check(nbody::norm(energy_after.center_of_mass) < tol_momentum, "orbita: centro de massa fixo");

  // --- condicoes iniciais -------------------------------------------------
  const nbody::BodySystem sphere =
      nbody::make_uniform_sphere(200, nbody::Real(1.0), nbody::Real(1.0), 7);
  check(sphere.size() == 200, "condicoes: esfera uniforme com 200 corpos");

  bool inside = true;
  double sphere_mass = 0.0;
  nbody::Vec3 sphere_com;
  for (std::size_t i = 0; i < sphere.size(); ++i) {
    const double r =
        std::sqrt(static_cast<double>(sphere.x[i] * sphere.x[i] + sphere.y[i] * sphere.y[i] +
                                      sphere.z[i] * sphere.z[i]));
    if (r > 1.0 + tol_tight) {
      inside = false;
    }
    sphere_mass += static_cast<double>(sphere.mass[i]);
    sphere_com += nbody::Vec3(sphere.mass[i] * sphere.x[i], sphere.mass[i] * sphere.y[i],
                              sphere.mass[i] * sphere.z[i]);
  }
  check(inside, "condicoes: corpos dentro do raio");
  check_close(sphere_mass, 1.0, tol_mass, "condicoes: massa total preservada");
  check(nbody::norm(sphere_com * static_cast<nbody::Real>(1.0 / sphere_mass)) < 0.2,
        "condicoes: centro de massa perto da origem");

  const nbody::BodySystem plummer =
      nbody::make_plummer_sphere(500, nbody::Real(1.0), nbody::Real(0.5), nbody::Real(1.0), 11);
  check(plummer.size() == 500, "condicoes: plummer com 500 corpos");
  check(all_finite(plummer), "condicoes: plummer com valores finitos");

  double plummer_kinetic = 0.0;
  double plummer_max_radius = 0.0;
  nbody::Vec3 plummer_com;
  for (std::size_t i = 0; i < plummer.size(); ++i) {
    plummer_kinetic += 0.5 * static_cast<double>(plummer.mass[i]) *
                       static_cast<double>(plummer.vx[i] * plummer.vx[i] +
                                           plummer.vy[i] * plummer.vy[i] +
                                           plummer.vz[i] * plummer.vz[i]);
    plummer_com += nbody::Vec3(plummer.mass[i] * plummer.x[i], plummer.mass[i] * plummer.y[i],
                               plummer.mass[i] * plummer.z[i]);
    const double r =
        std::sqrt(static_cast<double>(plummer.x[i] * plummer.x[i] + plummer.y[i] * plummer.y[i] +
                                      plummer.z[i] * plummer.z[i]));
    plummer_max_radius = std::max(plummer_max_radius, r);
  }
  check(plummer_kinetic > 0.0, "condicoes: plummer com velocidades nao nulas");
  check(nbody::norm(plummer_com) < 0.25 * plummer_max_radius,
        "condicoes: plummer com centro de massa centralizado");

  const nbody::BodySystem disk = nbody::make_disk(300, nbody::Real(1.0), nbody::Real(1.0),
                                                  nbody::Real(0.05), nbody::Real(1.0), 5);
  check(disk.size() == 300, "condicoes: disco com 300 corpos");

  bool flat = true;
  bool tangential = true;
  for (std::size_t i = 0; i < disk.size(); ++i) {
    if (std::fabs(disk.z[i]) > 0.5) {
      flat = false;
    }
    const double radial_velocity =
        static_cast<double>(disk.x[i] * disk.vx[i] + disk.y[i] * disk.vy[i]);
    if (std::fabs(radial_velocity) > tol_tight) {
      tangential = false;
    }
  }
  check(flat, "condicoes: disco fino (|z| pequeno)");
  check(tangential, "condicoes: disco com velocidade tangencial");

  // --- validacao e backends ----------------------------------------------
  bool empty_threw = false;
  try {
    nbody::Simulation invalid(nbody::BodySystem{}, params);
  } catch (const std::exception&) {
    empty_threw = true;
  }
  check(empty_threw, "simulacao: rejeita sistema vazio");

  nbody::Solver solver = nbody::Solver::BruteForce;
  check(nbody::parse_solver("barnes-hut", solver) && solver == nbody::Solver::BarnesHut,
        "solver: parse 'barnes-hut'");
  check(nbody::parse_solver("barnes-hut-cuda", solver) && solver == nbody::Solver::BarnesHutCuda,
        "solver: parse 'barnes-hut-cuda'");
  check(!nbody::parse_solver("gpu", solver), "solver: rejeita nome invalido");
  check(std::string(nbody::solver_name(nbody::Solver::BruteForce)) == "brute",
        "solver: nome do baseline");

  nbody::Distribution distribution = nbody::Distribution::Plummer;
  check(nbody::parse_distribution("disco", distribution) &&
            distribution == nbody::Distribution::Disk,
        "distribuicao: parse 'disco'");
  check(!nbody::parse_distribution("galaxia", distribution), "distribuicao: rejeita nome invalido");

  // --- barnes-hut sequencial ---------------------------------------------
  nbody::BodySystem bh_ref =
      nbody::make_plummer_sphere(250, nbody::Real(1.0), nbody::Real(0.5), nbody::Real(1.0), 42);
  nbody::compute_accelerations_bruteforce(bh_ref, nbody::Real(1.0), nbody::Real(0.05));

  nbody::SimulationParams bh_params;
  bh_params.gravity = nbody::Real(1.0);
  bh_params.softening = nbody::Real(0.05);
  bh_params.dt = nbody::Real(0.001);
  bh_params.solver = nbody::Solver::BarnesHut;

  nbody::BodySystem bh_exact = bh_ref;
  bh_params.theta = nbody::Real(0.0);
  nbody::compute_accelerations_barnes_hut(bh_exact, bh_params);
  check(mean_relative_accel_error(bh_ref, bh_exact) < tol_exact_bh,
        "barnes-hut: theta=0 reproduz forca bruta");

  nbody::BodySystem bh_t02 = bh_ref;
  bh_params.theta = nbody::Real(0.2);
  nbody::compute_accelerations_barnes_hut(bh_t02, bh_params);
  const double err_02 = mean_relative_accel_error(bh_ref, bh_t02);

  nbody::BodySystem bh_t05 = bh_ref;
  bh_params.theta = nbody::Real(0.5);
  nbody::compute_accelerations_barnes_hut(bh_t05, bh_params);
  const double err_05 = mean_relative_accel_error(bh_ref, bh_t05);

  nbody::BodySystem bh_t08 = bh_ref;
  bh_params.theta = nbody::Real(0.8);
  nbody::compute_accelerations_barnes_hut(bh_t08, bh_params);
  const double err_08 = mean_relative_accel_error(bh_ref, bh_t08);

  std::cout << "[info]   erro rel. medio CPU (N=250): theta=0.2 -> " << err_02
            << ", theta=0.5 -> " << err_05 << ", theta=0.8 -> " << err_08 << "\n";

  check(err_05 < 5e-2, "barnes-hut: erro relativo medio < 5% para theta=0.5");
  check(err_02 < err_05 && err_05 < err_08,
        "barnes-hut: erro cresce monotonicamente com theta (0.2 < 0.5 < 0.8)");

  nbody::BodySystem coincident;
  coincident.add_body(nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0),
                      nbody::Real(0.0), nbody::Real(0.0), nbody::Real(1.0));
  coincident.add_body(nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0),
                      nbody::Real(0.0), nbody::Real(0.0), nbody::Real(1.0));
  coincident.add_body(nbody::Real(1.0), nbody::Real(0.0), nbody::Real(0.0), nbody::Real(0.0),
                      nbody::Real(0.0), nbody::Real(0.0), nbody::Real(1.0));
  nbody::BodySystem coincident_ref = coincident;
  nbody::compute_accelerations_bruteforce(coincident_ref, nbody::Real(1.0), nbody::Real(0.1));
  bh_params.theta = nbody::Real(0.5);
  bh_params.softening = nbody::Real(0.1);
  nbody::compute_accelerations_barnes_hut(coincident, bh_params);
  check(mean_relative_accel_error(coincident_ref, coincident) < tol_exact_bh,
        "barnes-hut: trata corpos coincidentes corretamente");

  nbody::SimulationParams bh_orbit_params;
  bh_orbit_params.gravity = nbody::Real(1.0);
  bh_orbit_params.softening = nbody::Real(0.0);
  bh_orbit_params.dt = nbody::Real(0.001);
  bh_orbit_params.theta = nbody::Real(0.5);
  bh_orbit_params.solver = nbody::Solver::BarnesHut;

  nbody::Simulation bh_sim(orbit, bh_orbit_params);
  const nbody::Diagnostics bh_e0 =
      nbody::compute_diagnostics(bh_sim.bodies(), nbody::Real(1.0), nbody::Real(0.0));
  for (int i = 0; i < 500; ++i) {
    bh_sim.step();
  }
  const nbody::Diagnostics bh_e1 =
      nbody::compute_diagnostics(bh_sim.bodies(), nbody::Real(1.0), nbody::Real(0.0));
  const double bh_drift =
      std::fabs(static_cast<double>(bh_e1.total_energy - bh_e0.total_energy)) /
      std::fabs(static_cast<double>(bh_e0.total_energy));
  check(bh_drift < 5e-3, "barnes-hut: orbita conserva energia");

  bool negative_theta_threw = false;
  try {
    bh_params.theta = nbody::Real(-0.1);
    nbody::compute_accelerations_barnes_hut(bh_exact, bh_params);
  } catch (const std::exception&) {
    negative_theta_threw = true;
  }
  check(negative_theta_threw, "barnes-hut: rejeita theta negativo");

  // --- barnes-hut CUDA ---------------------------------------------------
  if (nbody::cuda_backend_available()) {
    nbody::SimulationParams cuda_params;
    cuda_params.gravity = nbody::Real(1.0);
    cuda_params.softening = nbody::Real(0.05);
    cuda_params.dt = nbody::Real(0.001);
    cuda_params.solver = nbody::Solver::BarnesHutCuda;

    nbody::BodySystem cuda_exact = bh_ref;
    cuda_params.theta = nbody::Real(0.0);
    nbody::compute_accelerations_barnes_hut_cuda(cuda_exact, cuda_params);
    check(mean_relative_accel_error(bh_ref, cuda_exact) < tol_exact_bh,
          "barnes-hut-cuda: theta=0 reproduz forca bruta");

    nbody::BodySystem cuda_t05 = bh_ref;
    cuda_params.theta = nbody::Real(0.5);
    nbody::compute_accelerations_barnes_hut_cuda(cuda_t05, cuda_params);
    const double err_cuda_vs_brute = mean_relative_accel_error(bh_ref, cuda_t05);
    const double err_cuda_vs_cpu = mean_relative_accel_error(bh_t05, cuda_t05);
    std::cout << "[info]   erro rel. medio CUDA (N=250, theta=0.5): vs brute -> "
              << err_cuda_vs_brute << ", vs BH-CPU -> " << err_cuda_vs_cpu << "\n";
    check(err_cuda_vs_brute < 5e-2, "barnes-hut-cuda: erro vs brute < 5% para theta=0.5");
    check(err_cuda_vs_cpu < 1e-3, "barnes-hut-cuda: concorda com Barnes-Hut CPU");

    nbody::BodySystem cuda_coincident = coincident_ref;
    cuda_params.softening = nbody::Real(0.1);
    cuda_params.theta = nbody::Real(0.5);
    nbody::compute_accelerations_barnes_hut_cuda(cuda_coincident, cuda_params);
    check(mean_relative_accel_error(coincident_ref, cuda_coincident) < tol_exact_bh,
          "barnes-hut-cuda: trata corpos coincidentes corretamente");

    nbody::SimulationParams cuda_orbit_params = bh_orbit_params;
    cuda_orbit_params.solver = nbody::Solver::BarnesHutCuda;
    nbody::Simulation cuda_sim(orbit, cuda_orbit_params);
    const nbody::Diagnostics cuda_e0 =
        nbody::compute_diagnostics(cuda_sim.bodies(), nbody::Real(1.0), nbody::Real(0.0));
    for (int i = 0; i < 500; ++i) {
      cuda_sim.step();
    }
    const nbody::Diagnostics cuda_e1 =
        nbody::compute_diagnostics(cuda_sim.bodies(), nbody::Real(1.0), nbody::Real(0.0));
    const double cuda_drift =
        std::fabs(static_cast<double>(cuda_e1.total_energy - cuda_e0.total_energy)) /
        std::fabs(static_cast<double>(cuda_e0.total_energy));
    check(cuda_drift < 5e-3, "barnes-hut-cuda: orbita conserva energia");
  } else {
    std::cout << "[info]   CUDA indisponivel neste ambiente; testes GPU pulados\n";
  }

  std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " verificacoes passaram\n";
  return g_failures == 0 ? 0 : 1;
}