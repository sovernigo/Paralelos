// Testes de corretude da base sequencial, sem dependencia externa: o
// executavel retorna 0 em caso de sucesso e integra direto com o CTest.
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>

#include "nbody/accelerations.hpp"
#include "nbody/body_system.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/initial_conditions.hpp"
#include "nbody/params.hpp"
#include "nbody/simulation.hpp"
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

}  // namespace

int main() {
  // --- vec3 ---------------------------------------------------------------
  const nbody::Vec3 a(1.0, 2.0, 3.0);
  const nbody::Vec3 b(4.0, 5.0, 6.0);

  check_close(nbody::dot(a, b), 32.0, 1e-12, "vec3: produto escalar");
  check_close(nbody::norm(nbody::Vec3(3.0, 4.0, 0.0)), 5.0, 1e-12, "vec3: norma");
  check_close(nbody::dot(a - b, nbody::Vec3(-3.0, -3.0, -3.0)), 27.0, 1e-12,
              "vec3: subtracao");
  check_close(nbody::norm(nbody::normalized(a)), 1.0, 1e-12, "vec3: normalizacao");

  // --- layout SoA ---------------------------------------------------------
  nbody::BodySystem soa;
  soa.add_body(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0);
  check(soa.size() == 1, "soa: tamanho apos add_body");
  check_close(soa.ax[0], 0.0, 0.0, "soa: aceleracao inicializada em zero");
  soa.resize(4);
  check(soa.size() == 4 && soa.mass[3] == 0.0 && soa.x[0] == 0.0, "soa: resize zera arrays");

  // --- forca de dois corpos -----------------------------------------------
  nbody::BodySystem pair;
  pair.add_body(-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  pair.add_body(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  nbody::compute_accelerations_bruteforce(pair, 1.0, 0.0);

  check_close(pair.ax[0], 1.0, 1e-12, "forca: corpo 1 atraido para +x");
  check_close(pair.ax[1], -1.0, 1e-12, "forca: corpo 2 atraido para -x (3a lei de Newton)");
  check_close(pair.ay[0], 0.0, 1e-12, "forca: componente y nula");

  nbody::BodySystem softened = pair;
  softened.ax.assign(2, 0.0);
  softened.ay.assign(2, 0.0);
  softened.az.assign(2, 0.0);
  nbody::compute_accelerations_bruteforce(softened, 1.0, 1.0);
  check(std::fabs(softened.ax[0]) < std::fabs(pair.ax[0]),
        "forca: softening reduz a aceleracao");

  // --- diagnosticos -------------------------------------------------------
  const nbody::Diagnostics diag_pair = nbody::compute_diagnostics(pair, 1.0, 0.0);
  check_close(diag_pair.potential, -1.0, 1e-12, "diagnostico: potencial de dois corpos");
  check_close(diag_pair.kinetic, 0.0, 1e-12, "diagnostico: energia cinetica nula");
  check_close(nbody::norm(diag_pair.momentum), 0.0, 1e-12, "diagnostico: momento nulo");
  check_close(diag_pair.total_energy, -1.0, 1e-12, "diagnostico: energia total");

  // --- orbita circular de dois corpos -------------------------------------
  // Massas iguais, separacao 1: v = sqrt(G*M_total/4 / r) = sqrt(0.5).
  nbody::BodySystem orbit;
  const double v = std::sqrt(0.5);
  orbit.add_body(-0.5, 0.0, 0.0, 0.0, v, 0.0, 1.0);
  orbit.add_body(0.5, 0.0, 0.0, 0.0, -v, 0.0, 1.0);

  nbody::SimulationParams params;
  params.gravity = 1.0;
  params.softening = 0.0;
  params.dt = 0.001;

  nbody::Simulation sim(orbit, params);
  const nbody::Diagnostics energy_before = nbody::compute_diagnostics(sim.bodies(), 1.0, 0.0);

  for (int i = 0; i < 1000; ++i) {
    sim.step();
  }

  const nbody::BodySystem& end = sim.bodies();
  const double dx = end.x[1] - end.x[0];
  const double dy = end.y[1] - end.y[0];
  const double dz = end.z[1] - end.z[0];
  check_close(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0, 0.01,
              "orbita: separacao preservada (1000 passos)");

  const nbody::Diagnostics energy_after = nbody::compute_diagnostics(end, 1.0, 0.0);
  const double drift =
      std::fabs(energy_after.total_energy - energy_before.total_energy) /
      std::fabs(energy_before.total_energy);
  check(drift < 5e-3, "orbita: energia conservada (drift < 0.5%)");
  check(nbody::norm(energy_after.momentum) < 1e-9, "orbita: momento conservado");
  check(nbody::norm(energy_after.center_of_mass) < 1e-9, "orbita: centro de massa fixo");

  // --- condicoes iniciais -------------------------------------------------
  const nbody::BodySystem sphere = nbody::make_uniform_sphere(200, 1.0, 1.0, 7);
  check(sphere.size() == 200, "condicoes: esfera uniforme com 200 corpos");

  bool inside = true;
  double sphere_mass = 0.0;
  nbody::Vec3 sphere_com;
  for (std::size_t i = 0; i < sphere.size(); ++i) {
    const double r =
        std::sqrt(sphere.x[i] * sphere.x[i] + sphere.y[i] * sphere.y[i] + sphere.z[i] * sphere.z[i]);
    if (r > 1.0 + 1e-12) {
      inside = false;
    }
    sphere_mass += sphere.mass[i];
    sphere_com += nbody::Vec3(sphere.mass[i] * sphere.x[i], sphere.mass[i] * sphere.y[i],
                              sphere.mass[i] * sphere.z[i]);
  }
  check(inside, "condicoes: corpos dentro do raio");
  check_close(sphere_mass, 1.0, 1e-12, "condicoes: massa total preservada");
  check(nbody::norm(sphere_com * (1.0 / sphere_mass)) < 0.2,
        "condicoes: centro de massa perto da origem");

  const nbody::BodySystem plummer = nbody::make_plummer_sphere(500, 1.0, 0.5, 1.0, 11);
  check(plummer.size() == 500, "condicoes: plummer com 500 corpos");
  check(all_finite(plummer), "condicoes: plummer com valores finitos");

  double plummer_kinetic = 0.0;
  double plummer_max_radius = 0.0;
  nbody::Vec3 plummer_com;
  for (std::size_t i = 0; i < plummer.size(); ++i) {
    plummer_kinetic += 0.5 * plummer.mass[i] *
                       (plummer.vx[i] * plummer.vx[i] + plummer.vy[i] * plummer.vy[i] +
                        plummer.vz[i] * plummer.vz[i]);
    plummer_com += nbody::Vec3(plummer.mass[i] * plummer.x[i], plummer.mass[i] * plummer.y[i],
                               plummer.mass[i] * plummer.z[i]);
    const double r = std::sqrt(plummer.x[i] * plummer.x[i] + plummer.y[i] * plummer.y[i] +
                               plummer.z[i] * plummer.z[i]);
    plummer_max_radius = std::max(plummer_max_radius, r);
  }
  check(plummer_kinetic > 0.0, "condicoes: plummer com velocidades nao nulas");
  // A cauda de Plummer tem raio grande, entao compara o deslocamento do
  // centro de massa com a extensao do sistema, nao com um valor absoluto.
  check(nbody::norm(plummer_com) < 0.25 * plummer_max_radius,
        "condicoes: plummer com centro de massa centralizado");

  const nbody::BodySystem disk = nbody::make_disk(300, 1.0, 1.0, 0.05, 1.0, 5);
  check(disk.size() == 300, "condicoes: disco com 300 corpos");

  bool flat = true;
  bool tangential = true;
  for (std::size_t i = 0; i < disk.size(); ++i) {
    if (std::fabs(disk.z[i]) > 0.5) {
      flat = false;
    }
    const double radial_velocity = disk.x[i] * disk.vx[i] + disk.y[i] * disk.vy[i];
    if (std::fabs(radial_velocity) > 1e-12) {
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
  check(!nbody::parse_solver("gpu", solver), "solver: rejeita nome invalido");
  check(std::string(nbody::solver_name(nbody::Solver::BruteForce)) == "brute",
        "solver: nome do baseline");

  nbody::Distribution distribution = nbody::Distribution::Plummer;
  check(nbody::parse_distribution("disco", distribution) &&
            distribution == nbody::Distribution::Disk,
        "distribuicao: parse 'disco'");
  check(!nbody::parse_distribution("galaxia", distribution),
        "distribuicao: rejeita nome invalido");

  params.solver = nbody::Solver::BarnesHut;
  bool barnes_hut_threw = false;
  try {
    nbody::Simulation barnes(orbit, params);
  } catch (const std::exception&) {
    barnes_hut_threw = true;
  }
  check(barnes_hut_threw, "barnes-hut: stub lanca erro explicativo");

  std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " verificacoes passaram\n";
  return g_failures == 0 ? 0 : 1;
}
