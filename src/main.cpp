#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nbody/accelerations.hpp"
#include "nbody/cuda_backend.hpp"
#include "nbody/diagnostics.hpp"
#include "nbody/initial_conditions.hpp"
#include "nbody/io.hpp"
#include "nbody/params.hpp"
#include "nbody/simulation.hpp"
#include "nbody/timer.hpp"
#include "nbody/types.hpp"

namespace {

constexpr double kTotalMass = 1.0;

struct Args {
  std::string init = "plummer";
  std::size_t n = 2000;
  std::size_t steps = 100;
  double dt = 0.002;
  double softening = 0.05;
  double gravity = 1.0;
  double theta = 0.5;
  std::string solver = "brute";
  unsigned int seed = 42;
  std::string metrics;
  std::string snapshot;
  std::size_t snapshot_every = 0;
  std::size_t check_every = 10;
  bool no_diagnostics = false;
  bool profile_kernels = false;
  bool verify_error = false;
  bool list = false;
};

void print_usage() {
  std::cout << "uso: nbody [opcoes]\n\n"
            << "  --init NAME        uniforme | plummer | disco         (padrao: plummer)\n"
            << "  --n N              numero de corpos                  (padrao: 2000)\n"
            << "  --steps S          passos de integracao              (padrao: 100)\n"
            << "  --dt D             passo de tempo                    (padrao: 0.002)\n"
            << "  --softening E      suavizacao de Plummer             (padrao: 0.05)\n"
            << "  --gravity G        constante gravitacional           (padrao: 1.0)\n"
            << "  --theta T          criterio do Barnes-Hut            (padrao: 0.5)\n"
            << "  --solver NAME      brute | barnes-hut | barnes-hut-cuda (padrao: brute)\n"
            << "  --seed S           semente aleatoria                 (padrao: 42)\n"
            << "  --metrics PATH     CSV com energia/momento por passo\n"
            << "  --snapshot PATH    CSV da configuracao dos corpos\n"
            << "  --snapshot-every K grava um quadro a cada K passos (0 = so o final)\n"
            << "  --check-every K    calcula diagnosticos a cada K passos (padrao: 10)\n"
            << "  --no-diagnostics   desativa diagnosticos O(N^2) (uso em benchmarks)\n"
            << "  --profile-kernels  mede tempo por kernel no solver CUDA\n"
            << "  --verify-error     compara aceleracao em t=0 contra brute e encerra\n"
            << "  --list             lista distribuicoes e solvers\n"
            << "  -h, --help         esta ajuda\n\n"
            << "exemplo:\n"
            << "  nbody --init plummer --n 5000 --steps 200 --metrics results/energia.csv\n";
}

bool parse_size(const std::string& text, std::size_t& out) {
  try {
    std::size_t pos = 0;
    const unsigned long long value = std::stoull(text, &pos);
    if (pos != text.size()) {
      return false;
    }
    out = static_cast<std::size_t>(value);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool parse_uint(const std::string& text, unsigned int& out) {
  try {
    std::size_t pos = 0;
    const unsigned long value = std::stoul(text, &pos);
    if (pos != text.size()) {
      return false;
    }
    out = static_cast<unsigned int>(value);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool parse_double(const std::string& text, double& out) {
  try {
    std::size_t pos = 0;
    const double value = std::stod(text, &pos);
    if (pos != text.size()) {
      return false;
    }
    out = value;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

Args parse_args(int argc, char** argv) {
  Args args;

  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("faltou o valor de ") + name);
      }
      return argv[++i];
    };

    if (flag == "-h" || flag == "--help") {
      print_usage();
      std::exit(0);
    } else if (flag == "--list") {
      args.list = true;
    } else if (flag == "--init") {
      args.init = value("--init");
    } else if (flag == "--n") {
      if (!parse_size(value("--n"), args.n)) {
        throw std::runtime_error("--n espera um inteiro positivo");
      }
    } else if (flag == "--steps") {
      if (!parse_size(value("--steps"), args.steps)) {
        throw std::runtime_error("--steps espera um inteiro positivo");
      }
    } else if (flag == "--dt") {
      if (!parse_double(value("--dt"), args.dt)) {
        throw std::runtime_error("--dt espera um numero");
      }
    } else if (flag == "--softening") {
      if (!parse_double(value("--softening"), args.softening)) {
        throw std::runtime_error("--softening espera um numero");
      }
    } else if (flag == "--gravity") {
      if (!parse_double(value("--gravity"), args.gravity)) {
        throw std::runtime_error("--gravity espera um numero");
      }
    } else if (flag == "--theta") {
      if (!parse_double(value("--theta"), args.theta)) {
        throw std::runtime_error("--theta espera um numero");
      }
    } else if (flag == "--solver") {
      args.solver = value("--solver");
    } else if (flag == "--seed") {
      if (!parse_uint(value("--seed"), args.seed)) {
        throw std::runtime_error("--seed espera um inteiro");
      }
    } else if (flag == "--metrics") {
      args.metrics = value("--metrics");
    } else if (flag == "--snapshot") {
      args.snapshot = value("--snapshot");
    } else if (flag == "--snapshot-every") {
      if (!parse_size(value("--snapshot-every"), args.snapshot_every)) {
        throw std::runtime_error("--snapshot-every espera um inteiro");
      }
    } else if (flag == "--check-every") {
      if (!parse_size(value("--check-every"), args.check_every)) {
        throw std::runtime_error("--check-every espera um inteiro");
      }
    } else if (flag == "--no-diagnostics") {
      args.no_diagnostics = true;
    } else if (flag == "--profile-kernels") {
      args.profile_kernels = true;
    } else if (flag == "--verify-error") {
      args.verify_error = true;
    } else {
      throw std::runtime_error("argumento desconhecido: " + flag);
    }
  }

  return args;
}

void print_list() {
  std::cout << "distribuicoes iniciais (--init):\n"
            << "  uniforme : esfera homogenea em repouso (colapso gravitacional)\n"
            << "  plummer  : esfera de Plummer em equilibrio (posicoes e velocidades)\n"
            << "  disco    : disco fino em rotacao circular\n\n"
            << "solvers (--solver):\n"
            << "  brute           : forca bruta O(N^2), baseline sequencial\n"
            << "  barnes-hut      : Octree linearizada em CPU, O(N log N)\n"
            << "  barnes-hut-cuda : Octree linearizada 100% em GPU (CUDA) ["
            << (nbody::cuda_backend_available() ? "disponivel" : "indisponivel") << "]\n\n"
            << "precisao compilada: " << (nbody::kUsingFloat ? "float (FP32)" : "double (FP64)")
            << "\n";
}

nbody::BodySystem make_initial_system(const Args& args, nbody::Distribution distribution) {
  const nbody::Real total_mass = static_cast<nbody::Real>(kTotalMass);
  const nbody::Real gravity = static_cast<nbody::Real>(args.gravity);

  switch (distribution) {
    case nbody::Distribution::UniformSphere:
      return nbody::make_uniform_sphere(args.n, total_mass, nbody::Real(1.0), args.seed);
    case nbody::Distribution::Plummer:
      return nbody::make_plummer_sphere(args.n, total_mass, nbody::Real(0.5), gravity, args.seed);
    case nbody::Distribution::Disk:
      return nbody::make_disk(args.n, total_mass, nbody::Real(1.0), nbody::Real(0.05), gravity,
                              args.seed);
  }
  throw std::runtime_error("distribuicao invalida");
}

int run_verify_error(const Args& args, nbody::Distribution distribution,
                     const nbody::SimulationParams& params) {
  nbody::BodySystem ref = make_initial_system(args, distribution);
  nbody::BodySystem approx = ref;

  nbody::compute_accelerations_bruteforce(ref, params.gravity, params.softening);
  nbody::compute_accelerations(approx, params);

  double sum_rel = 0.0;
  double max_rel = 0.0;
  double sum_err2 = 0.0;
  double sum_ref2 = 0.0;

  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double rx = static_cast<double>(ref.ax[i]);
    const double ry = static_cast<double>(ref.ay[i]);
    const double rz = static_cast<double>(ref.az[i]);
    const double dx = static_cast<double>(approx.ax[i]) - rx;
    const double dy = static_cast<double>(approx.ay[i]) - ry;
    const double dz = static_cast<double>(approx.az[i]) - rz;

    const double ref2 = rx * rx + ry * ry + rz * rz;
    const double err2 = dx * dx + dy * dy + dz * dz;
    const double rel = std::sqrt(err2) / std::max(std::sqrt(ref2), 1e-12);

    sum_rel += rel;
    max_rel = std::max(max_rel, rel);
    sum_err2 += err2;
    sum_ref2 += ref2;
  }

  const double mean_rel = sum_rel / static_cast<double>(ref.size());
  const double l2_rel = std::sqrt(sum_err2) / std::max(std::sqrt(sum_ref2), 1e-12);

  std::cout << "verificacao de erro (t = 0 vs. brute)\n"
            << "distribuicao : " << nbody::distribution_name(distribution) << " (seed = "
            << args.seed << ")\n"
            << "corpos       : " << args.n << "\n"
            << "solver       : " << nbody::solver_name(params.solver) << " ("
            << (nbody::kUsingFloat ? "FP32" : "FP64") << ") | theta = " << args.theta
            << " | softening = " << args.softening << "\n\n"
            << std::scientific << std::setprecision(4)
            << "  erro rel. medio : " << mean_rel << " (" << std::fixed << std::setprecision(4)
            << (mean_rel * 100.0) << "%)\n"
            << std::scientific << std::setprecision(4)
            << "  erro rel. L2    : " << l2_rel << " (" << std::fixed << std::setprecision(4)
            << (l2_rel * 100.0) << "%)\n"
            << std::scientific << std::setprecision(4)
            << "  erro rel. max   : " << max_rel << " (" << std::fixed << std::setprecision(4)
            << (max_rel * 100.0) << "%)\n";

  return 0;
}

int run(const Args& args) {
  nbody::Distribution distribution;
  if (!nbody::parse_distribution(args.init, distribution)) {
    throw std::runtime_error("distribuicao invalida: '" + args.init + "' (use --list)");
  }

  nbody::Solver solver;
  if (!nbody::parse_solver(args.solver, solver)) {
    throw std::runtime_error("solver invalido: '" + args.solver + "' (use --list)");
  }

  nbody::SimulationParams params;
  params.gravity = static_cast<nbody::Real>(args.gravity);
  params.softening = static_cast<nbody::Real>(args.softening);
  params.dt = static_cast<nbody::Real>(args.dt);
  params.theta = static_cast<nbody::Real>(args.theta);
  params.solver = solver;
  params.profile_kernels = args.profile_kernels;

  if (args.verify_error) {
    return run_verify_error(args, distribution, params);
  }

  const nbody::BodySystem initial = make_initial_system(args, distribution);
  nbody::Simulation simulation(initial, params);

  const bool run_diagnostics = !args.no_diagnostics;
  nbody::Diagnostics initial_diag;
  if (run_diagnostics) {
    initial_diag =
        nbody::compute_diagnostics(simulation.bodies(), params.gravity, params.softening);
  }

  std::cout << "distribuicao : " << nbody::distribution_name(distribution) << "\n"
            << "corpos       : " << args.n << "\n"
            << "solver       : " << nbody::solver_name(solver)
            << " (" << (nbody::kUsingFloat ? "FP32" : "FP64") << ")\n"
            << "passos       : " << args.steps << " | dt = " << args.dt
            << " | softening = " << args.softening << "\n\n";

  std::ofstream metrics;
  if (run_diagnostics && !args.metrics.empty()) {
    metrics.open(args.metrics, std::ios::trunc);
    if (!metrics) {
      throw std::runtime_error("nao foi possivel criar '" + args.metrics + "'");
    }
    nbody::write_metrics_header(metrics);
    nbody::write_metrics_row(metrics, 0, nbody::Real(0.0), initial_diag, 0.0);
  }

  const bool streaming = !args.snapshot.empty() && args.snapshot_every > 0;
  if (streaming) {
    nbody::write_snapshot_csv(args.snapshot, simulation.bodies(), 0, nbody::Real(0.0), false);
  }

  double total_ms = 0.0;
  nbody::Diagnostics latest = initial_diag;

  // Progresso em ~10 atualizacoes, independente do numero de passos.
  const std::size_t progress_every = std::max<std::size_t>(1, args.steps / 10);

  for (std::size_t step = 1; step <= args.steps; ++step) {
    nbody::Timer timer;
    timer.start();
    simulation.step();
    simulation.synchronize();
    total_ms += timer.stop_ms();

    const bool last = (step == args.steps);

    // A energia potencial custa O(N^2): em N grande, use --no-diagnostics
    // ou aumente --check-every.
    if (run_diagnostics && (step % args.check_every == 0 || last)) {
      latest = nbody::compute_diagnostics(simulation.bodies(), params.gravity, params.softening);
      if (metrics) {
        nbody::write_metrics_row(metrics, step, simulation.time(), latest, total_ms);
      }
    }

    if (streaming && (step % args.snapshot_every == 0 || last)) {
      nbody::write_snapshot_csv(args.snapshot, simulation.bodies(), step, simulation.time(), true);
    }

    if (step % progress_every == 0 || last) {
      std::cout << "\r  passo " << step << '/' << args.steps << " (" << (100 * step / args.steps)
                << "%)" << std::flush;
    }
  }
  std::cout << "\n";

  if (!args.snapshot.empty() && !streaming) {
    nbody::write_snapshot_csv(args.snapshot, simulation.bodies(), args.steps, simulation.time(),
                              false);
  }

  const double steps_d = static_cast<double>(args.steps);
  std::cout << std::fixed << std::setprecision(3)
            << "  tempo total : " << (total_ms / 1000.0) << " s (" << (total_ms / steps_d)
            << " ms/passo)\n";

  if (simulation.uses_cuda() && args.profile_kernels) {
    const nbody::CudaKernelTimings t = simulation.cuda_timings();
    std::cout << "  kernels GPU : "
              << "bbox=" << (t.bbox_ms / steps_d) << " ms"
              << " | sort=" << (t.sort_ms / steps_d) << " ms"
              << " | build=" << (t.build_tree_ms / steps_d) << " ms"
              << " | sum=" << (t.summarize_ms / steps_d) << " ms"
              << " | force=" << (t.force_ms / steps_d) << " ms"
              << " | verlet=" << (t.integrate_ms / steps_d) << " ms\n";
  }

  if (run_diagnostics) {
    const double scale = std::max(1.0, std::fabs(static_cast<double>(initial_diag.total_energy)));
    const double energy_drift =
        std::fabs(static_cast<double>(latest.total_energy - initial_diag.total_energy)) / scale;

    std::cout << std::scientific << std::setprecision(4)
              << "  energia     : " << initial_diag.total_energy << " -> " << latest.total_energy
              << " (deriva relativa " << energy_drift << ")\n"
              << "  momento     : (" << latest.momentum.x << ", " << latest.momentum.y << ", "
              << latest.momentum.z << ")\n"
              << "  centro mass : (" << latest.center_of_mass.x << ", " << latest.center_of_mass.y
              << ", " << latest.center_of_mass.z << ")\n";
  } else {
    std::cout << "  diagnosticos: desativados (--no-diagnostics)\n";
  }

  if (!args.metrics.empty()) {
    std::cout << "  metricas    : " << args.metrics << "\n";
  }
  if (!args.snapshot.empty()) {
    std::cout << "  snapshot    : " << args.snapshot << "\n";
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "erro: " << error.what() << "\n\n";
    print_usage();
    return 2;
  }

  if (args.list) {
    print_list();
    return 0;
  }

  if (args.n < 2) {
    std::cerr << "erro: --n deve ser >= 2 (a forca gravitacional precisa de dois corpos)\n";
    return 2;
  }
  if (args.steps < 1) {
    std::cerr << "erro: --steps deve ser >= 1\n";
    return 2;
  }
  if (!(args.dt > 0.0)) {
    std::cerr << "erro: --dt deve ser positivo\n";
    return 2;
  }
  if (args.softening < 0.0) {
    std::cerr << "erro: --softening nao pode ser negativo\n";
    return 2;
  }
  if (!args.no_diagnostics && args.check_every < 1) {
    std::cerr << "erro: --check-every deve ser >= 1\n";
    return 2;
  }
  if (args.no_diagnostics && !args.metrics.empty()) {
    std::cerr << "erro: --metrics nao pode ser usado junto com --no-diagnostics\n";
    return 2;
  }

  try {
    return run(args);
  } catch (const std::exception& error) {
    std::cerr << "erro: " << error.what() << "\n";
    return 1;
  }
}