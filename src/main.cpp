#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nbody/diagnostics.hpp"
#include "nbody/initial_conditions.hpp"
#include "nbody/io.hpp"
#include "nbody/params.hpp"
#include "nbody/simulation.hpp"
#include "nbody/timer.hpp"

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
  bool list = false;
};

void print_usage() {
  std::cout << "uso: nbody [opcoes]\n\n"
            << "  --init NAME        uniforme | plummer | disco   (padrao: plummer)\n"
            << "  --n N              numero de corpos            (padrao: 2000)\n"
            << "  --steps S          passos de integracao        (padrao: 100)\n"
            << "  --dt D             passo de tempo              (padrao: 0.002)\n"
            << "  --softening E      suavizacao de Plummer       (padrao: 0.05)\n"
            << "  --gravity G        constante gravitacional     (padrao: 1.0)\n"
            << "  --theta T          criterio do Barnes-Hut      (padrao: 0.5)\n"
            << "  --solver NAME      brute | barnes-hut          (padrao: brute)\n"
            << "  --seed S           semente aleatoria           (padrao: 42)\n"
            << "  --metrics PATH     CSV com energia/momento por passo\n"
            << "  --snapshot PATH    CSV da configuracao dos corpos\n"
            << "  --snapshot-every K grava um quadro a cada K passos (0 = so o final)\n"
            << "  --check-every K    calcula diagnosticos a cada K passos (padrao: 10)\n"
            << "  --list             lista distribuicoes e solvers\n"
            << "  -h, --help         esta ajuda\n\n"
            << "exemplo:\n"
            << "  nbody --init plummer --n 5000 --steps 200 --metrics results\\energia.csv\n";
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
            << "  brute      : forca bruta O(N^2), baseline sequencial\n"
            << "  barnes-hut : arvore com criterio theta, O(N log N) (a implementar)\n";
}

nbody::BodySystem make_initial_system(const Args& args, nbody::Distribution distribution) {
  switch (distribution) {
    case nbody::Distribution::UniformSphere:
      return nbody::make_uniform_sphere(args.n, kTotalMass, 1.0, args.seed);
    case nbody::Distribution::Plummer:
      return nbody::make_plummer_sphere(args.n, kTotalMass, 0.5, args.gravity, args.seed);
    case nbody::Distribution::Disk:
      return nbody::make_disk(args.n, kTotalMass, 1.0, 0.05, args.gravity, args.seed);
  }
  throw std::runtime_error("distribuicao invalida");
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
  params.gravity = args.gravity;
  params.softening = args.softening;
  params.dt = args.dt;
  params.theta = args.theta;
  params.solver = solver;

  const nbody::BodySystem initial = make_initial_system(args, distribution);
  nbody::Simulation simulation(initial, params);

  const nbody::Diagnostics initial_diag =
      nbody::compute_diagnostics(simulation.bodies(), params.gravity, params.softening);

  std::cout << "distribuicao : " << nbody::distribution_name(distribution) << "\n"
            << "corpos       : " << args.n << "\n"
            << "solver       : " << nbody::solver_name(solver) << "\n"
            << "passos       : " << args.steps << " | dt = " << args.dt
            << " | softening = " << args.softening << "\n\n";

  std::ofstream metrics;
  if (!args.metrics.empty()) {
    metrics.open(args.metrics, std::ios::trunc);
    if (!metrics) {
      throw std::runtime_error("nao foi possivel criar '" + args.metrics + "'");
    }
    nbody::write_metrics_header(metrics);
    nbody::write_metrics_row(metrics, 0, 0.0, initial_diag, 0.0);
  }

  const bool streaming = !args.snapshot.empty() && args.snapshot_every > 0;
  if (streaming) {
    nbody::write_snapshot_csv(args.snapshot, simulation.bodies(), 0, 0.0, false);
  }

  double total_ms = 0.0;
  nbody::Diagnostics latest = initial_diag;

  // Progresso em ~10 atualizacoes, independente do numero de passos.
  const std::size_t progress_every = std::max<std::size_t>(1, args.steps / 10);

  for (std::size_t step = 1; step <= args.steps; ++step) {
    nbody::Timer timer;
    timer.start();
    simulation.step();
    total_ms += timer.stop_ms();

    const bool last = (step == args.steps);

    // A energia potencial custa O(N^2): em N grande, amostre com menos
    // frequencia (--check-every).
    if (step % args.check_every == 0 || last) {
      latest = nbody::compute_diagnostics(simulation.bodies(), params.gravity, params.softening);
      if (metrics) {
        nbody::write_metrics_row(metrics, step, simulation.time(), latest, total_ms);
      }
    }

    if (streaming && (step % args.snapshot_every == 0 || last)) {
      nbody::write_snapshot_csv(args.snapshot, simulation.bodies(), step, simulation.time(), true);
    }

    if (step % progress_every == 0 || last) {
      std::cout << "\r  passo " << step << '/' << args.steps << " ("
                << (100 * step / args.steps) << "%)" << std::flush;
    }
  }
  std::cout << "\n";

  if (!args.snapshot.empty() && !streaming) {
    nbody::write_snapshot_csv(args.snapshot, simulation.bodies(), args.steps, simulation.time(),
                              false);
  }

  const double scale = std::max(1.0, std::fabs(initial_diag.total_energy));
  const double energy_drift =
      std::fabs(latest.total_energy - initial_diag.total_energy) / scale;

  std::cout << std::fixed << std::setprecision(3)
            << "  tempo total : " << (total_ms / 1000.0) << " s ("
            << (total_ms / static_cast<double>(args.steps)) << " ms/passo)\n"
            << std::scientific << std::setprecision(4)
            << "  energia     : " << initial_diag.total_energy << " -> " << latest.total_energy
            << " (deriva relativa " << energy_drift << ")\n"
            << "  momento     : (" << latest.momentum.x << ", " << latest.momentum.y << ", "
            << latest.momentum.z << ")\n"
            << "  centro mass : (" << latest.center_of_mass.x << ", " << latest.center_of_mass.y
            << ", " << latest.center_of_mass.z << ")\n";

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
  if (args.check_every < 1) {
    std::cerr << "erro: --check-every deve ser >= 1\n";
    return 2;
  }

  try {
    return run(args);
  } catch (const std::exception& error) {
    std::cerr << "erro: " << error.what() << "\n";
    return 1;
  }
}
