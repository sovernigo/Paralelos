#!/usr/bin/env python3
"""Gera graficos de desempenho, speedup, sensibilidade a theta e energia."""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Erro: matplotlib nao encontrado.\n"
        "No container Ubuntu, instale com: apt-get update && apt-get install -y python3-matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)

SOLVER_LABELS = {
    "brute": "Força Bruta (CPU)",
    "barnes-hut": "Barnes-Hut (CPU)",
    "barnes-hut-cuda": "Barnes-Hut (CUDA)",
}

SOLVER_COLORS = {
    "brute": "#d62728",
    "barnes-hut": "#1f77b4",
    "barnes-hut-cuda": "#2ca02c",
}

SOLVER_MARKERS = {
    "brute": "s",
    "barnes-hut": "o",
    "barnes-hut-cuda": "^",
}


def load_benchmarks(csv_path, metric):
    # Mantem a ultima execucao registrada para cada (solver, theta, n).
    latest = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                solver = row["solver"].strip()
                n = int(row["n"])
                theta = round(float(row["theta"]), 4)
                val = float(row[metric])
            except (KeyError, ValueError):
                continue
            latest[(solver, theta, n)] = val
    return latest


def plot_time_vs_n(data, target_theta, out_dir, metric):
    series = defaultdict(dict)
    for (solver, theta, n), val in data.items():
        if solver == "brute" or abs(theta - target_theta) < 1e-3:
            series[solver][n] = val

    if not series:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for solver in ("brute", "barnes-hut", "barnes-hut-cuda"):
        if solver not in series:
            continue
        ns = sorted(series[solver].keys())
        ts = [series[solver][n] for n in ns]
        ax.loglog(
            ns,
            ts,
            marker=SOLVER_MARKERS.get(solver, "o"),
            color=SOLVER_COLORS.get(solver),
            linewidth=2.0,
            markersize=6,
            label=SOLVER_LABELS.get(solver, solver),
        )

    # Guias visuais de inclinacao O(N^2) e O(N log N)
    if "brute" in series and len(series["brute"]) >= 2:
        n0 = min(series["brute"].keys())
        t0 = series["brute"][n0]
        ns = sorted(series["brute"].keys())
        ref_n2 = [t0 * (n / n0) ** 2 for n in ns]
        ax.loglog(ns, ref_n2, "--", color="#d62728", alpha=0.35, label="Ref. O(N²)")

    if "barnes-hut" in series and len(series["barnes-hut"]) >= 2:
        n0 = min(series["barnes-hut"].keys())
        t0 = series["barnes-hut"][n0]
        ns = sorted(series["barnes-hut"].keys())
        ref_nlogn = [
            t0 * (n * math.log2(n)) / (n0 * math.log2(n0)) for n in ns
        ]
        ax.loglog(ns, ref_nlogn, ":", color="#1f77b4", alpha=0.45, label="Ref. O(N log N)")

    ax.set_xlabel("Número de corpos (N)")
    ax.set_ylabel("Tempo por passo (ms)")
    ax.set_title(f"Escalabilidade: Tempo por Passo × N (θ = {target_theta:g}, {metric})")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "tempo_vs_n.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"gerado: {out_path}")


def plot_speedup_vs_n(data, target_theta, out_dir):
    cpu_bh = {}
    gpu_bh = {}
    brute = {}
    for (solver, theta, n), val in data.items():
        if solver == "brute":
            brute[n] = val
        elif abs(theta - target_theta) < 1e-3:
            if solver == "barnes-hut":
                cpu_bh[n] = val
            elif solver == "barnes-hut-cuda":
                gpu_bh[n] = val

    common_gpu_cpu = sorted(set(cpu_bh.keys()) & set(gpu_bh.keys()))
    common_bh_brute = sorted(set(brute.keys()) & set(cpu_bh.keys()))

    if not common_gpu_cpu and not common_bh_brute:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    if common_gpu_cpu:
        sp = [cpu_bh[n] / gpu_bh[n] for n in common_gpu_cpu]
        ax.plot(
            common_gpu_cpu,
            sp,
            marker="^",
            color="#2ca02c",
            linewidth=2.0,
            markersize=7,
            label="Speedup CUDA vs. Barnes-Hut CPU",
        )
        for n, s in zip(common_gpu_cpu, sp):
            ax.annotate(
                f"{s:.1f}×",
                (n, s),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    if common_bh_brute:
        sp_cpu = [brute[n] / cpu_bh[n] for n in common_bh_brute]
        ax.plot(
            common_bh_brute,
            sp_cpu,
            marker="o",
            color="#1f77b4",
            linestyle="--",
            linewidth=1.8,
            markersize=6,
            label="Speedup Barnes-Hut CPU vs. Força Bruta",
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Número de corpos (N)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"Speedup × N (θ = {target_theta:g})")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "speedup_vs_n.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"gerado: {out_path}")


def plot_theta_sensitivity(data, theta_n, out_dir):
    by_solver = defaultdict(dict)
    for (solver, theta, n), val in data.items():
        if solver == "brute":
            continue
        if theta_n is None or n == theta_n:
            by_solver[(solver, n)][theta] = val

    # Se theta_n nao foi fixado, escolhe o maior N com >= 2 valores de theta.
    if theta_n is None:
        candidates = [n for (_, n), tmap in by_solver.items() if len(tmap) >= 2]
        if not candidates:
            return
        theta_n = max(candidates)

    plotted = False
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for solver in ("barnes-hut", "barnes-hut-cuda"):
        tmap = by_solver.get((solver, theta_n), {})
        if len(tmap) < 2:
            continue
        thetas = sorted(tmap.keys())
        times = [tmap[t] for t in thetas]
        ax.plot(
            thetas,
            times,
            marker=SOLVER_MARKERS.get(solver, "o"),
            color=SOLVER_COLORS.get(solver),
            linewidth=2.0,
            label=SOLVER_LABELS.get(solver, solver),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Parâmetro de aproximação (θ)")
    ax.set_ylabel("Tempo por passo (ms)")
    ax.set_title(f"Desempenho × θ (N = {theta_n:,})".replace(",", "."))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "tempo_vs_theta.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"gerado: {out_path}")


def load_energy_csv(path):
    steps = []
    energies = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [c.strip() for c in (reader.fieldnames or [])]
        step_col = next((c for c in fields if c in ("step", "passo")), fields[0])
        energy_col = next(
            (c for c in fields if c in ("total_energy", "energia_total", "energy")),
            fields[4] if len(fields) > 4 else fields[-1],
        )
        for row in reader:
            try:
                steps.append(int(row[step_col]))
                energies.append(float(row[energy_col]))
            except (KeyError, ValueError):
                continue
    return steps, energies


def plot_energy(energy_specs, out_dir):
    if not energy_specs:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True)
    for spec in energy_specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = os.path.splitext(os.path.basename(path))[0]

        steps, energies = load_energy_csv(path)
        if not steps:
            continue
        e0 = energies[0]
        scale = max(1.0, abs(e0))
        drifts = [abs(e - e0) / scale for e in energies]

        ax1.plot(steps, energies, linewidth=1.8, label=label)
        ax2.plot(steps, drifts, linewidth=1.8, label=label)

    ax1.set_ylabel("Energia total (E)")
    ax1.set_title("Conservação de Energia ao Longo da Simulação")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.set_xlabel("Passo de integração")
    ax2.set_ylabel("Deriva relativa |E - E₀| / max(1, |E₀|)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "energia_vs_passos.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"gerado: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera graficos do projeto N-Corpos a partir dos CSVs em results/."
    )
    parser.add_argument(
        "--benchmark",
        default="results/benchmark.csv",
        help="CSV gerado por scripts/benchmark.sh (padrao: results/benchmark.csv)",
    )
    parser.add_argument(
        "--metric",
        default="best_ms_per_step",
        choices=("best_ms_per_step", "median_ms_per_step"),
        help="Coluna de tempo a plotar (padrao: best_ms_per_step)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Valor de theta para os graficos tempo/speedup x N (padrao: 0.5)",
    )
    parser.add_argument(
        "--theta-n",
        type=int,
        default=None,
        help="Valor de N para o grafico tempo x theta (padrao: maior N disponivel)",
    )
    parser.add_argument(
        "--energy",
        nargs="*",
        default=[],
        help="CSVs de metricas no formato 'Rotulo=caminho.csv'",
    )
    parser.add_argument(
        "--out-dir",
        default="results/plots",
        help="Diretorio de saida das imagens (padrao: results/plots)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.isfile(args.benchmark):
        data = load_benchmarks(args.benchmark, args.metric)
        plot_time_vs_n(data, args.theta, args.out_dir, args.metric)
        plot_speedup_vs_n(data, args.theta, args.out_dir)
        plot_theta_sensitivity(data, args.theta_n, args.out_dir)
    else:
        print(f"aviso: arquivo nao encontrado ({args.benchmark}), pulando graficos de benchmark.")

    if args.energy:
        plot_energy(args.energy, args.out_dir)


if __name__ == "__main__":
    main()