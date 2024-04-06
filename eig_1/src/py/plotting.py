import matplotlib.pyplot as plt
import numpy as np

# For nicer plots, requires a fairly full TeXlive install
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}",
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 24,
    "font.family": "serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})

# globals
x_max = 10
method_names = ["3 point", "3 point tridiag", "3 point tridiag parity", "5 point", "5 point parity"]


def main():
    #evs = load_evals("../../results/evs/evs_test.txt")
    #plot_evals(evs, "../../plots/evals/evals.pdf")
    #evecs = load_evecs("../../results/evecs/evecs_test.txt", 10)
    #plot_evecs(evecs, "../../plots/states/states_test.pdf", energies=evs)

    ns, times = load_bench_times("../../results/bench/bench_time.txt.txt")
    plot_bench_times(ns, times, "../../plots/bench/bench.pdf")
    ns, errs = load_bench_times("../../results/bench/bench_gs_erg_err.txt")
    plot_bench_errs(ns, errs, "../../plots/bench/bench_err.pdf")

    return 0


def load_evals(path):
    file = open(path)
    lines = file.readlines()
    evs = [float(ev) for ev in lines]
    evs_np = np.asarray(evs)
    file.close()

    return evs_np


def plot_evals(evals, path):
    fig, ax = plt.subplots()
    n = len(evals)

    ax.plot(np.arange(n) + 1, evals, label=f"Numerical eigenvalues, $n={n}$, " + "$z_\\text{max}=" + f"{x_max}$")
    ax.plot(np.arange(n) + 1, np.arange(n) + 0.5, color="forestgreen", alpha=0.6, label="Exact eigenvalues")
    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel("$i$-th Energy ($\\hbar \\omega$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1, n + 1)
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def load_evecs(path, n_evec=5):
    file = open(path)
    evecs = []
    for i in range(n_evec):
        line = next(file)
        components_str = line.split(",")[:-1]
        evecs.append([float(component) for component in components_str])
    evecs_np = np.asarray(evecs)
    file.close()

    return evecs_np


def plot_evecs(evecs, path, energies=None):
    n = len(evecs[0])
    n_evecs = len(evecs)
    xs = np.linspace(-x_max, x_max, n)
    fig, axes = plt.subplots(nrows=int(n_evecs / 2), ncols=2, figsize=(9, n_evecs))
    axes = axes.flatten()

    y_max = np.max(evecs[0] ** 2)
    for i, evec in enumerate(evecs):
        rho = evec ** 2
        label = f"State {i + 1}"
        if energies is not None:
            label += f", $E={np.round(energies[i], 5)}\\,\\hbar\\omega$"
        axes[i].plot(xs, rho, label=label, color="forestgreen")
        axes[i].set_xlabel("$z$")
        axes[i].set_ylabel(f"$|\\psi|^2$")
        axes[i].legend()
        axes[i].set_ylim(0, y_max + 0.001)  #
        axes[i].set_xlim(-x_max, x_max)
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def load_bench_times(path):
    file = open(path)

    head = next(file)
    ns_str = head.split(",")[:-1]
    ns = [int(n) for n in ns_str]

    times = []
    while True:
        try:
            line = next(file)
            times_str = line.split(",")[:-1]
            times_method = [float(time) for time in times_str]
            times.append(times_method)

        except StopIteration:
            break
    file.close()

    ns_np = np.asarray(ns)
    times_np = np.asarray(times)

    return ns_np, times_np


def plot_bench_times(ns, times, path):
    fig, ax = plt.subplots()
    for i, times_method in enumerate(times):
        ax.plot(ns, times_method, label=method_names[i])

    ax.set_xlabel("Number of bins $n$")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


    plt.close(fig)


def plot_bench_errs(ns, errs, path):
    fig, ax = plt.subplots()
    for i, errs_method in enumerate(errs):
        ax.plot(ns, errs_method, label=method_names[i])

    ax.set_xlabel("Number of bins $n$")
    ax.set_ylabel("GS Energy error ($\\hbar\\oemga$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


    plt.close(fig)


main()
