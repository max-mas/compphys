import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# For nicer plots, requires a fairly full TeXlive install
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}\usepackage{braket}",
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 24,
    "figure.titlesize": 24,
    "font.family": "serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})

# globals
x_max = 20
method_names = ["3 point full", "3 point tridiag full", "3 point tridiag parity full",
                "5 point full", "5 point parity full", "5 point parity inverse it. (GS only)"]


def main():
    #plot_potentials("../../plots/pot.pdf")

    #evs = load_evals("../../results/evs/evs_full_4000.txt")
    #plot_evals(evs, "../../plots/evals/evals.pdf")
    #plot_eval_error(evs, "../../plots/evals/evals_err.pdf")
    #evecs = load_evecs("../../results/evecs/evecs_full_4000.txt", 4000)
    #evecs_invit = load_evecs("../../results/evecs/evecs_invit.txt", 8)
    #plot_evecs(evecs[250:258],  "../../plots/states/states_low_middle.pdf", 20, energies=None)
    #plot_evec_deviation(evecs, evecs_invit, "../../plots/states/deviation.pdf")

    #plot_evec_orthonormality(evecs, "../../plots/orthonormal/orthonormality.pdf")

    #evals_0 = load_evals("../../results/evs/evs_full_4000.txt")
    #evals_1 = load_evals("../../results/evs/evs_small_bump.txt")
    #evals_2 = load_evals("../../results/evs/evs_large_bump.txt")
    evecs_0 = load_evecs("../../results/evecs/evecs_full_4000.txt", 4)
    evecs_1 = load_evecs("../../results/evecs/evecs_small_bump.txt", 4)
    evecs_2 = load_evecs("../../results/evecs/evecs_large_bump.txt", 4)
    plot_evecs_comparison([evecs_0, evecs_1, evecs_2], "../../plots/states/states_comp.pdf",
                          ["harmonic", "small perturbation", "large pertubation"], 6)
    #plot_evals_bump([evals_1, evals_2], ["small perturbation", "large pertubation"],
    #                "../../plots/evals/evals_bump.pdf", 100)

    #pos2 = load_evals("../../results/squared_positions.txt")
    #plot_pos2(pos2, "../../plots/pos2.pdf")
    #nums = load_evals("../../results/occupation.txt")
    #plot_num(nums, "../../plots/occupation.pdf")

    """
    evs = load_evals("../../results/evs/evs_morse_1000.txt")
    plot_evals(evs, "../../plots/evals/evals_morse.pdf")
    evecs = load_evecs("../../results/evecs/evecs_morse_1000.txt", 10)
    plot_evecs(evecs,  "../../plots/states/states_morse.pdf", 20, energies=evs)
    """

    #ns, times = load_bench_times("../../results/bench/bench_time_avgs_5.txt")
    #plot_bench_times(ns, times, "../../plots/bench/bench.pdf")
    #ns, errs = load_bench_times("../../results/bench/bench_gs_erg_err.txt")
    #plot_bench_errs(ns, errs, "../../plots/bench/bench_err.pdf")

    """
    xs1, gs_errs = load_errs_xmax_bench("../../results/bench/bench_xmax_err_state0_n_1000.txt")
    xs2, excited_errs = load_errs_xmax_bench("../../results/bench/bench_xmax_err_state50_n_1000.txt")
    xs3, excited_errs_bigger_n = load_errs_xmax_bench("../../results/bench/bench_xmax_err_state50_n_2500.txt")
    plot_xmax_bench_errs([xs1, xs2, xs3],
                         [np.take(gs_errs, [4, 5], axis=0),
                              np.take(excited_errs, [4], axis=0),
                              np.take(excited_errs_bigger_n, [4], axis=0)],
                         [0, 50, 50],
                         [1000, 1000, 2500],
                         [["5 point parity full", "5 point parity inverse it. (GS only)"],
                          ["5 point parity full"],
                          ["5 point parity full"]],
                         "../../plots/bench/bench_xmax_err.pdf")
    """

    return 0


def harmonic_bump(z, c1, c2):
    return 0.5 * z**2 + c1 * np.exp(-c2*z**2)

def plot_pos2(pos2s, path):
    n = len(pos2s)
    fig, ax = plt.subplots()
    ax.plot(np.arange(n), pos2s, label=f"$n={n}$, " + "$z_\\text{max}=" + f"{x_max}$")
    ax.legend()
    ax.set_ylabel("$\\braket{z^2}_i$")
    ax.set_xlabel("State label $i$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.savefig(path)
    plt.close(fig)

def plot_num(nums, path):
    n = len(nums)
    fig, ax = plt.subplots()
    ax.plot(np.arange(n), nums, label=f"$n={n}$, " + "$z_\\text{max}=" + f"{x_max}$")
    ax.legend()
    ax.set_ylabel("$\\braket{a^\\dag a}_i$")
    ax.set_xlabel("State label $i$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([1, np.max(nums)])

    fig.savefig(path)
    plt.close(fig)


def plot_potentials(path):
    n = 4000
    xs = np.linspace(-x_max/4, x_max/4, n)
    pot_0 = harmonic_bump(xs, 0, 0)
    pot_1 = harmonic_bump(xs, 1, 10)
    pot_2 = harmonic_bump(xs, 5, 10)
    fig, ax = plt.subplots()
    ax.plot(xs, pot_0, label="Harmonic")
    ax.plot(xs, pot_1, label="Small perturbation")
    ax.plot(xs, pot_2, label="Larger perturbation")
    ax.legend()
    ax.set_xlabel("$z$")
    ax.set_ylabel("$V(z)$")
    ax.set_ylim([0, np.max(pot_0)])

    fig.savefig(path)
    plt.close(fig)



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


def plot_eval_error(evals, path):
    fig, ax = plt.subplots()
    n = len(evals)

    ax.plot(np.arange(n) + 1, np.abs(evals - np.arange(n) - 0.5), label=f"Numerical error, $n={n}$, " + "$z_\\text{max}=" + f"{x_max}$")
    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel("Error of $i$-th eigenvalue ($\\hbar \\omega$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1, n + 1)
    ax.legend()

    fig.savefig(path)
    plt.close(fig)

def plot_evals_general(evals, path):
    fig, ax = plt.subplots()
    n = len(evals)

    ax.plot(np.arange(n) + 1, evals, label=f"Numerical eigenvalues, $n={n}$, " + "$z_\\text{max}=" + f"{x_max}$")
    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel("$i$-th Energy ($\\hbar \\omega$)")
    #ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1, n + 1)
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_evals_bump(evals_arr, potential_names, path, i_max=None):
    fig, ax = plt.subplots()
    n = len(evals_arr[0])
    if i_max is None:
        i_max = n

    for i, evals in enumerate(evals_arr):
        ax.plot(np.arange(i_max) + 1, np.abs(evals[:i_max] - (np.arange(i_max) + 0.5)), label=potential_names[i])
    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel("$|E_i - \\hbar\\omega (i + 1/2)|$")
    #ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1, i_max + 1)
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


def plot_evec_orthonormality(evecs, path):
    n = len(evecs)
    orthonormality_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            orthonormality_matrix[i, j] = np.abs(np.dot(evecs[i], evecs[j]))

    fig, ax = plt.subplots()
    ms = ax.matshow(orthonormality_matrix, norm=matplotlib.colors.LogNorm(), cmap="PuBuGn")
    fig.colorbar(ms)
    ax.set_xlabel("$i$")
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_ylabel("$j$")
    fig.suptitle("$O_{ij} = \\left|\\braket{i|j}\\right|$")

    fig.savefig(path)
    plt.close(fig)


def plot_evecs(evecs, path, x_max_plot, energies=None):
    n = len(evecs[0])
    n_evecs = len(evecs)
    xs = np.linspace(-x_max, x_max, n)
    fig, axes = plt.subplots(nrows=int(n_evecs / 2), ncols=2, figsize=(9, n_evecs))
    axes = axes.flatten()
    # xmin_plot = -xmax_plot = - x_max + i * h
    # i = (-xmax_plot + x_max)/h
    h = 2*x_max / n
    min_index = int( (x_max - x_max_plot) / h )
    max_index = int( (x_max + x_max_plot) / h )

    y_max = np.max(np.abs(evecs[0]))
    for i, evec in enumerate(evecs):
        if evec[int(n/2)] < 0 :
            evec *= -1
        rho = evec# ** 2
        label = f"State {i + 1}"
        if energies is not None:
            label += f", $E={np.round(energies[i], 5)}\\,\\hbar\\omega$"
        axes[i].plot(xs[min_index:max_index], rho[min_index:max_index], label=label, color="forestgreen")
        axes[i].set_xlabel("$z$")
        axes[i].set_ylabel(f"$\\psi$")
        axes[i].legend()
        axes[i].set_ylim(-y_max - 0.001, y_max + 0.001)  #
        axes[i].set_xlim(-x_max_plot, x_max_plot)
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def plot_evecs_comparison(evecs_arr, path, potential_names, x_max_plot):
    n = len(evecs_arr[0][0])
    n_evecs = len(evecs_arr[0])
    xs = np.linspace(-x_max, x_max, n)
    fig, axes = plt.subplots(nrows=int(n_evecs / 2), ncols=2, figsize=(9, 2*n_evecs))
    axes = axes.flatten()
    # xmin_plot = -xmax_plot = - x_max + i * h
    # i = (-xmax_plot + x_max)/h
    h = 2*x_max / n
    min_index = int( (x_max - x_max_plot) / h )
    max_index = int( (x_max + x_max_plot) / h )

    y_max = np.max(np.abs(evecs_arr[0][0]))
    for i in range(n_evecs):
        for j in range(len(evecs_arr)):
            evec = evecs_arr[j][i]
            if evec[int(n/2)] < 0:
                evec *= -1
            rho = evec# ** 2
            label = f"State {i + 1}, " + potential_names[j]
            axes[i].plot(xs[min_index:max_index], rho[min_index:max_index], label=label)
        axes[i].set_xlabel("$z$")
        axes[i].set_ylabel(f"$\\psi$")
        axes[i].legend()
        axes[i].set_ylim(-y_max - 0.001, y_max + 0.001)  #
        axes[i].set_xlim(-x_max_plot, x_max_plot)
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def plot_evec_deviation(evecs1, evecs2, path):
    n = len(evecs1[0])
    xs = np.linspace(-x_max, x_max, n)
    fig, ax = plt.subplots()
    for i, (evec1, evec2) in enumerate(zip(evecs1, evecs2)):
        if evec1[int(n/2)] < 0:
            evec1 *= -1
        if evec2[int(n/2)] < 0:
            evec2 *= -1
        ax.plot(xs, np.abs(evec1 - evec2), label=f"State ${i}$")
    ax.set_ylabel("Deviation $|\\psi_{i,\\text{ED}} - \\psi_{i,\\text{it.}}|$")
    ax.set_xlabel("z")
    ax.set_yscale("log")
    ax.legend()

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


def load_errs_xmax_bench(path):
    file = open(path)

    head = next(file)
    xmaxs_str = head.split(",")[:-1]
    xmaxs = [float(xmax) for xmax in xmaxs_str]

    errs = []
    while True:
        try:
            line = next(file)
            errs_str = line.split(",")[:-1]
            errs_method = [float(time) for time in errs_str]
            errs.append(errs_method)

        except StopIteration:
            break
    file.close()

    xmaxs_np = np.asarray(xmaxs)
    errs_np = np.asarray(errs)

    return xmaxs_np, errs_np


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
    ax.set_ylabel("GS energy error ($\\hbar\\omega$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


    plt.close(fig)


def plot_xmax_bench_errs(xmaxs_all, errs_states, state_ns, bins, method_names, path):
    fig, ax = plt.subplots()
    for xmaxs, errs_state, n, bin, method_name in zip(xmaxs_all, errs_states, state_ns, bins, method_names):
        for i in range(len(errs_state)):
            ax.plot(xmaxs, errs_state[i], label=method_name[i]+f", state {n}, {bin} bins")

    ax.set_xlabel("$z_\\text{max}$")
    ax.set_ylabel("Energy error ($\\hbar\\omega$)")
    ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


    plt.close(fig)


main()
