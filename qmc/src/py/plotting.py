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


def main():
    alphas, Es, deltas = load_summary("../../results/2d/summary.txt")
    alphas_s, Es_s, deltas_s = load_summary("../../results/2d_spaced_larger_step/summary.txt")
    plot_summary_2d(alphas_s, Es_s, deltas_s,
                    alphas, Es, deltas, "../../plots/summary/summary_2d_test.pdf")

    alphas, Es, deltas = load_summary("../../results/1d/summary.txt")
    alphas_s, Es_s, deltas_s = load_summary("../../results/1d_spaced/summary.txt")
    plot_summary_1d(alphas_s, Es_s, deltas_s,
                    alphas, Es, deltas, "../../plots/summary/summary_1d_test.pdf")

    return 0


def load_summary(path):
    file = open(path)
    lines = file.readlines()
    n_trials = len(lines)

    alphas = np.zeros(n_trials)
    Es = np.zeros(n_trials)
    deltas = np.zeros(n_trials)

    for i in range(n_trials):
        nums = lines[i].split(",")
        alphas[i] = float(nums[0])
        Es[i] = float(nums[1])
        deltas[i] = float(nums[2])

    # sort
    p = np.argsort(alphas)
    alphas = alphas[p]
    Es = Es[p]
    deltas = deltas[p]

    return alphas, Es, deltas


def plot_summary_2d(alphas_spaced, Es_spaced, deltas_spaced,
                    alphas_search, Es_search, deltas_search, path):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    min_erg_index = np.argmin(Es_search)
    min_erg = Es_search[min_erg_index]
    min_alpha = alphas_search[min_erg_index]

    plot_a, = ax.plot(alphas_spaced, Es_spaced, color="forestgreen", alpha=0.5, label="$E$")
    plot_a_search = ax.scatter(alphas_search, Es_search, color="red", marker="*", label="$E$, golden search")
    plot_b, = ax2.plot(alphas_spaced, deltas_spaced, color="pink", alpha=0.5, ls="--", label="$\\sigma$")
    plot_b_search = ax2.scatter(alphas_search, deltas_search, color="navy", marker="o",
                                label="$\\sigma$, golden search")
    min_line = ax.axvline(min_alpha, color="red", ls="-.",
                          label=f"$\\min E = {np.round(min_erg, 3)}$, $\\alpha={np.round(min_alpha, 3)}$")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$E$ $(\\hbar\\omega)$")
    ax2.set_ylabel("$\\sigma(E)$ $(\\hbar\\omega)$")
    ax.set_xlim([np.min(alphas_spaced), np.max(alphas_spaced)])
    ax.set_ylim([2.99, np.max(Es_spaced)])
    ax2.set_ylim([0, np.max(deltas_spaced)])
    ax.grid()
    plots = [plot_a, plot_a_search, plot_b, plot_b_search, min_line]
    ax.legend(plots, [plot.get_label() for plot in plots])
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def plot_summary_1d(alphas_spaced, Es_spaced, deltas_spaced,
                    alphas_search, Es_search, deltas_search, path):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    min_erg_index = np.argmin(Es_search)
    min_erg = Es_search[min_erg_index]
    min_alpha = alphas_search[min_erg_index]

    plot_a, = ax.plot(alphas_spaced, Es_spaced, color="forestgreen", alpha=0.5, label="$E$")
    plot_a_search = ax.scatter(alphas_search, Es_search, color="red", marker="*", label="$E$, golden search")
    plot_b, = ax2.plot(alphas_spaced, deltas_spaced, color="pink", alpha=0.5, ls="--", label="$\\sigma$")
    plot_b_search = ax2.scatter(alphas_search, deltas_search, color="navy", marker="o",
                                label="$\\sigma$, golden search")
    min_line = ax.axvline(min_alpha, color="red", ls="-.",
                          label=f"$\\min E = {np.round(min_erg, 3)}$, $\\alpha={np.round(min_alpha, 3)}$")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$E$ $(\\hbar\\omega)$")
    ax2.set_ylabel("$\\sigma(E)$ $(\\hbar\\omega)$")
    ax.set_xlim([np.min(alphas_spaced), np.max(alphas_spaced)])
    ax.set_ylim([0.49, 1])
    ax2.set_ylim([0, np.max(deltas_spaced)])
    ax.grid()
    plots = [plot_a, plot_a_search, plot_b, plot_b_search, min_line]
    ax.legend(plots, [plot.get_label() for plot in plots])
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def harmonic_trial_2d_2p(alpha, x):
    lambd = 1.0
    x1 = x[0]
    y1 = x[1]
    x2 = x[2]
    y2 = x[3]

    s = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return np.exp(-(x1 ** 2 + y1 ** 2 + x2 ** 2 + y2 ** 2) / 2) * np.exp((lambd * s) / (1 + alpha * s))


main()
