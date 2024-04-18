import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline

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
                    alphas, Es, deltas, "../../plots/summary/summary_2d_lambda1.pdf")

    alphas, Es, deltas = load_summary("../../results/2d_lambda0/summary.txt")
    alphas_s, Es_s, deltas_s = load_summary("../../results/2d_spaced_lambda0/summary.txt")
    plot_summary_2d(alphas_s, Es_s, deltas_s,
                    alphas, Es, deltas, "../../plots/summary/summary_2d_lambda0.pdf")

    alphas, Es, deltas = load_summary("../../results/2d_lambda2/summary.txt")
    alphas_s, Es_s, deltas_s = load_summary("../../results/2d_spaced_lambda2/summary.txt")
    plot_summary_2d(alphas_s, Es_s, deltas_s,
                    alphas, Es, deltas, "../../plots/summary/summary_2d_lambda2.pdf")

    alphas, Es, deltas = load_summary("../../results/2d_lambda8/summary.txt")
    alphas_s, Es_s, deltas_s = load_summary("../../results/2d_spaced_lambda8/summary.txt")
    plot_summary_2d(alphas_s, Es_s, deltas_s,
                    alphas, Es, deltas, "../../plots/summary/summary_2d_lambda8.pdf")

    alphas, Es, deltas = load_summary("../../results/1d/summary.txt")
    alphas_s, Es_s, deltas_s = load_summary("../../results/1d_spaced/summary.txt")
    plot_summary_1d(alphas_s, Es_s, deltas_s,
                    alphas, Es, deltas, "../../plots/summary/summary_1d.pdf")

    plot_wf(0, 0, 0, 0.0, "../../plots/wf/wf_0_0_lambda0.pdf")
    plot_wf(1, 0, 0, 0.0, "../../plots/wf/wf_1_0_lambda0.pdf")
    plot_wf(1, 1, 0, 0.0, "../../plots/wf/wf_1_1_lambda0.pdf")
    plot_wf(0.2, 0.2, 0, 0.0, "../../plots/wf/wf_02_02_lambda0.pdf")

    plot_wf(0, 0, 0.382, 1.0, "../../plots/wf/wf_0_0_lambda1.pdf")
    plot_wf(1, 0, 0.382, 1.0, "../../plots/wf/wf_1_0_lambda1.pdf")
    plot_wf(1, 1, 0.382, 1.0, "../../plots/wf/wf_1_1_lambda1.pdf")
    plot_wf(0.2, 0.2, 0.382, 1.0, "../../plots/wf/wf_02_02_lambda1.pdf")

    plot_wf(0, 0, 0.641, 8.0, "../../plots/wf/wf_0_0_lambda8.pdf")
    plot_wf(1, 0, 0.641, 8.0, "../../plots/wf/wf_1_0_lambda8.pdf")
    plot_wf(1, 1, 0.641, 8.0, "../../plots/wf/wf_1_1_lambda8.pdf")
    plot_wf(0.2, 0.2, 0.641, 8.0, "../../plots/wf/wf_02_02_lambda8.pdf")


    lambdas, alphas, Es = load_sweep("../../results/lambda_sweep.txt")
    plot_sweep(lambdas, alphas, Es, "../../plots/sweep.pdf")

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

    # remove nans
    ind1 = np.argwhere(np.isnan(Es_spaced))
    ind2 = np.argwhere(np.isnan(deltas_spaced))
    ind = np.concatenate((ind1, ind2))
    unique_ind = np.unique(ind)
    alphas_spaced = np.delete(alphas_spaced, unique_ind)
    Es_spaced = np.delete(Es_spaced, unique_ind)
    deltas_spaced = np.delete(deltas_spaced, unique_ind)

    ind1 = np.argwhere(np.isnan(Es_search))
    ind2 = np.argwhere(np.isnan(deltas_search))
    ind = np.concatenate((ind1, ind2))
    unique_ind = np.unique(ind)
    alphas_search = np.delete(alphas_search, unique_ind)
    Es_search = np.delete(Es_search, unique_ind)
    deltas_search = np.delete(deltas_search, unique_ind)

    min_erg_index = np.argmin(Es_search)
    min_erg = Es_search[min_erg_index]
    min_alpha = alphas_search[min_erg_index]

    plot_a, = ax.plot(alphas_spaced, Es_spaced, color="forestgreen", alpha=0.5, label="$E$", lw=3)
    plot_a_search = ax.scatter(alphas_search, Es_search, color="red", marker="*", label="$E$, golden search", s=80)
    plot_b, = ax2.plot(alphas_spaced, deltas_spaced, color="pink", alpha=0.8, ls="--", label="$\\sigma$")
    plot_b_search = ax2.scatter(alphas_search, deltas_search, color="navy", marker="o",
                                label="$\\sigma$, golden search")
    min_line = ax.axvline(min_alpha, color="red", ls="-.",
                          label=f"$\\min E = {np.round(min_erg, 3)}$, $\\alpha={np.round(min_alpha, 3)}$")
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$E$ $(\\hbar\\omega)$")
    ax2.set_ylabel("$\\sigma(E)$ $(\\hbar\\omega)$")
    ax.set_xlim([np.min(alphas_spaced), np.max(alphas_spaced)])
    ax.set_ylim([0, 8])
    ax2.set_ylim([0, 1])
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


def plot_wf(x2, y2, alpha, lambd, path):
    n = 100
    z_max = 3
    wfs = np.zeros((n, n))
    xs = np.linspace(-z_max, z_max, n)
    ys = np.linspace(-z_max, z_max, n)

    for i in range(n):
        for j in range(n):
            z = np.asarray([xs[j], ys[i], x2, y2])
            wfs[i, j] = harmonic_trial_2d_2p(alpha, z, lambd)

    fig, ax = plt.subplots()
    cf = ax.contourf(xs, ys, wfs, 100, cmap="BuPu")
    for c in cf.collections:  # deprecated but idc
        c.set_edgecolor("face")
    ax.scatter([x2], [y2], marker="2", s=250, color="red", alpha=0.5, label="Position of particle 2, " +  f"$\\lambda={lambd}\\,\\hbar\\omega$, $\\alpha={alpha}$")
    ax.set_xlabel("$\\overline{x}_1$")
    ax.set_ylabel("$\\overline{y}_1$")
    ax.legend()
    fig.colorbar(cf, label="$\\psi$ (not normalised)")

    fig.savefig(path)
    plt.close(fig)

def harmonic_trial_2d_2p(alpha, x, lambd):
    x1 = x[0]
    y1 = x[1]
    x2 = x[2]
    y2 = x[3]

    s = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return np.exp(-(x1 ** 2 + y1 ** 2 + x2 ** 2 + y2 ** 2) / 2) * np.exp((lambd * s) / (1 + alpha * s))


def load_sweep(path):
    file = open(path)
    lines = file.readlines()
    n_trials = len(lines)

    lambdas = np.zeros(n_trials)
    alphas = np.zeros(n_trials)
    Es = np.zeros(n_trials)

    for i in range(n_trials):
        nums = lines[i].split(",")
        lambdas[i] = float(nums[0])
        alphas[i] = float(nums[1])
        Es[i] = float(nums[2])

    return lambdas, alphas, Es


def plot_sweep(lambdas, alphas, Es, path):

    #lambdas_continuous = np.linspace(0, 10, 100)
    #alpha_interp = CubicSpline(lambdas, alphas)(lambdas_continuous)
    #E_interp = CubicSpline(lambdas, Es)(lambdas_continuous)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    p1, = ax.plot(lambdas, Es, label="$E(\\lambda)$", color="forestgreen")
    p2, = ax2.plot(lambdas, alphas, label="$\\alpha(\\lambda)$", color="mediumblue", alpha=0.7, ls="--")
    plots = [p1, p2]
    ax.legend(plots, [plot.get_label() for plot in plots])
    ax.set_xlabel("$\\lambda$ $(\\hbar\\omega)$")
    ax.set_ylabel("$E$ $(\\hbar\\omega)$")
    ax2.set_ylabel("$\\alpha$")
    ax.grid()
    ax.set_xlim([lambdas[0], lambdas[-1]])
    ax.set_ylim([0, 8])
    ax2.set_ylim([0, 1])

    fig.savefig(path)
    plt.close(fig)

main()
