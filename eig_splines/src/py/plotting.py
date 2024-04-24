import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from scipy.special import sph_harm

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
    xs, psi_rad = load_state("../../results/states/rad_wf_n1_l0_m0.txt")
    plot_wf(xs, psi_rad, 1, 0, "../../plots/psis/test1.pdf")
    xs, psi_rad = load_state("../../results/states/rad_wf_n1_l0_m0.txt")
    plot_wf_err(xs, psi_rad, 1, 0, "../../plots/psis_err/test1.pdf")

    xs, psi_rad = load_state("../../results/states/rad_wf_n2_l0_m0.txt")
    plot_wf(xs, psi_rad, 2, 0, "../../plots/psis/test2.pdf")
    xs, psi_rad = load_state("../../results/states/rad_wf_n2_l0_m0.txt")
    plot_wf_err(xs, psi_rad, 2, 0, "../../plots/psis_err/test2.pdf")

    xs, psi_rad = load_state("../../results/states/rad_wf_n3_l0_m0.txt")
    plot_wf(xs, psi_rad, 3, 0, "../../plots/psis/test3.pdf")
    xs, psi_rad = load_state("../../results/states/rad_wf_n3_l0_m0.txt")
    plot_wf_err(xs, psi_rad, 3, 0, "../../plots/psis_err/test3.pdf")
    return 0


def load_state(path):
    file = open(path)
    lines = file.readlines()
    n_xs = len(lines)

    xs = np.zeros(n_xs)
    psi_rad = np.zeros(n_xs)

    for i in range(n_xs):
        nums = lines[i].split(",")
        xs[i] = float(nums[0])
        psi_rad[i] = float(nums[1])

    return xs, psi_rad


def plot_wf(xs, psi_rad, n, l, path):
    sns.set_palette("colorblind")

    fig, ax = plt.subplots()
    n_str = str(n)
    l_str = str(l)
    sign = psi_rad[1] / np.abs(psi_rad[1])
    #ax.plot(xs[1:], sign * psi_rad[1:] / np.max(np.abs(psi_rad[1:])),
    #        label="$R_{" + n_str + "," + l_str + "}$, numerical")
    ax.plot(xs[1:], sign*psi_rad[1:]/np.abs(psi_rad[1]),
            label="$R_{" + n_str + "," + l_str + "}$, numerical")
    if n == 1:
        exact = np.exp(-xs[1:])
        exact /= exact[0]
        ax.plot(xs[1:], exact, label="$R_{" + n_str + "," + l_str + "}$, exact")
    elif n == 2 and l == 0:
        exact = (2 - xs[1:]) * np.exp(-xs[1:]/2) / 2
        exact /= exact[0]
        ax.plot(xs[1:], exact, label="$R_{" + n_str + "," + l_str + "}$, exact")
    elif n == 3 and l == 0:
        exact = (3 - 2*xs[1:] + 2/9 * xs[1:]**2) * np.exp(-xs[1:]/3) / 3
        exact /= exact[0]
        ax.plot(xs[1:], exact, label="$R_{" + n_str + "," + l_str + "}$, exact")
    #ax.scatter(xs, 0*xs, color="red", marker="*", label="Knots")


    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$R_{nl}$")
    ax.set_xlim(0, 20)
    #
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_wf_err(xs, psi_rad, n, l, path):
    sns.set_palette("colorblind")

    fig, ax = plt.subplots()
    n_str = str(n)
    l_str = str(l)
    sign = psi_rad[1] / np.abs(psi_rad[1])
    psi_rad = sign*psi_rad[1:]/np.abs(psi_rad[1])
    if n == 1:
        exact = np.exp(-xs[1:])
        exact /= exact[0]
        ax.plot(xs[1:], np.abs(((exact - psi_rad)/np.abs(exact))))
    elif n == 2 and l == 0:
        exact = (2 - xs[1:]) * np.exp(-xs[1:]/2) / 2
        exact /= exact[0]
        ax.plot(xs[1:], np.abs(((exact - psi_rad)/np.abs(exact))))
    elif n == 3 and l == 0:
        exact = (3 - 2*xs[1:] + 2/9 * xs[1:]**2) * np.exp(-xs[1:]/3) / 3
        exact /= exact[0]
        ax.plot(xs[1:], np.abs(((exact - psi_rad)/np.abs(exact))))

    #ax.scatter(xs, 0*xs, color="red", marker="*", label="Knots")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$|R_{nl}-R'_{nl}|/|R_{nl}|$")
    ax.set_xlim(0, 20)
    ax.set_yscale("log")
    #ax.legend()

    fig.savefig(path)
    plt.close(fig)

main()
