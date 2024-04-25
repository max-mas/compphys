import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.special
import seaborn as sns
from scipy.special import sph_harm, factorial, assoc_laguerre

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
    for l in range(5):
        for n in range(1, 10):
            try:
                path = "../../results/states/rad_wf_l" + str(l) + "_n" + str(n) + "_m0.txt"
                outpath1 = "../../plots/psis/l" + str(l) + "_n" + str(n) + ".pdf"
                outpath2 = "../../plots/psis_err/l" + str(l) + "_n" + str(n) + "l0.pdf"
                xs, psi_rad = load_state(path)
                plot_wf(xs, psi_rad, n, l, outpath1)
                xs, psi_rad = load_state(path)
                plot_wf_err(xs, psi_rad, n, l, outpath2)
            except FileNotFoundError:
                break


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
    sign = psi_rad[0] / np.abs(psi_rad[0])
    psi_rad[0] = psi_rad[1]
    #ax.plot(xs[1:], sign * psi_rad[1:] / np.max(np.abs(psi_rad[1:])),
    #        label="$R_{" + n_str + "," + l_str + "}$, numerical")
    ax.plot(xs, sign * psi_rad, label="$R_{" + n_str + "," + l_str + "}$, numerical", color="forestgreen")
    exact = psi_rad_exact(n, l, xs)
    ax.plot(xs, exact, label="$R_{" + n_str + "," + l_str + "}$, exact", color="mediumblue", ls="-.", alpha=0.7)


    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$R_{nl}$")
    ax.set_xlim(0, 20)
    ax.grid()
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_wf_err(xs, psi_rad, n, l, path):
    sns.set_palette("colorblind")

    fig, ax = plt.subplots()
    n_str = str(n)
    l_str = str(l)
    sign = psi_rad[0] / np.abs(psi_rad[0])
    psi_rad = sign * psi_rad / np.abs(psi_rad[0])
    exact = psi_rad_exact(n, l, xs)
    ax.plot(xs[1:], np.abs((exact - psi_rad) / np.abs(exact))[1:], color="indianred")

    #ax.scatter(xs, 0*xs, color="red", marker="*", label="Knots")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$|R_{nl}-R'_{nl}|/|R_{nl}|$")
    ax.set_xlim(0, 20)
    ax.set_yscale("log")
    ax.grid()
    #ax.legend()

    fig.savefig(path)
    plt.close(fig)


def psi_rad_exact(n, l, r):
    rho = 2 * r / n
    return np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * np.exp(-rho / 2) * rho ** l \
        * assoc_laguerre(rho, n - l - 1, 2 * l + 1)


main()
