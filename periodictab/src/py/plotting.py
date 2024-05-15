import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.special
import seaborn as sns
from scipy.special import sph_harm, factorial, assoc_laguerre
import pandas as pd
from tqdm.auto import tqdm

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
    """
    for i in tqdm(range(2, 20)):
        xs, V1 = load_potential("../../results/atoms/" + str(i) + "/direct_ne_" + str(i) + ".txt")
        xs, V2 = load_potential("../../results/atoms/" + str(i) + "/exchange_ne_" + str(i) + ".txt")

        plot_potential(xs, [V1, V2, V1+V2], ["dir", "ex", "tot"], "../../plots/potentials/potential_" + str(i) + ".pdf")
        plot_potential(xs, [V1, V2, V1+V2], ["dir", "ex", "tot"], "../../plots/potentials/png/potential_" + str(i) + ".png")

        xs, rho2 = load_potential("../../results/atoms/" + str(i) + "/rho_ne_" + str(i) + ".txt")
        xs, rho1 = load_potential("../../results/atoms/" + str(i) + "/rho_ne_" + str(i-1) + ".txt")
        plot_density(xs, [rho2, rho1], [i, i-1], "../../plots/density/density_" + str(i) + ".pdf")
        plot_density(xs, [rho2, rho1], [i, i-1], "../../plots/density/png/density_" + str(i) + ".png")
    """
    Zs, E = load_ionerg("../../results/ionergs.txt")
    plot_ionergs(Zs, E, "../../plots/ionerg.pdf")
    plot_ionergs_err(Zs, E, "../../plots/ionerg_err.pdf")

    xs, V1 = load_potential("../../results/atoms/55/direct_ne_55.txt")
    xs, V2 = load_potential("../../results/atoms/55/exchange_ne_55.txt")

    plot_potential(xs, [V1, V2, V1+V2], ["dir", "ex", "tot"], "../../plots/potentials/potential_55.pdf")

    xs, rho2 = load_potential("../../results/atoms/55/rho_ne_55.txt")
    xs, rho1 = load_potential("../../results/atoms/55/rho_ne_54.txt")
    plot_density(xs, [rho2, rho1], [55, 54], "../../plots/density/density_55.pdf")

    return 0


def load_ionerg(path):
    file = open(path)
    lines = file.readlines()
    n_Zs = len(lines)

    Zs = np.zeros(n_Zs)
    E = np.zeros(n_Zs)

    for i in range(n_Zs):
        nums = lines[i].split(",")
        Zs[i] = float(nums[0])
        E[i] = float(nums[1])

    return Zs, E


def load_potential(path):
    file = open(path)
    lines = file.readlines()
    n_xs = len(lines)

    xs = np.zeros(n_xs)
    V = np.zeros(n_xs)

    for i in range(n_xs):
        nums = lines[i].split(",")
        xs[i] = float(nums[0])
        V[i] = float(nums[1])

    return xs, V


def plot_potential(xs, Vs, types, path, rmax=10):
    sns.set_palette("colorblind")
    fig, ax = plt.subplots()

    for V, type in zip(Vs, types):
        V[0] = V[1] - (xs[1] - xs[0]) * (V[2] - V[1]) / (xs[1] - xs[0])
        ax.plot(xs, V, label="$V^\\text{" + type + "}_\\text{int}$")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$V(r)$ $(\\text{H}/e)$")

    ax.set_xlim([0, rmax])

    ax.legend()
    ax.grid()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_density(xs, rhos, elnums, path, xmax=5):
    sns.set_palette("colorblind")
    fig, ax = plt.subplots()

    for rho, elnum in zip(rhos, elnums):
        plotarr = 4 * np.pi * xs**2 * rho
        plotarr[0] = 0
        ax.plot(xs, plotarr, label=str(elnum) + " electrons")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$4\\pi r^2 \\rho(r)$ $(e/a_0)$")

    ax.set_xlim([0, xmax])

    ax.legend()
    ax.grid()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ionergs(Zs, E, path):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("../../results/exact.csv")
    selection = df[["At. num", "Ionization Energy (Hartree)"]]
    exact = selection.to_numpy()
    fig, ax = plt.subplots()
    H = 27.211386246

    ax.plot(Zs, H * E, label="Numerical", marker="o", color="mediumblue")
    ax.plot(exact[:,0], H * exact[:,1], ls="--", color="indianred", label="Experimental value", marker="*", alpha=0.8)

    ax.set_xlabel("$Z$")
    ax.set_ylabel("$E_\\text{ion}$ (eV)")

    ax.set_xlim([0, 20])
    ax.set_ylim([0, 25])

    ax.legend()
    ax.grid()
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def plot_ionergs_err(Zs, E, path):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("../../results/exact.csv")
    selection = df[["At. num", "Ionization Energy (Hartree)"]]
    exact = selection.to_numpy()
    fig, ax = plt.subplots()
    H = 27.211386246

    ax.plot(Zs[0:len(exact[:,1])], np.abs((E[0:len(exact[:,1])]-exact[:,1])) / exact[:,1], marker="o", color="forestgreen", ls="--")    

    ax.set_xlabel("$Z$")
    ax.set_ylabel("$\\Delta E_\\text{ion} / E_\\text{ion}$")

    ax.set_xlim([0, 120])
    ax.set_ylim([1e-4, 1e4])
    ax.set_yscale("log")

    ax.grid()
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)



main()