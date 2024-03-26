import matplotlib.pyplot as plt
import numpy as np

import physics

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


def plot_density(n, T, h_max, path):
    hs = np.linspace(0, 100e3, 100)
    rhos = physics.density_height(n = 100, T = 290, h_max = 100e3)

    hs_km = hs / 1e3
    rhos_scaled = rhos / physics.RHO_0

    fig, ax = plt.subplots()
    ax.plot(hs_km, rhos_scaled)
    #ax.set_ylim([0, 1])
    ax.set_xlim([0, h_max / 1e3])
    ax.set_xlabel("$h$ (km)")
    ax.set_ylabel("$\\rho / \\rho_0$")
    ax.set_yscale("log")

    fig.savefig(path)