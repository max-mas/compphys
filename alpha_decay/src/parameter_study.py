import alphadecay
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

# TODO plotting
def plot_density(a: alphadecay.Alphadecay, n, path):
    xs = np.linspace(0, a.coulomb_rng + 10, n, dtype=np.double)
    density = a.calculate_density(n)

    fig, ax = plt.subplots()
    ax.axvline(a.R, label="$R$", color="orange")
    ax.axvline(a.coulomb_rng, label="$E_\\alpha > E_\\text{c}$", color="orange")
    ax.plot(xs, density, label="$\\psi ^* \\psi$")
    #ax.set_ylim([0, 4])
    ax.set_yscale("log")
    ax.set_xlabel("$r$ (fm)")
    ax.set_ylabel("$\\psi ^* \\psi$ (not normalised)")
    ax.legend()

    fig.savefig(path)

# TODO numerics testing
    

# TODO performance scaling