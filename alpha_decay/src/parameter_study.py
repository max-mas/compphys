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
    dx = (a.coulomb_rng - a.R) / a.discr_steps
    density = a.calculate_density(n)

    V = a.piecewise_constant_potential()
    Vs = np.zeros(n)
    for i, x in enumerate(xs):
        if x < a.R:
            Vs[i] = V[0]
        elif x >= a.coulomb_rng:
            Vs[i] = V[-1]
        else:
            j = int((x - a.R) / dx) + 1
            Vs[i] = V[j]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    #ax.axvline(a.R, label="$R$", color="forestgreen")
    ax.axvline(a.coulomb_rng, label="$E_\\alpha > E_\\text{C}$", color="forestgreen", alpha=0.3)
    ax.plot(xs, density, label="$\\psi ^* \\psi$")
    ax2.plot(xs, Vs, label="$V$", color="orange", alpha=0.6)
    ax2.axhline(a.E_kin, label="$E_\\alpha$", color="red", alpha=0.6)
    #ax.set_ylim([0, 4])
    ax.set_yscale("log")
    ax.set_xlabel("$r$ (fm)")
    ax.set_ylabel("$\\psi ^* \\psi$ (not normalised)")
    ax2.set_ylabel("$V$ (MeV)")
    ax.legend(loc="upper right")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path)

# TODO numerics testing
    

# TODO performance scaling