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


def main():
    #evs = load_evals("../../results/evs/evs_test.txt")
    #plot_evals(evs, "../../plots/evals/evals.pdf")
    evecs = load_evecs("../../results/evecs/evecs_test.txt", 5)
    plot_evecs(evecs, 10, "../../plots/states/states_test.pdf")



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

    ax.plot(range(n), evals, label=f"Numerical eigenvalues, $n={n}$")
    ax.plot(range(n), range(n), color="forestgreen", alpha=0.6, label="Theoretical eigenvalues")
    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel("$i$-th Energy ($\\hbar \\omega$)")
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


def plot_evecs(evecs, x_max, path):
    n = len(evecs[0])
    xs = np.linspace(-x_max, x_max, n)
    fig, ax = plt.subplots()
    for i, evec in enumerate(evecs):
        ax.plot(xs, evec, label=f"State {i}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\psi$")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


main()
