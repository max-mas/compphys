import alphadecay
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm

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

true_half_lifes = {(92, 238): 1.41e17, # (s)
                   (92, 235): 2.22e16,
                   (90, 232): 4.4e17,
                   (86, 222): 330200,
                   (84, 212): 2.94e-7
                   }

true_Es = { (92, 238): 4.270, # (MeV)
            (92, 235): 4.678,
            (90, 232): 4.082,
            (86, 222): 5.590,
            (84, 212): 8.954,
            }
            

names = {   (92, 238): "U 238", 
            (92, 235): "U 235",
            (90, 232): "Th 232",
            (86, 222): "Rn 222", 
            (84, 212): "Po 212"
            }

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

def test_bin_dependence(Z, A, path, num_trials=10, bin_min=1, bin_max=3.5):
    rng = alphadecay.get_coulomb_range(Z, A)
    bins = np.logspace(bin_min, bin_max, num_trials, dtype=np.int64)
    t_12s = np.empty(num_trials, dtype=np.double)
    for i, bin in tqdm(enumerate(bins)):
        a = alphadecay.Alphadecay(Z, A, rng, bin)
        t_12s[i] = a.get_half_life()
        del a

    fig, ax = plt.subplots()
    ax.plot(bins, t_12s, label=f"Numerical result: relative error ${np.round(np.abs(t_12s[-1] - true_half_lifes[(Z, A)]) / true_half_lifes[(Z, A)] * 100, 2)}$\\%")
    ax.axhline(true_half_lifes[(Z, A)], color="orange", alpha=0.5, label="Experimental value")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of bins")
    ax.set_ylabel("Half life $t_{1/2}$ (s)")
    ax.legend()

    fig.savefig(path)


def test_R_dependence(Z, A, path, bins=1000, num_trials=10, R_factor_min=1, R_factor_max=2):
    rng = alphadecay.get_coulomb_range(Z, A)
    Rs = np.linspace(R_factor_min, R_factor_max, num_trials, dtype=np.double)
    t_12s = np.empty(num_trials, dtype=np.double)
    for i, R in tqdm(enumerate(Rs)):
        a = alphadecay.Alphadecay(Z, A, rng, bins, R_factor=R)
        t_12s[i] = a.get_half_life()
        del a

    true_half_life = true_half_lifes[(Z, A)]
    
    def intersect(x):
        return CubicSpline(Rs, t_12s)(x) - CubicSpline([Rs[0], Rs[-1]], [true_half_life, true_half_life])(x)
    R_intersect = fsolve(intersect, x0=1.3)

    fig, ax = plt.subplots()
    ax.plot(Rs, t_12s, label=f"Numerical result")
    ax.axhline(true_half_lifes[(Z, A)], color="orange", alpha=0.5, label="Experimental value")
    ax.axvline(R_intersect, color="forestgreen", alpha=0.5, label=f"Intersect: $\\tilde R = {np.round(R_intersect[0], 4)}$")
    ax.set_yscale("log")
    ax.set_xlabel("$R$ scaling factor")
    ax.set_ylabel("Half life $t_{1/2}$ (s)")
    ax.legend()

    fig.savefig(path)

def test_bin_dependence(Z, A, path, num_trials=10, bin_min=1, bin_max=3.5):
    rng = alphadecay.get_coulomb_range(Z, A)
    bins = np.logspace(bin_min, bin_max, num_trials, dtype=np.int64)
    t_12s = np.empty(num_trials, dtype=np.double)
    for i, bin in tqdm(enumerate(bins)):
        a = alphadecay.Alphadecay(Z, A, rng, bin)
        t_12s[i] = a.get_half_life()
        del a

    fig, ax = plt.subplots()
    ax.plot(bins, t_12s, label=f"Numerical result: relative error ${np.round(np.abs(t_12s[-1] - true_half_lifes[(Z, A)]) / true_half_lifes[(Z, A)] * 100, 2)}$\\%")
    ax.axhline(true_half_lifes[(Z, A)], color="orange", alpha=0.5, label="Experimental value")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of bins")
    ax.set_ylabel("Half life $t_{1/2}$ (s)")
    ax.legend()

    fig.savefig(path)


def test_V0_dependence(Z, A, path, bins=1000, num_trials=30, V0_min=-150, V0_max=-120):
    rng = alphadecay.get_coulomb_range(Z, A)
    Vs = np.linspace(V0_min, V0_max, num_trials, dtype=np.double)
    t_12s = np.empty(num_trials, dtype=np.double)
    for i, V0 in tqdm(enumerate(Vs)):
        a = alphadecay.Alphadecay(Z, A, rng, bins, V0=V0)
        t_12s[i] = a.get_half_life()
        del a

    true_half_life = true_half_lifes[(Z, A)]
    
    def intersect(x):
        return CubicSpline(Vs, t_12s)(x) - CubicSpline([Vs[0], Vs[-1]], [true_half_life, true_half_life])(x)
    V0_intersect = fsolve(intersect, x0=1.3)

    fig, ax = plt.subplots()
    ax.plot(Vs, t_12s, label=f"Numerical result")
    ax.axhline(true_half_lifes[(Z, A)], color="orange", alpha=0.5, label="Experimental value")
    #ax.axvline(V0_intersect, color="forestgreen", alpha=0.5, label=f"Intersect: $\\tilde R = {np.round(V0_intersect[0], 4)}$")
    ax.set_yscale("log")
    ax.set_xlabel("$V_0$ (MeV)")
    ax.set_ylabel("Half life $t_{1/2}$ (s)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)

cols = {(92, 238): "forestgreen", 
        (92, 235): "slateblue",
        (90, 232): "lightcoral",
        (86, 222): "darkorange", 
        (84, 212): "darkturquoise"
        }

def plot_t12s(Zs, As, t12s, path):
    fig, ax = plt.subplots()
    for Z, A, t12 in zip(Zs, As, t12s):
        ax.scatter(A, t12, label=names[(Z, A)] + " numerical", color=cols[(Z, A)])
        ax.scatter(A, true_half_lifes[(Z, A)], label=names[(Z, A)] + " experimental", color=cols[(Z, A)], alpha=0.5, marker="v")
    
    ax.set_xlabel("Mass number $A$")
    ax.set_ylabel("Half life $t_{1/2}$ (s)")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(path)

def plot_E(Zs, As, Es, path):
    fig, ax = plt.subplots()
    for Z, A, E in zip(Zs, As, Es):
        ax.scatter(A, E, label=names[(Z, A)] + " numerical", color=cols[(Z, A)])
        ax.scatter(A, true_Es[(Z, A)], label=names[(Z, A)] + " experimental", color=cols[(Z, A)], alpha=0.5, marker="v")
        ax.scatter(A, E - true_Es[(Z, A)])
    
    ax.set_xlabel("Mass number $A$")
    ax.set_ylabel("$E_{\\text{kin}}$ (MeV)")
    ax.legend()

    fig.savefig(path)
    


# TODO performance scaling