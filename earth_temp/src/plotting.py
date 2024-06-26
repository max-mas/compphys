import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import tqdm
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


def plot_density(n, Ts, h_max, path):
    hs = np.linspace(0, h_max, n)

    fig, ax = plt.subplots()
    for T in Ts:
        rhos = physics.density_height(n = n, T = T, h_max = h_max)

        hs_km = hs / 1e3
        rhos_scaled = rhos / physics.RHO_0

        
        ax.plot(hs_km, rhos_scaled, label=f"$\\rho$, $T={T}$ K")
    #ax.set_ylim([0, 1])
    ax.set_xlim([0, h_max / 1e3])
    ax.set_xlabel("$h$ (km)")
    ax.set_ylabel("$\\rho / \\rho_0$")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_visible_intensity(alphas, n, Ts, h_max, path):
    hs = np.linspace(0, h_max, n)

    fig, ax = plt.subplots()
    for T in Ts:
        for alpha in alphas:
            Is = physics.visible_intensity(alpha, n, T, h_max)

            hs_km = hs / 1e3
            
            ax.plot(hs, Is, label=f"$I$, $T={T}$ K, $\\alpha={np.round(alpha, 5)}$ / m")        
        
    ax.set_xlabel("$h$ (km)")
    ax.set_ylabel("$I/I_0$")
    ax.set_ylim([0, 1])
    #ax.set_xlim([0, h_max/1e3])
    ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_visible_absorption(alphas, n, T, h_max, path, surface_albedos=[0]): # vary albedo
    
    fig, ax = plt.subplots()
    for surface_albedo in surface_albedos:
        Ts = physics.temps_visible_light_absorption(alphas, n, T, h_max, surface_albedo=surface_albedo)
        ax.plot(alphas, Ts - 273.15, label=f"$T$, $\\epsilon={surface_albedo}$")

    ax.set_xlabel("$\\alpha$ ($1/\\text{m}$)")
    ax.set_ylabel("Surface temperature $T$ (° C)")
    ax.set_xlim([alphas[0], alphas[-1]])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 6))
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_N(alpha_V, alpha_IR, ns, T, h_max, path, surface_albedo=0):
    Ts = physics.temps_full_model_vary_N(alpha_V, alpha_IR, ns, T, h_max, surface_albedo=0)

    fig, ax = plt.subplots()
    ax.plot(ns, np.abs(Ts-Ts[-1]), label="$\\alpha_\\text{IR}=" + f"{alpha_IR}$ /m" + ", $T_{n_\\text{max}}" + f"={np.round(Ts[-1], 1)}$° C")
    ax.set_xlabel("Number of layers $n$")
    ax.set_ylabel("Deviation from final temperature $\\Delta T$ (° C)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_alpha_IR(alpha_V, alphas_IR, n, T, h_max, path, surface_albedo=0, log=False):
    Ts = physics.temps_full_model_vary_alpha_IR(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)

    fig, ax = plt.subplots()
    if log:
        ax.plot(alphas_IR, np.abs(Ts-15), label=f"$n={n}$")
        ax.set_xlabel("$\\alpha_\\text{IR}$ (1/m)")
        ax.set_ylabel("$|T-15\\text{° C}|$ (° C)")
        #ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.plot(alphas_IR, Ts, label=f"$n={n}$")
        ax.set_xlabel("$\\alpha_\\text{IR}$ (1/m)")
        ax.set_ylabel("$T$ (° C)")
        ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_sweeps(alpha_V, alpha_IR, n, T, h_max, n_sweeps_arr, path, surface_albedo=0):
    Ts = physics.temps_full_model_vary_sweeps(alpha_V, alpha_IR, n, T, h_max, n_sweeps_arr, surface_albedo=0)

    fig, ax = plt.subplots()
    ax.plot(n_sweeps_arr, np.abs(Ts - Ts[-1]), label=f"$n={n}$, " + "$\\alpha_\\text{IR}" + f"={alpha_IR}$ /m")
    ax.set_xlabel("Number of sweeps")
    ax.set_ylabel("Deviation from final temperature $\\Delta T$ (° C)")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_mat_N(alpha_V, alpha_IR, ns, T, h_max, path, surface_albedo=0):
    Ts = physics.temps_full_model_vary_N_mat(alpha_V, alpha_IR, ns, T, h_max, surface_albedo=0)

    fig, ax = plt.subplots()
    ax.plot(ns, Ts, label="$\\alpha_\\text{IR}=" + f"{alpha_IR}$ /m" + ", $T_{n_\\text{max}}" + f"={Ts[-1]}$° C")
    ax.set_xlabel("Number of layers $n$")
    ax.set_ylabel("Surface temperature $T$ (° C)")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_mat_alpha_IR(alpha_V, alphas_IR, n, T, h_max, path, surface_albedo=0):
    Ts = physics.temps_full_model_vary_alpha_IR_mat(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)

    fig, ax = plt.subplots()
    ax.plot(alphas_IR, Ts, label=f"$n={n}$")
    ax.set_xlabel("$\\alpha_\\text{IR}$ (1/m)")
    ax.set_ylabel("Surface temperature $T$ (° C)")
    #ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_comp_N(alpha_V, alpha_IR, ns, T, h_max, path, surface_albedo=0):
    Ts_mat = physics.temps_full_model_vary_N_mat(alpha_V, alpha_IR, ns, T, h_max, surface_albedo=0)
    Ts_it = physics.temps_full_model_vary_N(alpha_V, alpha_IR, ns, T, h_max, surface_albedo=0)

    fig, ax = plt.subplots()
    ax.plot(ns, Ts_mat, label="$\\alpha_\\text{IR}=" + f"{alpha_IR}$ /m" + ", $T_{n_\\text{max}}" + f"={np.round(Ts_mat[-1], 1)}$° C, matrix method")
    ax.plot(ns, Ts_it, label="$\\alpha_\\text{IR}=" + f"{alpha_IR}$ /m" + ", $T_{n_\\text{max}}" + f"={np.round(Ts_it[-1], 1)}$° C, iterative method")
    ax.set_xlabel("Number of layers $n$")
    ax.set_ylabel("Surface temperature $T$ (° C)")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_temps_full_comp_alpha_IR(alpha_V, alphas_IR, n, T, h_max, path, surface_albedo=0):
    Ts_mat = physics.temps_full_model_vary_alpha_IR_mat(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)
    Ts_it = physics.temps_full_model_vary_alpha_IR(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)

    fig, ax = plt.subplots()
    ax.plot(alphas_IR, Ts_mat, label=f"$n={n}$, matrix method")
    ax.plot(alphas_IR, Ts_it, label=f"$n={n}$, iterative method")
    ax.set_xlabel("$\\alpha_\\text{IR}$ (1/m)")
    ax.set_ylabel("Surface temperature $T$ (° C)")
    ax.set_xscale("log")
    ax.legend()

    fig.savefig(path)
    plt.close(fig)


def plot_performance(ns_it, ns_mat, path):
    alpha_V = 5e-5
    alpha_IR = 1.5e-4
    T = 288.2
    h_max = 300e3

    n_times_it  = len(ns_it)
    n_times_mat = len(ns_mat)

    times_it  = np.zeros(n_times_it)
    times_mat = np.zeros(n_times_mat)

    # jit compile everything!:
    discard = physics.temp_full_model(alpha_V, alpha_IR, 200, T, h_max, n_sweeps=10)
    discard = physics.temp_full_model_matrix_approach(alpha_V, alpha_IR, 200, T, h_max)

    for i in tqdm(range(n_times_it)):
        start = time.time_ns()
        discard = physics.temp_full_model(alpha_V, alpha_IR, ns_it[i], T, h_max, n_sweeps=10)
        times_it[i] = (time.time_ns() - start) / 1e9

    for i in tqdm(range(n_times_mat)):
        start = time.time_ns()
        discard = physics.temp_full_model_matrix_approach(alpha_V, alpha_IR, ns_mat[i], T, h_max)
        times_mat[i] = (time.time_ns() - start) / 1e9
    
    fig, ax = plt.subplots()
    ax.plot(ns_it, times_it, label="Iterative method")
    ax.plot(ns_mat, times_mat, label="LSE method")
    ax.legend()
    ax.set_xlabel("Number of layers $n$")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.set_xscale("log")

    fig.savefig(path)
    plt.close(fig)


def plot_temp_height_it(alpha_V, alpha_IR, n, T, h_max, path, surface_albedo=0, sweeps=15):
    Ts = physics.temps_full_model(alpha_V, alpha_IR, n, T, h_max, surface_albedo=surface_albedo, n_sweeps=sweeps)[:-1] - 273.15
    hs = np.linspace(0, h_max, n)

    fig, ax = plt.subplots()
    ax.plot(hs, Ts, label="$\\alpha_\\text{IR}=" + f"{alpha_IR}$ /m, $n={n}$")
    ax.legend()
    ax.set_xlabel("$h$ (m)")
    ax.set_ylabel("$T$ (° C)")
    #ax.set_xscale("log")

    fig.savefig(path)
    plt.close(fig)

def plot_temp_height_mat(alpha_V, alpha_IR, n, T, h_max, path, surface_albedo=0):
    Ts = physics.temps_full_model_matrix_approach(alpha_V, alpha_IR, n, T, h_max, surface_albedo=surface_albedo) - 273.15
    hs = np.linspace(0, h_max, n)

    fig, ax = plt.subplots()
    ax.plot(hs, Ts, label="$\\alpha_\\text{IR}=" + f"{alpha_IR}$ /m, $n={n}$")
    ax.legend()
    ax.set_xlabel("$h$ (m)")
    ax.set_ylabel("$T$ (° C)")
    #ax.set_xscale("log")

    fig.savefig(path)
    plt.close(fig)