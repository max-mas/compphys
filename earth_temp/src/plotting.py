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
            Is = physics.visible_intensity(alpha, T, n, h_max)

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