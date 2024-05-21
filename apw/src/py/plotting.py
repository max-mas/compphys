import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.special
import seaborn as sns
from scipy.special import sph_harm, factorial, assoc_laguerre
import pandas as pd
from tqdm.auto import tqdm

from scipy.signal import argrelextrema
from scipy.interpolate import splrep, splev
from os import listdir
import natsort
import copy

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
    xs_s = []
    psi_s = []
    
    #for l in range(11):
    #    xs, psi = load_function("../../results/muffintin_l" +  str(l) + ".txt")
    #    psi /= np.max(psi)
    #    xs_s.append(xs)
    #    psi_s.append(psi)
    
    #plot_muffin_tins(xs_s, psi_s, "../../plots/test.pdf")

    path = "../../results/det_2_med_R/"
    E_outer = []
    k_outer = []
    det_outer = []
    folders = ["g_h/", "h_n/", "n_g/", "g_p/", "p_h/"]
    point_names = ["$\\Gamma$", "$H$", "$N$", "$\\Gamma$", "$P$", "$H$"]
    for folder in folders:
        subfolder = path + folder

        Es_s = []
        ks = []
        det_s = []
        
        files = natsort.natsorted(listdir(subfolder))
        for file in files:
            Es, det = load_function(path + subfolder + file)
            s1 = str(file).split("k")[1].split(".")[0] + "." + str(file).split("k")[1].split(".")[1]
            k = float(s1)
            if k == 0 and folder == "g_h/":
                continue
            print(s1)
            Es_s.append(Es)
            ks.append(k)
            det_s.append(det)
        
        E_outer.append(Es_s)
        k_outer.append(ks)
        det_outer.append(det_s)

    plot_bandstructure(E_outer, k_outer, det_outer, point_names, "../../plots/bs_2_med_R.pdf", exp=False)
    #plot_bandstructure(E_outer, k_outer, det_outer, point_names, "../../plots/bs_exp.pdf", exp=True)

    """
    print("Plotting det")
    path = "../../results/det/g_h/"
    for file in natsort.natsorted(listdir(path)):
        s1 = str(file).split("k")[1].split(".")[0] + "." + str(file).split("k")[1].split(".")[1]
        k = float(s1)
        Es, det = load_function(path + file)
        print(k)
        plot_det(Es, det, "../../plots/det_g_h/det_k_" + str(k) + ".png")
    """
    return 0



def load_function(path):
    file = open(path)
    lines = file.readlines()
    n_xs = len(lines)

    xs = np.zeros(n_xs)
    F = np.zeros(n_xs)

    for i in range(n_xs):
        nums = lines[i].split(",")
        xs[i] = float(nums[0])
        F[i] = float(nums[1])

    return xs, F


def plot_muffin_tins(xs_s, psi_s, path):
    fig, ax = plt.subplots()

    for l in range(len(psi_s)):
        #if l == 0:
        #    continue
        ax.plot(xs_s[l], psi_s[l], label=f"$l={l}$")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$R_l(E, r)$")

    #ax.set_xscale("log")
    #ax.set_yscale("log")

    ax.legend()
    ax.grid()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_det(Es, det, path):
    fig, ax = plt.subplots()

    det_abs = np.abs(det)
    potential_zeros = Es[argrelextrema(det_abs, np.less)]
    #potential_zeros = Es[np.where(np.diff(np.sign(det)))[0]]
    for z in potential_zeros:
        p1 = ax.axvline(z, color="indianred", ls="--", alpha=0.7, label="Potential zero")

    p2, = ax.plot(Es, det_abs, color="mediumblue", label = "$|\\det (\\mathcal{H} - E\,I)|$")

    plots = [p2, p1]
    ax.legend(plots, [plot.get_label() for plot in plots])

    ax.set_xlabel("$E$ (Ha)")
    ax.set_ylabel("$|\\det (\\mathcal{H} - E\,I)|$ $(\\text{Ha}^n)$")

    ax.set_yscale("log")
    #ax.set_ylim([-1, 1])

    ax.grid()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bandstructure(E_outer, k_outer, det_outer, point_names, path, exp = False):
    fig, [ax, ax2] = plt.subplots(1, 2, width_ratios=[3, 1])

    points = []
    points_flat = []
    k_plot = []
    k_plot_flat = []
    k_offset = 0.0
    ticks = [0.0]

    for det_inner, E_inner, k_inner in zip(det_outer, E_outer, k_outer):
        for det, Es, k in zip(det_inner, E_inner, k_inner):
            det_abs = np.abs(det)
            potential_zeros = Es[argrelextrema(det_abs, np.less)]        
            #potential_zeros = Es[np.where(np.diff(np.sign(det)))[0]]
                
            k_plot.append(k + k_offset)
            points.append(potential_zeros)
            for E in potential_zeros:
                k_plot_flat.append(k + k_offset)
                points_flat.append(E)


        k_offset += k
        ticks.append(k_offset)


    H = 2*27.211386246 # TODO lying to myself?
    
    mini = np.min(H*np.asarray(points_flat))
    #EF = H/2*0.1742
    EF = 3.9
    bands = H*np.asarray(points_flat) - mini

    p2 = ax.axhline(EF, color="indianred", ls="--", label="$E_\\text{F}" + f"={np.round(EF, 2)}$ (eV)") # TODO s.o.
    #p3 = ax.axhline(mini, color="lightseagreen", ls="-.", label="$E_\\text{min} =" + f"{np.round(m, 2)}$ (eV)")
    
    p1 = ax.scatter(k_plot_flat, bands, marker="o", s=2, color="forestgreen", label="$E(k)$")
    
    #unique_E, dos = np.unique(points_flat, return_counts=True)
    #dos_spl = splrep(unique_E, dos, k=5, s=100)
    #dos_smooth = splev(unique_E, dos_spl)
    #ax2.hist(points_flat, bins=50, histtype="step", orientation="horizontal")
    sns.histplot(y=bands, bins=100, color="mediumblue", kde=True, kde_kws={"bw_method": 0.05}, line_kws={"color": "coral", "ls": "--"} ,ax=ax2)
    ax2.set_xlabel("DOS (a.u.)")
    ax2.set_xticks([])

    ax.set_xticks(ticks)
    ax.set_xticklabels(point_names)

    ax.set_xlabel("$k$")
    ax.set_ylabel("$E$ (eV)")

    ax.set_xlim([ticks[0], ticks[-1]])
    ax.set_ylim([0, 20])
    ax2.set_ylim([0, 20])

    plots = [p1, p2]
    ax.legend(plots, [plot.get_label() for plot in plots], loc="upper right")
    ax.grid()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def assign_bands(ks, Es_outer):
    Es_outer_mod = copy.deepcopy(Es_outer)
    n_bands = len(Es_outer[0])
    n_k = len(ks)
    #delta_k = ks[1] - ks[0]

    Es_bands = np.zeros((n_k, n_bands))
    #set first and second element of bands
    for i, E in enumerate(Es_outer_mod[0]):
        Es_bands[0][i] = E
    for i, E in enumerate(Es_outer_mod[1]):
        Es_bands[1][i] = E
    
    for i in range(2, n_k):
        pred_next = []
        for j in range(n_bands):
            pred_next.append(Es_bands[i-1][j] + (Es_bands[i-1][j] - Es_bands[i-2][j]) )
        pred_next = np.array(pred_next)
        for j in range(n_bands):
            if j >= len(Es_outer_mod[i]):
                continue
            E = Es_outer_mod[i][j]
            delta = np.abs(pred_next - E)
            closest_index = np.argmin(delta)
            Es_bands[i][j] = E
            np.delete(pred_next, closest_index)

    return Es_bands





main()