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
    """
    for l in range(7):
        xs, psi = load_function("../../results/mtins/0/muffintins_l" +  str(l) + ".txt")
        psi /= np.max(psi)
        xs_s.append(xs)
        psi_s.append(psi)
    
    plot_muffin_tins(xs_s, psi_s, "../../plots/mtins/0.pdf")
    """
    point_names = ["$\\Gamma$", "$H$", "$N$", "$\\Gamma$", "$P$", "$H$"]
    """
    

    path = "../../results/det_2_R_130/"
    E_outer, k_outer, det_outer = load_E_k_bs(path)
    path2 = "../../results/det_3_R_130/"
    E_outer2, k_outer2, det_outer2 = load_E_k_bs(path2)
    path3 = "../../results/det_4_R_130/"
    E_outer3, k_outer3, det_outer3 = load_E_k_bs(path3)
    plot_bandstructure_conv(E_outer, k_outer, det_outer, E_outer2, k_outer2, det_outer2, E_outer3, k_outer3, det_outer3, point_names, "../../plots/conv_24.pdf")
    """
    
    path = "../../results/det_2_R_130/"
    E_outer, k_outer, det_outer = load_E_k_bs(path)

    
    E_DOS = []
    E_kf = []
    k_kf = []
    path = "../../results/det_DOS_2_morepts/"
    files = natsort.natsorted(listdir(path))
    for file in files:
        Es, det = load_function(path + file)
        k = np.array([0, 0, 0], dtype=np.double)
        s = file.split("_")
        k[0] = float(s[1])
        k[1] = float(s[2])
        k[2] = float(s[3].split(".")[0])
        #print(k)
        det_abs = np.abs(det)
        potential_zeros = Es[argrelextrema(det_abs, np.less)]  
        
        for E in potential_zeros:
            H = 27.211386246
            E_DOS.append(E)
            #if np.abs(2 * (E - 0.015507753876938368) - 0.1742) < 2e-3:
            #if np.abs(2*H*E - 4.0) < 5e-3:
            #if True:
            #    print("Added vec to FS vecs", E)
            #    k_kf.append(k)
            #    E_kf.append(E)
    
    plot_bandstructure(E_outer, k_outer, det_outer, point_names, "../../plots/bs_2_R_130_full_dos.pdf", E_DOS=E_DOS)
    #print(k_kf)
    
    #plot_fs(E_kf, k_kf, "../../plots/fs_test.pdf")
    """
    #plot_bandstructure(E_outer, k_outer, det_outer, point_names, "../../plots/bs_exp.pdf", exp=True)

    
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

def load_E_k_bs(path):
    E_outer = []
    k_outer = []
    det_outer = []
    folders = ["g_h/", "h_n/", "n_g/", "g_p/", "p_h/"]

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
            #if k == 0 and folder == "g_h/":
            #    continue
            #print(s1)
            Es_s.append(Es)
            ks.append(k)
            det_s.append(det)
        
        E_outer.append(Es_s)
        k_outer.append(ks)
        det_outer.append(det_s)

    return E_outer, k_outer, det_outer

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
        ax.plot(xs_s[l][1:], psi_s[l][1:], label=f"$l={l}$")
    
    ax.axvline(1.3, color="red", ls="--", alpha=0.8, label="$R$")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$R_l(E, r)$")

    #ax.set_xscale("log")
    #ax.set_yscale("log")

    ax.legend(loc="upper right")
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


def plot_bandstructure(E_outer, k_outer, det_outer, point_names, path, E_DOS=None):
    plot_dos = False if E_DOS is None else True


    if plot_dos:
        fig, [ax, ax2] = plt.subplots(1, 2, width_ratios=[3, 1])
    else:
        fig, ax = plt.subplots()

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


    H = 27.211386246 # TODO lying to myself?
    
    mini = np.min(np.asarray(points_flat))
    print(mini)
    #EF = H*0.1742
    EF = 4
    #bands = 2*H*(np.asarray(points_flat) - mini)
    bands = 2*H*(np.asarray(points_flat))

    p2 = ax.axhline(EF, color="indianred", ls="--", label="$E_\\text{F}" + f"={np.round(EF, 2)}$ eV") # TODO s.o.
    #p3 = ax.axhline(mini, color="lightseagreen", ls="-.", label="$E_\\text{min} =" + f"{np.round(mini, 2)}$ eV")
    
    p1 = ax.scatter(k_plot_flat, bands, marker="o", s=2, color="forestgreen", label="$E(k)$")
    
    if plot_dos:
        #2*H*(np.asarray(E_DOS) - mini)
        sns.histplot(y=np.asarray(2*H*np.asarray(E_DOS)), bins=200, color="mediumblue", kde=True, kde_kws={"bw_method": 0.05}, line_kws={"color": "coral", "ls": "--"} ,ax=ax2)
        #sns.kdeplot(y=bands, color="mediumblue", fill=True, ax=ax2)
    
        ax2.set_xlabel("DOS (a.u.)")
        ax2.set_xticks([])
        ax2.set_ylim([0, 55])
        xs = np.linspace(0, 55, 200)
        ax2.plot(60*(2)**(3/2) * (6.632)**3/2 / (2*np.pi**2) * np.sqrt((xs)/H), xs+H*mini, color="indianred", label="Free e-gas")
        ax2.legend(loc="upper left")
        #ax2.set_xscale("log")

    ax.set_xticks(ticks)
    ax.set_xticklabels(point_names)

    ax.set_xlabel("$k$")
    ax.set_ylabel("$E$ (eV)")

    ax.set_xlim([ticks[0], ticks[-1]])
    ax.set_ylim([0, 55])
    

    plots = [p1, p2]
    ax.legend(plots, [plot.get_label() for plot in plots], loc="upper right")
    ax.grid()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bandstructure_conv(E_outer1, k_outer1, det_outer1, E_outer2, k_outer2, det_outer2, E_outer3, k_outer3, det_outer3, point_names, path):

    fig, ax = plt.subplots()

    E_min_1 = []
    E_min_2 = []
    E_min_3 = []
    k_plot = []
    k_offset = 0.0
    ticks = [0.0]

    for det_inner1, E_inner1, det_inner2, E_inner2, det_inner3, E_inner3, k_inner in zip(det_outer1, E_outer1, det_outer2, E_outer2, det_outer3, E_outer3, k_outer1):
        for det1, Es1, det2, Es2, det3, Es3, k in zip(det_inner1, E_inner1, det_inner2, E_inner2, det_inner3, E_inner3, k_inner):
            det_abs1 = np.abs(det1)
            potential_zeros1 = Es1[argrelextrema(det_abs1, np.less)]        
            det_abs2 = np.abs(det2)
            potential_zeros2 = Es2[argrelextrema(det_abs2, np.less)]     
            det_abs3 = np.abs(det3)
            potential_zeros3 = Es3[argrelextrema(det_abs3, np.less)]        
                
            #print(potential_zeros1)
            if len(potential_zeros1) == 0 or len(potential_zeros2) == 0 or len(potential_zeros3) == 0:
                continue
            E1 = np.min(potential_zeros1)
            E_min_1.append(E1)
            E2 = np.min(potential_zeros2)
            E_min_2.append(E2)
            E3 = np.min(potential_zeros3)
            E_min_3.append(E3)
            k_plot.append(k + k_offset)
            
        k_offset += k
        ticks.append(k_offset)


    H = 27.211386246 # TODO lying to myself?
    E_min_1_np = 2*H*(np.asarray(E_min_1))
    E_min_2_np = 2*H*(np.asarray(E_min_2))
    E_min_3_np = 2*H*(np.asarray(E_min_3))

    p1 = ax.scatter(k_plot, np.abs(E_min_1_np - E_min_3_np), marker="o", s=2, label="2 vs. 4")
    p2 = ax.scatter(k_plot, np.abs(E_min_2_np - E_min_3_np), marker="x", s=2, label="3 vs. 4")

    ax.set_xticks(ticks)
    ax.set_xticklabels(point_names)

    ax.set_xlabel("$k$")
    ax.set_ylabel("$\\Delta E$ (eV)")

    ax.set_xlim([ticks[0], ticks[-1]])
    #ax.set_ylim([0, 20])
    
    ax.set_yscale("log")
    ax.legend()
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


def plot_fs(E_kf, k_kf, path):
    fig, ax = plt.subplots()

    k_xy_plane = []
    for k, E in zip(k_kf, E_kf):
        if np.abs(k[2]) < 1e-12:
            k_xy_plane.append(k)
    
    k_x = [k[0] for k in k_xy_plane]
    k_y = [k[1] for k in k_xy_plane]

    EF = 2* 0.1742 #/ 2 # 27.211386246

    theta = np.linspace(0, 2*np.pi, 1000)
    x_fe = EF * np.cos(theta)
    y_fe = EF * np.sin(theta)

    ax.scatter(k_x, k_y, s=3, label="Numerical FS")
    ax.scatter(x_fe, y_fe, s=2, label="Free electrons")
    ax.axis("equal")

    ax.legend()
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")

    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


main()