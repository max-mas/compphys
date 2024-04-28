import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.special
import seaborn as sns
from scipy.special import sph_harm, factorial, assoc_laguerre

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
    for l in range(10):
        continue
        for n in range(l+1, 100):
            try:
                print("Plotting state l =", l, "n =", n)
                path = "../../results/states/rad_wf_l" + str(l) + "_n" + str(n) + "_m0.txt"
                outpath1 = "../../plots/psis/l" + str(l) + "_n" + str(n) + ".pdf"
                outpath2 = "../../plots/psis_err/l" + str(l) + "_n" + str(n) + "l0.pdf"
                xs, psi_rad = load_state(path)
                plot_wf(xs, psi_rad, n, l, outpath1)
                xs, psi_rad = load_state(path)
                plot_wf_err(xs, psi_rad, n, l, outpath2)
            except FileNotFoundError:
                print("State l =", l, "n =", n, "not found. Proceeding to next l.")
                break

    lmax = 10
    for l in range(lmax+1):
        continue
        print("Plotting summary: l =", l)
        psis = []
        for n in range(l+1, 100):
            try:
                print("Finding state n =", n)
                path = "../../results/states/rad_wf_l" + str(l) + "_n" + str(n) + "_m0.txt"
                xs, psi_rad = load_state(path)
            except FileNotFoundError:
                print("State l =", l, "n =", n, "not found. Plotting.")
                outpath = "../../plots/psis_comp/l" + str(l) + ".pdf"
                outpath_ortho = "../../plots/ortho/l" + str(l) + ".pdf"
                plot_multiple_wfs(xs, psis, l, np.arange(n)+1+l, outpath)
                plot_orthogonality(psis, xs[1]-xs[0], l, outpath_ortho)
                break
            else:
                psis.append(psi_rad)

    
    print("Plotting energies.")
    ns_outer = []
    ergs_outer = []
    lmax = 10
    ls = np.arange(lmax+1)
    for l in range(lmax+1):
        print("Searching l =", l, "energies.")
        path = "../../results/energies/energies_l" + str(l) + ".txt"
        try:
            ns, ergs = load_neg_ergs(path)
        except FileNotFoundError:
            print("Energies with l =", l, "not found. Stopping search.")
            break
        else:
            ns_outer.append(ns)
            ergs_outer.append(ergs)
    outpath = "../../plots/erg.pdf"
    outpath_log = "../../plots/erg_log.pdf"
    outpath_err = "../../plots/erg_err.pdf"
    plot_energies(ns_outer, ergs_outer, ls, outpath)
    plot_energies(ns_outer, ergs_outer, ls, outpath_log, log=True)
    plot_energies_err(ns_outer, ergs_outer, ls, outpath_err)
    

    print("Plotting number of bound states and GS errs.")
    ns, num_ls, num_exp = load_bound_nums("../../results/ns.txt")
    ns, errs_ls, errs_exp = load_gs_err("../../results/gs_err.txt")
    plot_ns_errs(ns, num_ls, num_exp, errs_ls, errs_exp, "../../plots/ns_errs.pdf")
    

def load_state(path):
    file = open(path)
    lines = file.readlines()
    n_xs = len(lines)

    xs = np.zeros(n_xs)
    psi_rad = np.zeros(n_xs)

    for i in range(n_xs):
        nums = lines[i].split(",")
        xs[i] = float(nums[0])
        psi_rad[i] = float(nums[1])

    return xs, psi_rad


def load_neg_ergs(path):
    file = open(path)
    lines = file.readlines()
    n_states = len(lines)

    ns = []
    neg_ergs = []

    for i in range(n_states):
        nums = lines[i].split(",")
        if float(nums[1]) >= 0:
            break
        ns.append(float(nums[0]))
        neg_ergs.append(float(nums[1]))

    return np.asarray(ns, dtype=np.int64), np.asarray(neg_ergs)


def load_bound_nums(path):
    file = open(path)
    lines = file.readlines()
    n = len(lines)

    ns = np.zeros(n, dtype=np.int64)
    num_ls = np.zeros(n, dtype=np.int64)
    num_exp = np.zeros(n, dtype=np.int64)

    for i in range(n):
        nums = lines[i].split(",")
        ns[i] = int(nums[0])
        num_ls[i] = int(nums[1])
        num_exp[i] = int(nums[2])

    return ns, num_ls, num_exp


def load_gs_err(path):
    file = open(path)
    lines = file.readlines()
    n = len(lines)

    ns = np.zeros(n, dtype=np.int64)
    errs_ls = np.zeros(n, dtype=np.double)
    errs_exp = np.zeros(n, dtype=np.double)

    for i in range(n):
        nums = lines[i].split(",")
        ns[i] = int(nums[0])
        errs_ls[i] = float(nums[1])
        errs_exp[i] = float(nums[2])

    return ns, errs_ls, errs_exp


def plot_wf(xs, psi_rad, n, l, path):
    sns.set_palette("colorblind")

    fig, ax = plt.subplots()
    n_str = str(n)
    l_str = str(l)
    sign = psi_rad[0] / np.abs(psi_rad[0])
    psi_rad[0] = psi_rad[1]
    #ax.plot(xs[1:], sign * psi_rad[1:] / np.max(np.abs(psi_rad[1:])),
    #        label="$R_{" + n_str + "," + l_str + "}$, numerical")
    ax.plot(xs, sign * psi_rad, label="$R_{" + n_str + "," + l_str + "}$, numerical", color="forestgreen")
    exact = psi_rad_exact(n, l, xs)
    ax.plot(xs, exact, label="$R_{" + n_str + "," + l_str + "}$, exact", color="mediumblue", ls="-.", alpha=0.7)


    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$R_{nl}(r)$")
    ax.set_xlim(0, xs[-1])
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_wf_err(xs, psi_rad, n, l, path):
    sns.set_palette("colorblind")

    fig, ax = plt.subplots()
    n_str = str(n)
    l_str = str(l)
    sign = psi_rad[0] / np.abs(psi_rad[0])
    psi_rad = sign * psi_rad / np.abs(psi_rad[0])
    exact = psi_rad_exact(n, l, xs)
    ax.plot(xs[1:], np.abs((exact - psi_rad) / np.abs(exact))[1:], color="indianred")

    #ax.scatter(xs, 0*xs, color="red", marker="*", label="Knots")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$|R_{nl}-R'_{nl}|/|R_{nl}|$")
    ax.set_xlim(0, xs[-1])
    ax.set_yscale("log")
    ax.grid()
    #ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_multiple_wfs(xs, wfs, l, ns, path):
    sns.set_palette("colorblind")
    fig, ax = plt.subplots()
    for i in range(len(wfs)):
        if i > 4:
            break
        n = ns[i]
        wf = wfs[i]
        n_str = str(n)
        l_str = str(l)
        sign = wf[0] / np.abs(wf[0])
        #wf[0] = wf[1]
        exact = psi_rad_exact(n, l, xs)
        p, = ax.plot(xs, sign * wf, label=f"$n,l={n},{l}$, numerical")
        ax.plot(xs, exact, color=p.get_color(), alpha=0.7, ls="-.", label=f"$n,l={n},{l}$, exact")

    ax.set_xlabel("$r$ $(a_0)$")
    ax.set_ylabel("$R_{nl}(r)$")
    upperlim_l = [10, 30, xs[-1], xs[-1], xs[-1], xs[-1], xs[-1], xs[-1], xs[-1], xs[-1], xs[-1]]
    ax.set_xlim(0, upperlim_l[l])
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_orthogonality(wfs, dx, l, path):
    n = len(wfs)
    orthonormality_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            orthonormality_matrix[i, j] = np.abs(np.dot(wfs[i], wfs[j])) * dx

    fig, ax = plt.subplots()
    ms = ax.matshow(orthonormality_matrix, norm=mpl.colors.LogNorm(), cmap="PuBuGn")
    fig.colorbar(ms)
    ax.set_xlabel("$n$")
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_ylabel("$n'$")
    fig.suptitle("$O_{nn'} = \\left|\\braket{nl|n'l}\\right|$")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_energies(ns_outer, ergs_outer, ls, path, log=False):
    fig, ax = plt.subplots()
    cols1 = sns.color_palette("husl", 11)
    cols2 = sns.color_palette("hls", 11)

    for l, ns, ergs in zip(ls, ns_outer, ergs_outer):
        num_ergs = len(ergs)
        H = 27.211386245988
        sign = 1
        if log:
            sign = -1
        lls = l * np.ones(num_ergs)
        ergs_num = H * sign * ergs
        ergs_exact = H * sign * energies_exact(ns)
        for nn, ll, erg_num, erg_exact in zip(ns, lls, ergs_num, ergs_exact):
            p = ax.scatter(ll - 0.1, erg_num, color=cols1[nn-1], label="numerical", marker=">")
            q = ax.scatter(ll+ 0.1, erg_exact, color=cols1[nn-1], label="exact", marker="<")

    r = ax.axhline(-sign*H/2, color="indianred", alpha=0.7, ls="-.", label="$-1$ Ry")
    f, = ax.plot(ls, -sign * H/2 * 1/(ls+1)**2, ls="--", alpha=0.7, color="mediumorchid",
                  label="$-1\\,\\text{Ry}/n_\\text{min}^2 = -1\\,\\text{Ry}/(l+1)^2$")

    norm = mpl.colors.Normalize(1, 12)
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    my_cmap = mpl.colors.ListedColormap(cols1.as_hex())
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=my_cmap, orientation='vertical', norm=norm, label="$n$")
    plt.gcf().add_axes(ax_cb)

    ax.set_xlabel("$l$")
    ax.set_ylabel("$E_{nl}$ (eV)")
    plots = [p, q, r, f]
    if not log:
        ax.legend(plots, [plot.get_label() for plot in plots])
    else:
        ax.legend(plots, [plot.get_label() for plot in plots], loc="upper right")

    ax.grid()
    if log:
        ax.set_yscale("log")
        ax.set_ylabel("$-E_{nl}$ (eV)")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_energies_err(ns_outer, ergs_outer, ls, path):
    fig, ax = plt.subplots()

    #cols = sns.color_palette("colorblind")
    cols = sns.color_palette("husl", 11)

    for l, ns, ergs in zip(ls, ns_outer, ergs_outer):
        num_ergs = len(ergs)
        lls = l * np.ones(num_ergs)
        errs = np.abs(ergs - energies_exact(ns))/np.abs(energies_exact(ns))
        for nn, ll, err in zip(ns, lls, errs):
            ax.scatter(ll, err, marker="v", color=cols[nn-1])

    norm = mpl.colors.Normalize(1, 12)
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    my_cmap = mpl.colors.ListedColormap(cols.as_hex())
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=my_cmap, orientation='vertical', norm=norm, label="$n$")
    plt.gcf().add_axes(ax_cb)

    ax.set_xlabel("$l$")
    ax.set_ylabel("$|E_{n} - E'_{nl}|/|E_n|$")
    ax.grid()
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ns_errs(ns, num_ls, num_exp, errs_ls, errs_exp, path):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    p1, = ax.plot(ns, num_ls, label="$n_\\text{bound}$ linspaced", color="mediumblue")
    p2, = ax.plot(ns, num_exp, label="$n_\\text{bound}$ exponential", color="firebrick")

    H = 27.211386245988
    p3, = ax2.plot(ns, H*errs_ls, label="$\\Delta E_\\text{GS}$ linspaced", color="mediumblue", ls="--")
    p4, = ax2.plot(ns, H*errs_exp, label="$\\Delta E_\\text{GS}$ exponential", color="firebrick", ls="--")

    ax.set_xlabel("Number of physical knots $N-k$")
    ax.set_ylabel("Number of bound states $n$")
    ax2.set_ylabel("GS energy error $\\Delta E_\\text{GS}$ (eV)")
    ax.grid()
    ax2.set_yscale("log")
    ax.set_xscale("log")
    plots = [p1, p2, p3, p4]
    ax2.legend(plots, [plot.get_label() for plot in plots], loc="upper right")
    ax.set_ylim([0, 9])
    ax2.set_ylim([1e-12, 1e2])
    #ax.set_xlim([1, 1200])

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)



def psi_rad_exact(n, l, r):
    rho = 2 * r / n
    return np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * np.exp(-rho / 2) * rho ** l \
        * assoc_laguerre(rho, n - l - 1, 2 * l + 1)


def energies_exact(n):
    return -0.5 / n**2


main()
