import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

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

    """
    plot_B_i(13, "../../results/B_i_k/", "../../plots/B_i_k/", nodes=np.arange(11))
    plot_B_i(13, "../../results/B_i_k_x/", "../../plots/B_i_k_x/", nodes=np.arange(11), deriv=1)
    plot_B_i(13, "../../results/B_i_k_xx/", "../../plots/B_i_k_xx/", nodes=np.arange(11), deriv=2)
    """



    xs, phis = load_B_i_k("../../results/solution/solution_solidsphere.txt")
    plot_potential(xs, phis, "../../plots/potential/potential_solidsphere.pdf", type="solid")
    xs2, phis2 = load_B_i_k("../../results/solution/solution_solidsphere_test.txt")
    plot_potential(xs2, phis2, "../../plots/potential/potential_solidsphere_test.pdf", type="solid")
    plot_error([xs, xs2], [phis, phis2], "../../plots/potential_error/error_solidsphere_test.pdf",
               type="solid", pts=["500 linspaced pts", "10 linspaced pts + 2 close to $R$"])


    xs, phis = load_B_i_k("../../results/solution/solution_shell.txt")
    plot_potential(xs, phis, "../../plots/potential/potential_shell.pdf", type="shell")
    xs2, phis2 = load_B_i_k("../../results/solution/solution_shell_test.txt")
    plot_potential(xs2, phis2, "../../plots/potential/potential_shell_test.pdf", type="shell")
    plot_error([xs, xs2], [phis, phis2], "../../plots/potential_error/error_shell_test.pdf", type="shell",
        pts=["500 linspaced pts", "50 linspaced pts + 2 close to $R_\\text{min},R_\\text{max}$"])

    xs, phis = load_B_i_k("../../results/solution/solution_hydrogen.txt")
    plot_potential(xs, phis, "../../plots/potential/potential_hydrogen.pdf", type="hydrogen")
    xs2, phis2 = load_B_i_k("../../results/solution/solution_hydrogen_test.txt")
    plot_potential(xs2, phis2, "../../plots/potential/potential_hydrogen_test.pdf", type="hydrogen")
    xs3, phis3 = load_B_i_k("../../results/solution/solution_hydrogen_test2.txt")
    plot_potential(xs3, phis3, "../../plots/potential/potential_hydrogen_test2.pdf", type="hydrogen")
    plot_error([xs, xs3, xs2], [phis, phis3, phis2], "../../plots/potential_error/error_hydrogen_test.pdf",
               type="hydrogen", pts=["500 linspaced pts", "50 linspaced points", "10 linspaced points"])

    xs, phis = load_B_i_k("../../results/solution/solution_hydrogen_2s.txt")
    plot_potential(xs, phis, "../../plots/potential/potential_hydrogen_2s.pdf", type="hydrogen2s")

    return 0


def load_B_i_k(path):
    file = open(path)
    lines = file.readlines()
    n_trials = len(lines)

    xs = np.zeros(n_trials)
    Bs = np.zeros(n_trials)

    for i in range(n_trials):
        nums = lines[i].split(",")
        xs[i] = float(nums[0])
        Bs[i] = float(nums[1])

    return xs, Bs


def plot_B_i(i_max, in_path, out_path, nodes, deriv=0):
    sns.set_palette("colorblind")
    xs_i = []
    Bs_i = []
    for i in range(i_max):
        xs, Bs = load_B_i_k(in_path + "B_" + str(i) + ".txt")
        xs_i.append(xs)
        Bs_i.append(Bs)
    B_sum = np.zeros(len(Bs_i[0]))
    for i in range(i_max):
        B_sum += Bs_i[i]

    fig, ax = plt.subplots()
    for i in range(i_max):
        ax.plot(xs_i[i], Bs_i[i])
    p1 = ax.scatter(nodes, np.zeros(len(nodes)), color="red", marker="*", label="Knots")

    if deriv == 1:
        ax.set_ylabel("$B'_{i,k}(x)$")
    elif deriv == 2:
        ax.set_ylabel("$B''_{i,k}(x)$")
    else:
        ax2 = ax.twinx()
        p2, = ax2.plot(xs_i[0], B_sum, color="indianred", ls="--", label="$\\sum_i B_{i,k}(x)$")
        plots = [p1, p2]
        ax.legend(plots, [plot.get_label() for plot in plots], loc=(0.405, 0.8))
        ax.set_ylim([-0.1, 1.1])
        ax2.set_ylim([-0.1, 1.1])
        ax.set_ylabel("$B_{i,k}(x)$")
        ax2.set_ylabel("$\\sum_i B_{i,k}(x)$")

    ax.set_xlabel("$x$")

    ax.set_xlim([xs_i[0][0], xs_i[0][-1]])
    ax.grid()
    fig.tight_layout()

    match deriv:
        case 1:
            fname = "B_i_x.pdf"
        case 2:
            fname = "B_i_xx.pdf"
        case _:
            fname = "B_i.pdf"

    fig.savefig(out_path + fname)
    plt.close(fig)


def plot_potential(xs, phis, path, type=None):
    sns.set_palette("colorblind")
    Vs = np.zeros(len(xs))
    Vs[1:] = phis[1:] / xs[1:]
    Vs[0] = Vs[1]

    fig, ax = plt.subplots()
    p1, = ax.plot(xs[:-1], Vs[:-1], label="$V$, collocation", color="forestgreen")
    ax2 = ax.twinx()


    if type=="solid":
        exact = np.zeros(len(xs))
        rho = np.zeros(len(xs))
        Q = 4/3 * np.pi * 1**3 * 1
        for i in range(len(xs)):
            if xs[i] <= 1:
                exact[i] = (3/2 - xs[i]**2 / (2 * 1**2)) * Q
                rho[i] = 1
            else:
                exact[i] = (1 / xs[i]) * Q
        p2, = ax.plot(xs, exact, color="mediumblue", ls="-.", alpha=0.8, label="$V$, exact")
        p3, = ax2.plot(xs, rho, color="mediumorchid", alpha=0.8, label="$\\rho$")
        p4 = ax.axvline(1.0, label="$r=R$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(R)$")
        ax.set_ylabel("$V(r)$ $(Q/(4\\pi\\varepsilon_0 R))$")
        ax2.set_ylabel("$\\rho(r)$ $(Q/V)$")
        plots = [p1, p2, p3, p4]
        ax2.set_ylim([0, np.max(rho)+0.1])

    if type=="shell":
        exact = np.zeros(len(xs))
        rho = np.zeros(len(xs))
        prefactor = 4 * np.pi * 1
        Q = 4/3 * np.pi * 1 * (1**3 - 0.8**3)
        for i in range(len(xs)):
            if xs[i] < 0.8:
                exact[i] = prefactor * (1**2/2 - 1/3 * (0.8**2 + 0.8**2/2))
            elif 0.8 <= xs[i] <= 1:
                exact[i] = prefactor * (1**2/2 - 1/3 * (0.8**3/xs[i] + xs[i]**2/2))
                rho[i] = 1
            else:
                exact[i] = (1 / xs[i]) * Q
        p2, = ax.plot(xs, exact, color="mediumblue", ls="-.", alpha=0.8, label="$V$, exact")
        p3, = ax2.plot(xs, rho, color="mediumorchid", alpha=0.8, label="$\\rho$")
        p4 = ax.axvline(1.0, label="$r=R_\\text{max}$", alpha=0.6, ls="--", color="indianred")
        p5 = ax.axvline(0.8, label="$r=R_\\text{min}$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(R_\\text{max})$")
        ax.set_ylabel("$V(r)$ $(Q/(4\\pi\\varepsilon_0 R_\\text{max}))$")
        ax2.set_ylabel("$\\rho(r)$ $(Q/V)$")
        plots = [p1, p2, p3, p4, p5]
        ax2.set_ylim([0, np.max(rho)+0.1])

    if type=="hydrogen":
        exact = np.zeros(len(xs))
        rho = np.zeros(len(xs))
        for i in range(len(xs)):
            exact[i] = (1/xs[i] - np.exp(-2*xs[i]) * (1/xs[i] + 1))
            rho[i] = 1 / (np.pi) * np.exp(-2*xs[i])
        p2, = ax.plot(xs, exact, color="mediumblue", ls="-.", alpha=0.8, label="$V$, exact")
        p3, = ax2.plot(xs, rho, color="mediumorchid", alpha=0.8, label="$\\rho$")
        #ax.axvline(1.0, label="$r=a_0$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(a_0)$")
        ax.set_ylabel("$V(r)$ $(e/(4\\pi\\varepsilon_0 a_0))$")
        ax2.set_ylabel("$\\rho(r)$ $(e/a_0^3)$")
        plots = [p1, p2, p3]
        ax2.set_ylim([0, np.max(rho)+0.05])

    if type=="hydrogen2s":
        rho = np.zeros(len(xs))
        for i in range(len(xs)):
            rho[i] = 1 / (32 * np.pi) * (2 - xs[i])**2 * np.exp(-xs[i])
        p2, = ax2.plot(xs, rho, color="mediumorchid", alpha=0.8, label="$\\rho$")
        #ax.axvline(1.0, label="$r=a_0$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(a_0)$")
        ax.set_ylabel("$V(r)$ $(e/(4\\pi\\varepsilon_0 a_0))$")
        ax2.set_ylabel("$\\rho(r)$ $(e/a_0^3)$")
        plots = [p1, p2]
        ax2.set_ylim([0, np.max(rho)])

    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([0, np.max(Vs)+0.1])
 

    ax.legend(plots, [plot.get_label() for plot in plots])
    ax.grid()
    fig.tight_layout()

    fig.savefig(path)
    plt.close(fig)


def plot_error(xss, phiss, path, type, pts):
    fig, ax = plt.subplots()

    if type=="solid":
        for xs, phis, pt in zip(xss, phiss, pts):
            Vs = np.zeros(len(xs))
            Vs[1:] = phis[1:] / xs[1:]
            Vs[0] = Vs[1]
            exact = np.zeros(len(xs))
            Q = 4/3 * np.pi * 1**3 * 1
            for i in range(len(xs)):
                if xs[i] <= 1:
                    exact[i] = (3/2 - xs[i]**2 / (2 * 1**2)) * Q
                else:
                    exact[i] = (1 / xs[i]) * Q
            ax.plot(xs[:-1], np.abs(Vs - exact)[:-1]/exact[:-1], label=pt)
        ax.axvline(1.0, label="$r=R$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(R)$")
        ax.set_ylabel("Error $|V'-V|/V$")
        ax.set_yscale("log")

    if type=="shell":
        for xs, phis, pt in zip(xss, phiss, pts):
            Vs = np.zeros(len(xs))
            Vs[1:] = phis[1:] / xs[1:]
            Vs[0] = Vs[1]
            exact = np.zeros(len(xs))
            prefactor = 4 * np.pi * 1
            Q = 4/3 * np.pi * 1 * (1**3 - 0.8**3)
            for i in range(len(xs)):
                if xs[i] < 0.8:
                    exact[i] = prefactor * (1**2/2 - 1/3 * (0.8**2 + 0.8**2/2))
                elif 0.8 <= xs[i] <= 1:
                    exact[i] = prefactor * (1**2/2 - 1/3 * (0.8**3/xs[i] + xs[i]**2/2))
                else:
                    exact[i] = (1 / xs[i]) * Q
            ax.plot(xs[:-1], np.abs(Vs - exact)[:-1]/exact[:-1], label=pt)
        ax.axvline(1.0, label="$r=R_\\text{max}$", alpha=0.6, ls="--", color="indianred")
        ax.axvline(0.8, label="$r=R_\\text{min}$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(R)$")
        ax.set_ylabel("Error $|V'-V|/V$")
        ax.set_yscale("log")

    if type=="hydrogen":
        for xs, phis, pt in zip(xss, phiss, pts):
            Vs = np.zeros(len(xs))
            Vs[1:] = phis[1:] / xs[1:]
            Vs[0] = Vs[1]
            exact = np.zeros(len(xs))
            for i in range(len(xs)):
                exact[i] = (1/xs[i] - np.exp(-2*xs[i]) * (1/xs[i] + 1))
            ax.plot(xs[:-1], np.abs(Vs - exact)[:-1]/exact[:-1], label=pt)
        #ax.axvline(1.0, label="$r=a_0$", alpha=0.6, ls="--", color="indianred")
        ax.set_xlabel("$r$ $(a_0)$")
        ax.set_ylabel("Error $|V'-V|/V$")
        ax.set_yscale("log")

    ax.set_xlim([xss[0][0], xss[0][-1]])

    #ax.set_ylim([np.min(Vs), np.max(Vs)])

    ax.legend()
    ax.grid()

    fig.savefig(path)
    plt.close(fig)



main()
