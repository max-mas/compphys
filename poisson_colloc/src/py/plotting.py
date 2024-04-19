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

    plot_B_i(13, "../../results/B_i_k/", "../../plots/B_i_k/", nodes=np.arange(11))
    plot_B_i(13, "../../results/B_i_k_x/", "../../plots/B_i_k_x/", nodes=np.arange(11), deriv=1)
    plot_B_i(13, "../../results/B_i_k_xx/", "../../plots/B_i_k_xx/", nodes=np.arange(11), deriv=2)

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


main()
