import alphadecay
import parameter_study
import numpy as np

def main():
    elements = [(92, 238), (92, 235), (90, 232), (86, 222), (84, 212)] # U 238, U235, Th 232, Rn 222, Po 212
    names = {(92, 238): "u238", 
             (92, 235): "u235",
             (90, 232): "th232",
             (86, 222): "rn222", 
             (84, 212): "po212"}
    
    # for final plots
    Es = []
    t12s = []
    Zs = []
    As = []

    # store main results in txt file
    file = open("./alpha_decay/results/res.txt", "w")
    
    
    for element in elements:
        name = names[element]
        (Z, A) = element

        # Solve system with many bins, print half life, plot density
        # Parameters: R_factor = 1.35, V0 = -134 MeV, 2000 bins
        rng = alphadecay.get_coulomb_range(Z, A)
        a = alphadecay.Alphadecay(Z, A, rng, 2000)
        print(40*"-")        
        E = a.E_kin
        Es.append(E)
        print("E_kin", name, E, "MeV")
        file.write("E_kin " + name + str(E) + " MeV\n")
        t12 = a.get_half_life()
        t12s.append(t12)
        As.append(A)
        Zs.append(Z)
        true_t12 = parameter_study.true_half_lifes[(Z,A)]
        print("Half life", name, t12, "s")
        file.write("Half life" + name + str(t12) + " s\n")
        print("Relative error:", np.round(np.abs(true_t12 - t12) / true_t12, 2))
        file.write("Relative error:" + str(np.round(np.abs(true_t12 - t12) / true_t12, 2)) + "\n")
        parameter_study.plot_density(a, n=1000, path=f"./alpha_decay/plots/density/" + name + ".pdf")
        
        print(40*"-")
        print("Testing:")
        # Plot dependence of half life on number of bins
        # Parameters: R_factor = 1.35, V0 = -134 MeV
        print("Bin dependence", name)
        parameter_study.test_bin_dependence(Z, A, path=f"./alpha_decay/plots/bin_dependence/bins_" + name + ".pdf")

        # Plot dependence of half life on nuclear radius
        # Parameters: bins = 1000, V0 = -134 MeV
        print("R dependence", name)
        parameter_study.test_R_dependence(Z, A, path=f"./alpha_decay/plots/R_dependence/R_" + name + ".pdf", num_trials=30, bins=1000)

        # Plot dependence of half life on strong force potential
        # Parameters: R_factor = 1.35, bins = 1000
        print("V0 dependence", name)
        parameter_study.test_V0_dependence(Z, A, path=f"./alpha_decay/plots/V0_dependence/V0_" + name + ".pdf", num_trials=30, bins=1000, V0_max=0)
    
    file.close()

    # Summary plot
    parameter_study.plot_t12s(Zs, As, t12s, path="./alpha_decay/plots/summary_t.pdf")
    parameter_study.plot_E(Zs, As, Es, path="./alpha_decay/plots/summary_E.pdf")

    # Benchmark
    parameter_study.time_scaling("./alpha_decay/plots/runtime.pdf", bin_max=4)

    return


if __name__ == "__main__":
    main()
