import alphadecay
import parameter_study

def main():
    rng = alphadecay.get_coulomb_range(92, 238)
    a = alphadecay.Alphadecay(92, 238, rng, 3000)
    #print(a.E_kin)
    #print(a.piecewise_constant_potential())
    #A = a.coeff_mat()
    #print(np.real(A))
    #print(A.shape)
    #print(a.get_transm_coeff())
    print(a.get_half_life())
    parameter_study.plot_density(a, n=1000, path="./alpha_decay/plots/u238.png")
    return


if __name__ == "__main__":
    main()
