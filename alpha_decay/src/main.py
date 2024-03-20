import alphadecay

def main():
    rng = alphadecay.get_coulomb_range(92, 238)
    a = alphadecay.Alphadecay(92, 238, rng, 1000)
    #print(a.E_kin)
    #print(a.piecewise_constant_potential())
    #A = a.coeff_mat()
    #print(np.real(A))
    #print(A.shape)
    #print(a.get_transm_coeff())
    print(a.get_half_life())
    return


if __name__ == "__main__":
    main()
