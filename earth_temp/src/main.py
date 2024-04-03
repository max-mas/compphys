import plotting
import numpy as np

def main():
    max_height = 100e3
    T_guess = 288.2
    alpha_v = 5e-5
    alpha_ir = 1.2e-4
    n = 10000
    """
    plotting.plot_density(n = n, Ts = np.linspace(270, 310, 5), h_max = max_height, path = "./earth_temp/plots/density/density.pdf")
    
    plotting.plot_visible_intensity(alphas=np.linspace(1e-5, 1e-4, 5), n = n, Ts = [T_guess], h_max=max_height, path="./earth_temp/plots/visible_intensity/visible_intensity.pdf")

    plotting.plot_visible_intensity(alphas=[alpha_v], n = n, Ts = np.linspace(270, 310, 5), h_max=max_height, path="./earth_temp/plots/visible_intensity/visible_intensity_T.pdf")

    plotting.plot_temps_visible_absorption(alphas=np.linspace(0, 1e-4, 1000), n=n, T=T_guess, h_max=max_height, path="./earth_temp/plots/temp_visible/temp_visible.pdf", surface_albedos=[0, 0.04])
    """
    
    alpha_irs = np.logspace(-5, -2, 40, base=10)
    ns = np.logspace(1.5, 6, 20, base=10, dtype=np.int64)
    n = int(1e5)
    """
    plotting.plot_temps_full_N(alpha_v, alpha_ir, ns, T_guess, max_height, "./earth_temp/plots/temp_full_it/full_N.pdf")

    plotting.plot_temps_full_alpha_IR(alpha_v, alpha_irs, n, T_guess, max_height, "./earth_temp/plots/temp_full_it/full_alpha.pdf")
    plotting.plot_temps_full_alpha_IR(alpha_v,  np.linspace(8e-5, 2e-4, 40), n, T_guess, max_height, "./earth_temp/plots/temp_full_it/full_alpha_log.pdf", log=True)
    
    sweeps = np.arange(21)
    plotting.plot_temps_full_sweeps(alpha_v, alpha_ir, n, T_guess, max_height, sweeps, path="./earth_temp/plots/temp_full_it/full_sweeps.pdf")
    """
    plotting.plot_temp_height_it(alpha_v, alpha_ir, n, T_guess, max_height, path="./earth_temp/plots/T_height/T_height_it.pdf")
    plotting.plot_temp_height_mat(alpha_v, alpha_ir, 1000, T_guess, max_height, path="./earth_temp/plots/T_height/T_height_mat.pdf")
    
    alpha_irs = np.logspace(-5, -2, 15, base=10)
    ns = np.logspace(1.5, 3.5, 20, base=10, dtype=np.int64)
    n = 3000
    """
    plotting.plot_temps_full_comp_N(alpha_v, alpha_ir, ns, T_guess, max_height, "./earth_temp/plots/temp_comp/full_N_comp.pdf")

    plotting.plot_temps_full_comp_alpha_IR(alpha_v, alpha_irs, n, T_guess, max_height, "./earth_temp/plots/temp_comp/full_alpha_comp.pdf")
    
    
    ns_it = np.logspace(2, 8, 20, base=10, dtype=np.int64)
    ns_mat = np.logspace(2, 3.8, 20, base=10, dtype=np.int64)
    plotting.plot_performance(ns_it, ns_mat, path="./earth_temp/plots/performance.pdf")
    """

if __name__ == "__main__":
    main()