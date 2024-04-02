import physics, plotting
import numpy as np

def main():
    #plotting.plot_density(n = 10000, Ts = np.linspace(270, 300, 4), h_max = 100e3, path = "./earth_temp/plots/density/density.pdf")
    #plotting.plot_visible_intensity(alphas=np.linspace(1e-5, 1e-4, 5), n = 10000, Ts = [290], h_max=100e3, path="./earth_temp/plots/visible_intensity/visible_intensity.pdf")
    #plotting.plot_visible_intensity(alphas=[5e-5], n = 10000, Ts = np.linspace(100, 1000, 10), h_max=100e3, path="./earth_temp/plots/visible_intensity/visible_intensity_T.pdf")
    #plotting.plot_temps_visible_absorption(alphas=np.linspace(0, 1e-4, 1000), n=10000, T=290, h_max=100e3, path="./earth_temp/plots/temp_visible/temp_visible.pdf", surface_albedos=[0, 0.04])
    
    alpha_ir = 0.2
    alpha_irs = np.logspace(-5, 1, 40, base=10)
    ns = np.logspace(2, 7, 30, base=10, dtype=np.int64)
    n = int(1e5)
    plotting.plot_temps_full_N(5e-5, alpha_ir, ns, 290, 100e3, "./earth_temp/plots/test_full_N.pdf")
    plotting.plot_temps_full_alpha_IR(5e-5, alpha_irs, n, 290, 100e3, "./earth_temp/plots/test_full_alpha.pdf")
    #plotting.plot_temps_full_comp_N(5e-5, alpha_ir, ns, 290, 10e3, "./earth_temp/plots/test_full_N_comp.pdf")
    #plotting.plot_temps_full_comp_alpha_IR(5e-5, alpha_irs, n, 290, 10e3, "./earth_temp/plots/test_full_alpha_comp.pdf")


if __name__ == "__main__":
    main()