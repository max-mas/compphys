import physics, plotting
import numpy as np

def main():
    plotting.plot_density(n = 10000, Ts = np.linspace(270, 300, 4), h_max = 100e3, path = "./earth_temp/plots/density/density.pdf")
    plotting.plot_visible_intensity(alphas=np.linspace(1e-5, 1e-4, 5), n = 10000, Ts = [290], h_max=100e3, path="./earth_temp/plots/visible_intensity/visible_intensity.pdf")
    plotting.plot_visible_intensity(alphas=[5e-5], n = 10000, Ts = np.linspace(100, 1000, 10), h_max=100e3, path="./earth_temp/plots/visible_intensity/visible_intensity_T.pdf")
    plotting.plot_temps_visible_absorption(alphas=np.linspace(1e-5, 1e-4, 1000), n=10000, T=290, h_max=100e3, path="./earth_temp/plots/temp_visible/temp_visible.pdf", surface_albedos=[0, 0.04])
    

if __name__ == "__main__":
    main()