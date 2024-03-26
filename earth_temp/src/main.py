import physics, plotting

def main():
    plotting.plot_density(n = 1000, T = 290, h_max = 100e3, path = "./earth_temp/plots/density/density.pdf")

    return

if __name__ == "__main__":
    main()