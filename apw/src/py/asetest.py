from ase.build import bulk
from ase.calculators.test import FreeElectrons
#from ase.calculators.espresso import Espresso

from ase.dft.kpoints import monkhorst_pack

import matplotlib.pyplot as plt
import matplotlib as mpl

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

a = bulk('Li')
a.calc = FreeElectrons(nvalence=1,
                       kpts={'path': 'GHNGPH', 'npoints': 200})
a.get_potential_energy()
bs = a.calc.band_structure()
ax = bs.plot(emin=0, emax=20, filename='li.pdf')
ax.set_ylabel("$E$ (eV)")

fig = plt.figure()
fig.add_axes(ax)

fig.savefig("li.pdf")

b = monkhorst_pack((5, 5, 5))
print(len(b))
print(b)