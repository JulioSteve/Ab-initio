import numpy as np
import matplotlib.pyplot as plt

ENERG = np.loadtxt("energy.dat")
R = np.linspace(0.8,3.0,30)

plt.plot(R,ENERG, lw=3, color="black")
plt.show()