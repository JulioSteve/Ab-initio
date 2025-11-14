import numpy as np
import matplotlib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


A, E = np.loadtxt("LATPARAM.dat", unpack=True)
V = A**3

def BM(V,E0,V0,B0,B0prime):
    coef = 9/16*V0*B0
    term1 = B0prime*((V0/V)**(2/3)-1)**3
    term2 = ((V0/V)**(2/3)-1)**2
    term3 = 6-4*(V0/V)**(2/3)

    return E0 + coef*(term1+term2*term3)


popt, pcov = curve_fit(BM, V, E, )