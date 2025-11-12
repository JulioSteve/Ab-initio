import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

A, E = np.loadtxt("LATPARAM.dat", unpack=True)
V = A**3

def BM(V,E0,V0,B0):
    coef = 9/16*V0*B0
    term1 = B0*((V0/V)**(2/3)-1)**3
    term2 = ((V0/V)**(2/3)-1)**2
    term3 = 6-4*(V0/V)**(2/3)

    return E0 + coef*(term1+term2*term3)


