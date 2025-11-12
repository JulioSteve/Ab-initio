import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

Xref, Yref = np.loadtxt("ENCUT20Kpoints.dat", unpack=True)
Xref = Xref[:-1]
Yref = Yref[:-1]

X4,Y4 = np.loadtxt("ENCUT_444.dat", unpack=True)
X6,Y6 = np.loadtxt("ENCUT_666.dat", unpack=True)
X8,Y8 = np.loadtxt("ENCUT_888.dat", unpack=True)
X10,Y10 = np.loadtxt("ENCUT_101010.dat", unpack=True)

def plot(flag):
    if flag:
        plt.rc('lines', linewidth=3, markersize=10, marker=".", linestyle="solid")
        fig,ax = plt.subplots(1,1, figsize=(2560/300, 1440/300), layout="constrained")
        ax.plot(Xref,Yref, c="black", label="20x20x20")
        ax.plot(X4,Y4, c="green", label="4x4x4")
        ax.plot(X6,Y6, c="blue", label="6x6x6")
        ax.plot(X8,Y8, c="red", label="8x8x8")
        ax.plot(X10,Y10, c="orange", label="10x10x10")

        ax.legend(loc="upper left")

        plt.show()

def plot2(flag):
    if flag:
        plt.rc('lines', linewidth=3, markersize=10, marker=".", linestyle="solid")
        fig,ax = plt.subplots(1,1, figsize=(2560/300, 1440/300), layout="constrained")
        ax.plot(Xref,Yref, c="black", label="20x20x20")
        ax.plot(X8,Y8, c="red", label="8x8x8")
        ax.plot(X10,Y10, c="orange", label="10x10x10")
        ax.set_ylim(-16.465, -16.46)

        ax.legend(loc="upper left")

        plt.show()

def plot3(flag):
    if flag:
        plt.rc('lines', linewidth=3, markersize=10, marker=".", linestyle="solid")
        fig,ax = plt.subplots(1,1, figsize=(2560/300, 1440/300), layout="constrained")
        # ax.plot(Xref,Yref, c="black", label="20x20x20")
        ax.plot(X8,Y8-Yref, c="red", label="8x8x8")
        ax.plot(X10,Y10-Yref, c="orange", label="10x10x10")
        # ax.hlines(1, X8.min(), X8.max(), color="black")

        ax.legend(loc="upper left")

        plt.show()

plot(False)
plot2(True)
plot3(True)

