import numpy as np
import matplotlib.pyplot as plt

def en(n,L):
    return n**2*np.pi**2/(2*L**2)


L0 = 1
N0 = 10
def E_exacte(N,L):
    E_ex = 0
    for n in range(1,N+1):
        E_ex += en(n,L)
    return E_ex

def psi(n,L,x):
    coef = np.sqrt(2/L)
    fac = n*np.pi*x/L
    return coef*np.sin(fac)

def edens(N,L,x):
    res = 0
    for n in range(1,N+1):
        res += abs(psi(n,L,x)**2)
    return res

def E_KDF(steps,N,L):
    coef = np.pi**2/6
    X = np.linspace(0,L,steps)
    dx = X[1]-X[0]
    res = 0
    for x in X:
        res += dx*edens(N,L,x)**3
    return res*coef

# print(f"For N = {N0}:"+"\n"+f"E_exact = {E_exacte(N0,L0)}"+"\n"+f"E_KDF = {E_KDF(10000,N0,L0)}")

E_ex_LIST = np.array([E_exacte(n,L0) for n in range(1,11)])
E_KDF_LIST = np.array([E_KDF(10000,n,L0) for n in range(1,11)])

diff_abs = np.abs((E_KDF_LIST-E_ex_LIST))
diff_rel = np.abs((E_KDF_LIST-E_ex_LIST)/E_ex_LIST)*100

def plot(flag):
    f,(ax1,ax2,ax3) = plt.subplots(3,sharex=True, layout="constrained")
    X = np.arange(1,11)
    if flag:
        ax1.set_title(r"Energy evolution w/r Number of electrons:")
        ax1.plot(X,E_ex_LIST, color="indianred", marker="o", label=r"E_{ex}", lw=3, markersize=10)
        ax1.plot(X,E_KDF_LIST, color="royalblue", marker="x", label=r"E_{KDF}", lw=3, markersize=10)
        ax1.grid()
        ax1.legend()

        ax2.plot(X,diff_abs, color="purple", marker="*", lw=3, markersize=10)
        ax2.set_title(r"Relative difference: $\left|E_{KDF}-E_{ex}\right|$")
        ax2.grid()

        ax3.plot(X,diff_rel, color="olivedrab", marker="*", lw=3, markersize=10)
        ax3.set_title(r"Relative difference: $\left|\frac{E_{KDF}-E_{ex}}{E_{ex}}\right|$")
        ax3.set_ylabel("in %")
        ax3.grid()
        ax3.set_xticks(X,[str(tick) for tick in X])

        plt.show()
plot(True)
