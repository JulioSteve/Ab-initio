import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# ==================================================
#   SECTION : Normalized Slater functions
# ==================================================
#region NZF
npts = 1000
r = np.linspace(-5,5,npts)

def chi(alpha,R):
    coef = np.sqrt(alpha**3/np.pi)
    RESULT = []
    for r in R:
        arg = -alpha*abs(r)
        res = coef*np.exp(arg)
        RESULT.append(res)
    return np.array(RESULT)

def plot_test1(flag):
    if flag:
        alpha_1 = 1.45 ; alpha_2 = 2.91
        plt.plot(r,chi(alpha_1,r), c='red', label=r"$\alpha_1=1.45$")
        plt.plot(r,chi(alpha_2,r), c='blue', label=r"$\alpha_2=2.91$")
        plt.legend()
        plt.show()
plot_test1(flag=False)
#endregion

def S(ap,aq):
    return (2*np.sqrt(ap*aq)/(ap+aq))**3

def I(ap,aq):
    return 4*(ap*aq-2*(ap+aq))*(np.sqrt(ap*aq)/(ap+aq))**3

def scalprod(ap,aq,ar,als):
    coef = 32*(ap*aq*ar*als)**(3/2)
    term1 = 1/((ap+aq)**3*(ar+als)**2)
    term2 = 1/((ap+aq)**3*(ap+aq+ar+als)**2)
    term3 = 1/((ap+aq)**2*(ap+aq+ar+als)**3)
    return coef*(term1-term2-term3)

def F_el(ap,aq,CLIST,ALIST):
    if len(ALIST)!=len(CLIST):
        raise ValueError("TAILLES LISTES DIFFERENTES DANS CALCUL ELEMENT F !")
    f = 0
    for i in range(len(CLIST)):
        for j in range(len(CLIST)):
            f += CLIST[i]*CLIST[j]*scalprod(ap,aq,ALIST[i],ALIST[j])
    return f+I(ap,aq)

def Energ(CLIST,ALIST):
    el = 0
    for i in range(len(CLIST)):
        for j in range(len(CLIST)):
            el += CLIST[i]*CLIST[j]*(I(ALIST[i],ALIST[j])+F_el(ALIST[i],ALIST[j],CLIST,ALIST))
    return el*27.2114

alpha_1 = 1.45 ; alpha_2 = 2.91
res = np.round(Energ([1,0],[alpha_1,alpha_2]),5)
print("Total initial energy is "+str(res)+" eV.\n")

S = np.array([[S(alpha_1,alpha_1), S(alpha_1,alpha_2)],[S(alpha_2,alpha_1), S(alpha_2,alpha_2)]])
Sd, P = LA.eigh(S)
Sd = np.diag(Sd**(-1/2))
Pm = LA.inv(P)
SLowdin = P@Sd@Pm

def buildFLowdin(CLIST, ALIST):
    F11 = F_el(alpha_1,alpha_1,CLIST,[alpha_1,alpha_2])
    F12 = F_el(alpha_1,alpha_2,CLIST,[alpha_1,alpha_2])
    F21 = F_el(alpha_2,alpha_1,CLIST,[alpha_1,alpha_2])
    F22 = F_el(alpha_2,alpha_2,CLIST,[alpha_1,alpha_2])
    F = np.array([[F11,F12],[F21,F22]])

    FLowd = SLowdin@F@SLowdin
    return FLowd

def OneRound(C_old, ALIST):
    eps, Cnew = LA.eigh(buildFLowdin(C_old, ALIST))
    Cnew = SLowdin@Cnew[:,0]

    energ = Energ(Cnew,ALIST)
    # print(energ)
    return Cnew, energ

def integration(eps=1e-8, maxsteps=100, ALIST=[alpha_1, alpha_2], debug_print=False):
    C = [1,0]
    E = Energ(C, ALIST)
    compteur = 1
    if debug_print:
        print("Iteration n°1:")
    Cnew, Enew = OneRound(C, ALIST)
    breakflag = False

    EList = [E, Enew]

    while abs(E-Enew)>=eps:
        if debug_print:
            print(f"DEBUG: New energy is {Enew:.2f} eV.\n")
        E = Enew
        C = Cnew
        Cnew, Enew = OneRound(C, ALIST)
        EList.append(Enew)

        compteur += 1
        if debug_print:
            print(f"DEBUG: energy difference was {abs(Enew-E):.1e} for iteration n°{compteur}")
        if compteur>=maxsteps:
            breakflag = True
            print("Breaking while for maxsteps.")
            break

    if breakflag == False:
        if debug_print:
            print("") #visual improvement of in the terminal
        print(f"Ab-initio simulation gives a total energy of {Enew:.5f} eV after {compteur} iterations.")
    else:
        print("Correct function, convergence not acquired, no energy results.")
    
    return EList

Conv = np.array(integration(debug_print=True))

def Convplot(plot):
    if plot:
        f,ax = plt.subplots(3,2, layout="constrained")
        f.suptitle("Energy convergence and comparison:", size=20, color="darkred", fontweight='bold')

        ax[0,0].hlines(-79, 0, 8, label="Experimental value", ls="dashed", color="olivedrab", lw=4)
        ax[0,0].plot(np.arange(0, len(Conv)), Conv, marker="o", label="Hartree-Fock-Roothan simulation", color="indianred", markersize=10, lw=5)
        ax[0,0].hlines(-77.879, 0, 8, label="Hartee-Fock limit", ls="dotted", color="royalblue", lw=4)
        ax[0,0].set_title("Global comparison and convergence:", size=20, fontweight='bold')
        ax[0,0].set_ylabel(r'E (eV)', size=20)
        ax[0,0].legend(fontsize='xx-large')
        ax[0,0].set_xticks(np.arange(0,8), [])

        ax[1,0].plot(np.arange(0, len(Conv)), abs(Conv+77.879), marker="o", color="rebeccapurple", markersize=10, lw=5)
        ax[1,0].set_title(r"$\left| E_{HFR}-E_{limHF} \right|$", size=20)
        ax[1,0].set_ylabel(r'$\Delta E$ (no units)', size=20)
        ax[1,0].set_xticks(np.arange(0,8), [])

        ax[2,0].plot(np.arange(0, len(Conv)), abs(Conv+79), marker="o", color="goldenrod", markersize=10, lw=5)
        ax[2,0].set_title(r"$\left| E_{HFR}-E_{exp} \right|$", size=20)
        ax[2,0].set_ylabel(r'$\Delta E$ (no units)', size=20)
        ax[2,0].set_xlabel("Number of iterations", size=20)
        ax[2,0].set_ylim(-0.5,3.5)

        # ZOOM
        cutval = 3

        ax[0,1].hlines(-79, cutval, 8, label="Experimental value", ls="dashed", color="olivedrab", lw=4)
        ax[0,1].plot(np.arange(cutval, len(Conv)), Conv[cutval:], marker="o", label="Hartree-Fock-Roothan simulation", color="indianred", markersize=10, lw=5)
        ax[0,1].hlines(-77.879, cutval, 8, label="Hartee-Fock limit", ls="dotted", color="royalblue", lw=4)
        ax[0,1].set_title("ZOOM:", size=20, fontweight='bold')
        ax[0,1].set_ylabel(r'E (eV)', size=20)
        ax[0,1].set_xticks(np.arange(cutval,8), [])

        ax[1,1].plot(np.arange(cutval, len(Conv)), abs(Conv[cutval:]+77.879), marker="o", color="rebeccapurple", markersize=10, lw=5)
        ax[1,1].set_title(r"$\left| E_{HFR}-E_{limHF} \right|$", size=20)
        ax[1,1].set_ylabel(r'$\Delta E$ (no units)', size=20)
        ax[1,1].set_xticks(np.arange(cutval,8), [])

        ax[2,1].plot(np.arange(cutval, len(Conv)), abs(Conv[cutval:]+79), marker="o", color="goldenrod", markersize=10, lw=5)
        ax[2,1].set_title(r"$\left| E_{HFR}-E_{exp} \right|$", size=20)
        ax[2,1].set_ylabel(r'$\Delta E$ (no units)', size=20)
        ax[2,1].set_xlabel("Selection of iterations", size=20)
        ax[2,1].set_xticks(np.arange(cutval,8), [str(n) for n in np.arange(cutval,8)])


        for ax in f.get_axes():
            ax.tick_params(axis='both', which="major", labelsize=20)
            ax.grid()
        plt.show()

Convplot(plot=False)