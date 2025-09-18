import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

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
print("Energy is "+str(res)+" eV.")

S = np.array([[S(alpha_1,alpha_1), S(alpha_1,alpha_2)],[S(alpha_2,alpha_1), S(alpha_2,alpha_2)]])
Sd, P = LA.eigh(S)
Sd = np.diag(Sd**(-1/2))
Pm = LA.inv(P)
SLowdin = P@Sd@Pm

def buildFLowdin(CLIST):
    alpha_1 = 1.45 ; alpha_2 = 2.91
    F11 = F_el(alpha_1,alpha_1,CLIST,[alpha_1,alpha_2])
    F12 = F_el(alpha_1,alpha_2,CLIST,[alpha_1,alpha_2])
    F21 = F_el(alpha_2,alpha_1,CLIST,[alpha_1,alpha_2])
    F22 = F_el(alpha_2,alpha_2,CLIST,[alpha_1,alpha_2])
    F = np.array([[F11,F12],[F21,F22]])

    FLowd = SLowdin@F@SLowdin
    return FLowd

# print(buildFLowdin([1,0]))
eps, Cnew = LA.eigh(buildFLowdin([1,0]))
print(Cnew[:,0])
print(SLowdin@Cnew[:,0])