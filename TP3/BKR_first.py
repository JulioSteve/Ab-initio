import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

np.random.seed(0)

sigma_w = 0.025
sigma_v = 0.1
ksi = 1


PHI = np.linspace(0,10,2000)
phi_test = np.random.rand(10)*10

def ground_truth(phi):
    t1 = 1.5*np.exp(-0.8*(phi-1.5)**2)
    t2 = 2.0*np.exp(-1*(phi-5)**2)
    t3 = 1.8*np.exp(-0.5*(phi-8)**2)
    return t1+t2+t3

def plot_GT(flag):
    if flag:
        fig,ax = plt.subplots(1,1, figsize=(1920/300, 1080/300), layout="constrained")
        ax.plot(PHI, ground_truth(PHI), ls="solid", color="red", lw=3, alpha=0.9)
        ax.plot(phi_test, ground_truth(phi_test), marker=".", ms=15, ls="None", color="black")

        plt.show()

def RBF(phi1,phi2, ksi):
    return np.exp(
        -np.abs(phi1-phi2)**2/(2*ksi**2)
    )

def psi_rbf(phi1, phi2, ksi):
    n1 = len(phi1)
    n2 = len(phi2)

    PSI = np.zeros(shape=(n1 ,n2))

    for i in range(n1):
        for j in range(n2):
            PSI[i][j] = RBF(phi1[i], phi2[j], ksi)

    return PSI

Psi_train = psi_rbf(phi_test, phi_test, ksi)
# print(np.around(Psi_train,2))
# print(f"\nIs symmetric: PSI_train =? PSI_train.T\n {Psi_train==Psi_train.T}")

def train_bkr(PSI, Y,sigma_w, sigma_v):
    n = len(PSI)

    sw = sigma_w**(-2)
    sv = sigma_v**(-2)

    Sm = sw*np.identity(n)+sv*(PSI.T@PSI)
    S = np.linalg.inv(Sm)

    wbar = sv*S@(PSI.T@Y)
    return wbar, S

Y = ground_truth(phi_test)
wbar, SIGMA = train_bkr(Psi_train, Y, sigma_w, sigma_v)

PSI_pred = psi_rbf(PHI, phi_test, ksi)

def predict_bkr(psi_pred, S, wbar, sigma_v):
    mu = psi_pred@wbar
    s2 = sigma_v**2+np.array([psi_x.T@S@psi_x for psi_x in psi_pred])
    
    return mu,s2

mu,s2 = predict_bkr(PSI_pred, SIGMA, wbar, sigma_v)
Ytot = ground_truth(PHI)

def plot_predict(flag):
    if flag:
        fig, ax = plt.subplots(1,1, figsize=(1920/300,1080/300), layout="constrained")
        ax.plot(PHI, ground_truth(PHI), ls="solid", color="red", lw=3, alpha=0.9)
        ax.plot(phi_test, ground_truth(phi_test), marker=".", ms=15, ls="None", color="black")
        ax.plot(PHI, mu, color="black")
        ax.plot(PHI, mu+np.sqrt(s2), color="grey", ls="dashed")
        ax.plot(PHI, mu-np.sqrt(s2), color="grey", ls="dashed")

        plt.savefig("TP3/ksi30.png", format="png", dpi=300)
        plt.show()


plot_GT(False)
plot_predict(False)

def crit(s2pred, phi_pred, phi_test):
    sigma = s2pred**(1/2)
    idx = np.argmax(sigma)

    phi_crit = phi_pred[idx]
    if not(phi_crit in phi_test):
        np.append(phi_test, phi_crit)
