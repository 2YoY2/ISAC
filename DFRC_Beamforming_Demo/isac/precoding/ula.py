
import numpy as np

def steering_vector(M, theta_rad, d_lam=0.5):
    m = np.arange(M)
    return np.exp(1j*2*np.pi*d_lam*m*np.sin(theta_rad))

def beampattern(w, thetas_rad, d_lam=0.5, normalize=True):
    M = len(w)
    A = np.exp(1j*2*np.pi*d_lam*np.arange(M)[:,None]*np.sin(thetas_rad)[None,:])
    y = np.conj(w).T @ A
    p = np.abs(y)**2
    if normalize and np.max(p) > 0:
        p = p/np.max(p)
    return p.real

def channel_ue(M, theta_rad, d_lam=0.5, kappa=0.1, seed=0):
    rng = np.random.default_rng(seed)
    a = steering_vector(M, theta_rad, d_lam)
    h_nlos = (rng.normal(size=M)+1j*rng.normal(size=M))/np.sqrt(2)
    h = a + kappa*h_nlos
    return h/np.linalg.norm(h)
