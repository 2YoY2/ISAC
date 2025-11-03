
import numpy as np
from .ula import steering_vector

def design_weights_ridge(M, theta_targets, gains, theta_sidelobes=None, sl_weight=5.0, lam=1e-3, d_lam=0.5):
    A_rows = []
    b = []
    for th, g in zip(theta_targets, gains):
        A_rows.append(steering_vector(M, th, d_lam))
        b.append(g)
    if theta_sidelobes is not None and len(theta_sidelobes) > 0:
        for th in theta_sidelobes:
            A_rows.append(sl_weight*steering_vector(M, th, d_lam))
            b.append(0.0)
    A = np.vstack(A_rows)
    b = np.array(b, dtype=np.complex128)
    AhA = A.conj().T @ A
    rhs = A.conj().T @ b
    AhA += lam*np.eye(AhA.shape[0])
    w = np.linalg.solve(AhA, rhs)
    w = w/np.linalg.norm(w)
    return w

def sinr_ue(w, h_ue, noise_var=1.0):
    num = np.abs(np.conj(h_ue).T @ w)**2
    return (num/noise_var).real
