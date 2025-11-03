
import numpy as np
import matplotlib.pyplot as plt
from precoding.ula import beampattern, channel_ue, steering_vector
from precoding.pattern_fit import design_weights_ridge, sinr_ue

def run(M=12, d_lam=0.5, theta_ue_deg=10.0, theta_sense_deg=40.0,
        sidelobe_span=(-90,90), sidelobe_exclude=8.0, grid_deg=0.25,
        g_sense_db_list=( -10, -7, -5, -3, 0, 3, 6 ),
        sl_weight=8.0, lam=1e-3, noise_var=1.0, seed=0):
    theta_ue = np.deg2rad(theta_ue_deg)
    theta_s = np.deg2rad(theta_sense_deg)

    thetas_all = np.deg2rad(np.arange(sidelobe_span[0], sidelobe_span[1]+grid_deg, grid_deg))
    mask_sl = np.ones_like(thetas_all, dtype=bool)
    for th_deg in (theta_ue_deg, theta_sense_deg):
        mask_sl &= np.abs(np.rad2deg(thetas_all) - th_deg) > sidelobe_exclude
    theta_sl = thetas_all[mask_sl]

    h_ue = channel_ue(M, theta_ue, d_lam=d_lam, kappa=0.1, seed=seed+1)

    pareto = []
    weights = {}
    for g_db in g_sense_db_list:
        g_lin = 10**(g_db/20)
        w = design_weights_ridge(
            M,
            theta_targets=[theta_ue, theta_s],
            gains=[1.0+0j, g_lin+0j],
            theta_sidelobes=theta_sl,
            sl_weight=sl_weight,
            lam=lam,
            d_lam=d_lam
        )
        sinr = sinr_ue(w, h_ue, noise_var=noise_var)
        a_s = steering_vector(M, theta_s, d_lam)
        g_sense = np.abs(np.conj(w).T @ a_s)**2
        pareto.append((g_db, 10*np.log10(sinr+1e-12)))
        weights[g_db] = w

    pareto = np.array(pareto)
    mid_idx = len(g_sense_db_list)//2
    w_mid = weights[g_sense_db_list[mid_idx]]

    thetas = np.deg2rad(np.linspace(-90, 90, 1441))
    patt = beampattern(w_mid, thetas, d_lam=d_lam, normalize=True)
    plt.figure(figsize=(7,4.5))
    plt.plot(np.rad2deg(thetas), 10*np.log10(patt+1e-9))
    plt.xlabel("Angle (deg)")
    plt.ylabel("Normalized pattern (dB)")
    plt.title(f"DFRC Beampattern (M={M}, UE={theta_ue_deg}°, Sense={theta_sense_deg}°)")
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(6.2,4.2))
    plt.plot(pareto[:,0], pareto[:,1], marker='o')
    plt.xlabel("Sensing mainlobe target (V gain, dB)")
    plt.ylabel("UE SINR (dB)")
    plt.title("Pareto: UE SINR vs Sensing Gain")
    plt.grid(True)
    plt.tight_layout()

    return pareto, w_mid, thetas, patt

if __name__ == "__main__":
    run()
