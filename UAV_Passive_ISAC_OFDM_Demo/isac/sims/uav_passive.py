
import numpy as np
from waveform.ofdm import ofdm_tx, ofdm_symbol_len
from channel.delay_doppler import surveillance_mix
from radar_rx.rdmap import rd_map
from metrics.plots import plot_rd
import matplotlib.pyplot as plt

def run(seed=0, n_sc=256, n_sym=64, cp_len=32, pilot_spacing=16, fs=15.36e6,
        K=48, L=2048, overlap=0.5, snr_db=20.0):
    # Generate OFDM reference
    tx, X, params = ofdm_tx(n_sc=n_sc, n_sym=n_sym, cp_len=cp_len, pilot_spacing=pilot_spacing, seed=seed)
    # Normalize power
    tx = tx/np.sqrt(np.mean(np.abs(tx)**2))

    # Define targets: one main target and a weaker second
    targets = [
        {'delay_s': 15e-6, 'doppler_hz': 300.0, 'amp': 0.5},
        {'delay_s': 40e-6, 'doppler_hz': -150.0, 'amp': 0.2},
    ]
    # Noise power from desired SNR relative to reference power (reference power is ~1)
    noise_lin = 10**(-snr_db/10)

    surv = surveillance_mix(tx, fs, targets, clutter_amp=0.25, noise_w=noise_lin, seed=seed+1)

    # RD processing
    RD, rbin, dbin = rd_map(tx, surv, L=L, K=K, overlap=overlap)
    plot_rd(RD, title="Passive ISAC UAV — Range–Doppler")
    return RD, rbin, dbin, tx, surv, params

if __name__ == "__main__":
    RD, rbin, dbin, tx, surv, params = run()
    plt.show()
