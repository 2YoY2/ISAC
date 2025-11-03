
import numpy as np
import matplotlib.pyplot as plt
from waveform.otfs import otfs_tx, otfs_simple_rx
from comm_rx.simple_ofdm import tx as ofdm_tx, rx as ofdm_rx
from channel.parametric_dd import multipath_dd
from metrics.ambiguity import ambiguity
from metrics.plots2 import heat

def run(fs=15.36e6, dopplers_hz=(0, 100, 200, 300, 400), snr_db=20.0, seed=0):
    ofdm_x, Xref, _ = ofdm_tx(n_sc=256, n_sym=32, cp_len=32, seed=seed)
    ofdm_x = ofdm_x/np.sqrt(np.mean(np.abs(ofdm_x)**2))

    otfs_x, Xdd, params, _ = otfs_tx(Nf=32, Nt=32, seed=seed)
    otfs_x = otfs_x/np.sqrt(np.mean(np.abs(otfs_x)**2))

    bler_ofdm = []
    bler_otfs = []
    noise_lin = 10**(-snr_db/10)

    for fd in dopplers_hz:
        paths = [{'delay_s':15e-6, 'doppler_hz':fd, 'amp':0.8},
                 {'delay_s':40e-6, 'doppler_hz':-fd/2, 'amp':0.3}]
        y_ofdm = multipath_dd(ofdm_x, fs, paths, noise_w=noise_lin, seed=seed+1)
        y_otfs = multipath_dd(otfs_x, fs, paths, noise_w=noise_lin, seed=seed+2)

        Y = ofdm_rx(y_ofdm, n_sc=256, n_sym=32, cp_len=32)
        ref = np.fft.fftshift(Xref,axes=1)
        bler_bits = np.mean(np.sign(np.real(Y))!=np.sign(np.real(ref))) \
                  + np.mean(np.sign(np.imag(Y))!=np.sign(np.imag(ref)))
        bler_ofdm.append(bler_bits/2.0)

        ytf = y_otfs.reshape(params['M'], params['N'])
        Ydd = otfs_simple_rx(ytf, params['M'], params['N'])
        errs = np.mean(np.sign(np.real(Ydd))!=np.sign(np.real(Xdd))) \
             + np.mean(np.sign(np.imag(Ydd))!=np.sign(np.imag(Xdd)))
        bler_otfs.append(errs/2.0)

    # Use frame-length-dependent L for ambiguity
    L_ofdm = min(2048, len(ofdm_x))
    L_otfs = min(1024, len(otfs_x))
    RD_ofdm = ambiguity(ofdm_x, L=L_ofdm, K=32, overlap=0.5)
    RD_otfs = ambiguity(otfs_x, L=L_otfs, K=32, overlap=0.5)

    heat(RD_ofdm, "OFDM Ambiguity (proxy)", "Range bins", "Doppler bins",
         path="data/results/ofdm_ambiguity.png")
    heat(RD_otfs, "OTFS Ambiguity (proxy)", "Range bins", "Doppler bins",
         path="data/results/otfs_ambiguity.png")

    plt.figure(figsize=(6.4,4.2))
    xs = np.array(dopplers_hz)
    plt.plot(xs, bler_ofdm, marker='o', label='OFDM')
    plt.plot(xs, bler_otfs, marker='s', label='OTFS')
    plt.xlabel("Doppler (Hz)")
    plt.ylabel("Proxy BLER")
    plt.title("BLER vs Doppler (hard-decision proxies)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/results/bler_vs_doppler.png")

    return dict(bler_ofdm=bler_ofdm, bler_otfs=bler_otfs, dopplers=dopplers_hz)
