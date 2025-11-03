
import numpy as np

def apply_single_path(sig, fs, delay_s, doppler_hz, amp=1.0):
    N = len(sig)
    k = np.arange(N)
    S = np.fft.fft(sig)
    phase = np.exp(-1j*2*np.pi*np.arange(N)*delay_s*fs/N)
    y = np.fft.ifft(S*phase)
    y *= np.exp(1j*2*np.pi*doppler_hz*k/fs)
    return amp*y

def multipath_dd(sig, fs, paths, noise_w=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(sig, dtype=np.complex128)
    for p in paths:
        y += apply_single_path(sig, fs, p.get('delay_s',0.0), p.get('doppler_hz',0.0), p.get('amp',1.0))
    y += (rng.normal(scale=np.sqrt(noise_w/2), size=sig.shape)
          +1j*rng.normal(scale=np.sqrt(noise_w/2), size=sig.shape))
    return y
