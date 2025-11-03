
import numpy as np

def fractional_delay(sig, delay_samples):
    # Fractional delay via frequency-domain linear phase
    N = len(sig)
    n = np.arange(N)
    # FFT
    S = np.fft.fft(sig)
    # linear phase shift: exp(-j*2pi*k*delay/N) in freq domain
    k = np.arange(N)
    phase = np.exp(-1j*2*np.pi*k*delay_samples/N)
    y = np.fft.ifft(S*phase)
    return y

def apply_delay_doppler(sig, fs, delay_s, doppler_hz, amp=1.0):
    # Shift in time and apply Doppler
    delay_samples = delay_s * fs
    y = fractional_delay(sig, delay_samples)
    n = np.arange(len(sig))
    y *= np.exp(1j*2*np.pi*doppler_hz*n/fs)
    return amp*y

def surveillance_mix(reference, fs, targets, clutter_amp=0.3, noise_w=1e-3, seed=0):
    """
    targets: list of dicts { 'delay_s':..., 'doppler_hz':..., 'amp': ... }
    Returns surveillance channel as sum(reference through targets) + clutter + noise
    """
    rng = np.random.default_rng(seed)
    y = np.zeros_like(reference, dtype=np.complex128)
    for t in targets:
        y += apply_delay_doppler(reference, fs, t.get('delay_s',0.0), t.get('doppler_hz',0.0), t.get('amp',1.0))
    # simple clutter as attenuated reference + small random multipath
    y += clutter_amp*reference
    # add a couple of random weak multipaths
    for _ in range(3):
        d_s = rng.uniform(0.0, 5e-5)  # up to 50 microseconds
        a = rng.uniform(0.02, 0.05)
        y += apply_delay_doppler(reference, fs, d_s, rng.uniform(-50,50), a)
    # AWGN
    y += (rng.normal(scale=np.sqrt(noise_w/2), size=reference.shape) 
          + 1j*rng.normal(scale=np.sqrt(noise_w/2), size=reference.shape))
    return y
