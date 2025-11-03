
import numpy as np

def rd_map(reference, surveillance, L, K, overlap=0.5, window=None):
    """
    Passive radar-style processing: split signals into K blocks of length L with overlap.
    For each block, compute cross-correlation via FFT (range), then FFT over slow-time (Doppler).
    Returns RD map [K x L] (Doppler x Range), range_lags, doppler_bins
    """
    if window is None:
        window = np.hanning(L)

    step = int(L*(1-overlap))
    # build blocks
    ref_blocks = []
    sur_blocks = []
    for start in range(0, min(len(reference), len(surveillance))-L+1, step):
        ref_blocks.append(reference[start:start+L]*window)
        sur_blocks.append(surveillance[start:start+L]*window)
        if len(ref_blocks) >= K:
            break
    ref_blocks = np.stack(ref_blocks, axis=0)  # [K,L]
    sur_blocks = np.stack(sur_blocks, axis=0)

    # FFT for fast-time correlations
    Nfft = int(2**np.ceil(np.log2(L)))
    Rf = np.fft.fft(ref_blocks, n=Nfft, axis=1)
    Sf = np.fft.fft(sur_blocks, n=Nfft, axis=1)
    # cross-correlation via frequency domain
    X = Sf*np.conj(Rf)
    x_tau = np.fft.ifft(X, axis=1)[:, :L]  # keep positive lags

    # Doppler via FFT across slow-time K
    RD = np.fft.fftshift(np.fft.fft(x_tau, axis=0), axes=0)  # [K,L]
    # magnitude
    RDmag = np.abs(RD)

    range_lags = np.arange(L)
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(K))
    return RDmag, range_lags, doppler_bins
