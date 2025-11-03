
import numpy as np

def ambiguity(x, L, K, overlap=0.5, window=None):
    if window is None:
        window = np.hanning(L)
    step = int(L*(1-overlap))
    xs = []
    for start in range(0, len(x)-L+1, step):
        xs.append(x[start:start+L]*window)
        if len(xs)>=K:
            break
    X = np.stack(xs, axis=0)
    Nfft = int(2**np.ceil(np.log2(L)))
    Xf = np.fft.fft(X, n=Nfft, axis=1)
    R = Xf*np.conj(Xf)
    r_tau = np.fft.ifft(R, axis=1)[:, :L]
    RD = np.fft.fftshift(np.fft.fft(r_tau, axis=0), axes=0)
    return np.abs(RD)
