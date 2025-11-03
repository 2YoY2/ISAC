
import numpy as np

def qpsk_mod(bits):
    bits = bits.reshape(-1,2)
    syms = (1-2*bits[:,0]) + 1j*(1-2*bits[:,1])
    syms /= np.sqrt(2)
    return syms

def tx(n_sc=256, n_sym=32, cp_len=32, seed=0):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0,2,size=(n_sc*n_sym,2))
    X = qpsk_mod(bits).reshape(n_sym, n_sc)
    x_syms = np.fft.ifft(np.fft.ifftshift(X, axes=1), axis=1)
    x_cp = np.concatenate([x_syms[:,-cp_len:], x_syms], axis=1)
    x = x_cp.reshape(-1)
    return x.astype(np.complex128), X, bits

def rx(y, n_sc=256, n_sym=32, cp_len=32):
    symlen = n_sc+cp_len
    y = y[:n_sym*symlen].reshape(n_sym, symlen)
    y_no_cp = y[:, cp_len:]
    Y = np.fft.fftshift(np.fft.fft(y_no_cp, axis=1), axes=1)
    return Y
