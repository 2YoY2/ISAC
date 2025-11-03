
import numpy as np

def qpsk_mod(bits):
    bits = bits.reshape(-1,2)
    syms = (1-2*bits[:,0]) + 1j*(1-2*bits[:,1])
    syms /= np.sqrt(2)
    return syms

def qpsk_demod(syms):
    b0 = (np.real(syms)<0).astype(int)
    b1 = (np.imag(syms)<0).astype(int)
    return np.vstack([b0,b1]).T.reshape(-1)

def otfs_tx(Nf=32, Nt=32, seed=0):
    rng = np.random.default_rng(seed)
    M = Nt; N = Nf
    n_syms = M*N
    bits = rng.integers(0,2, size=(n_syms,2))
    Xdd = qpsk_mod(bits).reshape(M,N)
    Xtf = (1/np.sqrt(N))*np.fft.ifft((1/np.sqrt(M))*np.fft.ifft(Xdd, axis=0), axis=1)
    tx_syms = np.fft.ifft(np.fft.ifftshift(Xtf, axes=1), axis=1)
    x = tx_syms.reshape(-1)
    params = dict(M=M, N=N)
    return x.astype(np.complex128), Xdd, params, bits

def otfs_simple_rx(ytf, M, N):
    Ydd = (1/np.sqrt(M))*np.fft.fft((1/np.sqrt(N))*np.fft.fft(ytf, axis=1), axis=0)
    return Ydd
