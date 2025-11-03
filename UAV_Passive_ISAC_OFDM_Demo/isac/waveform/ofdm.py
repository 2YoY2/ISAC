
import numpy as np

def qpsk_mod(bits):
    bits = bits.reshape(-1,2)
    symbols = (1-2*bits[:,0]) + 1j*(1-2*bits[:,1])
    symbols /= np.sqrt(2)
    return symbols

def ofdm_tx(n_sc=256, n_sym=64, cp_len=32, pilot_spacing=16, snr_db=None, seed=0):
    """
    Generate a simple CP-OFDM baseband time-domain signal with QPSK data and comb pilots.
    Returns: tx (complex samples), X (freq grid), params dict
    """
    rng = np.random.default_rng(seed)
    n_data = n_sc*n_sym
    bits = rng.integers(0,2, size=(n_data,2))
    X = qpsk_mod(bits).reshape(n_sym, n_sc)

    # insert pilots (BPSK) every pilot_spacing subcarriers
    pilot_idx = np.arange(0, n_sc, pilot_spacing)
    pilots = (2*rng.integers(0,2,size=(n_sym, pilot_idx.size))-1).astype(np.complex128)
    X[:, pilot_idx] = pilots

    # IFFT per symbol
    x_syms = np.fft.ifft(np.fft.ifftshift(X, axes=1), axis=1)
    # add CP
    x_cp = np.concatenate([x_syms[:,-cp_len:], x_syms], axis=1)
    tx = x_cp.reshape(-1)

    params = dict(n_sc=n_sc, n_sym=n_sym, cp_len=cp_len, pilot_spacing=pilot_spacing, pilot_idx=pilot_idx)
    return tx.astype(np.complex128), X, params

def ofdm_symbol_len(n_sc, cp_len):
    return n_sc + cp_len
