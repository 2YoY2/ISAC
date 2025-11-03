
import numpy as np
import matplotlib.pyplot as plt

def plot_rd(RD, vmax=None, title="Range–Doppler Map", xlabel="Range bins", ylabel="Doppler bins"):
    plt.figure(figsize=(7,4.5))
    if vmax is None:
        vmax = np.percentile(RD, 99.5)
    plt.imshow(RD, aspect='auto', origin='lower', vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
