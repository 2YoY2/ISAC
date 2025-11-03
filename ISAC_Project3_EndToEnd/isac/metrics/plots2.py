
import numpy as np
import matplotlib.pyplot as plt

def heat(R, title, xlabel, ylabel, vmax=None, path=None):
    plt.figure(figsize=(7,4.5))
    if vmax is None:
        vmax = np.percentile(R, 99.5)
    plt.imshow(R, aspect='auto', origin='lower', vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    if path is not None:
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
