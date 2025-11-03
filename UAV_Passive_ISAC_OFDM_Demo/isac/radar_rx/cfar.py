
import numpy as np

def ca_cfar(rd, guard_cells=(2,2), ref_cells=(8,8), pfa=1e-3):
    """
    Simple 2D CA-CFAR on RD map magnitude.
    rd: 2D array [K x L]
    Returns detection mask of same shape (bool).
    """
    K, L = rd.shape
    gD, gR = guard_cells
    rD, rR = ref_cells
    det = np.zeros_like(rd, dtype=bool)

    # scale factor (approximate) using cell-averaging: threshold = alpha * noise_mean
    # Alpha from Pfa for CA-CFAR 1D; we approximate in 2D by using product window count
    nref = (2*rD+2*rR+1)*(2*rR+2*rD+1) - (2*gD+1)*(2*gR+1)
    if nref <= 0:
        nref = 1
    alpha = nref*(pfa**(-1/nref)-1)  # rough

    for i in range(rD+gD+1, K-(rD+gD+1)):
        for j in range(rR+gR+1, L-(rR+gR+1)):
            # extract reference window excluding guard and CUT
            rd_win = rd[i-(rD+gD+1):i+(rD+gD+2), j-(rR+gR+1):j+(rR+gR+2)].copy()
            rd_win[(rD):(rD+2*gD+1), (rR):(rR+2*gR+1)] = 0.0  # zero-out guard+CUT approx
            noise = np.mean(rd_win[rd_win>0])
            thresh = alpha*noise
            det[i,j] = rd[i,j] > thresh
    return det
