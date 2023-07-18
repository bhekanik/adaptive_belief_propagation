import numpy as np
from scipy.linalg import lu
from scipy.special import logit


def ADP(H, r, iter, alfa, mod):
    """
    This function performs the Adopting parity Check matrix (ADP) decoding
    algorithm for Reed Solomon codes.
    The algorithm was adopted for the paper 'Iterative Soft-Input Soft-Output
    Decoding of Reed-Solomon Codes by Adopting the Parity Check Matrix' which
    was authored by Jing Jiang and Krishna R. Narayanan
    NOTE: Works for BPSK modulation i.e works with bits not symbols
    where
    H = The parity check matrix in the GF field
    r = The received vector from the noisy channel
    iter = The number of iterations
    mod = 1 = BPSK modulation and 2 = 16-QAM modulation
    """

    N0 = 1
    if mod == 1:  # for BPSK
        LLR = 4 * r / N0
    elif mod == 2:  # for 16-QAM
        LLR = logit(r)  # equivalent to LLR16 in MATLAB

    llr = LLR
    hbt = np.zeros_like(H)  # preallocation for transformed matrix

    for i in range(iter):
        it = i
        # The matrix updating stage
        ll = np.abs(llr)
        v = np.argsort(ll)  # indexing from lowest to highest
        P, L, U = lu(H[:, v])  # LU decomposition, equivalent to gfrref in MATLAB
        hb = np.dot(P, U)
        hbt[:, v] = hb  # transformed H matrix
        hbz = hbt != 0  # using non-gf hb

        # The bit reliability updating stage
        C, new_LLR = logSPA(hbz, llr, 1, alfa)  # decoding
        llr = new_LLR

        # EARLY STOPPING CRITERION
        S = np.dot(C, hbt.T)
        if np.all(S == 0):
            break

    return C, it, llr, hbz, S
