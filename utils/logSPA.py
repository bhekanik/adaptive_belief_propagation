import numpy as np


def logSPA(H, LLR, iter, alfa):
    """
    This function performs the SPA decoding algorithm in the log domain. It's
    a simplification of the SPA in the log domain. This is the algorithm
    done in the book 'Channel Codes Classical and Modern' by Shu Lin and
    Willian E Ryan

    Parameters:
    H : The Binary Parity check matrix
    LLR : The log likelihood ratios
    iter : Number of iterations
    alfa : Damping factor. Not used in decoding of LDPC However it is used in the ADP
    """

    # Initialization
    mm, nn = H.shape
    Z = np.zeros((mm, nn))
    Lrij = Z
    Lqji = H * np.tile(LLR, (mm, 1))
    Lj = np.zeros(nn)
    Lext = np.zeros(nn)

    for it in range(iter):
        # The horizontal step
        for i in range(mm):
            cj = np.nonzero(H[i, :])[0]  # indices of nonzeros in each column
            for ij in range(len(cj)):
                lij = 1
                for in_ in range(len(cj)):
                    if in_ != ij:  # all bits except bit n
                        lij = lij * np.tanh(Lqji[i, cj[in_]] / 2)
                Lrij[i, cj[ij]] = 2 * np.arctanh(lij)  # horizontal step(CN) update

        # The vertical step
        for j in range(nn):
            ri = np.nonzero(H[:, j])[0]  # indices of nonzeros in each row
            Lext[j] = np.sum(Lrij[ri, j])
            Lj[j] = LLR[j] + alfa * np.sum(Lrij[ri, j])

    V = Lj > 0
    return V, Lj
