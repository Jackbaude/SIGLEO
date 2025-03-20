import numpy as np

def genPss():
    """
    genPss generates the Starlink downlink primary synchronization sequence (PSS).

    Returns:
    --------
    pss : np.ndarray
        (N + Ng) x 1 PSS. For Starlink, N = 1024 and Ng = 32. Returned sampled at 240e6 Hz.
    """
    # Duration of OFDM symbol guard interval (cyclic prefix), expressed in intervals of 1/Fs.
    Ng = 32
    # Number of sub-segments in the primary synchronization sequence (PSS)
    NpssSeg = 8

    # Fibonacci LFSR
    ciVec = np.array([3, 7])  # Feedback taps
    a0Vec = np.array([0, 0, 1, 1, 0, 1, 0])  # Initial state
    n = 7  # Number of bits in LFSR
    m = 2**n - 1  # Length of LFSR sequence

    # Generate LFSR sequence
    lfsrSeq = np.zeros(m, dtype=int)
    for idx in range(m):
        buffer = a0Vec[ciVec - 1]  # Python uses 0-based indexing
        val = np.sum(buffer) % 2  # Cascaded XOR is the same as parity
        a0Vec = np.roll(a0Vec, 1)
        a0Vec[0] = val
        lfsrSeq[idx] = val

    # Modify LFSR sequence
    lfsrSeqMod = np.flip(lfsrSeq)
    lfsrSeqMod = np.insert(lfsrSeqMod, 0, 0)  # Add a leading zero
    seq = 2 * lfsrSeqMod - 1  # Convert binary sequence to ±1

    # Generate pkSegVec
    pkSegVec = np.exp(-1j * np.pi / 4 - 1j * 0.5 * np.pi * np.cumsum(seq))

    # Construct pkVec
    pkVec = np.concatenate([-pkSegVec, np.tile(pkSegVec, NpssSeg - 1)])

    # Add cyclic prefix
    pkCP = pkVec[-Ng:]  # Last Ng samples
    pss = np.concatenate([-pkCP, pkVec])

    return pss


# Example usage
if __name__ == "__main__":
    pss = genPss()
    print("PSS shape:", pss.shape)
    print("PSS samples:", pss[:10])  # Print first 10 samples