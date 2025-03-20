import numpy as np

def genSss():
    """
    genSss generates the Starlink downlink secondary synchronization sequence (SSS).

    Returns:
    --------
    sssTimeDomainVec : np.ndarray
        (N + Ng)-by-1 SSS expressed in the time domain with a cyclic prefix.
    sssFreqDomainVec : np.ndarray
        N-by-1 SSS expressed in the frequency domain with 4QAM symbols having unit coordinates.
    """
    # Number of subcarriers
    N = 1024
    # Duration of OFDM cyclic prefix (CP), expressed in intervals of 1/Fs
    Ng = 32

    # Load precomputed frequency-domain SSS vector
    # Replace this with the actual loading mechanism for sssVecFull
    # For now, we'll simulate a placeholder vector
    sssVecFull = np.ones(N, dtype=complex)  # Placeholder: replace with actual data
    sssVecFull[1::2] = -1  # Simulate 4QAM symbols (±1 ±1j)

    # Validate the loaded SSS vector
    if len(sssVecFull) != N:
        raise ValueError('Frequency domain SSS expression must be of length N.')
    if np.any(np.abs(sssVecFull[2:-2]) != 1):  # Check for unit magnitude (excluding gutter)
        raise ValueError('sssVecFull expected to have 4QAM symbols with unit coordinates.')

    # Frequency-domain SSS vector
    sssFreqDomainVec = sssVecFull

    # Convert to time domain
    sssTimeDomainVec = (N / np.sqrt(N)) * np.fft.ifft(sssFreqDomainVec)

    # Add cyclic prefix
    sssTimeDomainVec = np.concatenate([sssTimeDomainVec[-Ng:], sssTimeDomainVec])

    return sssTimeDomainVec, sssFreqDomainVec


# Example usage
if __name__ == "__main__":
    sssTimeDomainVec, sssFreqDomainVec = genSss()
    print("SSS Time Domain Shape:", sssTimeDomainVec.shape)
    print("SSS Frequency Domain Shape:", sssFreqDomainVec.shape)
    print("First 10 Time Domain Samples:", sssTimeDomainVec[:10])
    print("First 10 Frequency Domain Samples:", sssFreqDomainVec[:10])