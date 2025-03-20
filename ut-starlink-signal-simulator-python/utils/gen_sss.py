''' gen_sss.py
    Generate the Secondary Synchronization Sequence (SSS) for Starlink.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

import numpy as np
"""
-- Input --
None. This function does not take in any input parameters.

-- Output --
- **sssTimeDomainVec ** (numpy array): (N + ng)-by-1 SSS.
Expressed in the time domain with a length-ng cyclic prefix
Scaled such that the norm squared of the N-by-1 time domain vector before cyclic prefix pre-pending is N - 4.
This scaling, which accounts for the absent modulation in the -subcarrier mid-channel gutter, ensures that the magnitude of the SSS produced by this function is commensurate with that of the the PSS produced by gen_pss.

- **sssFreqDomainVec** (numpy array): N-by-1 SSS expressed in the frequency domain with 4QAM symbols having unit coordinates.
"""

def gen_sss():
    ''' Number of subcarriers. '''
    N = 1024

    ''' Duration of OFDM cyclic prefix. Expressed in intervals of 1/fs. '''
    ng = 32

    ''' TODO :
        The original MATLAB file contains the comment. % Run estimateSSS.m first.
        I have zero idea what this means. The ut-starlink-signal-simulator does not contain this file.
        As a placeholder I assume that sss_vec_full has been saved as a NumPy (.npy) file.
        Of course sss_vec_full is not actually defined or loaded here. 
        The specifics of how we manage the data WILL CHANGE based on actual implementation details.
    '''
    try:
        sss_vec_full = np.load('sss_vec.npy')  #Assumes the array is already saved in a NumPy (.npy) file
        if sss_vec_full.shape[0] != N:
            raise ValueError('Frequency domain SSS expression must be of length N')
        if not np.all(np.abs(sss_vec_full[2:-2]) == 1):
            raise ValueError('sssVecFull expected to have 4QAM symbols with unit coordinates')
    except FileNotFoundError:
        print("File not found. Ensure 'sssVec.npy' is in the current directory.")
        return None, None

    sss_freq_domain_vec = sss_vec_full
    
    ''' Scale the IFFT output to match the specific power requirements. '''
    sss_time_domain_vec = (N / np.sqrt(N)) * np.fft.ifft(sss_freq_domain_vec)

    ''' Prepend the cyclic prefix. '''
    sss_time_domain_vec = np.concatenate([sss_time_domain_vec[-ng:], sss_time_domain_vec])

    return sss_time_domain_vec, sss_freq_domain_vec

