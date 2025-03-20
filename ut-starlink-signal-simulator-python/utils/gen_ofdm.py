''' gen_ofdm.py
    Generate an OFDM signal.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

import numpy as np
"""
-- Input --
A Python dictionary (`s`) with the following keys:

- **fs** (float): Channel bandwidth (not including channel guard intervals) in Hz.

- **N** (int): Number of subcarriers, size of FFT/IFFT without CP (cyclic prefix).

- **ng** (int): Number of guard subcarriers in the cyclic prefix.

- **Midx** (int): Subcarrier constellation size (e.g., 2 = BPSK, 4 = 4QAM, 16 = 16QAM, etc.).
  This value is **overwritten if data is provided** (see `data` below).

- **type** (str): Modulation type ('PSK' or 'QAM' only).

- **SNRdB** (float or NaN): Simulated Signal-to-Noise Ratio (SNR) in dB.
  Pass in `nan` for no noise.

- **fsr** (float): Receiver sample rate. If it is less than `Fs`, the signal is filtered 
  before resampling to prevent aliasing. Set `Fsr` to `Fs` to skip resampling and get 
  the full `Fs` signal.

- **fcr** (float): Receiver center frequency.

- **fc** (float): OFDM signal center frequency in Hz.

- **beta** (float): Doppler factor. Doppler shift is computed as `FD = -beta * Fc`, where
  `beta = vlos / c`. Simulated by resampling at `(1 + beta) * Fs` and shifting by `FD`.
  - Set `beta` to zero for no Doppler shift.
  - For approaching satellites (beta < 0), the measured inter-frame interval is shorter
    (compressed) compared to the ideal inter-frame interval of `1/750`.
  - Assuming a polynomial model of the form `tF(t) = p3*t^2 + p2*t + p1`, then 
    `dtF/dt|_0 = p1 = beta(0)`. Doppler frequency shift for a tone at `Fc` is given by 
    `FD = -beta * Fc`.

- **gutter** (bool, optional): Enables a gutter of `4F` at the center, as observed in 
  Starlink signals.

- **n_sym** (int, optional): Number of consecutive symbols to generate.
  - Default is `1`.
  - If `data` is provided, `Nsym` defaults to its length.

- **data** (numpy array, optional): `1024 x K` matrix, where each column corresponds to 
  the serial data transmitted on a symbol’s subcarriers.
  - Each column should have elements in the range `[0, M-1]`, where `M` determines the 
    constellation size for the symbol.
  - Data per symbol should be structured such that:
    - The first index corresponds to the `ceil(N/2)` subcarrier.
    - The element at index `ceil(N/2)` corresponds to the 1st subcarrier.
    - The final index corresponds to the `floor(N/2)` subcarrier value.
  - The number of symbols whose data can be set (`K`) should be in the range `[1, Nsym]`.
  - If `K < n_sym`, the symbols will take the first `K` columns of `data`, and the 
    remaining symbols will wrap, using the first `Nsym-K` columns of `data`.


-- Output --
- **y_vec** (numpy array): OFDM symbol in time, sampled at `fsr`, centered at `fsr`, 
  and expressed at baseband.
"""

def gen_ofdm(s):
    # ------------------------------------ 
    # ----------- Optional parameters and checks. -----------
    # ------------------------------------ 
    
    ''' Check if the 'gutter' field exists in the dictionary 's'. 
        If not initialize it.
    '''
    if 'gutter' not in s:
        s['gutter'] = 0

    ''' Check if the 'nysm' field exists in the dictionary 's'. 
        If not initialize it.
    '''
    if 'nysm' not in s:
        s['nysm'] = 1

    ''' Check if 'midx' is either 2 or an even power of 2.
        If not the input is invalid so we raise an error.
    '''
    if 'data' not in s and (np.log2(s['Midx']) % 2 != 0) and (s['Midx'] != 2):
        raise ValueError('Midx must be 2 or an even power of 2')
    
    ''' Initialize or process 'data' fields based on existence. '''
    if 'data' not in s:
        x = np.random.randint(0, s['midx'], size=(s['N'], s['nsym']))
        l, w = x.shape
        s['midx'] = 2 ** np.ceil(np.log2(s['midx'])) * np.ones(w, dtype=int)
    else:
        l, w = s['data'].shape
        if l != s['N'] or w > s['nsym']:
            raise ValueError('s[data] must be an N x nsym matrix')
        s['Midx'] = 2 ** np.ceil(np.log2(np.max(s['data'])))
        x = s['data']

    ''' Generate data symbols each with midx bits of information. '''

    # ------------------------------------ 
    # ----------- Dependent parameters. -----------
    # ------------------------------------ 

    ''' Calculate the symbol duration T.
        Based on the number of samples N and the sampling frequency fs.
        This repressents the time taken for a single complete symbol to be transmitted.
        The original Matlab line was T = s.N/s.Fs;
    '''
    t = s['N'] / s['fs']  #Symbol duration, non-cyclic.

    ''' Calculate the guard duration.
        This is length of time that a cyclic prefix (guard interval) is applied to a signal.
    '''
    tg = s['ng'] / s['fs']  #Guard duration.

    ''' Calculate the total duration of an OFDM symbol.
        Crucially this includes BOTH the symbol duration and the guard interval duration.
        The total duration (tsym) defines the time slots in which OFDM symbols are transmitted.
    '''
    tsym = t + tg

    ''' Calculate the subcarrier spacing within the OFDM system.
        This defines the frequency separation between individual subcarriers in the OFDM signal.
    '''
    f = s['fs'] / s['N']  #Subcarrier spacing.

    # ------------------------------------ 
    # ----------- Generate simulated serial data symbols. -----------
    # ------------------------------------ 

    ''' Generate these symbols as complex numbers with unit average energy. '''
    
    # ------------------------------------ 
    # ----------- Generate the OFDM symbols from Serial Data. -----------
    # ------------------------------------ 

    # ------------------------------------ 
    # ----------- Simulate Doppler & Receiver bias to center. -----------
    # ------------------------------------ 

    # ------------------------------------ 
    # ----------- Pass through AWGN channel. -----------
    # ------------------------------------ 

    # ------------------------------------ 
    # ----------- Resample to simulate receiver capture. -----------
    # ------------------------------------ 