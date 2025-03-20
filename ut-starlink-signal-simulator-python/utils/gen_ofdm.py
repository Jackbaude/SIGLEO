''' gen_ofdm.py
    Generate an OFDM signal.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

"""
-- Input --
A Python dictionary (`s`) with the following keys:

- **fs** (float): Channel bandwidth (not including channel guard intervals) in Hz.
- **N** (int): Number of subcarriers, size of FFT/IFFT without CP (cyclic prefix).
- **Ng** (int): Number of guard subcarriers in the cyclic prefix.
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
- **y_vec** (numpy array): OFDM symbol in time, sampled at `fsr`, centered at `Fsr`, 
  and expressed at baseband.
"""
def gen_ofdm(s):
    ''' Optional parameters and checks. '''
    if 'gutter' not in s:
        s['gutter'] = 0
    if 'nysm' not in s:
        s['nysm'] = 1

    ''' Generate data symbols each with midx bits of information. '''