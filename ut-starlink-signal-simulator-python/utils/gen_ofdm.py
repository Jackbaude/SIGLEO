import numpy as np
from scipy.signal import resample
from typing import Dict, Union

def genOFDM(s: Dict[str, Union[float, int, np.ndarray, str]]) -> np.ndarray:
    """
    genOFDM is an OFDM signal generator.

    Parameters:
    -----------
    s : dict
        A dictionary with the following keys:
        - Fs : float
            Channel bandwidth (not including guard intervals) in Hz.
        - N : int
            Number of subcarriers, size of FFT/IFFT without CP (cyclic prefix).
        - Ng : int
            Number of guard subcarriers CP (cyclic prefix).
        - Midx : int
            Subcarrier constellation size (2 = BPSK, 4 = 4QAM, 16 = 16QAM, etc.).
            OVERWRITTEN IF DATA IS PROVIDED (see 'data' below).
        - type : str
            Modulation type ('PSK' or 'QAM' only).
        - SNRdB : float
            Simulated Signal-to-Noise Ratio, in dB. For no noise, pass `np.nan`.
        - Fsr : float
            Receiver sample rate. If less than Fs, the signal is filtered before resampling.
        - Fcr : float
            Receiver center frequency.
        - Fc : float
            OFDM signal center frequency, in Hz.
        - beta : float
            Doppler factor. Doppler shift is FD = -beta * Fc.
        - gutter : bool, optional
            Enables a gutter of 4F at the center (default is False).
        - Nsym : int, optional
            Number of consecutive symbols to generate (default is 1).
        - data : np.ndarray, optional
            1024 x K vector. Each column corresponds to the serial data transmitted on a symbol's subcarriers.

    Returns:
    --------
    yVec : np.ndarray
        OFDM symbol in time, sampled at Fsr, centered at Fsr, expressed at baseband.
    """
    # Optional parameters & checks
    s.setdefault('gutter', False)
    s.setdefault('Nsym', 1)

    if 'data' not in s and (np.log2(s['Midx']) % 2 != 0 and s['Midx'] != 2):
        raise ValueError('Midx must be 2 or an even power of 2.')

    if 'data' not in s:
        x = np.random.randint(0, s['Midx'], size=(s['N'], s['Nsym']))
        s['Midx'] = 2 ** int(np.ceil(np.log2(s['Midx'])))
    else:
        x = s['data']
        if x.shape[0] != s['N'] or x.shape[1] > s['Nsym']:
            raise ValueError('s.data must be an N x Nsym vector.')
        s['Midx'] = 2 ** int(np.ceil(np.log2(np.max(x) + 1)))

    # Dependent parameters
    T = s['N'] / s['Fs']  # Symbol duration, non-cyclic
    Tg = s['Ng'] / s['Fs']  # Guard duration
    Tsym = T + Tg  # OFDM symbol duration
    F = s['Fs'] / s['N']  # Subcarrier spacing

    # Generate simulated serial data symbols
    XVec = np.zeros_like(x, dtype=complex)
    if s['type'].upper() == 'PSK':
        for ii in range(x.shape[1]):
            XVec[:, ii] = pskmod(x[:, ii], s['Midx'])
    elif s['type'].upper() == 'QAM':
        for ii in range(x.shape[1]):
            XVec[:, ii] = qammod(x[:, ii], s['Midx'], unit_avg_power=True)
    else:
        raise ValueError('Type must be "QAM" or "PSK".')

    if s['Nsym'] > x.shape[1]:
        Nreps = s['Nsym'] // x.shape[1]
        Nremain = s['Nsym'] % x.shape[1]
        XVec = np.tile(XVec, (1, Nreps))
        XVec = np.hstack((XVec, XVec[:, :Nremain]))

    # Generate the OFDM symbols from serial data
    if s['gutter']:
        Ngut = 4  # Starlink has a 4-subcarrier gutter at the center
        XVec = np.fft.fftshift(XVec, axes=0)
        XVec[(s['N'] - Ngut) // 2 : (s['N'] + Ngut) // 2, :] = 0
        XVec = np.fft.fftshift(XVec, axes=0)

    # Transform to time domain
    Mx = np.sqrt(s['N']) * np.fft.ifft(XVec, axis=0)
    # Prepend each symbol with cyclic prefix
    MxCP = np.vstack((Mx[-s['Ng']:, :], Mx))
    # Unroll into serial sample stream
    xVec = MxCP.flatten()

    # Simulate Doppler & receiver bias to center
    if s['beta'] != 0 or s['Fc'] != s['Fcr']:
        tVec = np.arange(len(xVec)) / s['Fs']
        FD = -s['beta'] * s['Fc']
        Fshift = FD + s['Fc'] - s['Fcr']
        xVec = xVec * np.exp(1j * 2 * np.pi * Fshift * tVec)

    # Pass through AWGN channel
    yVec = xVec
    if not np.isnan(s['SNRdB']):
        SNR = 10 ** (s['SNRdB'] / 10)
        sigmaIQ = np.sqrt(1 / (2 * SNR))
        Nsamps = len(xVec)
        nVec = sigmaIQ * (np.random.randn(Nsamps) + 1j * np.random.randn(Nsamps))
        yVec = xVec + nVec

    # Resample to simulate receiver capture
    if s['Fsr'] != s['Fs'] and len(yVec) > 0:
        yVec = resample(yVec, int(len(yVec) * s['Fsr'] / s['Fs']))

    return yVec

import numpy as np
import matplotlib.pyplot as plt

# Helper functions (assume these are implemented or use libraries like commpy)
def pskmod(data: np.ndarray, M: int) -> np.ndarray:
    """PSK modulation."""
    return np.exp(1j * 2 * np.pi * data / M)

def qammod(data: np.ndarray, M: int, unit_avg_power: bool = True) -> np.ndarray:
    """QAM modulation."""
    constellation = np.arange(M)
    constellation = constellation - np.mean(constellation)
    if unit_avg_power:
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    return constellation[data]
    

import numpy as np
import matplotlib.pyplot as plt
from gen_ofdm import genOFDM  # Assuming genOFDM is implemented

def plot_example():
    # Example parameters for the OFDM signal
    params = {
        'Fs': 20e6,  # Channel bandwidth (20 MHz)
        'N': 1024,   # Number of subcarriers
        'Ng': 256,   # Number of guard subcarriers (cyclic prefix)
        'Midx': 16,  # 16-QAM modulation
        'type': 'QAM',  # Modulation type
        'SNRdB': 30,  # Signal-to-Noise Ratio in dB
        'Fsr': 20e6,  # Receiver sample rate (same as Fs)
        'Fcr': 2.4e9,  # Receiver center frequency (2.4 GHz)
        'Fc': 2.4e9,  # OFDM signal center frequency (2.4 GHz)
        'beta': 0,  # No Doppler shift
        'gutter': True,  # Enable gutter
        'Nsym': 4,  # Number of consecutive symbols
    }

    # Generate the OFDM signal
    ofdm_signal = genOFDM(params)

    # Time vector for plotting
    t = np.arange(len(ofdm_signal)) / params['Fsr']

    # Plot the time-domain signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, np.real(ofdm_signal), label='Real Part')
    plt.plot(t, np.imag(ofdm_signal), label='Imaginary Part')
    plt.title('Time-Domain OFDM Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Compute and plot the spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(ofdm_signal, Fs=params['Fsr'], NFFT=1024, noverlap=512, cmap='viridis', scale='dB')
    plt.title('Spectrogram of OFDM Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.ylim(-params['Fsr'] *2, params['Fsr'] * 2)  # Set frequency range to ±Fsr/2
    plt.colorbar(label='Intensity (dB)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_example()