import numpy as np
from typing import Dict, Union
from gen_ofdm import genOFDM
import matplotlib.pyplot as plt

def genStrlkOFDM(s: Dict[str, Union[float, int, np.ndarray, str]]) -> np.ndarray:
    """
    genStrlkOFDM is an OFDM symbol generator with specific Starlink parameters.

    Parameters:
    -----------
    s : dict
        A dictionary with the following keys:
        - Midx : int
            Modulation index (2 = BPSK, 4 = 4QAM/4PSK, 16 = 16QAM/16PSK, etc.).
            Only M = 4 and M = 16 are allowed.
        - type : str
            Modulation type ('PSK' or 'QAM' only). For PSK, only M = 4 is allowed.
        - SNRdB : float
            Simulated Signal-to-Noise Ratio, in dB. For no noise, use `np.nan`.
        - Fsr : float
            Receiver sample rate. If less than Fs, the signal is filtered before resampling.
        - Fcr : float
            Receiver center frequency, in Hz.
        - beta : float
            Doppler factor. Doppler shift is FD = -beta * Fc.
        - Nsym : int, optional
            Number of consecutive symbols to generate (default is 1).
        - data : np.ndarray, optional
            1024 x K vector. Each column corresponds to the serial data transmitted on a symbol's subcarriers.

    Returns:
    --------
    y : np.ndarray
        Starlink OFDM symbol in time, sampled at Fsr, centered at Fsr, expressed at baseband.
    """
    # Fixed Starlink parameters
    s['Fs'] = 240e6  # Channel bandwidth (240 MHz)
    s['N'] = 1024  # Number of subcarriers
    s['Ng'] = 32  # Number of guard subcarriers (cyclic prefix)
    s['gutter'] = True  # Enable 4-subcarrier gutter at center

    # Optional parameters
    s.setdefault('Nsym', 1)

    # Validate input data
    if 'data' in s and 'type' not in s:
        raise ValueError("A constellation type should also be provided as s.type (i.e., 'PSK' or 'QAM').")

    if 'data' in s:
        l, w = s['data'].shape
        if l != s['N'] or w > 300:
            raise ValueError('s.data must be an N x 300 vector at most.')

    if 'data' not in s and (np.log2(s['Midx']) % 2 != 0 and s['Midx'] != 2):
        raise ValueError('Midx must be 2 or an even power of 2.')

    # Calculate Starlink channel center frequency
    F = s['Fs'] / s['N']
    chIdx = round((s['Fcr'] / 1e9 - 10.7 - F / 2 / 1e9) / 0.25 + 0.5)
    Fcii = 10.7e9 + F / 2 + 250e6 * (chIdx - 0.5)
    s['Fc'] = Fcii

    # Generate the OFDM signal
    y = genOFDM(s)
    return y


if __name__ == '__main__':
    # Example parameters for Starlink OFDM signal
    params = {
        'Midx': 16,  # 16-QAM modulation
        'type': 'QAM',  # Modulation type
        'SNRdB': 30,  # Signal-to-Noise Ratio in dB
        'Fsr': 240e6,  # Receiver sample rate (240 MHz)
        'Fcr': 11e9,  # Receiver center frequency (11 GHz)
        'beta': 0,  # No Doppler shift
        'Nsym': 4,  # Number of consecutive symbols
    }

    # Generate the Starlink OFDM signal
    starlink_signal = genStrlkOFDM(params)

    # Time vector for plotting
    t = np.arange(len(starlink_signal)) / params['Fsr']

    # Plot the time-domain signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, np.real(starlink_signal), label='Real Part')
    plt.plot(t, np.imag(starlink_signal), label='Imaginary Part')
    plt.title('Time-Domain Starlink OFDM Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Compute and plot the spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(starlink_signal, Fs=params['Fsr'], NFFT=1024, noverlap=512, cmap='viridis', scale='dB')
    plt.title('Spectrogram of Starlink OFDM Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(-params['Fsr'] * 2, params['Fsr'] * 2)  # Set frequency range to ±Fsr/2
    plt.colorbar(label='Intensity (dB)')

    plt.tight_layout()
    plt.show()