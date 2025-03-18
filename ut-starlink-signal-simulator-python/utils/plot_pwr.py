''' Ported from Matlab to Python by Jack R. Tschetter on 03/13/2025.
    The following key considerations were made.

    1. ) Uses SciPy's Welch Method for PSD
        Uses scipy.signal.welch() to estimate power spectral density (PSD).
        Applies Kaiser windowing to match MATLAB behavior.
    2. ) Handles FFT Processing
        Uses numpy.fft.fft() for computing the FFT-based power estimation.
    3. ) Power vs. Time Calculation
        Segments data into blocks, computes power per block, and plots power vs. time.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, kaiser

def plot_pwr(data, Nblocks, tstart, tdur, Fs, NFFT):
    """
    Plots the PSD and Power vs Time of the provided signal starting at tstart for a duration of tdur.

    Parameters:
    data (np.array): Vector of complex samples.
    Nblocks (int): Number of blocks to sweep for power vs time plot.
    tstart (float): Start of data to process relative to the first sample.
    tdur (float): Duration of data to process starting from tstart.
    Fs (float): Sampling rate of provided signal.
    NFFT (int): Number of FFT points for PSD plot.
    """
    if Nblocks > 10 * tdur * Fs:
        raise ValueError("Provided number of blocks is too high for duration specified.")

    # Extract the relevant segment of the data
    seekOffset = int(tstart * Fs)
    Nblock = int(tdur * Fs)
    data_segment = data[seekOffset:seekOffset + Nblock]

    # Time vector for plotting
    tVec = np.arange(len(data_segment)) / Fs + tstart

    # Compute PSD using Welch's method
    fVec, Syy = welch(data_segment, fs=Fs, window=kaiser(NFFT, 3), nperseg=NFFT, noverlap=NFFT//2, nfft=NFFT)

    # Plot PSD
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(fVec / 1e6, 10 * np.log10(Syy))
    plt.grid(True)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power density (dB/Hz)')
    plt.title('Power spectral density estimate')

    # Calculate power vs time
    L_seg = len(data_segment) // Nblocks
    Pwrdb = []
    for i in range(Nblocks):
        segment = data_segment[i * L_seg:(i + 1) * L_seg]
        f, pxx = welch(segment, fs=Fs, window=kaiser(L_seg//2, 3), nperseg=L_seg//2, nfft=NFFT)
        Pwrdb.append(max(10 * np.log10(np.sum(pxx * Fs)), -180))

    # Time vector for power vs time plot
    tVec = np.linspace(tstart, tstart + tdur, Nblocks)

    # Plot power vs time
    plt.subplot(2, 1, 2)
    plt.plot(tVec, Pwrdb)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('dB')
    plt.title('Power vs Time')

    plt.tight_layout()
    plt.show()