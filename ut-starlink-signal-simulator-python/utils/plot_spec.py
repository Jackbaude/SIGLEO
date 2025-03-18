''' Ported from Matlab to Python by Jack R. Tschetter on 03/13/2025.
    The following key considerations were made.

    1. ) Handles Data Resampling
        Uses scipy.signal.resample() if the provided sampling rate (Fsr) differs from the desired spectrogram rate (Fs).
    2. ) Applies Frequency Shift if Needed
        If Fc ≠ Fcr, applies a complex exponential to shift the center frequency.
    3. ) Computes and Plots the Spectrogram
        Uses scipy.signal.spectrogram() with a Kaiser window.
        Normalizes data to match MATLAB’s behavior.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, spectrogram
from numpy.fft import fftshift

def plot_spec(data, tstart, tdur, Fcr, Fsr, Fc, Fs, NFFT, Stitle=None):
    """
    Plots a spectrogram of the provided data.

    Parameters:
    data (np.array): Vector of complex samples.
    tstart (float): Start of plot in seconds relative to beginning of input data.
    tdur (float): Duration of plot in seconds. If 0, the whole section is plotted.
    Fcr (float): Center frequency of Passband equivalent signal.
    Fsr (float): Sampling rate of provided signal.
    Fc (float): Center frequency desired to be plotted.
    Fs (float): Desired plot sampling rate. Will resample provided signal to Fs if Fs != Fsr.
    NFFT (int): Number of FFT points for spectrogram.
    Stitle (str): Optional title for plot.
    """
    if Stitle is None:
        Stitle = f"Spectrogram centered at {Fc}"

    if tdur == 0:
        tdur = len(data) / Fsr

    # Calculate the indices for the data segment
    seekOffset = int(tstart * Fsr)
    Nblock = int(tdur * Fsr)
    data_segment = data[seekOffset:seekOffset + Nblock]

    # Time vector for the data segment
    tyVec = np.arange(len(data_segment)) / Fsr

    # Resample if necessary
    if Fs != Fsr:
        data_segment, tyVec = resample(data_segment, int(len(data_segment) * Fs / Fsr), t=tyVec)

    # Apply frequency shift if necessary
    if Fc != Fcr:
        Fshift = Fcr - Fc
        data_segment *= np.exp(1j * 2 * np.pi * Fshift * tyVec)

    # Normalize data to have unit power
    data_normalized = data_segment / np.sqrt(np.mean(np.abs(data_segment)**2))

    # Spectrogram calculation
    f, t, Sxx = spectrogram(data_normalized, Fs, window='kaiser', nperseg=NFFT, noverlap=NFFT//2, nfft=NFFT, scaling='spectrum', mode='psd')
    
    # Plotting
    plt.figure()
    plt.pcolormesh(t, fftshift(f), fftshift(10 * np.log10(Sxx), axes=0), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(Stitle)
    plt.colorbar(label='PSD (dB/Hz)')
    plt.show()