''' gen_ofdm.py
    This file defines a comprehensive function for generating Orthogonal Frequency Division Multiplexing Signals (OFDM).
    Function is highly configurable, allowing various parameters to be set to simulate different transmission conditions.
'''
import numpy as np
from scipy.signal import resample
from scipy.stats import norm
from numpy.fft import fft, ifft, fftshift

"""
    Generates an OFDM signal with specified parameters.

    Parameters:
    s (dict): Configuration dictionary with keys corresponding to OFDM settings.

    Returns:
    np.array: OFDM signal in time domain, sampled at Fsr, centered at Fsr, expressed at baseband.
"""

def gen_ofdm(s):
    # Optional parameters & Checks
    if 'gutter' not in s:
        s['gutter'] = 0
    if 'Nsym' not in s:
        s['Nsym'] = 1
    if 'data' not in s and (np.log2(s['Midx']) % 2 != 0 and s['Midx'] != 2):
        raise ValueError('Midx must be 2 or an even power of 2')

    # Generate data symbols each with Midx bits of information
    if 'data' not in s:
        x = np.random.randint(0, s['Midx'], (s['N'], s['Nsym']))
        s['Midx'] = 2 ** np.ceil(np.log2(s['Midx']))
    else:
        x = s['data']
        if x.shape[0] != s['N'] or x.shape[1] > s['Nsym']:
            raise ValueError('data must be an N x Nsym matrix')

    # Dependent parameters
    T = s['N'] / s['Fs']  # symbol duration, non cyclic
    Tg = s['Ng'] / s['Fs']  # guard duration
    Tsym = T + Tg  # ofdm symbol duration
    F = s['Fs'] / s['N']  # Subcarrier spacing

    # Generate simulated serial data symbols
    XVec = np.zeros((s['N'], s['Nsym']), dtype=complex)
    for ii in range(s['Nsym']):
        if s['type'].upper() == 'PSK':
            XVec[:, ii] = np.exp(2j * np.pi * x[:, ii] / s['Midx'])
        elif s['type'].upper() == 'QAM':
            XVec[:, ii] = (2 * x[:, ii] - s['Midx'] + 1) + 1j * (2 * x[:, ii] - s['Midx'] + 1)
        else:
            raise ValueError('Type must be "QAM" or "PSK".')

    # Generate the OFDM symbols from Serial Data
    if s['gutter']:
        Ngut = 4  # 4 subcarrier gutter at center
        XVec = fftshift(XVec, axes=0)
        XVec[s['N']//2 - Ngut//2:s['N']//2 + Ngut//2, :] = 0
        XVec = fftshift(XVec, axes=0)

    # Transform to time domain. Multiply by sqrt(N) to preserve energy
    Mx = np.sqrt(s['N']) * ifft(XVec, axis=0)
    # Prepend each symbol with cyclic prefix
    MxCP = np.vstack([Mx[-s['Ng']:, :], Mx])

    # Simulate Doppler & Receiver bias to center
    xVec = MxCP.flatten()
    if s['beta'] != 0 or s['Fc'] != s['Fcr']:
        tVec = np.arange(len(xVec)) / s['Fs']
        FD = -s['beta'] * s['Fc']
        Fshift = FD + s['Fc'] - s['Fcr']
        xVec *= np.exp(1j * 2 * np.pi * Fshift * tVec)

    # Pass through AWGN channel
    yVec = xVec
    if not np.isnan(s['SNRdB']):
        SNR = 10 ** (s['SNRdB'] / 10)
        sigmaIQ = np.sqrt(1 / (2 * SNR))
        Nsamps = len(xVec)
        nVec = sigmaIQ * (np.random.randn(Nsamps) + 1j * np.random.randn(Nsamps))
        yVec += nVec

    # Resample to simulate receiver capture
    if s['Fsr'] != s['Fs'] and len(yVec) > 0:
        yVec = resample(yVec, int(len(yVec) * s['Fsr'] / s['Fs']))

    return yVec
