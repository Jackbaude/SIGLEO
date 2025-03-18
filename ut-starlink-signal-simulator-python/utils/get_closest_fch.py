''' get_closest_fch.py
    Defines a function that calculates the closest Starlink OFDM channel to a given receiver frequency.
'''

import numpy as np

def get_closest_fch(fc):
    """
    Returns the center frequency of the closest Starlink OFDM channel.

    Parameters:
    fc : float
        Center frequency of receiver in Hz.

    Returns:
    fcii : float
        Center frequency of the closest Starlink OFDM channel in Hz.
    """

    ''' Calculate the frequency increment. Identical to Matlab, where F is computed as 240e6 / 1024.'''
    F = 240e6 / 1024

    ''' Calculate the channel index (ch_idx) using numpy.round for rounding. 
        The formula calculates the channel index based on input frequency converted to gigahertz.
    '''
    ch_idx = np.round(((fc / 1e9) - 10.7 - (F / 2 / 1e9)) / 0.25 + 0.5)

    ''' Compute closest channel frequency (fcii)
        This formula is directly translated from Matlab.
    '''
    fcii = 10.7e9 + F / 2 + (250e6 * (ch_idx - 0.5))
    return fcii
