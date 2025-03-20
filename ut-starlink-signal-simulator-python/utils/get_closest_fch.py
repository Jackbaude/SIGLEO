''' gen_closest_fch.py
    Return the center frequency of the closest StarLink OFDM channel.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

"""
-- Input --
- **fc** (float): The center frequency of the receiver in Hz.

-- Output --
- **grid** (numpy array): The center frequency of the closest Starlink OFDM channel in units of Hertz (Hz).
"""
def get_closest_fch(fc):

    ''' Calculate F.
        Divide a fixed frequency of 240 MHz by 1024. Giving a small frequency increment.
        This line of code is identical between MATLAB and Python.
        NOTE : In Python the data type of F will be float. In MATLAB it is double.
        Both Python and MATLAB use floating point precision for such calculations.
        In Python floating point numbers are repressented using double precision (64 bits).
        This is similar to MATLAB's double type.
    '''
    F = 240e6 / 1024

    ''' Determine the channel index (ch_idx) 
        Calculate the index for the channel by adjusting the input frequency fc in GHz.
        Subtracting a fixed offset (10.7 GHz).
        Adjusting for half of the frequency increment F.
        And scaling by the channel bandwidth (250 MHz).
        Then round the result to the nearest integer representing the nearest channel.
        NOTE : The Python round() function behaves differently then MATLAB's round() function.
        MATLAB's round() will round to the nearest integer, with halfway cases rounding away from zero.
        Python's round() also rounds to the nearest integer, but for halfway cases uses something "banker's rounding".
        The Python version rounds 0.5 to the nearest even number.
        TODO : We may need to adjust the rounding behavior manually in cases where it differs significantly between MATLAB and Python.
    '''
    ch_idx = round((fc / 1e9 - 10.7 - F / 2 / 1e9) / 0.25 + 0.5)

    ''' Calculate the center frequency of the closest channel.
        Compute the center frequency using the previously calculated index.
        It adds a base frequency of 10.7 GHz.
        Half the frequency increment, and the product of 250 MHz.
        And the channel index offset by 0.5.
    '''
    fcii = (10.7e9 + F/2 + 250e6*(ch_idx - 0.5))