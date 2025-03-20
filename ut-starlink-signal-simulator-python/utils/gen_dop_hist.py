''' gen_dop_hist.py
    Generate a Doppler history interpolated from the provided parameters.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

from scipy.constants import speed_of_light

"""
-- Input --
A Python dictionary (`s`) with the following keys:

- **epoch_datetime** (numpy array): Nx1 vector of datetimes.
- **epoch_state** (numpy array): Nx6 array representing the corresponding ECEF state 
  [x, y, z, vx, vy, vz] in km.
- **start_datetime** (datetime): Datetime corresponding to the beginning of the desired 
  simulated Doppler history.
- **tdur** (float): Duration of the simulated Doppler history.
- **fc** (float): Center frequency of the signal.
- **fs** (float): Sampling frequency of the desired simulated Doppler history.
- **rx_state** (numpy array): 1x6 array representing the corresponding ECEF state of 
  the receiver [x, y, z, vx, vy, vz] in km.

-- Output --
- **fdop** (numpy array): Doppler shift values over time.
"""
def gen_dop_hist(s):
    NMAX = 32000000

    c = speed_of_light / 1e3  # Convert to km/s

    lamda = c / s["fc"]

    n_samples = s["fs"] * s["tdur"]