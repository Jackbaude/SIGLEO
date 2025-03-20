''' gen_dop_hist.py
    Generate a Doppler history interpolated from the provided parameters.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

from scipy.constants import speed_of_light
import numpy as np
import pandas as pd

"""
-- Input --
A Python dictionary (`s`) with the following keys:

- **epoch_datetime** (pandas datetime Series): Nx1 pandas datetime Series.
- **epoch_state** (numpy array): Nx6 array representing the corresponding ECEF state 
  [x, y, z, vx, vy, vz] in km.
- **start_datetime** (datetime): Datetime corresponding to the beginning of the desired 
  simulated Doppler history.
- **t_dur** (float): Duration of the simulated Doppler history.
- **fc** (float): Center frequency of the signal.
- **fs** (float): Sampling frequency of the desired simulated Doppler history.
- **rx_state** (numpy array): 1x6 array representing the corresponding ECEF state of 
  the receiver [x, y, z, vx, vy, vz] in km.

-- Output --
- **f_dop** (numpy array): The Doppler shift values for each sample.
- **epoch_datetimes** (pandas datetime Series): The datetime for each sample.
- **epoch_state_interp** (numpy array): The interpolated satellite states for each sample.
"""
def gen_dop_hist(s):
    # ------------------------------------ 
    # ----------- Constants -----------
    # ------------------------------------ 
    NMAX = 32000000

    c = speed_of_light / 1e3  #Speed of light in km/s.

    lamda = c / s["fc"] #Wavelength in km.

    n_samples = s["fs"] * s["tdur"] #The total number of samples.

    ''' Generate datetimes for each sample. '''
    time_indices = pd.date_range(s['start_datetime'], periods=n_samples, freq=pd.DateOffset(seconds=1/s['fs']))
    
    ''' Interpolate the state vectors. '''
    epoch_state_interp = np.zeros((6, len(time_indices)))
    for i in range(6):
        epoch_state_interp[i, :] = np.interp(
            time_indices.view(np.int64),  #Convert datetimes to integer for interpolation.
            s['epoch_datetime'].view(np.int64),
            s['epoch_state'][:, i]
        )

    ''' Calculate the relative velocity and position. '''
    relative_position = s['rx_state'][:3] - epoch_state_interp[:3, :]
    distance = np.linalg.norm(relative_position, axis=0)
    rg = relative_position / distance  #Unit vector from SV to RX.
    relative_velocity = s['rx_state'][3:6] - epoch_state_interp[3:6, :]  #Relative velocity from SV to Rx.
    
    ''' Calculate the doppler shift. '''
    fdop = -1 / lamda * np.einsum('ij,ij->j', rg, relative_velocity)  #Dot product along each column.

    return fdop, time_indices, epoch_state_interp.T

''' TODO :
  Set up a sample dictionary for 's' here. To use for testing.
'''