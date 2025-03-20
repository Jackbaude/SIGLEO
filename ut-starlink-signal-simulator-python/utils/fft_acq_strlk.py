''' fft_acq_strlk.py
    Performs an FFT acquisition based on PSS and SSS.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

import numpy as np
import matplotlib.pyplot as plt
from get_closest_fch import get_closest_fch
from gen_pss import gen_pss
from gen_sss import gen_sss
from scipy.signal import resample

"""
-- Input --
A Python dictionary (`s`) with the following keys:

- **fsr** (float): Receiver sampling frequency in Hz.
- **fcr** (float): Receiver center frequency in Hz.
- **fmax** (float): Maximum frequency for frequency search in Hz.
- **fstep** (float): Frequency step for frequency search in Hz.
- **fstepFine** (float): Frequency step for fine frequency search in Hz.
- **fmin** (float): Minimum frequency for frequency search in Hz.
- **y** (numpy array): Ns x 1 vector of baseband data, where Ns corresponds to
  samples at least equal in length to a Starlink frame in time.
- **pfa** (float): False alarm probability used for the threshold to determine
  if a signal is present for a PRN. The total acquisition false alarm probability.
  The pfa for each search cell can be derived from this.

-- Output --
A Python dictionary containing:

- **grid** (numpy array): F x Nk acquisition grid where F is created from `fmax`, `fmin`,
  and `fstep`, and Nk is the number of samples of a Starlink frame sampled at 240 Msps.
- **tauvec** (numpy array): Nk x 1 vector of considered delay times of Starlink time of
  arrival of the first frame with respect to the data input beginning.
- **fdvec** (numpy array): F x 1 vector of considered Doppler frequencies.
- **fdfine** (float): Fine Doppler frequency estimate from peak.
- **tau** (float): Time offset estimate from peak.
- **nc** (int): Length of local replica.
- **sigma2_c** (float): Variance of local replica (non-zero samples).
- **pkvec** (numpy array): Local replica used for acquisition.
"""

def fft_acq_strlk(s):
    ''' Debugging flags
        TODO : I am unsure if these are meant to be in function scope or global scope. Matlab is confusing.
    '''
    debugPltHT_enable = 0
    debugPltGrid_en = 0
    fontsize = 12

    # ------------------------------------ 
    # ----------- Params -----------
    # ------------------------------------ 
    n = 1024
    ng = 32
    ns = n + ng
    ndsym = 300
    fs = 240e6
    nk = 1/750*fs

    ''' Call the function gen_pss()
        This generates the Primary Synchronization Signal (PSS).
        PSS is used in signal acquisition. 
        PSS is expected to be a vector in Matlab.
        TODO : ensure gen_pss is implemented correctly.
        TODO : What datatype will the Matlab vector be in Python?
    '''
    pss = gen_pss(s)

    ''' Call the function gen_sss()
        This generates the Secondary Synchronization Signal (SSS).
        SSS is used in signal acquisition. 
        SSS is expected to be a vector in Matlab.
        TODO : ensure gen_sss is implemented correctly.
        TODO : What datatype will the Matlab vector be in Python?
    '''
    sss = gen_sss(s)

    ''' Concatenate the two signals vertically into a single column vector c.
        This combined signal represents a known Starlink synchronization frame.
        The original Matlab line was c = [PSS; SSS];
        TODO : Might need to use np.concatenate instead.
    '''
    c = np.vstack((pss, sss))

    ''' Assigns c to pkvec.
        pkvec will store the known synchronization frame before zero padding is applied.
        We use c.copy() though since Python passes by references whilst Matlab passes by value.
        The original Matlab line was pkVec = c;
    '''
    pkvec = c.copy()

    ''' Zero padding
        Extends c to match a required frame length nk.
        If nk is greateer than length(c) this ensures c has exactly nk elements by appending zeros.
        This is IMPORTANT as it ensures proper signal processing and alignment in acquisition algorithms.
        The original Matlab line was c = [c; zeros(Nk - length(c), 1)];
    '''
    c = np.vstack((c, np.zeros(nk - len(c), 1)))

    # ------------------------------------ 
    # ----------- Resample to full BW -----------
    # ------------------------------------ 
    ''' TODO : What datatype is s in Python?
        In Matlab s is a struct.
        In Python we have the following options
        1. ) s is a dictionary
        2. ) s is an object (Using a Class)
        3. ) s is a NamedTuple (Immutable Structure)
        4. ) s is a NumPy Structured Array. Something like MATLAB Struct Arrays.
        For simplicity I assume s is a dict in Python.
    '''
    y = s['y']

    if s['fsr'] != fs:
        t_vec = np.arange(len(y)).reshape(-1, 1) / s['fsr']

        #TODO : This is still the original Matlab line. I am unsure how to convert to Python.
        y = resample(y,t_vec,fs)
    buffer = 100
    if len(y) >= nk - buffer:
        y = np.vstack((y, np.zeros((nk - len(y), 1))))
    elif len(y) - nk < buffer:
        raise ValueError("Error! The data should be long enough to contain at least 1 frame.")

    # ------------------------------------ 
    # ----------- Remove receiver bias -----------
    # ------------------------------------ 
    if get_closest_fch(s['fcr']) - s['fcr'] != 0:
        t_vec = np.arange(len(y)).reshape(-1, 1) / fs
        f_shift = get_closest_fch(s['fcr']) - s['fcr']
        y = y * np.exp(-1j * 2 * np.pi * f_shift * t_vec)

    # ------------------------------------ 
    # ----------- Derived Params -----------
    # ------------------------------------ 
    
    ''' Create a range of values from s['fmin'] to s['fmax'] with step size s['fstep']
        Convert to column vector.
        This line was originally fdvec = (s.min : s.fstep : s.fmax)'; in Matlab.
    '''
    fdvec = np.arange(s['fmin'], s['fmax'] + s['fstep'], s['fstep']).reshape(-1, 1)
    
    ''' Create a column vector ranging from 0 to nk - 1 (inclusive)
        Element wise divide by fs to scale the values appropriately.
        This line was originally tauvec = (0:(Nk-1))'/Fs; in Matlab.
    '''
    tauvec = np.arange(0, nk).reshape(-1, 1) / fs

    ''' Create a logical mask where elements of c that are not equal to zero are marked as true
        Zeros are marked as false.
        This line was originally known = c(c ~= 0); in Matlab.
    '''
    known = c[c != 0]
    nc = len(known)

    ''' Compute the variance of the known array.
        In MATLAB the default var() computes the sample variance (i.e. dividing by N-1 instead of N).
        NumPy default is to compute the population variance (i.e., divide by N).
        To match MATLAB we set ddof=1 (Delta Degrees of Freedom) in NumPy.
        The line was originally sigma2_c = var(known); in Matlab.
    '''
    sigma2_c = np.var(known, ddof=1)

    ''' Computes the total number of cells in a grid.
        Return the number of elements in fdvec
        Return the number of elements in tauvec.
        Element wise multiply to get the total number of cells in the grid.
        The line was originally Ngrid = length(fdvec) .* length(tauvec); in Matlab.
    '''
    ngrid = len(fdvec) * len(tauvec)  # Number of cells in the grid

    ''' Compute the Fast Fourier Transform (FFT) of the signal. 
        In MATLAB fft() computes the 1D FFT as default.
        The line was originally C = fft(c); in Matlab.
        TODO : np.fft.fft(c) expects a 1D array. If c needs to be a multi-dimensional array we may need to specify an axis.
    '''
    C = np.fft.fft(c)

    ''' Return the number of elements in the vector y.
        Divide by nk.
        round down to the nearest integer.
        The line was originally Nac = floor(length(y)/Nk); in Matlab.
    '''
    nac = np.floor(len(y) / nk).astype(int)

    ''' Using numpy slicing.
        MATLAB uses 1-based indexing, while Python uses 0-based indexing.
        In Python, y[:Nac * Nk] extracts elements from index 0 to Nac * Nk - 1.
        The line was originally y = y(1:(Nac*Nk)); in Matlab.
    '''
    y = y[:nac * nk]

    #Initializations

    ''' Create a 4x1 column vector.
        MATLAB's zeros(4,1) creates a 4x1 column vector.
        To achieve the same results with numpy use np.zeros((4, 1))
        The line was originally values = zeros(4,1); in Matlab.
    '''
    values = np.zeros((4, 1))

    ''' Create a 2D NumPy array of shape (len(fdvec), nk) filled with zeros.
        np.zeros(shape) creates a NumPy array filled with zeros.
        The shape argument specifies the dimensions of the array.
        The number of rows in the array is defined by len(fdvec).
        nk repressents the number of columns. It is assumed to be predefined (above) as an integer.
        The line was originally grids = zeros(length(fdvec), Nk); in Matlab.
    '''
    grids = np.zeros(len(fdvec), nk)

    sk = np.zeros((len(fdvec), len(c), nac))

    # ------------------------------------ 
    # ----------- Generate acq. grid -----------
    # ------------------------------------ 
    for j in range(1, nac + 1):  # MATLAB 1-based indexing to Python 0-based
        for i in range(1, len(fdvec) + 1):

            pre_idcs = np.arange((j - 1) * nk, j * nk)  # Equivalent to MATLAB ((j-1)*nk+1):(j*nk)
            tau_pre = tauvec[:len(pre_idcs)]  # Extract corresponding tau values
            x = y[pre_idcs]  # Extract x values

            beta = -fdvec[i - 1] / get_closest_fch(s['fcr'])  # Adjust for MATLAB 1-based indexing
            fsd = (1 + beta) * fs  # Adjusted sample rate

            # Resample using SciPy
            x_resampled, tau_post = resample(x, len(tau_pre), t=tau_pre, window=None)

            th_hat = 2 * np.pi * fdvec[i - 1] * tau_post  # Compute phase shift

            x_tilde = x_resampled * np.exp(-1j * th_hat)  # Apply phase correction

            # Zero-padding to Nk length
            x_tilde = np.concatenate([x_tilde, np.zeros(nk - len(x_tilde), dtype=complex)])
            x_tilde = x_tilde[:nk]  # Ensure length Nk

            # FFT and processing
            X_tilde = np.fft.fft(x_tilde)
            Zr = X_tilde * np.conj(C)
            sk = np.fft.ifft(Zr).T  # Transpose to match MATLAB behavior

            sk[i - 1, :, j - 1] = sk[:len(c)]  # Store in sk

    # Compute grid values
    grids[:, :] = np.sum(np.abs(sk), axis=2)  # Summing over the 3rd dimension

    # ------------------------------------ 
    # ----------- Process Acquisition -----------
    # ------------------------------------ 

    #Find Peak, tau, and doppler
    tau_prob = np.max(grids) #Find the maximum value.
    fidx = np.argmax(grids) #Find the index of the maximum value.
    
    mx_val = np.max(tau_prob) #Find the maximum value.
    tau_idx = np.argmax(tau_prob) #Find the index of the maximum value.

    tau = tauvec[tau_idx]
    fdcoarse = fdvec[fidx[tau_idx]]
    
    #Find finer doppler
    fdfine = fdcoarse  # Initial assignment (same value)

    expSk2_fine = mx_val / nac

    if s.fstep != 1:  # Finer look if step is not 1
        fdvec_fine = np.arange(fdcoarse - np.floor(s.fstep / 2), 
                            fdcoarse + np.floor(s.fstep / 2) + s.fstepFine, 
                            s.fstepFine)

        # Each column represents a different phase shift
        thmat = 2 * np.pi * (tauvec - tauvec[nk - 1])[:, None] * fdvec_fine  # Match MATLAB broadcasting

        z_fine = np.zeros(len(fdvec_fine))

        ctau = np.roll(c, tau_idx - 1)  # Equivalent to MATLAB circshift(c, tau_idx-1)

        for jj in range(1, nac + 1):
            iidum = np.arange((jj - 1) * nk, jj * nk)  # MATLAB 1-based indexing to Python 0-based
            
            x1_shift = y[iidum] * np.exp(-1j * thmat[:len(iidum), :])
            
            z_fine[:] += np.abs(x1_shift.T @ ctau)

        mx, fd_fine_idx = np.max(z_fine), np.argmax(z_fine)
        expSk2_fine = mx / nac

        fdfine = fdvec_fine[fd_fine_idx]  # Update fdfine if s.fstep != 1

    # ------------------------------------ 
    # ----------- Output -----------
    # ------------------------------------

    ''' Create a dictionary to store the data (equivalent to MATLAB struct). '''
    data = {
        "grid": grids,
        "tauvec": tauvec,
        "fdvec": fdvec,
        "fdfine": fdfine,
        "tau": tau,
        "nc": nc,
        "sigma2_c": sigma2_c,
        "pkvec": pkvec,
        "sk": sk
    }

    # Plot if debugPltGrid_en is enabled
    if debugPltGrid_en:
        X, Y = np.meshgrid(tauvec, fdvec)  # Equivalent to MATLAB's meshgrid
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, grids, cmap='viridis', edgecolor='none')
        
        # Optional: Configure lighting-like effect (not exactly like MATLAB's 'gouraud')
        ax.set_xlabel('Tau')
        ax.set_ylabel('Fd')
        ax.set_zlabel('Grid Value')
        plt.show()
