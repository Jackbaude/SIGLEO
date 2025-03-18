import numpy as np
from get_closest_fch import get_closest_fch
from gen_pss import gen_pss
from gen_sss import gen_sss

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

    # ------------------------------------ 
    # ----------- Generate acq. grid -----------
    # ------------------------------------ 

    # ------------------------------------ 
    # ----------- Process Acqiosition -----------
    # ------------------------------------ 

    #Find Peak, tau, and doppler

    #Find finer doppler

    # ------------------------------------ 
    # ----------- Output -----------
    # ------------------------------------ 