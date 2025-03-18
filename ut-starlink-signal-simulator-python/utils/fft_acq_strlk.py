'''' ftt_acq_strlk(s)
     Converted from the Matlab file fttAcqStrlk.m.
'''

import numpy as np

def fft_acq_strlk(s):
    ''' Debugging flags
        TODO : I am unsure if these are meant to be in function scope or global scope. Matlab is confusing.
    '''
    debugPltHT_enable = 0
    debugPltGrid_en = 0
    fontsize = 12

    ''' Params '''
    n = 1024
    ng = 32
    ns = n + ng
    ndsym = 300
    fs = 240e6
    nk = 1/750*fs

    '''TODO :
        Ensure gen_pss, and gen_sss are implemented/converted correctly.
        Ensure gen_pss and gen_sss are imported correctly into this file.
        Ensure gen_pss, and gen_sss are used correctly in this file.
    '''
    
    pss = gen_pss()
    sss = gen_sss()

    ''' Concatenates the two signals vertically into a single column vector c.
        This combined signal represents a known Starlink synchronization frame.
        This was originally c = [PSS; SSS]; in Matlab.
        TODO : Might need to use np.concatenate instead.
    '''
    c = np.vstack((pss, sss))

    ''' Assigns c to pkvec.
        pkvec will store the known synchronization frame before zero padding is applied.
        The original Matlab line was pkVec = c;
        We use c.copy() though since Python passes by references whilst Matlab passes by value.
    '''
    pkvec = c.copy()

    ''' Zero padding
        Extends c to match a required frame length nk.
        If nk is greateer than length(c) this ensures c has exactly nk elements by appending zeros.
        This is IMPORTANT as it ensures proper signal processing and alignment in acquisition algorithms.
        The original Matlab line was c = [c; zeros(Nk - length(c), 1)];
    '''
    c = np.vstack((c, np.zeros(nk - len(c), 1)))

    ''' Resample to full BW '''
    y = s['y']

    if s['fsr'] != fs:
        tVec = np.arange(len(y)).reshape(-1, 1) / s['Fsr']

        #TODO : This is still the original Matlab line. I am unsure how to convert to Python.
        y = resample(y,tVec,Fs)
    buffer = 100

    ''' Remove receiver bias '''

    ''' Derived params '''

    ''' Generate acq. grid '''

    ''' Process Acqiosition '''

    #Find Peak, tau, and doppler

    #Find finer doppler

    '''Output'''