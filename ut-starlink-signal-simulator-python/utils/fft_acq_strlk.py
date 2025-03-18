'''' ftt_acq_strlk(s)
     Converted from the Matlab file fttAcqStrlk.m.
'''

import numpy as np

def fft_acq_strlk(s):
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

    pss = gen_pss()
    sss = gen_sss()

    #TODO : Might need to use np.concatenate instead.
    c = np.vstack((pss, sss))
    pkvec = c.copy()
    c = np.vstack((c, np.zeros(nk - len(c), 1)))

    ''' Resample to full BW '''
    y = s['y']

    if s['fsr'] != fs:
        tVec = np.arange(len(y)).reshape(-1, 1) / s['Fsr']

        #TODO : This is the original Matlab line. I am unsure how to convert.
        y = resample(y,tVec,Fs)
    buffer = 100

    ''' Remove receiver bias '''

    ''' Derived params '''

    ''' Generate acq. grid '''

    ''' Process Acqiosition '''

    #Find Peak, tau, and doppler

    #Find finer doppler

    '''Output'''