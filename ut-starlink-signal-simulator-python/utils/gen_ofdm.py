def gen_ofdm(s):
    ''' Optional parameters and checks. '''
    if 'gutter' not in s:
        s['gutter'] = 0
    if 'nysm' not in s:
        s['nysm'] = 1

    ''' Generate data symbols each with midx bits of information. '''