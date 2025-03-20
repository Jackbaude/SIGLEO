''' gen_strlk_stream.py
    Generate a stream of Starlink frames.
    Data is packaged into frames transmitted at 750 Hz.
    Consisting of 302 symbols.
    Two of which are known.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

def gen_strlk_stream(s):