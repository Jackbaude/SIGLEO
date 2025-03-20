''' gen_strlk_frame.py
    Generate a Starlink frame.
    Data is packaged into frames.
    Transmitted at 750 Hz consisting of 302 symbols. 2 of which are known.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

def gen_strlk_frame(s):