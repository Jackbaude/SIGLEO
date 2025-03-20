''' gen_pss.py
    Generate the Primary Synchronization Sequence (PSS) for Starlink.
    Ported to Python from MATLAB by Jack R. Tschetter on 03/19/2025.
'''

import numpy as np
"""
-- Input --
None. This function does not take in any input parameters.


-- Output --
- **pss** (numpy array): (N + ng) x 1 PSS. For Starlink specifically N = 1024, ng = 32.
Returned sampled at 240e6 Hz.
"""
def gen_pss():
    ''' The duration of OFDM symbol guard interval (cyclic prefix)
        Expressed in intervals of 1/fs.
    '''
    ng = 32

    ''' Number of sub-segments in the primary synchronization sequence (PSS).
    '''
    npss_seg = 8

    # ------------------------------------ 
    # ----------- Implementation of Fibonacci Linear Feedback Shift Register (LFSR) -----------
    # ------------------------------------ 

    ''' Create a column vector with the elements 3 and 7.
        The transpose operation from MATLAB is not needed with Python.
        Python does not automatically treat lists as column or row vectors in the way MATLAB does with its arrays.
        The original Matlab line was ciVec = [3 7]'; 
     '''
    ci_vec = [3, 7]

    ''' Initialize a column vector with specific sequence of binary values (0s and 1s).
        The sequence [0 0 1 1 0 1 0] is the initial state or "seed" for a shift register. In this case for Linear Feedback Shift Register (LFSR).
        This is then used for generating pseudo-random binary sequences.
        Each position of the vector represents a bit within a register.
        Shifting operations and feedback mechanisms operate based on this initial configuration.
    '''
    a_0_vec = [0, 0, 1, 1, 0, 1, 0]

    n = 7

    m = 2**n - 1

    ''' Initialize a column vector of length m filled with zeros.
        This vector will store sequential data generated within a loop.
        In our case the outputs from an LFSR (Linear Feedback Shift Register)
        In this Python code np.zeros((m, 1)) creates an m-by-1 array (i.e., a column vector) of zeros.
        The tuple (m, 1) specifies the shape of the array.
        NOTE : Using NumPy requires explicit tuple notation for array dimensions.
        Whereas MATLAB directly uses the argument in the zeros function for dimensions.
        TODO : Do we need a column vector? Does NumPy explicitly require a column vector shape?
    '''
    lfsr_seq = np.zeros((m, 1))

    ''' This loop is part of our Linear Feedback Shift Register (LFSR) implementation.
        We can break it down into 4 parts.

        1. ) Loop structure.
            • In the old MATLAB code for idx=1:m iterates from 1 to m.
            • m is the length of the sequence that the LFSR can generate before it starts repeating.
        2. ) Feedback Calculation.
            • In the old MATLAB code buffer = a0Vec(ciVec); extracts elements from a0Vec at the positions specified in ciVec.
            • These positions are the "taps" where feedback is applied. The values extracted by ciVec are used to compute the next bit.
            • In the old MATLAB code val = rem(sum(buffer),2); computes the next bit to be fed back into the register.
            • The sum of the selected bits from buffer is taken, and the remainder when divided by 2 gives a result of either 0 or 1. 
            • Effectively this just calculates the parity of the selected bits (XOR).
        3. ) State Update.
            • In the old MATLAB code a0Vec = [val; a0Vec(1:end-1)]; updates the state of the LFSR.
            • The new value val is inserted at the beginning of the vector, and the last element of the vector is discarded (this is the shift operation). 
            • This shifts all bits to the right, with the new bit entering from the left.
        4. ) Storing the Output.
            • In the old MATLAB code lfsrSeq(idx) = val; stores the computed value val in the lfsrSeq array at position idx.
            • This array captures the output sequence from the LFSR.
    '''
    for idx in range(m):
        buffer = a_0_vec[ci_vec]

        ''' Get the new bit.
            This is XOR of the selected bits.
        '''
        val = sum(buffer) % 2

        ''' Insert the new bit at the start.
            Then remove the last bit.
        '''
        a_0_vec = np.insert(a_0_vec[:-1], 0, val)

        ''' Store the output bit. '''
        lfsr_seq[idx] = val
    
    ''' Flip up-down.
        This MATLAB function reverses the order of the elements in a column vector or matrix vertically.
        In Python we can acheive the same operation just using NumPy's slicing capabilities.
        The original Matlab line was lfsrSeqMod = flipud(lfsrSeq);
     '''
    lfsr_seq_mod = lfsr_seq[::-1]

    ''' Add a zero at the beginning of the vector lfsr_seq_mod.
        MATLAB does this by using semicolons (;) within brackets to stack arrays vertically.
        In MATLAB a new row containing the value 0 is inserted above the existing contents of lfsrSeqMod. 
        This makes 0 the first element in the updated vector.
        In python we can achieve this result with NumPy functions.
        TODO : What data type do we want for lfsr_seq_mod? 
        Currently I assume lfsr_seq_mod is a numpy array.
        If it is a simple list use lfsrSeqMod = [0] + lfsrSeqMod instead.
    '''
    lfsr_seq_mod = np.insert(lfsr_seq_mod, 0, 0)

    ''' Convert elements of the lfsr_seq_mod array from a binary sequence to a bipolar sequence.
        This is done by multiplying each element by 2 and then subtracting 1.
        We use bipolar encoding where elements are -1 and 1 for signal modulation purposes.
        NOTE : This implementation assumes lfsr_seq_mod is already a numpy array.
        TODO : If lfsr_seq_mod is a simple list and not numpy array convert it to numpy array in order to perform elementwise multiplication.
    '''
    seq = 2 * lfsr_seq_mod - 1

    ''' Generate a sequence of complex numbers based on the exponential function.
        This is used to generate Phase-Shift-Keyed (PSK) signals.
        seq is the bipolar sequence from above.
        cumsum(seq) calculates the cumulative sum of the elements in seq. This is a phase accumulator for the modulation.
        The line 1j*0.5*pi*cumsum(seq) computes a phase shift where 1j repressents the imaginary unit.
        0.5*pi is used to scale the cumulative sum appropriately to the type of modulation.
        -1j*pi/4 is a constant phase offset. Subtracting one quarter of pi, adjusted by the imaginary unit 1j.
        exp(...) Converts phase information into a complex exponential form. This represents the phase modulated signal.
        The original Matlab line was pkSegVec = [exp(-1j*pi/4 - 1j*0.5*pi*cumsum(seq))];
   '''
    pk_seg_vec = np.exp(-1j * np.pi / 4 - 1j * 0.5 * np.pi * np.cumsum(seq))

    ''' Construct a sequence which is part of a patter in the synchronization sequence.
        In MATLAB -pkSegVec is taking the complex conjugate of each element in the vector pkSegVec.  
        Negating a complex vector negates both the real and imaginary parts. Essentially this reflects the vector about the origin of the complex plane.
        
        In MATLAB repmat(pkSegVec, NpssSeg - 1, 1) replicates the vector pkSegVec vertically (NpssSeg - 1) times.
        The 1 indicates that the replication is column-wise (the shape of pkSegVec is preserved).
        NpssSeg defines how many segments or repeats of the signal to use.
        
        In MATLAB the semicolon (;) is used for vertical concatenation.
        The negative of the segment vector pkSegVec is placed first.
        Followed by the replicated segments of pkSegVec.
        This creates a signal with specific initial phase inversion.

        Offsetting different parts of a signal is useful for interference avoidance and pattern diversity.
    '''
    pk_vec = np.vstack([-pk_seg_vec, np.tile(pk_seg_vec, (npss_seg - 1, 1))])

    ''' Extract a portion of the vector pk_vec to serve as a cyclic prefix.
        In OFDM a cyclic prefix is added to each OFDM symbol to mitigate inter symbol interference caused by multipath propogation.
    
        In MATLAB end-Ng+1:end specifies a range starting from ng elements from the end of pkVec to the end of pkVec.
        This slice is effectively the last Ng elements of the vector.
        Basically Ng is the length of the cyclic prefix.
        This is IMPORTANT. It defines how many of the last samples of the OFDM symbol are repeated at the beginning to create a guard interval.
    '''
    pk_cp = pk_vec[-ng:]

    ''' Create the Primary Synchronization Sequence (PSS).
        To do this construct a new vector by vertically concatenating the two vectors -pk_cp and pk_vec.
        The original Matlab line was pss = [-pkCP; pkVec];
    '''
    pss = np.vstack([-pk_cp, pk_vec])