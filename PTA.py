import numpy as np
import  galois
from scipy.special import erfc
from numpy.random import default_rng
from numpy import binary_repr
import math
n=7
k=4
t=n-k
rs=galois.ReedSolomon(7,4)
GF=rs.field
GF=galois.GF(2**3)
HH=rs.H
GG=rs.G
HHH=np.array(HH)
print(HHH)
HHHH=np.flip(HHH,axis=1)
print(HHHH)
GGG=np.array(GG)
message=GF.Random(rs.k)
c=rs.encode(message)
print(c)
A=np.array(c)
snr_db =range(0,10)
SNR=0
def awgn_channel(x, snr):
    rng = default_rng()
    nvar = 1 / (2 * 10**(snr/10))
    n = rng.normal(scale=np.sqrt(nvar/2), size=x.shape) +1j * rng.normal(scale=np.sqrt(nvar/2), size=x.shape)
    return x + n
def expand_matrix(A, m):
    n = 2**m - 1
    B = np.zeros((len(A), m), dtype=int)
    for i in range(len(A)):
        binary = np.binary_repr(A[i], width=m)
        for j in range(m):
            if binary[j] == '1':
                B[i][j] =1
    cexpanded = B.flatten().tolist()
    return cexpanded
m=3
y=expand_matrix(A,m)
cb=np.array(y)
print(y)
T=2*(cb)-1 
print(T)
R= awgn_channel(T,SNR)
def compute_rel(r):
    p0 = np.exp(-np.abs(r-1)**2/(2*1/(10**(SNR/10))))
    p1 = np.exp(-np.abs(r+1)**2/(2*1/(10**(SNR/10))))
    return  np.stack([p0/(p0+p1), p1/(p0+p1)], axis=1)
rel=compute_rel(R)
print(rel)
sorted=np.argsort(rel,axis=0)
print(sorted)
hard_decisions = np.argmax(rel, axis=1)
print(hard_decisions)
def Rmatrix_BPSK(r):
    # This function finds the scaled reliability based on the Euclidean distance
    # of a received signal from a noisy channel in BPSK modulation.
    #
    # Input:
    # r: The received signal from the noisy channel.
    #
    # Output:
    # scrm: The scaled reliability matrix.

    K = np.array([1, -1])   # Constellation symbols for BPSK.

    rl = len(r)
    rm = np.zeros((2, rl))   # Reliability matrix pre-allocation.

    for i in range(rl):
        for j in range(len(K)):
            rx = abs(np.real(r[i]) - K[j])   # Euclidean distance calculation.
            rm[j,i] = rx

    exprm = np.exp(-rm)   # Finding the negative exponentials values so that the closest distances have the highest values.
    rs = np.sum(exprm, axis=0)   # Finding the sum used for scaling.
    scrm = np.zeros((2, rl))   # Pre-allocation of the scaled matrix.

    for i in range(rl):
        scc = exprm[:,i] / rs[i]
        scrm[:,i] = scc

    return scrm
ftt=Rmatrix_BPSK(R)#scaled relibility matrix
print('ftt: \n',ftt) 
sorted_indices = np.argsort(ftt[0])

result = (ftt[0][sorted_indices], ftt[1][sorted_indices])
print("Sorted indices of maximum values in each column:\n", result)
#arrange the H matrix according to the the indices in ftt
#perform gaussian elimination
#return the indices to their orginal form before feeding it to the decoder 
#the syndrome and the harddecision vectoer are updated at every iteration
#compute the syndrome
def gfrref(A,x):
    """
    Compute the row-reduced echelon form of a matrix A over GF(2).

    Parameters:
    A (numpy.ndarray): the matrix to be reduced.
    tol (float): the tolerance for determining if a column is negligible.

    Returns:
    A (numpy.ndarray): the row-reduced echelon form of A.
    jb (list): a list of column indices corresponding to the pivot columns.
    """
      
    A = np.array(A, dtype=int) 
    
    mq, nq = A.shape
    t*x==mq
    n*x==nq
    jb = []  # List of pivot column indices
    i, j = 0, 0
    
    while i < mq and j < nq:
        # Find the index of the pivot row
        pivot = np.argmax(A[i:, j])
        if A[i + pivot, j]:
            # Column is negligible, zero it out
            A[i:, j] = 0
            j += 1
        else:
            # Remember the pivot column index
            jb.append(j)
            # Swap the current row with the pivot row
            A[[i, i + pivot], j:] = A[[i + pivot, i], j:]
            # Divide the pivot row by the pivot element
            A[i, j:] = A[i, j:] / A[i, j]
            # Subtract multiples of the pivot row from the other rows
            for k in range(mq):
                if k != i:
                    A[k, j:] -= A[k, j] * A[i, j:]
            i += 1
            j += 1
            
    A = np.array(A, dtype=int)
    
    return A, jb
def PTA(H, rel, iter, sigma):
    """
    This function performs the SPA decoding algorithm in the log domain.
    It is a simplification of the SPA in the log domain.
    
    Arguments:
    S -- Syndrome calculation
    H -- The Binary Parity check matrix transformed
    rel -- The reliablities ratios
    pm- parity check matrix 
    ri-- indices for the row reliablities
    iter -- Number of iterations
    delta -- The correction factor.
    
    Returns:
    new Rlb   -- The outputted reliabilities 
    """
    # Initialization 
     
    t,n =H.shape
    Z = np.ones((t, n))
    r_prime=np.argmax(ftt, axis=1)
    S=np.dot(r_prime*H)
    for i in range(iter):
        for i in range(n):
                if S[i,:]==0:
                    Z[i,:]=Z[i,:]+sigma
                else:
                    Z[i,:]=Z[i,:]-sigma/2
    
                if H[i][j]==1:
                    if S[i][j]==0:
                        Z[i][j]=Z[i][j]+sigma
                    else:
                        Z[i][j]=Z[i][j]-sigma/2
        for j in range(n):
            if S[i][j]==0:
                Vj=Z[i][j]+sigma
            else:
                Vj=Z[i][j]-sigma/2
   
        for j in range(n):
            for k in range(t):
                if H[k][j]==1:
                    if S[k][j]==0:
                        Vj=Vj+sigma
                    else:
                        Vj=Vj-sigma/2
            Lj=rel[j]+Vj
            if Lj>0:
                r_prime[j]=0
            else:
                r_prime[j]=1
            S=np.dot(r_prime*H)

    
    return Vj, Lj
def bitexpan(S, b):
    """
    This function converts decimal and GF symbols into binary bits.
    This binary bits aren't the ones generated by matlab and are in reality
    just 0s and 1s. The function works for up to eight bits only (i.e., b=8).
    
    Parameters:
        S: The array or matrix to be converted
        b: The number of bits
        
    Returns:
        B: The output matrix in bits (i.e., 0s and 1s)
    """
    
    nn = len(S)
    k = np.arange(1, b+1)
    z = np.zeros((b, nn), dtype=int)
    
    if b  == 3:
        for j in range(nn):
            if S[j] == 0:
                z[k-1, j] = [0, 0, 0]
            elif S[j] == 1:
                z[k-1, j] = [1, 0, 0]
            elif S[j] == 2:
                z[k-1, j] = [0, 1, 0]
            elif S[j] == 3:
                z[k-1, j] = [1, 1, 0]
            elif S[j] == 4:
                z[k-1, j] = [0, 0, 1]
            elif S[j] == 5:
                z[k-1, j] = [1, 0, 1]
            elif S[j] == 6:
                z[k-1, j] = [0, 1, 1]
            elif S[j] == 7:
                z[k-1, j] = [1, 1, 1]
    return z
def BIE(Hs, b):
    o, n = Hs.shape
    # Generating GF(2^b) primitive element and bit representation
    a = galois.GF(2**b).primitive_element
    aa = a**np.arange(n)
    cx = np.concatenate((aa, aa[:b-1]))
    print(cx)
    v = bitexpan(cx, b)
    
    # Generating 3 by 3 square matrices for each element of the field
    
    
    sq = np.zeros((n, b, b), dtype=int) 
    for i in range(n):
        if cx[i] != a**(i-1):
            print("t")
            #sq1=sq[i]
            #v1=v[:,i:i+b]
            #sq1[i,:]=v1
            #sq[i]=sq1
            sq[i,:,:] = v[:,i:i+b]

    print('sq: \n',sq)
    print('V: \n',v)
    # Generating Hb matrix
    Hb = np.zeros((t*b, n*b), dtype=int)
    for i in range(t):
        for j in range(n):
            for f in range(n):
                 if Hs[i,j] == a**(f):
                       if i==1 and j==1:
                           Hb[:i*b,:j*b:]=sq[f,:,:]
                       elif i==1 and j>1:
                           Hb[:i*b,j*b:(j+1)*b]=sq[f,:,:]   
                       elif i>1 and j==1:
                           Hb[i*b:(i+1)*b,:j*b]=sq[f,:,:]
                       elif i>1 and j>1:
                           Hb[i*b:(i+1)*b,j*b:(j+1)*b]=sq[f,:,:]
        
                            
    return Hb                                  
b=3
BB=BIE(HHHH,b) 
#calculate the syndrome 
def syndrome(Ht, rr, p=None):
    if p is None:
        S = np.dot(rr, Ht.T)
    else:
        m, n = Ht.shape
        k = n - m
        S = np.zeros((1, n-k), dtype=np.object)%2
        for ll in range(1, n-k+1):
            s = np.dot(rr, Ht[ll-1, :])
            S[0, ll-1] = s
    return S
def syndrome(H, r):
    """
    This function calculates the syndrome of a given codeword r.
    
    Parameters:
        H: The parity check matrix
        r: The codeword
        
    Returns:
        s: The syndrome
    """
    
    s = np.mod(np.dot(H, r), 2)
    
    return s
def participating_R(Ht, R, rn):
    # This function finds the participating R symbols in each row of the syndrome check.
    # Rp = participating_R(Ht, R, rn)
    # Rp = participating R symbols for a given row
    # Ht = the corresponding Transformed H matrix based on U and R
    # R = The Reliable symbols vector
    # rn = The row number of the H matrix
    
    Hr = Ht[:, R]
    Rx = Hr[rn, :] != 0
    Rp = R[Rx]
    
    return Rp
#multiply the reliablities across th H matrix
def Beta(S, Ht, pm, ri, delta):
    # This function is used in the new LDPC construction with the PTA.
    # The main difference is that it uses the participating R symbols in each correction
    # step instead of all the R symbols (*****)
    #include the syndrome in the loop 

    Ci = list(range(1, len(ri) + 1))  # codeword indexes or number of columns of H
    for kk in range(len(S)):
        Pi = participating_R(Ht, Ci, kk)  # ***************** Participating indexes
        if S[kk] == 0:
            for jj in range(len(Pi)):
                pm[ri[Pi[jj]] - 1, Pi[jj] - 1] += delta
        elif S[kk] != 0:
            for jj in range(len(Pi)):
                pm[ri[Pi[jj]] - 1, Pi[jj] - 1] -= delta

    return pm
def compute_BER(received_bits, transmitted_bits):
    if len(received_bits) != len(transmitted_bits):
        raise ValueError("The lengths of received_bits and transmitted_bits must be equal.")
    
    error_count = 0
    total_bits = len(received_bits)
    
    for i in range(total_bits):
        if received_bits[i] != transmitted_bits[i]:
            error_count += 1
    
    BER = error_count / total_bits
    return BER   
#Rlb=newRmatrix4(S,Ht,Rlb,ri,delta)
Nerr=compute_BER(PTA,cb)
#compute the hard decision for chat
# apply the PTA 
import numpy as np
import  galois
from scipy.special import erfc
from numpy.random import default_rng
from numpy import binary_repr
import math
#parameters
n=7
k=4
t=n-k
rs=galois.ReedSolomon(7,4)
GF=rs.field
GF=galois.GF(2**3)
HH=rs.H
GG=rs.G
HHH=np.array(HH)
print(HHH)
HHHH=np.flip(HHH,axis=1)
print(HHHH)
GGG=np.array(GG)
message=GF.Random(rs.k)
c=rs.encode(message)
print(c)
A=np.array(c)
snr_db =range(0,10)
SNR=0
def awgn_channel(x, snr):
    rng = default_rng()
    nvar = 1 / (2 * 10**(snr/10))
    n = rng.normal(scale=np.sqrt(nvar/2), size=x.shape) #1j * rng.normal(scale=np.sqrt(nvar/2), size=x.shape)
    return x + n
def expand_matrix(A, m):
    n = 2**m - 1
    B = np.zeros((len(A), m), dtype=int)
    for i in range(len(A)):
        binary = np.binary_repr(A[i], width=m)
        for j in range(m):
            if binary[j] == '1':
                B[i][j] =1
    cexpanded = B.flatten().tolist()
    return cexpanded
m=3
y=expand_matrix(A,m)
cb=np.array(y)
print(y)
T=2*(cb)-1 
print(T)
R= awgn_channel(T,SNR)
#probability matrix (pm)
def compute_rel(r):
    p0 = np.exp(-np.abs(r-1)**2/(2*1/(10**(SNR/10))))
    p1 = np.exp(-np.abs(r+1)**2/(2*1/(10**(SNR/10))))
    return  np.stack([p0/(p0+p1), p1/(p0+p1)], axis=1)
rel=compute_rel(R)
print(rel)
#hard decision of pm
hard_decision = np.argmax(rel, axis=1)
print(hard_decision)
#SORTING SECTION
#ORIGINAL INDICES
indices = np.arange(len(rel))  # Get indices from 0 to the number of rows in the matrix
rightmost_column_indices = indices[np.newaxis, -1]  # Get the indices of the rightmost column
print(rightmost_column_indices)
#SORTED INDICES FROM LOWEST TO HIGHEST
sorted_indices = np.argsort(rel[:, -1])
sorted_matrix = rel[sorted_indices]
print(sorted_matrix)
#other section 
#relt=np.transpose(rel)
#sorted=np.argsort(rel,axis=0)
#print(sorted)
#hard_decisions = np.argmax(rel, axis=1)
#print(hard_decisions)
#def Rmatrix_BPSK(r):
    # This function finds the scaled reliability based on the Euclidean distance
    # of a received signal from a noisy channel in BPSK modulation.
    #
    # Input:
    # r: The received signal from the noisy channel.
    #
    # Output:
    # scrm: The scaled reliability matrix.

    #K = np.array([1, -1])   # Constellation symbols for BPSK.

    #rl = len(r)
    #rm = np.zeros((2, rl))   #Reliability matrix pre-allocation.

    #for i in range(rl):
        #for j in range(len(K)):
            #rx = abs(np.real(r[i]) - K[j])   # Euclidean distance calculation.
            #rm[j,i] = rx

    #exprm = np.exp(-rm)   # Finding the negative exponentials values so that the closest distances have the highest values.
    #rs = np.sum(exprm, axis=0)   # Finding the sum used for scaling.
    #scrm = np.zeros((2, rl))   # Pre-allocation of the scaled matrix.

    #for i in range(rl):
        #scc = exprm[:,i] / rs[i]
        #scrm[:,i] = scc

    #return scrm
#ftt=Rmatrix_BPSK(R)#scaled relibility matrix
#print('ftt: \n',ftt) 
#sorted_indices = np.argsort(relt[1])
#print(sorted_indices)
#result = (relt[0][sorted_indices], relt[1][sorted_indices])
#print("Sorted indices of maximum values in each column:\n", result)

#arrange the H matrix according to the the indices in relt

#perform gaussian elimination
#return the indices to their orginal form before feeding it to the decoder 
#the syndrome and the harddecision vectoer are updated at every iteration
#compute the syndrome
#add the  scaling factor within this code to ensure that when adding subtracting beta to ensure that the column adds up to 1
def gfrref(A,x):
    """
    Compute the row-reduced echelon form of a matrix A over GF(2).

    Parameters:
    A (numpy.ndarray): the matrix to be reduced.
    tol (float): the tolerance for determining if a column is negligible.

    Returns:
    A (numpy.ndarray): the row-reduced echelon form of A.
    jb (list): a list of column indices corresponding to the pivot columns.
    """
      
    A = np.array(A, dtype=int) 
    
    mq, nq = A.shape
    t*x==mq
    n*x==nq
    jb = []  # List of pivot column indices
    i, j = 0, 0
    
    while i < mq and j < nq:
        # Find the index of the pivot row
        pivot = np.argmax(A[i:, j])
        if A[i + pivot, j]:
            # Column is negligible, zero it out
            A[i:, j] = 0
            j += 1
        else:
            # Remember the pivot column index
            jb.append(j)
            # Swap the current row with the pivot row
            A[[i, i + pivot], j:] = A[[i + pivot, i], j:]
            # Divide the pivot row by the pivot element
            A[i, j:] = A[i, j:] / A[i, j]
            # Subtract multiples of the pivot row from the other rows
            for k in range(mq):
                if k != i:
                    A[k, j:] -= A[k, j] * A[i, j:]
            i += 1
            j += 1
            
    A = np.array(A, dtype=int)
    
    return A, jb
def PTA(H, rel, iter, sigma):
    """
    This function performs the SPA decoding algorithm in the log domain.
    It is a simplification of the SPA in the log domain.
    
    Arguments:
    S -- Syndrome calculation
    H -- The Binary Parity check matrix transformed
    rel -- The reliablities ratios
    pm- parity check matrix 
    ri-- indices for the row reliablities
    iter -- Number of iterations
    delta -- The correction factor.
    
    Returns:
    new Rlb   -- The outputted reliabilities 
    """
    # Initialization 
     
    t,n =H.shape
    Z = np.ones((t, n))
    r_prime=np.argmax(ftt, axis=1)
    S=np.dot(r_prime*H)
    for i in range(iter):
        for i in range(n):
                if S[i,:]==0:
                    Z[i,:]=Z[i,:]+sigma
                else:
                    Z[i,:]=Z[i,:]-sigma/2
    
                if H[i][j]==1:
                    if S[i][j]==0:
                        Z[i][j]=Z[i][j]+sigma
                    else:
                        Z[i][j]=Z[i][j]-sigma/2
        for j in range(n):
            if S[i][j]==0:
                Vj=Z[i][j]+sigma
            else:
                Vj=Z[i][j]-sigma/2
   
        for j in range(n):
            for k in range(t):
                if H[k][j]==1:
                    if S[k][j]==0:
                        Vj=Vj+sigma
                    else:
                        Vj=Vj-sigma/2
            Lj=rel[j]+Vj
            if Lj>0:
                r_prime[j]=0
            else:
                r_prime[j]=1
            S=np.dot(r_prime*H)

    
    return Vj, Lj
def bitexpan(S, b):
    """
    This function converts decimal and GF symbols into binary bits.
    This binary bits aren't the ones generated by matlab and are in reality
    just 0s and 1s. The function works for up to eight bits only (i.e., b=8).
    
    Parameters:
        S: The array or matrix to be converted
        b: The number of bits
        
    Returns:
        B: The output matrix in bits (i.e., 0s and 1s)
    """
    
    nn = len(S)
    k = np.arange(1, b+1)
    z = np.zeros((b, nn), dtype=int)
    
    if b  == 3:
        for j in range(nn):
            if S[j] == 0:
                z[k-1, j] = [0, 0, 0]
            elif S[j] == 1:
                z[k-1, j] = [0, 0, 1]
            elif S[j] == 2:
                z[k-1, j] = [0, 1, 0]
            elif S[j] == 3:
                z[k-1, j] = [0, 1, 1]
            elif S[j] == 4:
                z[k-1, j] = [1, 0, 0]
            elif S[j] == 5:
                z[k-1, j] = [1, 0, 1]
            elif S[j] == 6:
                z[k-1, j] = [1, 1, 0]
            elif S[j] == 7:
                z[k-1, j] = [1, 1, 1]
    return z
def BIE(Hs, b):
    o, n = Hs.shape
    # Generating GF(2^b) primitive element and bit representation
    a = galois.GF(2**b).primitive_element
    aa = a**np.arange(n)
    cx = np.concatenate((aa, aa[:b-1]))
    print(cx)
    v = bitexpan(cx, b)
    
    # Generating 3 by 3 square matrices for each element of the field
    
    
    sq = np.zeros((n, b, b), dtype=int) 
    for i in range(n):
        if cx[i] != a**(i-1):
            print("t")
            #sq1=sq[i]
            #v1=v[:,i:i+b]
            #sq1[i,:]=v1
            #sq[i]=sq1
            sq[i,:,:] = v[:,i:i+b]

    print('sq: \n',sq)
    print('V: \n',v)
    # Generating Hb matrix
    Hb = np.zeros((t*b, n*b), dtype=int)
    for i in range(t):
        for j in range(n):
            for f in range(n):
                 if Hs[i,j] == a**(f):
                     Hb[i*b:(i+1)*b,j*b:(j+1)*b]=sq[f,:,:]
                       #if i==1 and j==1:
                           #Hb[:i*b,:j*b:]=sq[f,:,:]
                       #elif i==1 and j>1:
                           #Hb[:i*b,j*b:(j+1)*b]=sq[f,:,:]   
                       #elif i>1 and j==1:
                           #Hb[i*b:(i+1)*b,:j*b]=sq[f,:,:]
                       #elif i>1 and j>1:                         
    return Hb                                          
b=3
BB=BIE(HHHH,b) 
print('BB: \n',BB)
Hsorted=BB[:,sorted_indices]
print('Hsorted: \n',Hsorted)
original_indices = np.argsort(sorted_indices)
print('original_indices: \n',original_indices)
def gaussian_elimination(matrix):
    # Perform Gaussian elimination
    num_rows, num_cols = matrix.shape
    for col in range(num_cols):
        for row in range(col + 1, num_rows):
            if matrix[row, col] == 1:
                matrix[row] = (matrix[row] + matrix[col]) % 2

    # Back-substitution to obtain reduced row echelon form
    for col in range(num_cols - 1, -1, -1):
        pivot_row = -1
        for row in range(num_rows):
            if matrix[row, col] == 1 and pivot_row == -1:
                pivot_row = row
            elif matrix[row, col] == 1:
                matrix[row] = (matrix[row] + matrix[pivot_row])%2 
        if pivot_row != -1:
            matrix[pivot_row] = matrix[pivot_row] / matrix[pivot_row, col]
            for row in range(pivot_row):
                if matrix[row, col] == 1:
                    matrix[row] = (matrix[row] + matrix[pivot_row]) %2

    return matrix  
Ht=gaussian_elimination(Hsorted)
print('Ht:\n',Ht)
Htt=Ht[:,original_indices]
print('Htt:\n',Htt)
#calculate the syndrome 
def syndrome(Ht, rr, p=None):
    if p is None:
        S = np.dot(rr, Ht.T)
    else:
        m, n = Ht.shape
        k = n - m
        S = np.zeros((1, n-k), dtype=np.object)%2
        for ll in range(1, n-k+1):
            s = np.dot(rr, Ht[ll-1, :])
            S[0, ll-1] = s
    return S
#def syndrome(H, r):
    """
    This function calculates the syndrome of a given codeword r.
    
    Parameters:
        H: The parity check matrix
        r: The codeword
        
    Returns:
        s: The syndrome
    """
    
    #s = np.mod(np.dot(H, r), 2)
    
    #return s
maximum_values = np.amax(relt, axis=0)
print('maximum_values:\n',maximum_values)
indices = np.argmax(relt, axis=0)
print('indices:\n',indices)
column_indices = np.where(indices== 1)[0]
print('column_indices:\n',column_indices)
selected_columns = Htt[:, column_indices]
print('selected_columns:\n',selected_columns)
def participating_R(Ht, xc,rn ):
    # This function finds the participating R symbols in each row of the syndrome check.
    # Rp = participating_R(Ht, R, rn)
    # Rp = participating R symbols for a given row
    # Ht = the corresponding Transformed H matrix based on U and R
    # R = The Reliable symbols vector
    # rn = The row number of the H matrix
    # xc = The column indexes of the H matrix
    Hr=Ht[:,column_indices]
    Rx = Hr[rn,:] != 0
    Rp = column_indices[Rx]
    #Hr=Ht(:,R);
    #Rx = Hr(rn,:)~=0;
    #Rp=R(Rx);
    
    return Rp
#multiply the reliablities across th H matrix
def Beta(Ht, pm, delta,rr):
    #This function is used in the new LDPC construction with the PTA.
    #The main difference is that it uses the participating R symbols in each correction
    #step instead of all the R symbols (*****)
    #include the syndrome in the loop
    #rr is the hard decision vector  
    q,w=Ht.shape
    ri=np.arange(Ht.shape[0])[:, np.newaxis]
    print('ri:\n',ri)
    Ci = list(range(1, len(ri) + 1))  # codeword indexes or number of columns of H
    S=syndrome(Ht,rr,p=None)
    for kk in range(len(S)):
        Pi = participating_R(Ht, Ci, kk)  # ***************** Participating indexes
        print('Pi:\n',Pi)
        #scale the probability matrix 
        #pm = pm * (S[kk, :] / (np.sum(S[kk, :]) * (np.sum(S[kk, :]) - 1)))
        for j in range(len(Pi)):
            if S[kk] != 0:
                 print('Pi:\n',Pi)
                 if S[kk] == 0:
                      print('S[kk]:\n',S[kk])
                      #correctionstage=pm[ri[Pi[j]], Pi[j]]+delta 
                      pm[:,ri[Pi[j]]]=pm[:,ri[Pi[j]]]+delta
                      #pm[ri[Pi[j]],Pi[j]]/np.sum(pm[Pi[j]])  
                 elif S[kk] != 0:
                      #correctionstagee=pm[ri[Pi[j]], Pi[j]] - delta
                      pm[:,ri[Pi[j]]]=pm[:,ri[Pi[j]]]-delta
                      #pm[ri[Pi[j]],Pi[j]]/np.sum(pm[Pi[j]]-1)               
                #termination citerion
    return pm

Rlb=Beta(Htt,relt,delta=0.05,rr=hard_decisions)
print('Rlb:\n',Rlb)
#decoded codeword
#rr=np.argmax(Beta,axis=0)
#def compute_BER(received_bits, transmitted_bits):
    #if len(received_bits) != len(transmitted_bits):
        #raise ValueError("The lengths of received_bits and transmitted_bits must be equal.")
    
    #error_count = 0
    #total_bits = len(received_bits)
    
    #for i in range(total_bits):
        #if received_bits[i] != transmitted_bits[i]:
            #error_count += 1
    
    #BER = error_count / total_bits
    #return BER   
#Nerr=compute_BER(rr,cb)
#compute the hard decision for chat
# apply the PTA 
