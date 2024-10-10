import numpy as np 

def tridiagsolver(lowerdiag, upperdiag, maindiag, b):

    d = maindiag.copy()
    u = upperdiag.copy()
    b_tilde = b.copy()
    
    m = len(d)
    x = np.zeros(m)

    for i in range (1, m):
        m_i = lowerdiag[i-1] / d[i-1]
        d[i] = d[i] - m_i*u[i-1] 
        b_tilde[i] = b_tilde[i] - m_i * b_tilde[i-1]

    x[m - 1] = b[m - 1] / maindiag[m - 1]

    for i in range(m-2, -1, -1):
        x[m - 1] = (b[m - 1] - upperdiag[i]*x[i + 1]) / maindiag[i]

    return x




