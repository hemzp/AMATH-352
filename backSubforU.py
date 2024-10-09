import numpy as np

def backSubstitution(upperdiag, b, maindiag):
    
    m = len(upperdiag)
    x = np.zeros(m)

    x[m - 1] = b[m - 1] / maindiag[m - 1]

    for i in range(m-2, -1, -1):

        x[m - 1] = (b[m - 1] - upperdiag[i]*x[i + 1]) / maindiag[i]

    return x
