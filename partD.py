import numpy as np
import time
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

def tridiagsolver(lowerdiag, maindiag, upperdiag, b):
    d = maindiag.copy()
    u = upperdiag.copy()
    b_tilde = b.copy()
    
    m = len(d)
    x = np.zeros(m)

    
    for i in range(1, m):
        m_i = lowerdiag[i - 1] / d[i - 1]
        d[i] = d[i] - m_i * u[i - 1]
        b_tilde[i] = b_tilde[i] - m_i * b_tilde[i - 1]

    
    x[m - 1] = b_tilde[m - 1] / d[m - 1]

    for i in range(m - 2, -1, -1):
        x[i] = (b_tilde[i] - u[i] * x[i + 1]) / d[i]

    return x


M = np.linspace(500, 6000, 10)
M = np.round(M).astype(int)


timetridiag = np.zeros(len(M))
timestandard = np.zeros(len(M))

for j in range(len(M)):
    m = M[j]

    upperdiag = np.random.rand(m - 1)
    lowerdiag = np.random.rand(m - 1)

    maindiag = np.concatenate(([0], upperdiag)) + np.concatenate((lowerdiag, [0])) + np.random.rand(m) + 2
    b = np.random.rand(m)

    diagonals = [lowerdiag, maindiag, upperdiag]
    offsets = [-1, 0, 1]
    A_sparse = diags(diagonals, offsets, shape=(m, m))
    

    
    start_time = time.perf_counter()
    x_tridiag = tridiagsolver(lowerdiag, maindiag, upperdiag, b)
    end_time = time.perf_counter()
    timetridiag[j] = end_time - start_time

    
    start_time = time.perf_counter()
    x_standard = spsolve(A_sparse, b)
    end_time = time.perf_counter()
    timestandard[j] = end_time - start_time


plt.figure(figsize=(10, 6))
plt.semilogy(M, timetridiag, label='Tridiagonal Solver', marker='o')
plt.semilogy(M, timestandard, label='Standard Solver', marker='s')
plt.xlabel('Matrix Size (m)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Matrix Size')
plt.legend()
plt.grid(True)
plt.show()
