import numpy as np 
from scipy.sparse import diags

def tridiagsolver(lowerdiag, upperdiag, maindiag, b):
    d = maindiag.copy()
    u = upperdiag.copy()
    b_tilde = b.copy()
    
    m = len(d)
    x = np.zeros(m)

    # Forward elimination
    for i in range(1, m):
        m_i = lowerdiag[i - 1] / d[i - 1]
        d[i] = d[i] - m_i * u[i - 1]
        b_tilde[i] = b_tilde[i] - m_i * b_tilde[i - 1]

    # Back substitution
    x[m - 1] = b_tilde[m - 1] / d[m - 1]

    for i in range(m - 2, -1, -1):
        x[i] = (b_tilde[i] - u[i] * x[i + 1]) / d[i]

    return x

m = 5000

upperdiag = np.random.rand(m - 1)
lowerdiag = np.random.rand(m - 1)
maindiag = np.concatenate(([0], upperdiag)) + np.concatenate((lowerdiag, [0])) + np.random.rand(m) + 2
b = np.random.rand(m)


diagonals = [lowerdiag, maindiag, upperdiag]
offsets = [-1, 0, 1]
A_sparse = diags(diagonals, offsets, shape=(m, m))
A = A_sparse.toarray()
true_x = np.linalg.solve(A, b)

# Solve using tridiagsolver
x = tridiagsolver(lowerdiag, upperdiag, maindiag, b)

# Calculate relative error
error = np.max(np.abs(true_x - x)) / np.max(np.abs(true_x))
print(f"Relative Error: {error}")

# Additional print statements
print("First 10 elements of the computed solution x:")
print(x[:10])

print("\nFirst 10 elements of the true solution true_x:")
print(true_x[:10])

print("\nDifference between computed and true solutions (first 10 elements):")
print((true_x - x)[:10])
