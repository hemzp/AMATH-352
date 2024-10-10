import numpy as np

def forward_elimination(lowerdiag, upperdiag, maindiag, b):
    
    
    d = maindiag.copy()
    u = upperdiag.copy()
    b_tilde = b.copy()
    
    m = len(d)


    for i in range (1, m):
        m_i = lowerdiag[i-1] / d[i-1]

        d[i] = d[i] - m_i*u[i-1] 

        b_tilde[i] = b_tilde[i] - m_i * b_tilde[i-1]

    return d, u, b_tilde


# Example values for the tridiagonal matrix
lowerdiag = np.array([1, 1, 1])
maindiag = np.array([4, 4, 4, 4])
upperdiag = np.array([1, 1, 1])
b = np.array([5, 5, 5, 5])

# Call the forward elimination function
d, u, b_tilde = forward_elimination(lowerdiag, maindiag, upperdiag, b)

# Output the result
print("Updated main diagonal (d):", d)
print("Updated upper diagonal (u):", u)
print("Updated right-hand side (b_tilde):", b_tilde) 