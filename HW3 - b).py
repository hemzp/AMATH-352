import numpy as np
from scipy.linalg import lu

m = 20

W1 = -np.ones((m, m))
W2 = np.eye(m)
W = np.tril(W1, -1) + W2
W[:, -1] = np.ones(m)

P, L, U = lu(W)

last_column_U = U[:, -1]

print("Last column of U when m = 10:")
print(last_column_U)

