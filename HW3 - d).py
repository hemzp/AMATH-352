import numpy as np
from scipy.linalg import lu

for m in [10, 50, 100]:

    W1 = -np.ones((m, m))
    W2 = np.eye(m)
    W = np.tril(W1, -1) + W2
    W[:, -1] = np.ones(m)


    cond_W = np.linalg.cond(W)
    

    P, L, U = lu(W)

    cond_U = np.linalg.cond(U)
    
    print(f"m = {m}")
    print(f"Condition number of W: {cond_W:.2e}")
    print(f"Condition number of U: {cond_U:.2e}")
    print("")
