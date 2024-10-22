import numpy as np

m = 5000
W1 = -np.ones((m, m))
W2 = np.eye(m)
W = np.tril(W1, -1) + W2
W[:, -1] = np.ones(m)


true_x = np.random.rand(m, 1)

b = np.dot(W, true_x)

x = np.linalg.solve(W, b)

abs_error = np.abs(true_x - x)
max_error = np.max(abs_error)

print(f"Maximum entrywise absolute error: {max_error}")