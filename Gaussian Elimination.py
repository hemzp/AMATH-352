import numpy as np

# Example augmented matrix [A | b]
matrix = np.array([[2, 1, -1, 8],
                   [-3, -1, 2, -11],
                   [-2, 1, 2, -3]], dtype=float)

def gaussian_elimination(matrix):
    n = matrix.shape[0]

    for i in range(n):
        # Pivot: Ensuring the diagonal is non-zero
        if matrix[i, i] == 0:
            for j in range(i + 1, n):
                if matrix[j, i] != 0:
                    matrix[[i, j]] = matrix[[j, i]]  # Swap rows
                    break

        # Elimination: Making the elements below the diagonal zero
        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] -= factor * matrix[i, i:]
    
    return matrix


# Perform Gaussian Elimination
upper_triangular_matrix = gaussian_elimination(matrix.copy())


# Print the results
print("Upper Triangular Matrix:")
print(upper_triangular_matrix)
