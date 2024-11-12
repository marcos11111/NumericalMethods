import numpy as np

def matrix_vector_multiply(matrix, vector):
    result = []
    for row in matrix:
        row_result = sum(row[i] * vector[i] for i in range(len(vector)))
        result.append(row_result)
    return result

# Define the matrix
matrix = [
    [6, 5, -5],
    [2, 6, -2],
    [2, 5, -1]
]

# Define a tolerance for convergence
tolerance = 1e-6
max_iterations = 1000

# Initialize the eigenvector with arbitrary values (non-zero)
n = len(matrix)
eigenvector = [1.0] * n

# Power Iteration
for _ in range(max_iterations):
    # Multiply matrix by the current vector
    next_eigenvector = matrix_vector_multiply(matrix, eigenvector)
    
    # Normalize the vector
    norm = max(abs(x) for x in next_eigenvector)  # Find the largest absolute value for scaling
    next_eigenvector = [x / norm for x in next_eigenvector]
    
    # Check for convergence
    diff = sum(abs(next_eigenvector[i] - eigenvector[i]) for i in range(n))
    if diff < tolerance:
        break
    
    eigenvector = next_eigenvector

# The largest eigenvalue is approximated by the Rayleigh quotient
numerator = sum(
    next_eigenvector[i] * matrix_vector_multiply(matrix, next_eigenvector)[i]
    for i in range(n)
)
denominator = sum(x ** 2 for x in next_eigenvector)
largest_eigenvalue = numerator / denominator

# Print the largest eigenvalue and corresponding eigenvector
print("Largest Eigenvalue:", largest_eigenvalue)
print("Corresponding Eigenvector:", next_eigenvector)




# Define tolerance for convergence and maximum iterations
tolerance = 1e-6
max_iterations = 1000

# Initialize eigenvector with arbitrary values (non-zero)
n = len(matrix)
eigenvector = [1.0] * n

def normalize(vector):
    #Normalize a vector by its maximum absolute component.
    max_val = max(abs(x) for x in vector)
    return [x / max_val for x in vector]

def aitken_acceleration(x_k, x_k1, x_k2):
    #Apply Aitken’s Delta-squared acceleration to approximate a better eigenvector.
    
    delta_x = [x_k1[i] - x_k[i] for i in range(len(x_k))]
    delta_x1 = [x_k2[i] - 2 * x_k1[i] + x_k[i] for i in range(len(x_k))]

    accelerated = [x_k[i] - (delta_x[i] ** 2) / delta_x1[i] if delta_x1[i] != 0 else x_k[i] for i in range(len(x_k))]
    return normalize(accelerated)

# Power Iteration with Aitken's Acceleration
for iteration in range(max_iterations):
    # Perform three consecutive Power Iteration steps
    x_k = normalize(eigenvector)
    x_k1 = normalize(matrix_vector_multiply(matrix, x_k))
    x_k2 = normalize(matrix_vector_multiply(matrix, x_k1))
    
    accelerated_vector = aitken_acceleration(x_k, x_k1, x_k2)
    
    # Check for convergence
    diff = sum(abs(accelerated_vector[i] - x_k[i]) for i in range(n))
    if diff < tolerance:
        break
    
    eigenvector = accelerated_vector

# Compute the largest eigenvalue using the Rayleigh quotient
numerator = sum(
    accelerated_vector[i] * matrix_vector_multiply(matrix, next_eigenvector)[i]
    for i in range(n)
)
denominator = sum(x ** 2 for x in accelerated_vector)
largest_eigenvalue = numerator / denominator

# Print the largest eigenvalue and the corresponding eigenvector
print("Largest Eigenvalue (with Aitken's acceleration):", largest_eigenvalue)
print("Corresponding Eigenvector (with Aitken's acceleration):", accelerated_vector)




def get_minor(matrix, i, j):
    minor = np.delete(np.delete(matrix, i, 0), j, 1)
    return minor

def determinant(matrix):
    # Base case for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]
    
    # Base case for 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Recursive case for NxN matrix
    det = 0
    for col in range(len(matrix)):
        minor = get_minor(matrix, 0, col)
        det += (-1)**col * matrix[0][col] * determinant(minor)
    return det

def eigenvalues(matrix,X):
    eigenvalues = []
    for x in X:
        new_matrix = -np.array(matrix)
        for i in range(n):
            new_matrix[i][i] += x
        if abs(determinant(new_matrix)) < 1e-6:
            eigenvalues.append(x)
        
    return eigenvalues

X=np.linspace(0, 10, 11)
print(eigenvalues(matrix,X))

