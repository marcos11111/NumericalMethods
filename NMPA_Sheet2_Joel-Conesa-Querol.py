# Define the function to multiply matrix and vector without np.dot
def matrix_vector_multiply(matrix, vector):
    result = []
    for row in matrix:
        row_result = sum(row[i] * vector[i] for i in range(len(vector)))
        result.append(row_result)
    print("Resulting vector:", result)
    return result

# Define the matrix and vector as given in the test case
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

vector = [10, 11, 12]

# Use the function to calculate the result
result = matrix_vector_multiply(matrix, vector)


def back_substitution(M, b):
    n = len(b)
    x = [0] * n  # Initialize the solution vector with zeros

    # Start from the last row and move upwards
    for i in range(n - 1, -1, -1):
        if M[i][i] == 0:
            raise ValueError("Matrix is singular or system has no unique solution.")
        
        # Back-substitution calculation
        x[i] = (b[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]

    return x

# Test case
M = [
    [1, 2, 3],
    [0, 5, 6],
    [0, 0, 9]]

b = [10, 11, 12]

# Solve the system
solution = back_substitution(M, b)

# Print the solution
print("Solution vector:", solution)

# Validate the solution by calculating M * x and comparing with b
def validate_solution(M, x, b):
    calculated_b = [sum(M[i][j] * x[j] for j in range(len(x))) for i in range(len(b))]
    is_valid = all(abs(calculated_b[i] - b[i]) < 1e-6 for i in range(len(b)))
    print("Validation passed." if is_valid else "Validation failed.")
    return is_valid

# Validate the solution
validate_solution(M, solution, b)
