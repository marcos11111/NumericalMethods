import numpy as np


max_iterations = 1000
tolerance = 1e-6

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


def poly_char(matrix, x):
    new_matrix = - np.array(matrix)
    for i in range(len(matrix)):
        new_matrix[i][i] += x
    return determinant(new_matrix)


def eigenvalues(matrix, X):
    eigenvalues = []
    
    for i in range(len(X)-1):
        a = X[i]; b = X[i+1]
        
        if poly_char(matrix, a) * poly_char(matrix, b) >= 0:
            continue
        else:
            for _ in range(max_iterations):
                # Compute the midpoint
                mid = (a + b) / 2
                f_mid = poly_char(matrix, mid)
                
                # Check for convergence
                if abs(f_mid) < tolerance or abs(b - a) < tolerance:
                    eigenvalues.append(mid)
                    break
                
                # Narrow the interval
                if poly_char(matrix, a) * f_mid < 0:
                    b = mid
                else:
                    a = mid
        
    return eigenvalues


def schroedinger_solver(potential, dx, hbar=1, mass=1):
    
    # Number of points
    n = len(potential)
    
    # Define the kinetic energy matrix (using finite differences)
    kinetic = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            kinetic[i, i - 1] = -1
        kinetic[i, i] = 2
        if i < n - 1:
            kinetic[i, i + 1] = -1
    kinetic *= -(hbar**2) / (2 * mass * dx**2)
    
    # Define the potential energy matrix (diagonal)
    potential_matrix = np.diag(potential)
    
    # The Hamiltonian is the sum of the kinetic and potential matrices
    hamiltonian = kinetic + potential_matrix
    
    return hamiltonian

# Parameters
dx = 0.1  # Spatial step size
x = np.arange(-5, -4.5 + dx, dx)  # Grid of spatial points
potential = [0.5 * (xi**2) for xi in x]  # Harmonic oscillator potential


hamiltonian = schroedinger_solver(potential, dx, hbar=1, mass=1)
X_Ss = np.linspace(-200, 200, 1000)

# Solve the Schrödinger equation

lambdas = eigenvalues(hamiltonian, X_Ss)

# Print the first few energy eigenvalues
print("Energy Eigenvalues:", lambdas)

