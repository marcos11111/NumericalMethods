import matplotlib.pyplot as plt
import numpy as np


'''
# Exercise 1
# Define the function f(x)
def f(x):
    return np.exp(np.sqrt(5) * x) - 13.5 * np.cos(0.1 * x) + 25 * x**4

    # Set the initial interval where a root is suspected
a, b = -1, 0

# Set the tolerance for the root approximation and a maximum number of iterations
tolerance = 1e-6
max_iterations = 100

# Perform linear interpolation to find the root
for i in range(max_iterations):
    # Calculate the interpolated point
    x_new = a - f(a) * (b - a) / (f(b) - f(a))
    
    # Check if x_new is a root within tolerance
    if abs(f(x_new)) < tolerance:
        print(f"Root found: x = {x_new}, f(x) = {f(x_new)}")
        break
    
    # Update the interval based on the sign of f(x_new)
    if f(a) * f(x_new) < 0:
        b = x_new
    else:
        a = x_new
else:
    # If no convergence after max iterations
    print("Root not found within the maximum number of iterations.")
    

# Define the derivative f'(x)
def f_prime(x):
    return np.sqrt(5) * np.exp(np.sqrt(5) * x) + 1.35 * np.sin(0.1 * x) + 100 * x**3

# Initial guess
x0 = -0.5

# Perform Newton's method to find the root
for i in range(max_iterations):
    x_new = x0 - f(x0) / f_prime(x0)
    
    # Check if x_new is a root within tolerance
    if abs(f(x_new)) < tolerance:
        print(f"Root found: x = {x_new}, f(x) = {f(x_new)}")
        break
    
    x0 = x_new
    
else:
    print("Root not found within the maximum number of iterations.")



# Linear interpolation method
def linear_interpolation(a, b, tolerance):
    errors = []
    iterations = 0
    while iterations < max_iterations:
        x_new = a - f(a) * (b - a) / (f(b) - f(a))
        errors.append(abs(f(x_new)))
        
        if abs(f(x_new)) < tolerance:
            return x_new, iterations + 1, errors
        
        if f(a) * f(x_new) < 0:
            b = x_new
        else:
            a = x_new
        
        iterations += 1
    return None, iterations, errors

# Newton's method
def newtons_method(x0, tolerance):
    errors = []
    iterations = 0
    while iterations < max_iterations:
        x_new = x0 - f(x0) / f_prime(x0)
        errors.append(abs(f(x_new)))
        
        if abs(f(x_new)) < tolerance:
            return x_new, iterations + 1, errors
        
        x0 = x_new
        iterations += 1
    return None, iterations, errors

# Set the parameters
tolerance = 1e-14
max_iterations = 100

# Linear Interpolation
a, b = -1, 0
lin_root, lin_iter, lin_errors = linear_interpolation(a, b, tolerance)

# Newton's Method
newton_root, newton_iter, newton_errors = newtons_method(-0.5, tolerance)

# Plotting the errors for both methods
plt.figure(figsize=(12, 6))
plt.semilogy(lin_errors, label='Linear Interpolation', marker='o')
plt.semilogy(newton_errors, label='Newton\'s Method', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Convergence Comparison: Linear Interpolation vs Newton\'s Method')
plt.legend()
plt.grid()
plt.show()

print(lin_root, lin_iter, newton_root, newton_iter)



# Set the parameters
tolerances = [1e-6, 1e-12, 1e-15, 1e-20]  # Varying tolerances
initial_guesses = [-0.5, -0.75, -1]  # Varying initial guesses
max_iterations = 10000

# Store results
results = {}

# Loop over tolerances and initial guesses
for tol in tolerances:
    for guess in initial_guesses:
        # Linear Interpolation
        a, b = -1, 0
        lin_root, lin_iter, lin_errors = linear_interpolation(a, b, tol)

        # Newton's Method
        newton_root, newton_iter, newton_errors = newtons_method(guess, tol)

        # Store results
        results[(tol, guess)] = {
            'Linear': (lin_root, lin_iter, lin_errors),
            'Newton': (newton_root, newton_iter, newton_errors)
        }

# Display results
for (tol, guess), res in results.items():
    print(f"Tolerance: {tol}, Initial Guess: {guess}")
    for method, (root, iter_count, _) in res.items():
        print(f"  {method}: Root = {root}, Iterations = {iter_count}")
'''






'''
# Exercise 2
# Define the function and its derivative
def f(lambda_val):
    return np.tan(lambda_val) - lambda_val

def f_prime(lambda_val):
    return np.cos(lambda_val)**(-2) - 1

# Implementing Newton's method
def newton_method_2(initial_guess, tolerance=1e-6, max_iterations=100):
    lambda_val = initial_guess
    for _ in range(max_iterations):
        lambda_new = lambda_val - f(lambda_val) / f_prime(lambda_val)
        
        # Check for convergence
        if abs(lambda_new - lambda_val) < tolerance:
            return lambda_new
        
        lambda_val = lambda_new
    return None  # No convergence

initial_guesses = [4.5, 7.7, 10.9, 14]
eigenvalues = []

for i in initial_guesses:
    root = newton_method_2(i)
    if root is not None:
        eigenvalues.append(root)

# Display the found eigenvalues
print(eigenvalues)
'''






'''
# Exercise 3
# Define the functions
def f1(x, y):
    return x * y - 0.1

def f2(x, y):
    return x**2 + 3 * y**2 - 2

# Define the Jacobian matrix
def jacobian(x, y):
    return np.array([[y, x], [2 * x, 6 * y]])

# Newton's method for systems of equations
def newton_system(initial_guess, tol=1e-6, max_iter=100):
    x, y = initial_guess
    for _ in range(max_iter):
        # Evaluate functions and Jacobian
        F = np.array([f1(x, y), f2(x, y)])
        J = jacobian(x, y)
        
        # Check if Jacobian is invertible (determinant should not be too small)
        det = np.linalg.det(J)
        if abs(det) < 1e-10:
            print("Jacobian is nearly singular.")
            return None

        # Compute Newton update
        delta = np.linalg.solve(J, -F)
        x, y = x + delta[0], y + delta[1]
        
        # Check for convergence
        if np.linalg.norm(delta) < tol:
            return (x, y)
        
        # Check for NaN or divergence
        if np.isnan(x) or np.isnan(y) or np.linalg.norm([x, y]) > 1e6:
            print("Divergence or NaN encountered.")
            return None
    print("Maximum iterations reached without convergence.")
    return None

# Initial guesses to find multiple solutions
initial_guesses = [(-1, -0.05), (1, 0.05), (0.1, 0.8), (-0.1, -0.8)]
solutions = []

# Find all solutions
for guess in initial_guesses:
    sol = newton_system(guess)
    if sol is not None and sol not in solutions:
        solutions.append(sol)

# Display the solutions
for i, sol in enumerate(solutions, 1):
    print(f"Solution {i}: x = {sol[0]:.6f}, y = {sol[1]:.6f}")



# Define the functions for the improved x = g(x) method
def g1(y):
    if y == 0:
        return None  # Prevent division by zero
    return 0.1 / y

def g2(x, sign=1):
    value = 2 - x**2
    if value < 0:
        return None  # Square root of negative is not real
    return sign * np.sqrt(value / 3)

# Fixed-point iteration method
def fixed_point_method(initial_guess, tol=1e-6, max_iter=100):
    x, y = initial_guess
    for _ in range(max_iter):
        x_new = g1(y)
        y_new = g2(x, sign=np.sign(y))  # Use sign of initial y to select root
        
        # Check for NaNs or divergence
        if x_new is None or y_new is None or np.isnan(x_new) or np.isnan(y_new):
            print("Non-convergent or NaN encountered.")
            return None

        # Check for convergence
        if abs(x_new - x) < tol and abs(y_new - y) < tol:
            return (x_new, y_new)
        
        x, y = x_new, y_new

    print("Maximum iterations reached without convergence.")
    return None

# Initial guesses to find multiple solutions
initial_guesses = [(-1, -0.05), (1, 0.05), (0.1, 0.8), (-0.1, -0.8)]
solutions_fixed_point = []

# Find all solutions using fixed-point iteration
for guess in initial_guesses:
    sol = fixed_point_method(guess)
    if sol is not None and sol not in solutions_fixed_point:
        solutions_fixed_point.append(sol)

# Display the solutions
for i, sol in enumerate(solutions_fixed_point, 1):
    print(f"Solution {i} (Fixed-Point): x = {sol[0]:.6f}, y = {sol[1]:.6f}")

# Test with (x, y) = (2, 1)
test_guess = (2, 1)
sol_test = fixed_point_method(test_guess)
print(f"Result for initial guess (2, 1): {sol_test}")
'''






'''
# Exercise 4
# Function to return f(z) and f'(z) for f(z) = z^3 - 1
def get_fdf(z):
    fz = z**3 - 1
    dfz = 3 * z**2
    return fz, dfz

# Newton-Raphson solver for complex numbers
def solve_cnewton(initial_guess, tol=1e-6, max_iter=100):
    z = initial_guess
    for i in range(max_iter):
        fz, dfz = get_fdf(z)  # Get f(z) and f'(z)
        
        if dfz == 0:
            # If the derivative is zero, return None (no convergence)
            return None, i, np.abs(fz)
        
        # Newton-Raphson update for complex numbers
        z_new = z - fz / dfz
        
        # Check if the update is within tolerance
        if np.abs(z_new - z) < tol:
            # Converged: return the root, number of iterations, and quality of solution
            return z_new, i + 1, np.abs(fz)
        
        z = z_new
    
    # If we reach the max number of iterations, return None to indicate failure to converge
    return None, max_iter, np.abs(fz)



# Example usage
initial_guesses = [2 - 0.5j, -1 + 0.5j, -1 - 0.5j]  # Example starting point

for guess in initial_guesses:
    root, iterations, quality = solve_cnewton(guess)
    print(f"Root: {root}")
    print(f"Iterations: {iterations}")
    print(f"Quality of solution (|f(z)|): {quality}")



# Function to create the convergence map
def create_convergence_map(N=400, tol=1e-9, max_iter=200):
    # Define grid boundaries
    x_min, x_max, y_min, y_max = -2, 2, -2, 2
    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    with open("convergence_data.txt", "w") as file:
        file.write("x0, y0, k(z), f(z), log10(n_itns)\n")

        # Arrays to store the results
        roots = np.zeros(Z0.shape, dtype=complex)
        iterations = np.zeros(Z0.shape, dtype=int)
    
        # Iterate over each point in the grid
        for i in range(N):
            for j in range(N):
                z0 = Z0[i, j]
                root, iter_count, quality = solve_cnewton(z0, tol=tol, max_iter=max_iter)
                if root is not None:
                    roots[i, j] = root
                    iterations[i, j] = iter_count
                else:
                    iterations[i, j] = max_iter  # Mark as non-convergent
        
            # Write the data in the specified format
            fz_at_root, _ = get_fdf(root)
            log_iter_count = np.log10(iter_count) if iter_count > 0 else 0
            file.write(f"{z0.real}, {z0.imag}, {root}, {fz_at_root}, {log_iter_count}\n")

    # Plotting the convergence map based on number of iterations
    plt.figure(figsize=(10, 10))
    plt.imshow(iterations, extent=(x_min, x_max, y_min, y_max), cmap="hot")
    plt.colorbar(label="Number of Iterations to Converge")
    plt.title("Convergence Map of Newton-Raphson Method")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

    print("Data saved to convergence_data.txt")

# Run the program with specified parameters
create_convergence_map()



# Function to create and plot the color-scaled map
def plot_imaginary_part_map(N=400, tol=1e-9, max_iter=200, zoom_area=None):
    # Define grid boundaries (zoom in if specified)
    if zoom_area:
        x_min, x_max, y_min, y_max = zoom_area
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2

    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    # Array to store the imaginary parts of the roots
    imag_part_roots = np.zeros(Z0.shape, dtype=float)

    # Iterate over each point in the grid
    for i in range(N):
        for j in range(N):
            z0 = Z0[i, j]
            root, iter_count, quality = solve_cnewton(z0, tol=tol, max_iter=max_iter)
            imag_part_roots[i, j] = root.imag  # Store the imaginary part of the root

    # Plotting the imaginary part with color scaling
    plt.figure(figsize=(8, 8))
    plt.imshow(imag_part_roots, extent=(x_min, x_max, y_min, y_max), cmap="viridis", origin="lower")
    plt.colorbar(label="Imaginary part of k(z)")
    plt.title("Imaginary Part of Roots in the Complex Plane")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

# Run the program with specified parameters
plot_imaginary_part_map()

# Optionally, to explore an interesting region, you could use:
plot_imaginary_part_map(zoom_area=(-0.5, 0.5, -0.5, 0.5))



def plot_log_iterations_map(N=400, tol=1e-9, max_iter=200, zoom_area=None):
    # Define grid boundaries (zoom in if specified)
    if zoom_area:
        x_min, x_max, y_min, y_max = zoom_area
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2

    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    # Array to store log10 of iteration counts
    log_iterations = np.zeros(Z0.shape, dtype=float)

    # Iterate over each point in the grid
    for i in range(N):
        for j in range(N):
            z0 = Z0[i, j]
            root, iter_count, quality = solve_cnewton(z0, tol=tol, max_iter=max_iter)
            log_iterations[i, j] = np.log10(iter_count) if iter_count > 0 else 0  # Avoid log(0)

    # Plotting the log10(iterations) with color scaling
    plt.figure(figsize=(8, 8))
    plt.imshow(log_iterations, extent=(x_min, x_max, y_min, y_max), cmap="plasma", origin="lower")
    plt.colorbar(label="log10(n_itns)")
    plt.title("Log10 of Iteration Count in the Complex Plane")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

# Run the program with specified parameters
plot_log_iterations_map()

# Optionally, to explore an interesting region, you could use:
plot_log_iterations_map(zoom_area=(-0.5, 0.5, -0.5, 0.5))



def get_fdf_2(z):
    fz = 35 * z**9 - 180 * z**7 + 378 * z**5 - 420 * z**3 + 315 * z
    dfz = 315 * z**8 - 1260 * z**6 + 1890 * z**4 - 1260 * z**2 +315
    return fz, dfz

def solve_cnewton_2(initial_guess, tol=1e-6, max_iter=100):
    z = initial_guess
    for i in range(max_iter):
        fz, dfz = get_fdf_2(z)  # Get f(z) and f'(z)
        
        if dfz == 0:
            # If the derivative is zero, return None (no convergence)
            return None, i, np.abs(fz)
        
        # Newton-Raphson update for complex numbers
        z_new = z - fz / dfz
        
        # Check if the update is within tolerance
        if np.abs(z_new - z) < tol:
            # Converged: return the root, number of iterations, and quality of solution
            return z_new, i + 1, np.abs(fz)
        
        z = z_new
    
    # If we reach the max number of iterations, return None to indicate failure to converge
    return None, max_iter, np.abs(fz)

initial_guesses = [0.1 + 0.1j, 0.9 + 0.6j, 0.9 - 0.6j, -0.9 + 0.6j, -0.9 - 0.6j, 1.5 + 0.3j, 1.5 - 0.3j, -1.5 + 0.3j, -1.5 - 0.3j]  # Example starting point

for guess in initial_guesses:
    root, iterations, quality = solve_cnewton_2(guess)
    print(f"Root: {root}")
    print(f"Iterations: {iterations}")
    print(f"Quality of solution (|f(z)|): {quality}")


# Function to create the convergence map
def create_convergence_map(N=400, tol=1e-9, max_iter=200):
    # Define grid boundaries
    x_min, x_max, y_min, y_max = -2, 2, -2, 2
    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    with open("convergence_data_2.txt", "w") as file:
        file.write("x0, y0, k(z), f(z), log10(n_itns)\n")

        # Arrays to store the results
        roots = np.zeros(Z0.shape, dtype=complex)
        iterations = np.zeros(Z0.shape, dtype=int)
    
        # Iterate over each point in the grid
        for i in range(N):
            for j in range(N):
                z0 = Z0[i, j]
                root, iter_count, quality = solve_cnewton_2(z0, tol=tol, max_iter=max_iter)
                if root is not None:
                    roots[i, j] = root
                    iterations[i, j] = iter_count
                else:
                    iterations[i, j] = max_iter  # Mark as non-convergent
        
            # Write the data in the specified format
            fz_at_root, _ = get_fdf_2(root)
            log_iter_count = np.log10(iter_count) if iter_count > 0 else 0
            file.write(f"{z0.real}, {z0.imag}, {root}, {fz_at_root}, {log_iter_count}\n")

    # Plotting the convergence map based on number of iterations
    plt.figure(figsize=(10, 10))
    plt.imshow(iterations, extent=(x_min, x_max, y_min, y_max), cmap="hot")
    plt.colorbar(label="Number of Iterations to Converge")
    plt.title("Convergence Map of Newton-Raphson Method")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

    print("Data saved to convergence_data.txt")

# Run the program with specified parameters
create_convergence_map()



# Function to create and plot the color-scaled map
def plot_imaginary_part_map(N=400, tol=1e-9, max_iter=200, zoom_area=None):
    # Define grid boundaries (zoom in if specified)
    if zoom_area:
        x_min, x_max, y_min, y_max = zoom_area
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2

    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    # Array to store the imaginary parts of the roots
    imag_part_roots = np.zeros(Z0.shape, dtype=float)

    # Iterate over each point in the grid
    for i in range(N):
        for j in range(N):
            z0 = Z0[i, j]
            root, iter_count, quality = solve_cnewton_2(z0, tol=tol, max_iter=max_iter)
            imag_part_roots[i, j] = 0 if root == None else root.imag  # Store the imaginary part of the root

    # Plotting the imaginary part with color scaling
    plt.figure(figsize=(8, 8))
    plt.imshow(imag_part_roots, extent=(x_min, x_max, y_min, y_max), cmap="viridis", origin="lower")
    plt.colorbar(label="Imaginary part of k(z)")
    plt.title("Imaginary Part of Roots in the Complex Plane")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

# Run the program with specified parameters
plot_imaginary_part_map()



def plot_log_iterations_map(N=400, tol=1e-9, max_iter=200, zoom_area=None):
    # Define grid boundaries (zoom in if specified)
    if zoom_area:
        x_min, x_max, y_min, y_max = zoom_area
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2

    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    # Array to store log10 of iteration counts
    log_iterations = np.zeros(Z0.shape, dtype=float)

    # Iterate over each point in the grid
    for i in range(N):
        for j in range(N):
            z0 = Z0[i, j]
            root, iter_count, quality = solve_cnewton_2(z0, tol=tol, max_iter=max_iter)
            log_iterations[i, j] = np.log10(iter_count) if iter_count > 0 else 0  # Avoid log(0)

    # Plotting the log10(iterations) with color scaling
    plt.figure(figsize=(8, 8))
    plt.imshow(log_iterations, extent=(x_min, x_max, y_min, y_max), cmap="plasma", origin="lower")
    plt.colorbar(label="log10(n_itns)")
    plt.title("Log10 of Iteration Count in the Complex Plane")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

# Run the program with specified parameters
plot_log_iterations_map()



def get_fdf_3(z):
    fz = 0 if z == None else z**4 - 1
    dfz = 0 if z == None else 4 * z**3
    return fz, dfz

def solve_cnewton_3(initial_guess, tol=1e-6, max_iter=100):
    z = initial_guess
    for i in range(max_iter):
        fz, dfz = get_fdf_3(z)  # Get f(z) and f'(z)
        
        if dfz == 0:
            # If the derivative is zero, return None (no convergence)
            return None, i, np.abs(fz)
        
        # Newton-Raphson update for complex numbers
        z_new = z - fz / dfz
        
        # Check if the update is within tolerance
        if np.abs(z_new - z) < tol:
            # Converged: return the root, number of iterations, and quality of solution
            return z_new, i + 1, np.abs(fz)
        
        z = z_new
    
    # If we reach the max number of iterations, return None to indicate failure to converge
    return None, max_iter, np.abs(fz)

initial_guesses = [0.9 + 0.1j, -0.9 + 0.1j, 0.1 + 0.9j, 0.1 - 0.9j]  # Example starting point

for guess in initial_guesses:
    root, iterations, quality = solve_cnewton_3(guess)
    print(f"Root: {root}")
    print(f"Iterations: {iterations}")
    print(f"Quality of solution (|f(z)|): {quality}")


# Function to create the convergence map
def create_convergence_map(N=400, tol=1e-9, max_iter=200):
    # Define grid boundaries
    x_min, x_max, y_min, y_max = -2, 2, -2, 2
    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    with open("convergence_data_3.txt", "w") as file:
        file.write("x0, y0, k(z), f(z), log10(n_itns)\n")

        # Arrays to store the results
        roots = np.zeros(Z0.shape, dtype=complex)
        iterations = np.zeros(Z0.shape, dtype=int)
    
        # Iterate over each point in the grid
        for i in range(N):
            for j in range(N):
                z0 = Z0[i, j]
                root, iter_count, quality = solve_cnewton_3(z0, tol=tol, max_iter=max_iter)
                if root is not None:
                    roots[i, j] = root
                    iterations[i, j] = iter_count
                else:
                    iterations[i, j] = max_iter  # Mark as non-convergent
        
            # Write the data in the specified format
            fz_at_root, _ = get_fdf_3(root)
            log_iter_count = np.log10(iter_count) if iter_count > 0 else 0
            file.write(f"{z0.real}, {z0.imag}, {root}, {fz_at_root}, {log_iter_count}\n")

    # Plotting the convergence map based on number of iterations
    plt.figure(figsize=(10, 10))
    plt.imshow(iterations, extent=(x_min, x_max, y_min, y_max), cmap="hot")
    plt.colorbar(label="Number of Iterations to Converge")
    plt.title("Convergence Map of Newton-Raphson Method")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

    print("Data saved to convergence_data.txt")

# Run the program with specified parameters
create_convergence_map()



# Function to create and plot the color-scaled map
def plot_imaginary_part_map(N=400, tol=1e-9, max_iter=200, zoom_area=None):
    # Define grid boundaries (zoom in if specified)
    if zoom_area:
        x_min, x_max, y_min, y_max = zoom_area
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2

    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    # Array to store the imaginary parts of the roots
    imag_part_roots = np.zeros(Z0.shape, dtype=float)

    # Iterate over each point in the grid
    for i in range(N):
        for j in range(N):
            z0 = Z0[i, j]
            root, iter_count, quality = solve_cnewton_3(z0, tol=tol, max_iter=max_iter)
            imag_part_roots[i, j] = 0 if root == None else root.imag  # Store the imaginary part of the root

    # Plotting the imaginary part with color scaling
    plt.figure(figsize=(8, 8))
    plt.imshow(imag_part_roots, extent=(x_min, x_max, y_min, y_max), cmap="viridis", origin="lower")
    plt.colorbar(label="Imaginary part of k(z)")
    plt.title("Imaginary Part of Roots in the Complex Plane")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

# Run the program with specified parameters
plot_imaginary_part_map()



def plot_log_iterations_map(N=400, tol=1e-9, max_iter=200, zoom_area=None):
    # Define grid boundaries (zoom in if specified)
    if zoom_area:
        x_min, x_max, y_min, y_max = zoom_area
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2

    x_vals = np.linspace(x_min, x_max, N)
    y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z0 = X + 1j * Y  # Grid of complex initial guesses

    # Array to store log10 of iteration counts
    log_iterations = np.zeros(Z0.shape, dtype=float)

    # Iterate over each point in the grid
    for i in range(N):
        for j in range(N):
            z0 = Z0[i, j]
            root, iter_count, quality = solve_cnewton_3(z0, tol=tol, max_iter=max_iter)
            log_iterations[i, j] = np.log10(iter_count) if iter_count > 0 else 0  # Avoid log(0)

    # Plotting the log10(iterations) with color scaling
    plt.figure(figsize=(8, 8))
    plt.imshow(log_iterations, extent=(x_min, x_max, y_min, y_max), cmap="plasma", origin="lower")
    plt.colorbar(label="log10(n_itns)")
    plt.title("Log10 of Iteration Count in the Complex Plane")
    plt.xlabel("Re(z0)")
    plt.ylabel("Im(z0)")
    plt.show()

# Run the program with specified parameters
plot_log_iterations_map()
'''
