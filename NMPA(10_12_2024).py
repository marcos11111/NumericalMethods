# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:16:15 2024

@author: Joel
"""
'''
import numpy as np

def fill_array_with_function(array, interval, num_points, func):
    # Validate that the array has the correct size
    if len(array) != num_points:
        raise ValueError("The array size must match the number of data points.")
    
    start, end = interval
    x_values = np.linspace(start, end, num_points)
    
    # Fill the array with evaluated values
    for i, x in enumerate(x_values):
        array[i] = func(x)
    
    return array

def newton_cotes_2(a, b, num_points, array):
    if num_points % 2 == 0:
        raise ValueError("The number of points must be odd.")
    
    # Step size
    h = (b - a) / (num_points - 1)
    
    # Generate x values
    x = np.linspace(a, b, num_points)
    
    # Evaluate the function at these points
    interval = [a,b]
    y = array
    
    # Newton-Cotes weights for a 2nd-order polynomial
    weights = np.array([1, 4, 1]) * (h / 3)
    
    # Apply the weights in segments (Trapezoidal or Simpson-like)
    integral = 0
    for i in range(0, num_points - 1, 2):
        integral += np.dot(weights, y[i:i+3])
    
    # Error estimate (using the fourth derivative at the midpoint)
    f4_approx = (y[2] - 4 * y[1] + 6 * y[0] - 4 * y[-2] + y[-1]) / h**4
    error = (1 / 90) * (h**4) * f4_approx
    
    return integral, error

n = 5
array0 = np.zeros(n)
interval = [0,1]
def f(x):
    return x + x**3

array = fill_array_with_function(array0, interval, n, f)

print(newton_cotes_2(interval[0], interval[1], n, array))



import matplotlib.pyplot as plt

# Define the constants
L = 2
omega_1 = 3
omega_2 = 4.5
delta_omega = omega_2 - omega_1

# Define the wavefunction |Ψ(x, t)|
def wavefunction(x, t):
    if 0 < x < L:
        return (1 / np.sqrt(L)) * (np.sin(np.pi * x / L) * np.exp(-1j * omega_1 * t) +
                                   np.sin(2 * np.pi * x / L) * np.exp(-1j * omega_2 * t))
    else:
        return 0

# Probability distribution P(x, t) = |Ψ(x, t)|^2
def probability_density(x, t):
    psi = wavefunction(x, t)
    return np.abs(psi)**2



# Function to compute the probability P(t | x ∈ [3L/4, L])
def probability_in_right_quarter(t, num_points):
    a = 3 * L / 4  # Start of the interval
    b = L  # End of the interval
    def P(x):
        return probability_density(x,t)
    array_b = fill_array_with_function(np.zeros(num_points), [a, b], num_points, P)
    integral, error = newton_cotes_2(a, b, num_points, array_b)
    return integral, error

# Run the simulation for t = 0 and t = π / Δω
times = [0, np.pi / delta_omega]
num_points_range = np.arange(5, 502, 2)

# Initialize storage for results
results = []

for t in times:
    errors = []
    log_h = []
    for n in num_points_range:
        h = (L / 4) / (n - 1)
        log_h.append(np.log(h))
        # Compute probability using the given n
        P = probability_in_right_quarter(t, n)[0]
        errors.append(probability_in_right_quarter(t, n)[1])  # Placeholder for exact result computation
    results.append((log_h, np.log(errors)))

# Plotting log(E) vs log(h) for both times
for i, t in enumerate(times):
    log_h, log_E = results[i]
    plt.plot(log_h, log_E, label=f"t = {t:.2f}")
    plt.xlabel("log(h)")
    plt.ylabel("log(E)")
    plt.legend()
plt.title("log(E) vs log(h)")
plt.show()


'''




def simpsons_3_8_rule(func, a, b, num_points):
    if (num_points - 1) % 3 != 0:
        raise ValueError("Number of intervals must be a multiple of 3 for Simpson's 3/8 Rule.")
    
    h = (b - a) / (num_points - 1)  # Step size
    x = np.linspace(a, b, num_points)  # Subdivide the interval
    y = func(x)  # Evaluate the function at the points
    
    integral = 0
    for i in range(0, num_points - 1, 3):
        integral += (3 * h / 8) * (y[i] + 3 * y[i+1] + 3 * y[i+2] + y[i+3])
    
    return integral



def gauss_legendre_quadrature(func, a, b, order=4):
    # Weights and nodes for Gauss-Legendre quadrature (order 4)
    if order == 4:
        nodes = np.array([-0.8611363115940526, -0.3399810435848563, 
                           0.3399810435848563,  0.8611363115940526])
        weights = np.array([0.34785484513745385, 0.6521451548625461, 
                            0.6521451548625461,  0.34785484513745385])
    else:
        raise ValueError("Currently only order 4 is supported.")
    
    # Map nodes from [-1, 1] to [a, b]
    mapped_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    mapped_weights = 0.5 * (b - a) * weights
    
    # Evaluate the integral
    integral = np.sum(mapped_weights * func(mapped_nodes))
    
    return integral


from scipy.integrate import quad

# Define the functions
def f1(x):
    return np.exp(x) * np.cos(x)

def f2(x):
    return np.exp(x)

def f3(x):
    return np.piecewise(x, [x < 0, x >= 0], [lambda x: np.exp(2*x), lambda x: x - 2*np.cos(x) + 4])

# Define intervals
intervals = {
    "f1": (0, np.pi / 2),
    "f2": (-1, 3),
    "f3": (-1, 1)
}

# Exact integrals using scipy's quad
exact_integrals = {
    "f1": quad(f1, *intervals["f1"])[0],
    "f2": quad(f2, *intervals["f2"])[0],
    "f3": quad(f3, *intervals["f3"])[0]
}


# Compute errors for each method and function
methods = {
    "Simpson's 3/8": simpsons_3_8_rule,
    "Gauss-Legendre": gauss_legendre_quadrature
}
results = {}

for func_name, func in [("f1", f1), ("f2", f2), ("f3", f3)]:
    a, b = intervals[func_name]
    exact = exact_integrals[func_name]
    results[func_name] = {}
    for method_name, method in methods.items():
        errors = []
        log_h = []
        for n in range(5, 502, 2):
            h = (b - a) / (n - 1)
            log_h.append(np.log(h))
            try:
                if method_name == "Simpson's 3/8":
                    integral = method(func, a, b, n)
                else:  # Gauss-Legendre
                    integral = method(func, a, b)
                error = np.abs(integral - exact)
                errors.append(error)
            except ValueError:
                continue
        results[func_name][method_name] = (log_h[:len(errors)], np.log(errors))

# Plotting results
for func_name in results:
    plt.figure()
    for method_name, (log_h, log_E) in results[func_name].items():
        plt.plot(log_h, log_E, label=method_name)
    plt.title(f"log(E) vs log(h) for {func_name}")
    plt.xlabel("log(h)")
    plt.ylabel("log(E)")
    plt.legend()
    plt.show()
