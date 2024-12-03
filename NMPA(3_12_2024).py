import numpy as np


#Pronlem 2

def f(x):
    return 2**x * np.sin(x)/x

def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1  # Ensure n is even
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    fx = f(x)
    result = fx[0] + fx[-1] + 4 * sum(fx[1:n:2]) + 2 * sum(fx[2:n-1:2])
    result *= h / 3
    return result

def romberg(f, a, b, levels):
    R = np.zeros((levels, levels))
    for k in range(levels):
        n = 2**k
        R[k, 0] = simpson(f, a, b, n)
        for j in range(1, k+1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
    return R[levels-1, levels-1]

def gauss_legendre_4(f, a, b, n):
    h = (b - a) / n  # Step size for subintervals
    integral = 0.0
    for i in range(n):
        # Map the nodes to the subinterval [x_i, x_{i+1}]
        x_i = a + i * h
        x_ip1 = x_i + h
        mid = (x_i + x_ip1) / 2
        half_width = (x_ip1 - x_i) / 2
        # Evaluate the function at transformed nodes
        integral += sum(gauss_weights * f(mid + half_width * gauss_nodes)) * half_width
    return integral


# Parameters
n_simpson = 100  # Number of intervals for Simpson's Rule
levels_romberg = 4  # Levels for Romberg integration
gauss_nodes = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
gauss_weights = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
subintervals = 4

a = 1-14
b = np.pi/2

# Calculations
simpson_result = simpson(f, a, b, n_simpson)
romberg_result = romberg(f, a, b, levels_romberg)
gauss_result = gauss_legendre_4(f, a, b, subintervals)


print(simpson_result, romberg_result, gauss_result)



#Problem 3

def fx_a(x):
    return x

def fy_a(y):
    return gauss_legendre_4(fx_a, 0, 2, subintervals) * y**2

result_a = gauss_legendre_4(fy_a, 0, 1, subintervals)
print(result_a)


def fx_b(x):
    return x

def fy_b(y):
    return gauss_legendre_4(fx_b, 2 * y, 2, subintervals) * y**2

result_b = gauss_legendre_4(fy_b, 0, 1, subintervals)
print(result_b)


def fy_c(y):
    return y**2

def fx_c(x):
    return gauss_legendre_4(fy_c, 0, x/2, subintervals) * x

result_c = gauss_legendre_4(fx_c, 0, 2, subintervals)
print(result_c)
