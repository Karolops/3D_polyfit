import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Your Z matrix
Z = np.array([
    [-0., -5.12, -16., -24.32, -28.64, -27.68, -19.92, -14.16, -5.76, 1.28],
    [0.72, -5.12, -15.6, -23.36, -27.76, -27.2, -22.32, -12.96, -6.64, -0.],
    [-1.36, -5.92, -15.6, -21.28, -27.04, -27.52, -21.04, -12.16, -4.08, 1.84],
    [-7.6, -16.08, -24.48, -29.36, -29.44, -27.2, -22.8, -12.88, -8.32, -0.32],
    [-1.6, -1.28, -6.88, -16.4, -24., -29.28, -29.04, -23.12, -14.96, -7.6]
])

# Provided X and Y matrices
X_vals = np.array([0., 1261., 2521., 3782., 5042., 6303., 7563., 8824., 10084., 11345.])
Y_vals = np.array([0., 808., 1615., 2422., 3230.])

# Generate meshgrid for X and Y
X, Y = np.meshgrid(X_vals, Y_vals)

# Flatten arrays for curve fitting
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Define the 4th order polynomial function
def polynomial_surface_4th_order(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
    x, y = xy
    return (a * x**4 + b * y**4 + c * x**3 * y + d * x * y**3 + e * x**3 + f * y**3 + 
            g * x**2 * y + h * x * y**2 + i * x**2 + j * y**2 + k * x * y + l * x + 
            m * y + n)


# Fit the polynomial
params, _ = curve_fit(polynomial_surface_4th_order, (X_flat, Y_flat), Z_flat)

# Print the polynomial with 15 decimal places
polynomial_str = (f"Z = {params[0]:.15f}*x^4 + {params[1]:.15f}*y^4 + {params[2]:.15f}*x^3*y + "
                  f"{params[3]:.15f}*x*y^3 + {params[4]:.15f}*x^3 + {params[5]:.15f}*y^3 + {params[6]:.15f}*x^2*y + "
                  f"{params[7]:.15f}*x*y^2 + {params[8]:.15f}*x^2 + {params[9]:.15f}*y^2 + {params[10]:.15f}*x*y + "
                  f"{params[11]:.15f}*x + {params[12]:.15f}*y + {params[13]:.15f}")

polynomial_str = (f"Z = {params[0]:.15f}*x**4 + {params[1]:.15f}*y**4 + {params[2]:.15f}*x**3*y + "
                  f"{params[3]:.15f}*x*y**3 + {params[4]:.15f}*x**3 + {params[5]:.15f}*y**3 + {params[6]:.15f}*x**2*y + "
                  f"{params[7]:.15f}*x*y**2 + {params[8]:.15f}*x**2 + {params[9]:.15f}*y**2 + {params[10]:.15f}*x*y + "
                  f"{params[11]:.15f}*x + {params[12]:.15f}*y + {params[13]:.15f}")
print("Fitted polynomial:")
print(polynomial_str)

# Generate Z values from the fitted model
Z_fit = polynomial_surface_4th_order((X, Y), *params)

# Calculate error for each point
error_matrix = np.abs(Z_fit - Z)

# Print the error for each point
print("\nError for each point:")
print(error_matrix)

# Calculate least squares error
LSE = np.sum((Z_fit - Z)**2)
print(f"Least Squares Error: {LSE:.15f}")

# Plotting
fig = plt.figure(figsize=(12, 6))

# Original data and surface fit
ax1 = fig.add_subplot(121, projection='3d')
sc = ax1.scatter(X_flat, Y_flat, Z_flat, c=error_matrix.flatten(), cmap='viridis', s=50)
ax1.plot_surface(X, Y, Z_fit, color='b', alpha=0.3)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('4th Order Polynomial Surface Fit with Error Coloring')
cbar = fig.colorbar(sc, ax=ax1)
cbar.set_label('Error Magnitude')

# Residuals
ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(X, Y, error_matrix, cmap='coolwarm')
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Error')
ax2.set_title('Polyfit Error')

plt.tight_layout()
plt.show()