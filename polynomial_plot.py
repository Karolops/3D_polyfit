import numpy as np
import matplotlib.pyplot as plt

def z_func(x, y):
    return (
        -0.000000000000019*x**4 + 
        0.000000000001721*y**4 + 
        0.000000000000023*x**3*y + 
        -0.000000000000536*x*y**3 + 
        0.000000000409352*x**3 + 
        -0.000000007085036*y**3 + 
        -0.000000000454927*x**2*y + 
        0.000000002173308*x*y**2 + 
        -0.000001651419093*x**2 + 
        0.000004809570246*y**2 + 
        0.000000508109387*x*y + 
        -0.005579597965373*x + 
        0.000197600352962*y + 
        1.860460148953467
    )

# Create a grid of x and y values
x = np.linspace(0, 11345, 400)  # Increase number for better resolution
y = np.linspace(0, 3230, 400)
x, y = np.meshgrid(x, y)
z = z_func(x, y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface plot of the function')

plt.show()