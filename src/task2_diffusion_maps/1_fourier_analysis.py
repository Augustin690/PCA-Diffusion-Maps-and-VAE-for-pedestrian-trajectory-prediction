import numpy as np
import matplotlib.pyplot as plt
import utils

""" Task2.1: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Create a periodic dataset with the details described in the task-sheet
N = 1000
t = 2 * np.pi * np.arange(1, N+1) / (N+1)
X = np.vstack((np.cos(t), np.sin(t))).T

# TODO: Visualize data-set
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data-set')
plt.axis('equal')
plt.show()
# TODO: Plot 5 eigenfunctions associated to the largest eigenvalues using the function diffusion_map() implemented in utils.py
lambdas, phi = utils.diffusion_map(X)
# TODO: Plot 5 eigenfunctions
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.plot(t, phi[:, i], label=f'phi_{i}')
plt.xlabel('t')
plt.ylabel('Phi')
plt.legend()
plt.title('Eigenfunctions')
plt.show()


