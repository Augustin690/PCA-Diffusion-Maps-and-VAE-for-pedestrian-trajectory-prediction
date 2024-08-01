import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import utils

""" Task2.2: In this script, we compute eigenfunctions of the Laplace Beltrami operator on the
“swiss roll” manifold.  We need functions defined in utils.py for this script.
"""

# TODO: Generate swiss roll dataset
N = 5000
X, _ = make_swiss_roll(n_samples=N, noise=0.0)
# TODO: Visualize data-set
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0])
ax.set_title("Swiss Roll dataset")
plt.show()
# TODO: Use function diffusion_map() defined in utils to compute first ten eigenfunctions (corresponding to 10 largest eigenvalues) of the Laplace Beltrami operator on the “swiss roll” manifold
lambdas, phi = utils.diffusion_map(X, n_eig_vals=10)
# TODO: Plot the first non-constant eigenfunction φ1 against the other eigenfunctions
phi_1 = phi[:, 1]
for i in range(2, 11):
    plt.figure()
    plt.scatter(phi_1, phi[:, i], s=2)
    plt.xlabel('φ1')
    plt.ylabel('φ{}'.format(i))
    plt.title(f'φ1 vs φ{i}')
    plt.show()