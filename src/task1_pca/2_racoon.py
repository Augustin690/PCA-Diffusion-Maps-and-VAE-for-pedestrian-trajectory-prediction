from scipy.linalg import svd
import utils

""" Task 1.2: In this script, we apply principal component analysis to a racoon image. 
We need functions defined in utils.py for this script.
"""

# TODO: Load and resize the racoon image in grayscale
racoon  = utils.load_resize_image()
data = racoon.T
# TODO: Compute Singular Value Decomposition (SVD) using utils.compute_svd()
import numpy as np
mean = np.mean(data, axis=0)
face_centered = data - mean
U, S, Vt = utils.compute_svd(face_centered)

# TODO: Reconstruct images using utils.reconstruct_images
utils.reconstruct_images(U, S, Vt)
# TODO: Compute the number of components where energy loss is smaller than 1% using utils.compute_num_components_capturing_threshold_energy()
num_components = utils.compute_num_components_capturing_threshold_energy(S)
print(num_components)