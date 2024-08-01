import utils
import numpy as np
import matplotlib.pyplot as plt

""" Task 1.1: In this script, we apply principal component analysis to two-dimensional data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load the dataset from the file pca_dataset.txt
data = np.loadtxt('../../data/pca_dataset.txt')
# TODO: Compute mean of the data
mean = np.mean(data, axis=0)
# TODO: Center data
data_centered = utils.center_data(data)
# TODO: Compute SVD
U, S, Vt = utils.compute_svd(data_centered)
# TODO: Plot principal components
x = []
y = []
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
plt.scatter(x, y)

for i in range(2):
    # FIXME vt是不是v的转置
    plt.arrow(mean[0], mean[1], Vt[i, 0]*S[i], Vt[i, 1]*S[i], head_width=0.05, head_length=0.1, fc='blue', ec='blue')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('PCA on pca_dataset.txt')
plt.show()

# TODO: Analyze the energy captured by the first two principal components using utils.compute_energy()
print("The energy captured by the 1st principal component is {}".format(utils.compute_energy(S, 1)))
print("The energy captured by the 2st principal component is {}".format(utils.compute_energy(S, 2)))
