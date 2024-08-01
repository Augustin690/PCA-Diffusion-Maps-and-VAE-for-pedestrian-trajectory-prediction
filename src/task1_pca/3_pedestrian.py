import numpy as np
import matplotlib.pyplot as plt
import utils

""" Task 1.3: In this script, we apply principal component analysis to pedestrian trajectory data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load trajectory data in data_DMAP_PCA_Vadere.txt. (Hint: You may need to use a space as delimiter)
data = np.loadtxt('../../data/data_DMAP_PCA_Vadere.txt', delimiter=' ')
# TODO: Center the data by subtracting the mean
mean = np.mean(data, axis=0)
data_centered = data - mean
# TODO: Extract positions of pedestrians 1 and 2
pedestrian_1 = data[:, :2]
pedestrian_2 = data[:, 2:4]
# TODO: Visualize trajectories of first two pedestrians (Hint: You can optionally use utils.visualize_traj_two_pedestrians() )
utils.visualize_traj_two_pedestrians(pedestrian_1, pedestrian_2, ('trajectories', 'x', 'y'))
plt.show()
# TODO: Compute SVD of the data using utils.compute_svd()
U, S, Vt = utils.compute_svd(data_centered)
# TODO: Reconstruct data by truncating SVD using utils.reconstruct_data_using_truncated_svd()
num_components = 2
data_reconstructed_T = utils.reconstruct_data_using_truncated_svd(U, S, Vt, num_components)
data_reconstructed = data_reconstructed_T
# TODO: Visualize trajectories of the first two pedestrians in the 2D space defined by the first two principal components
pedestrian_1_re_centered = data_reconstructed[:, :2]
pedestrian_2_re_centered = data_reconstructed[:, 2:4]
utils.visualize_traj_two_pedestrians(pedestrian_1_re_centered, pedestrian_2_re_centered, ('trajectories', 'x', 'y'))
plt.show()
# TODO: Answer the questionsin the worksheet with the help of utils.compute_cumulative_energy(), utils.compute_num_components_capturing_threshold_energy()
cumulative_energy = utils.compute_cumulative_energy(S, 2)
print(cumulative_energy)
num_components = utils.compute_num_components_capturing_threshold_energy(S, 0.1)
print(num_components)


