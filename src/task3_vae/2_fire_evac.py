import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import VAE
from utils import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" This script is used to train and test the VAE on the fire_evac dataset. 
(Bonus: You can simulate trajectories with Vadere, for bonus points.) 
Not included in automated tests, as it's open-ended.
"""

# TODO: Download the FireEvac dataset
train_data_numpy = np.load('../../data/FireEvac_train_set.npy')
test_data_numpy = np.load('../../data/FireEvac_test_set.npy')
train_data_n = 2 * (train_data_numpy - train_data_numpy.min(axis=0)) / (train_data_numpy.max(axis=0) - train_data_numpy.min(axis=0)) - 1
test_data_n = 2 * (test_data_numpy - test_data_numpy.min(axis=0)) / (test_data_numpy.max(axis=0) - test_data_numpy.min(axis=0)) - 1
print(train_data_n)
train_data = torch.tensor(train_data_n, dtype=torch.float32)
test_data = torch.tensor(test_data_n, dtype=torch.float32)

from torch.utils.data import TensorDataset
train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data), batch_size=64, shuffle=False)
batch_size = 64

# TODO: Make a scatter plot to visualise it.
plt.scatter(train_data[:, 0], train_data[:, 1], marker='x', s=35, color='blue', label='Training Data')
plt.scatter(test_data[:, 0], test_data[:, 1], marker='+', s=35, color='red', label='Test Data')
plt.title('Training Data Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# TODO: Train a VAE on the FireEvac data

# TODO: Need to change the activation function in model.py follow the instruction in the Reports before training
num_epochs = 200
input_dim = 2
latent_dim = 2
hidden_dim = 64
learning_rate = 0.0005
vae = instantiate_vae(input_dim, hidden_dim, latent_dim, device).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# train_losses, test_losses = training_loop(vae, optimizer, train_loader, test_loader, num_epochs, plots_at_epochs, device)
# plot_loss(train_losses, test_losses)
train_losses = []
test_losses = []
reconstructed = []
for epoch in range(num_epochs):
    vae.train()
    total_train_loss = 0
    for data in train_loader:

        optimizer.zero_grad()
        data = data[0].to(device)
        recon_x, mu, logvar = vae(data)
        loss = elbo_loss(data*100, recon_x*100, mu, logvar)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    total_train_loss /= len(train_loader.dataset)
    train_losses.append(total_train_loss)

    vae.eval()

    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data[0].to(device)
            recon_x, mu, logvar = vae(data)
            #loss = mse_loss(data, recon_x)
            loss = elbo_loss(data*100, recon_x*100, mu, logvar)
            total_test_loss += loss.item()
    total_test_loss /= len(test_loader.dataset)
    train_losses.append(total_test_loss)
    print(f'Epoch {epoch + 1}, Train Loss: {total_train_loss:.4f}, Test Loss: {total_test_loss:.4f}')

# TODO: Make a scatter plot of the reconstructed test set

vae.eval()

with torch.no_grad():
    for data in test_loader:
        data = data[0].to(device)
        recon_x, _, _ = vae(data)
        reconstructed.append(recon_x.cpu().detach().numpy())

reconstructed = np.vstack(reconstructed)

plt.scatter(test_data[:, 0], test_data[:, 1], marker='+', s=15, color='red', label='test Data')
plt.scatter(reconstructed[:, 0], reconstructed[:, 1], marker='x', s=15, color='green', label='recover Data')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Reconstructed Test Set')
plt.show()

# TODO: Make a scatter plot of 1000 generated samples.
vae.eval()
mean = [-0.5, 1]
covariance_matrix = [[1, 0.5], [0.5, 1]]
data_points = np.random.multivariate_normal(mean, covariance_matrix, 1000)
with torch.no_grad():
    z = torch.tensor(data_points).to(vae.device)
    generate_data = vae(z)
    generate_data = generate_data[0].cpu().detach().numpy()
    print(generate_data)
plt.figure(figsize=(10, 8))
plt.scatter(test_data[:, 0], test_data[:, 1], marker='+', s=15, color='red', label='test Data')
plt.scatter(generate_data[:, 0], generate_data[:, 1], marker='x', s=15, color='green', label='generate Data')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('generate_digits')
plt.show()
# TODO: Generate data to estimate the critical number of people for the MI building
critical_area = ((130, 70), (150, 50))

def in_critical_area(x, y, area):
    (x1, y1), (x2, y2) = area
    return x1 <= x <= x2 and y2 <= y <= y1

critical_count = 0
vae.eval()
mean = [-0.5, 1]
covariance_matrix = [[1, 0.5], [0.5, 1]]
data_points = np.random.multivariate_normal(mean, covariance_matrix, 1000)
with torch.no_grad():
    z = torch.tensor(data_points).to(vae.device)
    critical_data = vae(z)
    critical_data = critical_data[0].cpu().detach().numpy()
    critical_data_min = train_data_numpy.min(axis=0)
    critical_data_max = train_data_numpy.max(axis=0)
    critical_data_original = (critical_data + 1) / 2 * (critical_data_max - critical_data_min) + critical_data_min
    for x, y in critical_data_original:
        if in_critical_area(x, y, critical_area):
            critical_count += 1

print(f'Estimated number of people in the critical area: {critical_count}')
print(critical_data_original)

