import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
import numpy.typing as npt
import time
# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def reconstruction_loss(x_reconstructed:npt.NDArray[torch.float32], x:npt.NDArray[torch.float32]) -> torch.float32:
    """Compute the reconstruction loss.

    Args:
        x_reconstructed (npt.NDArray[np.float64]): Reconstructed data
        x (npt.NDArray[np.float64]): raw/original data

    Returns:
        np.float64: reconstruction loss
    """
    # TODO: Implement method!
    recon_diff = (x_reconstructed - x) ** 2
    recon_loss = recon_diff.sum()
    # FIXME
    return recon_loss
def kl_loss(logvar:npt.NDArray[torch.float32], mu:npt.NDArray[torch.float32]) -> torch.float32:
    """ Compute the Kullback-Leibler (KL) divergence loss using the encoded data into the mean and log-variance.

    Args:
        logvar (npt.NDArray[np.float64]): log of variance (from the output of the encoder)
        mu (npt.NDArray[np.float64]): mean (from the output of the encoder)

    Returns:
        np.float64: KL loss
    """
    # TODO: Implement method!
    return -0.5 * torch.sum((1 + logvar - mu ** 2 - torch.exp(logvar)))

# Function to compute ELBO loss
def elbo_loss(x:npt.NDArray[torch.float32], x_reconstructed:npt.NDArray[torch.float32], mu:npt.NDArray[torch.float32], logvar:npt.NDArray[torch.float32]):
    """Compute Evidence Lower BOund (ELBO) Loss by combining the KL loss and reconstruction loss. 

    Args:
        x (npt.NDArray[np.float64]): raw/original data
        x_reconstructed (npt.NDArray[np.float64]): Reconstructed data
        mu (npt.NDArray[np.float64]): mean (from the output of the encoder)
        logvar (npt.NDArray[np.float64]): log of variance (from the output of the encoder)

    Returns:
        np.float64: ELBO loss
    """
    # TODO: Implement method! Hint(You may need to reshape x using x.view(. , .)!)
    recon_loss = reconstruction_loss(x_reconstructed, x)
    kl_div = kl_loss(logvar, mu)
    loss = (recon_loss + kl_div) / x.size(0)
    return loss

# Function for training the VAE
def train_epoch(model:object, optimizer:object, dataloader:object, device) -> np.float64:
    """ Train the vae for one epoch and return the training loss on the epoch. 

    Args:
        model (object): The model (of class VAE)
        optimizer (object): Adam optimizer (from torch.optim)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu') on which the training is to be done.

    Returns:
        np.float64: training loss
    """
    model.train()
    total_loss = 0
    for data, _ in dataloader:
        data = data.view(-1, int(np.shape(data)[-1] * np.shape(data)[-2])).to(device)
        # TODO: Set gradient to zero! You can use optimizer.zero_grad()!
        optimizer.zero_grad()
        # TODO: Perform forward pass of the VAE
        x_reconstructed, mu, logvar = model(data)
        # TODO: Compute ELBO loss
        # loss = (F.binary_cross_entropy(x_reconstructed.view(-1, 28*28), data.view(-1, 28*28), reduction='sum') + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / data.size(0)
        loss = elbo_loss(data, x_reconstructed, mu, logvar)
        # #print("\rrecon_loss: {}".format(recon_loss / len(data)), end='')
        # loss = torch.tensor(loss, requires_grad=True)
        # TODO: Compute gradients
        loss.backward()
        # TODO: Perform an optimization step
        optimizer.step()
        # TODO: Compute total_loss and return the total_loss/len(dataloader.dataset)
        total_loss += loss.item()

        # FIXME parms.grad = None
        #for name, parms in model.named_parameters():
            # print(parms.grad)
            # print(parms.grad_fn)
            #print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data))
    return total_loss / len(dataloader.dataset)

def evaluate(model:object, dataloader:object, device)-> np.float64:
    """ Evaluate the model on the test data and return the test loss.

    Args:
        model (object): The model (of class VAE)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        np.float64: test loss.
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    # return total_loss/len(dataloader.dataset)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(-1, int(np.shape(data)[-1] * np.shape(data)[-2])).to(device)
            x_reconstructed, mu, logvar = model(data)
            # loss = (F.binary_cross_entropy(x_reconstructed.view(-1, 28 * 28), data.view(-1, 28 * 28),
            #                    reduction='sum') + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / data.size(0)
            loss = elbo_loss(data, x_reconstructed, mu, logvar)
            # loss = torch.tensor(loss, requires_grad=True)
            total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

def latent_representation(model:object, dataloader:object, device) -> None:
    """Plot the latent representation of the data.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(x.size(0), -1).to(device)
            mu, _ = model.encode_data(x)
            mu = mu.cpu().detach().numpy()
            latents.append(mu)
            labels.append(y.cpu())

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(10, 10))

    plt.scatter(latents[:, 0], latents[:, 1], c=labels)
    plt.title('latent_representation')
    plt.colorbar()
    plt.show()

# Function to plot reconstructed digits
def reconstruct_digits(model:object, dataloader:object, device, num_digits:int =15) -> None:
    """ Plot reconstructed digits. 

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
        num_digits (int, optional): No. of digits to be re-constructed. Defaults to 15.
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    model.eval()
    original_images = []
    reconstruct_images = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.view(x.size(0), -1).to(device)
            x_reconstructed, _, _ = model(x)
            original_images.append(x.cpu())
            reconstruct_images.append(x_reconstructed.cpu().detach().numpy())

            # FIXME * x.size(0) ?
            if len(original_images) > num_digits:
                break

    original_images = np.concatenate(original_images, axis=0)[:num_digits]
    reconstruct_images = np.concatenate(reconstruct_images, axis=0)[:num_digits]

    plt.figure(figsize=(15, 3))
    for i in range(num_digits):
        plt.subplot(2, num_digits, i+1)
        plt.imshow(original_images[i].reshape(28, 28))
        plt.axis('off')
        plt.subplot(2, num_digits, i + 1 + num_digits)
        plt.imshow(reconstruct_images[i].reshape(28, 28))
        plt.axis('off')
    plt.title('reconstruct_digits')
    plt.show()

# Function to plot generated digits
def generate_digits(model:object, num_samples:int =15) -> None:
    """ Generate 'num_samples' digits.

    Args:
        model (object): The model (of class VAE).
        num_samples (int, optional): No. of samples to be generated. Defaults to 15.
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    model.eval()
    with torch.no_grad():
        z = torch.rand(num_samples, model.d_latent).to(model.device)
        generate_images = model.decode_data(z)
        generate_images = generate_images.cpu().detach().numpy()
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(generate_images[i].reshape(28, 28))
        plt.axis('off')
    plt.title('generate_digits')
    plt.show()


# Function to plot the loss curve
def plot_loss(train_losses, test_losses):
    epochs = len(train_losses)
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train')
    plt.plot(range(1, epochs+1), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    plt.savefig('loss_curve.png')


def training_loop(vae:object, optimizer:object, train_loader:object, test_loader:object, epochs:int, plots_at_epochs:list, device) -> tuple [list, list]:
    """ Train the vae model. 

    Args:
        vae (object): The model (of class VAE).
        optimizer (object): Adam optimizer (from torch.optim).
        train_loader (object): A data loader that combines the training dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        test_loader (object): A data loader that combines the test dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        epochs (int): No. of epochs to train the model.
        plots_at_epochs (list): List of integers containing epoch numbers at which the plots are to be made.
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple [list, list]: Lists train_losses, test_losses containing train and test losses at each epoch.
    """
    # Lists to store the training and test losses
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # TODO: Compute training loss for one epoch
        train_loss = train_epoch(vae, optimizer, train_loader, device)
        # TODO: Evaluate loss on the test dataset
        test_loss = evaluate(vae, test_loader, device)
        # TODO: Append train and test losses to the lists train_losses and test_losses respectively
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        # TODO: For specific epoch numbers described in the worksheet, plot latent representation, reconstructed digits, generated digits after specific epochs
        if epoch in plots_at_epochs:
            latent_representation(vae, test_loader, device)
            reconstruct_digits(vae, test_loader, device)
            generate_digits(vae)

    # TODO: return train_losses, test_losses
    return train_losses, test_losses


def instantiate_vae(d_in, d_latent, d_hidden_layer, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """Instantiate the variational autoencoder.

    Args:
        d_in (int): Input dimension.
        d_latent (int): Latent dimension.
        d_hidden_layer (int): Number of neurons in each hidden layer of the encoder and decoder.
        device: e.g., 'cuda' or 'cpu'. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        object: An object of class VAE
    """
    return VAE(d_in, d_latent, d_hidden_layer, device).to(device)