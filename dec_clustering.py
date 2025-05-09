# %% ------------------------------- imports -----------------------------------

# TODO: add a semi-supervised loss component 

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Parameter

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from scipy.optimize import linear_sum_assignment

from tqdm import *

# %% ------------------------------- settings -----------------------------------

# set the device for running deep learning
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from pathlib import Path

# Get the directory where this script (dec_clustering.py) is located
script_dir = Path(__file__).resolve().parent

# Construct the path to the plots folder (assumed to be in the project root)
plot_dir = script_dir.parent / 'plots'

# Create the folder if it doesn't exist
plot_dir.mkdir(parents=True, exist_ok=True)

# %% ------------------------------- utility functions --------------------------

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    # Create the directory if it doesn't exist
    save_dir = os.path.dirname(filename)
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist. Creating it.")
        os.makedirs(save_dir, exist_ok=True)
    
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # Count the occurrences of predicted and true label pairs
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # Use scipy's linear_sum_assignment instead of deprecated linear_assignment_
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # Calculate the clustering accuracy
    accuracy = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
    
    return accuracy

def plot_clusters(X, y, title):
    """
    Plots 2D data points coloured by cluster or label.

    Parameters:
    - X (np.ndarray or torch.Tensor): 2D array of shape (n_samples, 2)
    - y (array-like): Cluster labels or ground truth labels
    - title (str): Title of the plot
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()
    
# %% ------------------------------- define autoencoder--------------------------

# TODO: add drop out
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 10))  # Final latent space dimension

        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, input_dim))  # Output dimension should match input dimension

        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x

# %% ------------------------------- Clustering Layer --------------------------

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).to(device)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist

# %% ------------------------------- DEC model ------------------------------------------
# 
class DEC(nn.Module):
    def __init__(self, n_clusters=10, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x) 
        return self.clusteringlayer(x)

    def visualize(self, epoch, x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach() 
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:,0], x_embedded[:,1])
        fig.savefig(plot_dir / 'tabular_{}.png'.format(epoch))
        plt.close(fig)

# %% ------------------------------- pre-train --------------------------
 
def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    batch_size = kwargs['batch_size']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            img  = data.float()
            noisy_img = add_noise(img)
            noisy_img = noisy_img.to(device)
            img = img.to(device)
            output = model(noisy_img)
            output = output.squeeze(1)
            output = output.view(output.size(0), -1)
            loss = nn.MSELoss()(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch [{}/{}], MSE_loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, savepath, is_best)

# %% ------------------------------- train --------------------------
 
def train(**kwargs):
    
    data = kwargs['data']
    labels = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    batch_size = kwargs['batch_size']
    lr = kwargs['lr']
    features = []
    train_loader = DataLoader(dataset=data, 
                              batch_size=batch_size, 
                              shuffle=False)

    for i, batch in enumerate(train_loader):
        img = batch.float().to(device)
        features.append(model.autoencoder.encode(img).detach().cpu())
    features = torch.cat(features)
    features_np = features.numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=model.n_clusters, random_state=0).fit(features_np)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)

    y_pred = kmeans.predict(features_np)
    accuracy = acc(labels.cpu().numpy(), y_pred)
    print('Initial Accuracy: {}'.format(accuracy))

    # 🔧 Save to global DataFrame
    cluster_assignments_df = pd.DataFrame({
        'sample_idx': np.arange(len(features_np)),
        'cluster': y_pred,
        'label': labels.cpu().numpy()
    })
    for i in range(data.shape[1]):
        cluster_assignments_df[f'f{i}'] = data[:, i]

    # Continue with DEC training
    loss_function = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    print('Training')
    row = []

    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float().to(device)
        output = model(img)
        target = model.target_distribution(output).detach()
        out = output.argmax(1)

        if epoch % 20 == 0:
            print('plotting')
            model.visualize(epoch, img)

        loss = loss_function(output.log(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = acc(labels.cpu().numpy(), out.cpu().numpy())
        row.append([epoch, accuracy])
        print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch, num_epochs, accuracy, loss))

        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, savepath, is_best)

    df = pd.DataFrame(row, columns=['epochs', 'accuracy'])
    df.to_csv('log.csv')

    return cluster_assignments_df

# %% ------------------------------- main section to run clustering --------------------------

# Load tabular data 
# def load_tabular_data(df):
#     data = df.values
#     return torch.tensor(data, dtype=torch.float)

# Example tabular dataset (replace with actual data)
# df = pd.read_csv("your_data.csv")
# x = load_tabular_data(df)
# y = df['label'].values  # Assuming you have a label column

# Generate synthetic data with 10 distinct clusters
# X, y = make_blobs(n_samples=10000, n_features=2, centers=10, cluster_std=1.0)
    
# plot_clusters(X, y, title="Synthetic 2D Data with 10 Clusters")

# # Convert to PyTorch tensors
# x = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.long)

# n_clusters = 10

# %% ------------------------------- run clustering --------------------------

# # Hyperparameters
# batch_size = 256
# pretrain_epochs = 5
# train_epochs = 5
# save_dir = os.path.abspath('saves')
# lr = 1e-3
# weight_decay = 1e-5

# # Save paths
# ae_save_path = f'{save_dir}/autoencoder_tabular.pth'
# dec_save_path = f'{save_dir}/dec_tabular.pth'

# # Initialize AutoEncoder with appropriate input dimension
# input_dim = x.shape[1]
# autoencoder = AutoEncoder(input_dim).to(device)

# checkpoint = {
#     "epoch": 0,
#     "best": float("inf")
# }

# # pre-train the autoencoder on the iniput data
# pretrain(data=x, model=autoencoder, num_epochs=pretrain_epochs, savepath=ae_save_path, checkpoint=checkpoint,
#          batch_size=batch_size, lr=lr, weight_decay=weight_decay)

# # Initialize DEC
# dec = DEC(n_clusters=n_clusters, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)

# # train DEC
# checkpoint = {
#     "epoch": 0,
#     "best": float("inf")
# }
# cluster_assignments_df = train(data=x, labels=y, model=dec, num_epochs=train_epochs, savepath=dec_save_path,        
#                                checkpoint=checkpoint, batch_size=batch_size, lr=lr, weight_decay=weight_decay)  
    
# %% ------------------------------- print and plot the results --------------------------

# # print accuracy
# if not cluster_assignments_df.empty:
#     my_accuracy = acc(cluster_assignments_df['label'].to_numpy(),            
#                     cluster_assignments_df['cluster'].to_numpy(),
#                     )
#     print(my_accuracy)
    
# plot_clusters(x, cluster_assignments_df['cluster'].to_numpy(), title="DEC Clustering Results (Predicted Clusters)")
# plot_clusters(x, cluster_assignments_df['label'].to_numpy(), title="Ground Truth Labels")

# %% function to call
def run_dec_clustering_from_dataframe(df: pd.DataFrame,
                                      target_column: str = 'label',
                                      n_clusters: int = 10,
                                      pretrain_epochs: int = 5,
                                      train_epochs: int = 5,
                                      batch_size: int = 256,
                                      lr: float = 1e-3,
                                      weight_decay: float = 1e-5,
                                      save_dir: str = 'saves') -> pd.DataFrame:
    """
    Run DEC clustering on a tabular DataFrame.

    Parameters:
    - df: Input DataFrame with features and target column
    - target_column: Name of the column containing ground truth targets
    - n_clusters: Number of clusters to learn
    - pretrain_epochs: Number of epochs to pretrain the autoencoder
    - train_epochs: Number of epochs to train the DEC model
    - batch_size: Batch size for training
    - lr: Learning rate
    - weight_decay: Weight decay (L2 regularization)
    - save_dir: Directory to save models

    Returns:
    - cluster_assignments_df: DataFrame with cluster assignments, targets and features
    """
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.abspath(save_dir)

    # Split features and targets
    features_df = df.drop(columns=[target_column])
    targets = df[target_column].to_numpy()

    # Convert to tensors
    x = torch.tensor(features_df.values, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)

    input_dim = x.shape[1]
    autoencoder = AutoEncoder(input_dim).to(device)

    # Pretrain AE
    ae_save_path = f'{save_dir}/autoencoder_tabular.pth'
    checkpoint = {"epoch": 0, "best": float("inf")}
    pretrain(data=x, model=autoencoder, num_epochs=pretrain_epochs, savepath=ae_save_path,
             checkpoint=checkpoint, batch_size=batch_size, lr=lr, weight_decay=weight_decay)

    # Train DEC
    dec = DEC(n_clusters=n_clusters, autoencoder=autoencoder, hidden=10,
              cluster_centers=None, alpha=1.0).to(device)
    checkpoint = {"epoch": 0, "best": float("inf")}
    dec_save_path = f'{save_dir}/dec_tabular.pth'
    cluster_assignments_df = train(data=x, labels=y, model=dec, num_epochs=train_epochs,
                                   savepath=dec_save_path, checkpoint=checkpoint,
                                   batch_size=batch_size, lr=lr, weight_decay=weight_decay)

    return cluster_assignments_df

# %%

# # 1. Generate synthetic data
# X, y = make_blobs(
#     n_samples=10000, n_features=2, centers=10, 
#     cluster_std=1.0, random_state=42
# )

# # 2. Visualise ground truth
# plot_clusters(X, y, title="Synthetic 2D Data with 10 Clusters")

# # 3. Prepare DataFrame for DEC
# df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
# df['target'] = y

# # 4. Run DEC clustering
# df_dec = run_dec_clustering_from_dataframe(
#     df=df,
#     target_column='target',
#     n_clusters=10,
#     pretrain_epochs=5,
#     train_epochs=5,
#     batch_size=256,
#     lr=1e-3,
#     weight_decay=1e-5,
#     save_dir='saves'
# )

# # 5. Visualise DEC clustering results
# X_tensor = torch.tensor(X, dtype=torch.float32)
# plot_clusters(X_tensor, df_dec['cluster'].to_numpy(), title="DEC Clustering Results")
# plot_clusters(X_tensor, df_dec['label'].to_numpy(), title="Ground Truth Labels")
# %%
