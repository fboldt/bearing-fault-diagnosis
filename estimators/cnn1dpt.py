import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import label_binarize
import numpy as np

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class VibrationDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        label = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        return x, label

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )
        self.double()
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
    print(f"loss: {loss:>7f}", end='')

def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    validation_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    validation_loss /= num_batches
    correct /= size
    print(f", val_accuracy: {(correct):>0.4f}, val_loss: {validation_loss:>8f}")

def predict(row, model):
    # make prediction
    yhat = model(row.to(device))
    # retrieve numpy array
    yhat = yhat.detach().cpu().numpy()
    return yhat

class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=10, checkpoint="model.checkpoint.pth", verbose=2):
        super().__init__()
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.verbose = verbose
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop
    
    def fit(self, Xtr, ytr, Xva, yva):
        input_size = Xtr.shape[1]*Xtr.shape[2]
        Xtr = torch.from_numpy(Xtr)
        Xva = torch.from_numpy(Xva)
        self.labels = np.unique(ytr)
        ytr = label_binarize(ytr, classes=self.labels)
        ytr = torch.from_numpy(np.array(ytr, dtype=float))
        yva = label_binarize(yva, classes=self.labels)
        yva = torch.from_numpy(np.argmax(np.array(yva, dtype=float), axis=1))
        output_size = len(self.labels)
        self.model = NeuralNetwork(input_size, output_size).to(device)
        optimizer = self.optimizer(self.model.parameters(), lr=1e-3)
        tr = VibrationDataset(Xtr, ytr)
        tr = DataLoader(tr, batch_size=64, shuffle=True)
        va = VibrationDataset(Xva, yva)
        va = DataLoader(va, batch_size=64, shuffle=True)
        for t in range(self.epochs):
            print(f"Epoch {t+1}: ", end='')
            train(tr, self.model, self.loss_fn, optimizer)
            validation(va, self.model, self.loss_fn)

    def predict(self, X):
        self.model.eval()
        X = Tensor(X).to(torch.double)
        yprob = predict(X, self.model)
        ypred = self.labels[np.argmax(yprob, axis=1)]
        return ypred
    