import numpy as np
import math
import torch
from sklearn.datasets import make_friedman3
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def makeSinLoader(num_examples, batch_size, random_state = None,
                  noise = 0.1, num_gaps = 0, shuffle = True, dom_range = (-3,3),
                  tr_ratio = 0.6, val_ratio = 0.2, test_ratio = 0.2):
    
    input_length = dom_range[1] - dom_range[0]

    #Make random number generator with seed
    if random_state==None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=random_state)

    #determine where the gaps in the data will be
    gap_locs = np.zeros((num_gaps,2))
    for i in range(num_gaps):
        lower_bound = np.round(rng.uniform(),2)
        upper_bound = np.round(min(1.0, lower_bound + rng.uniform(low=0.05, high=0.10), 2),2)

        lower_bound = dom_range[0] + (lower_bound * input_length)
        upper_bound = dom_range[0] + (upper_bound * input_length)
        gap_locs[i] = (lower_bound, upper_bound)
    print(gap_locs)

    candidates = np.random.uniform(dom_range[0], dom_range[1], num_examples)

    # Build mask for excluded ranges
    mask = np.ones_like(candidates, dtype=bool)
    for start, end in gap_locs:
        mask &= ~((candidates >= start) & (candidates <= end))

    valid = candidates[mask]

    # If not enough valid points, resample until we have enough
    while valid.size < num_examples:
        extra = np.random.uniform(dom_range[0], dom_range[1], num_examples)
        extra_mask = np.ones_like(extra, dtype=bool)
        for start, end in gap_locs:
            extra_mask &= ~((extra >= start) & (extra <= end))
        valid = np.concatenate([valid, extra[extra_mask]])

    #get data
    x = valid[:num_examples]
    y = np.sin(2 * x) - np.cos(x) + noise * rng.normal(size = num_examples)

    X_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    n_train = int(num_examples * tr_ratio)
    n_val   = int(num_examples * val_ratio)

    indices = rng.permutation(num_examples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    #Return data loaders
    return train_loader, val_loader, test_loader

def createFriedmanLoader(n_samples, batch_size, random_state=None,
                         noise = 0.1, shuffle = True,
                         tr_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    
    if random_state==None:
        X, y = make_friedman3(n_samples=n_samples, noise=noise)
    else:
        X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    n_train = int(n_samples * tr_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = makeSinLoader(300, 300, num_gaps=3)

# --- Plotting code ---
# Helper to extract full tensors from a DataLoader
def get_full_batch(loader):
    for xb, yb in loader:
        return xb, yb

x_train, y_train = get_full_batch(train_loader)
x_val, y_val     = get_full_batch(val_loader)
x_test, y_test   = get_full_batch(test_loader)

# Plot train data
plt.figure(figsize=(8,5))
plt.scatter(x_train.numpy(), y_train.numpy(), color='blue', s=10)
plt.title("Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot validation data
plt.figure(figsize=(8,5))
plt.scatter(x_val.numpy(), y_val.numpy(), color='green', s=10)
plt.title("Validation Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot test data
plt.figure(figsize=(8,5))
plt.scatter(x_test.numpy(), y_test.numpy(), color='red', s=10)
plt.title("Test Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

