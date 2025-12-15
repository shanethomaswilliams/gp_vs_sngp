import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

'''
Inputs:
    file_path: filename to load data from
    sample_size: number of samples to receive
    train_percentage: What percent of data should be in training loader
                      the rest will be allocated
    random_state: Seed used for reproducable randomness
    Shuffle: to Shuffle the data loaders or not

Returns: 
    GP_train_loader: Train loader formated for GP x is (N, F) 
    GP_valid_loader: Val loader formated for GP y is (N, 1)
    SNGP_train_loader: Train loader formated for SNGP x is (N, F)
    SNGP_valid_loader: Val Loader formated for SNGP y is (N)
    
'''
def load_data_train(file_path, sample_size, train_percentage = 0.75, random_state=42, shuffle = True):
    #Makes sure the splits make sense
    assert train_percentage < 1.0


    #Load data
    tensor_DF = torch.load(file_path)
    x_DF = tensor_DF["x"]
    y_D = tensor_DF["y"]

    
    # Random shuffle of row ids corresponding to all L provided examples
    D = len(x_DF)
    rng = np.random.default_rng(seed=random_state)
    shuffled_ids_D = rng.permutation(np.arange(D))

    #Train = N, Validation = V, Test = T
    #Determine N, V, T
    N = int(np.round(sample_size * (train_percentage)))
    V = sample_size - N
    assert D >= N + V

    #Make x_data
    x_train_NF = x_DF[shuffled_ids_D[0:N]].unsqueeze(-1)
    x_valid_VF = x_DF[shuffled_ids_D[N:N+V]].unsqueeze(-1)

    #Make SNGP y_data
    y_SNGP_train_N1 = y_D[shuffled_ids_D[0:N]]
    y_SNGP_valid_V1 = y_D[shuffled_ids_D[N:N+V]]

    #Make GP y_data
    y_GP_train_N = y_SNGP_train_N1.unsqueeze(-1)
    y_GP_valid_V = y_SNGP_valid_V1.unsqueeze(-1)

    #Creates datasets
    GP_train_dataset = TensorDataset(x_train_NF, y_GP_train_N)
    GP_valid_dataset = TensorDataset(x_valid_VF, y_GP_valid_V)

    SNGP_train_dataset = TensorDataset(x_train_NF, y_SNGP_train_N1)
    SNGP_valid_dataset = TensorDataset(x_valid_VF, y_SNGP_valid_V1)

    GP_train_loader = DataLoader(GP_train_dataset, batch_size=N, shuffle=shuffle)
    GP_valid_loader = DataLoader(GP_valid_dataset, batch_size=V, shuffle=shuffle)

    SNGP_train_loader = DataLoader(SNGP_train_dataset, batch_size=N, shuffle=shuffle)
    SNGP_valid_loader = DataLoader(SNGP_valid_dataset, batch_size=V, shuffle=shuffle)

    return GP_train_loader, GP_valid_loader, SNGP_train_loader, SNGP_valid_loader


'''
Inputs:
    file_path: filename to load data from
    sample_size: number of samples to receive
    random_state: Seed used for reproducable randomness
    Shuffle: to Shuffle the data loaders or not

Returns: 
    GP_test_loader: Train loader formated for GP
    SNGP_test_loader: Train loader formated for SNGP 
'''
def load_data_test(file_path, sample_size, random_state=42, shuffle = True):

    #Load data
    tensor_DF = torch.load(file_path)
    x_DF = tensor_DF["x"]
    y_D = tensor_DF["y"]

    
    # Random shuffle of row ids corresponding to all L provided examples
    D = len(x_DF)
    N = sample_size
    rng = np.random.default_rng(seed=random_state)
    shuffled_ids_D = rng.permutation(np.arange(D))

    #Train = N, Validation = V, Test = T
    #Determine N, V, T
    assert D >= N

    #Make x_data
    x_test_NF = x_DF[shuffled_ids_D[0:N]].unsqueeze(-1)

    #Make SNGP y_data
    y_SNGP_test_N1 = y_D[shuffled_ids_D[0:N]]

    #Make GP y_data
    y_GP_test_N = y_SNGP_test_N1.unsqueeze(-1)

    #Creates datasets
    GP_test_dataset = TensorDataset(x_test_NF, y_GP_test_N)
    SNGP_test_dataset = TensorDataset(x_test_NF, y_SNGP_test_N1)

    GP_test_loader = DataLoader(GP_test_dataset, batch_size=N, shuffle=shuffle)

    SNGP_test_loader = DataLoader(SNGP_test_dataset, batch_size=N, shuffle=shuffle)

    return GP_test_loader, SNGP_test_loader
