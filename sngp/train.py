import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import math
import tqdm # type: ignore
from sklearn.datasets import make_moons # type: ignore
from sklearn import metrics # type: ignore
import torch # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import numpy as np # type: ignore
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loss import square_loss


def flatten_params(model, excluded_params=['lengthscale_param', 'outputscale_param', 'sigma_param']):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if name not in excluded_params])

def train_model(model, device, tr_loader, va_loader, optimizer=None,
                n_epochs=10, lr=0.001, l2pen_mag=0.0, data_order_seed=42,
                model_filename=None,
                do_early_stopping=True,
                n_epochs_without_va_improve_before_early_stop=15,
                ):
    ''' Train model via stochastic gradient descent.

    Assumes provided model's trainable params already set to initial values.

    Returns
    -------
    best_model : PyTorch model
        Model corresponding to epoch with best validation loss (xent)
        seen at any epoch throughout this training run
    info : dict
        Contains history of this training run, for diagnostics/plotting
    '''
    # Make sure tr_loader shuffling reproducible
    torch.manual_seed(data_order_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(data_order_seed)
    model.to(device)
    
    if optimizer is None:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr)
        
    # Allocate lists for tracking progress each epoch
    tr_info = {'loss':[], 'sqloss':[], 'mse':[]}
    va_info = {'loss':[], 'sqloss':[], 'mse':[]}
    epochs = []

    # Init vars needed for early stopping
    best_va_loss = float('inf')
    curr_wait = 0 # track epochs we are waiting to early stop

    # Count size of datasets, for adjusting metric values to be per-example
    n_train = float(len(tr_loader.dataset))
    n_batch_tr = float(len(tr_loader))
    n_valid = float(len(va_loader.dataset))

    # Progress bar
    progressbar = tqdm.tqdm(range(n_epochs + 1))
    pbar_info = {}

    for epoch in progressbar:
        if epoch > 0:
            model.train()
            tr_loss = 0.0  # aggregate total loss
            tr_sqloss = 0.0 # aggregate squared loss
            tr_mse = 0     # count average error
            pbar_info['batch_done'] = 0
            for x, y in tr_loader:
                optimizer.zero_grad()
                x_BF = x.to(device)
                y_B = y.to(device)

                #get model prediction
                pred_B = model(x_BF)
       
                #Compute loss according to paper
                loss_sqerr = square_loss(pred_B, y_B)

                params = flatten_params(model)
                l2_loss = (params**2).sum()
        
                loss = loss_sqerr + (float(l2pen_mag) / n_train) * l2_loss
                #Backprop
                loss.backward()
                optimizer.step()

                pbar_info['batch_done'] += 1        
                progressbar.set_postfix(pbar_info)
    
                # Increment loss metrics we track for debugging/diagnostics
                tr_loss += loss.item() / n_batch_tr
                tr_sqloss += loss_sqerr.item() / n_batch_tr
                tr_mse += metrics.mean_squared_error(y_B.detach().cpu().numpy(), 
                                                     pred_B.detach().cpu().numpy())
        else:
            # First epoch (0) doesn't train, just measures initial perf on val
            tr_loss = np.nan
            tr_sqloss = np.nan
            tr_mse = np.nan

        #Track metrics on validation set
        with torch.no_grad():
            model.eval()
            va_total_loss = 0.0
            va_sqloss = 0.0
            va_mse = 0.0

            for x_va, y_va in va_loader:
                x_va_BF = x_va.to(device)
                y_va_B = y_va.to(device)


                pred_va_B = model(x_va_BF)


                #Compute loss according to paper
                va_loss_sq = square_loss(pred_va_B, y_va_B)

                # params = flatten_params(model)
                # l2_loss = (params**2).sum()
        
                va_loss_total = va_loss_sq  # + (float(l2pen_mag) / n_train) * l2_loss

                va_total_loss += va_loss_total.item()
                va_sqloss += va_loss_sq.item()
                va_mse += metrics.mean_squared_error(y_va_B.detach().cpu().numpy(), 
                                                     pred_va_B.detach().cpu().numpy())
                                                     
        epochs.append(epoch)
        tr_info['loss'].append(tr_loss)
        tr_info['sqloss'].append(tr_sqloss)
        tr_info['mse'].append(tr_mse)     
        va_info['loss'].append(va_loss_total)   
        va_info['sqloss'].append(va_loss_sq)
        va_info['mse'].append(va_mse)
        pbar_info.update({
            "tr_loss": tr_loss, "tr_sqloss":tr_sqloss, "tr_mse":tr_mse,
            "va_loss": va_loss_total, "va_sqloss":va_sqloss, "va_mse":va_mse
            })
        
        # Early stopping logic
        # If loss is dropping, track latest weights as best
        if va_loss_sq < best_va_loss:
            best_epoch = epoch
            best_va_loss = va_loss_sq
            best_tr_err_rate = tr_mse
            best_va_err_rate = va_mse
            curr_wait = 0
            if model_filename != None:
                model = model.cpu()
                torch.save(model.state_dict(), model_filename)
                model.to(device)
            
        else:
            curr_wait += 1

        wait_enough = curr_wait >= n_epochs_without_va_improve_before_early_stop
        if do_early_stopping and wait_enough:
            print("Stopped early.")
            break


    print(f"Finished after epoch {epoch}, best epoch={best_epoch}")
    model.to(device)
    if model_filename != None:
        model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))    
    result = { 
        'data_order_seed':data_order_seed,
        'lr':lr, 'n_epochs':n_epochs, 'l2pen_mag':l2pen_mag,
        'tr':tr_info,
        'va':va_info,
        'best_tr_mse': best_tr_err_rate,
        'best_va_mse': best_va_err_rate,
        'best_va_loss': best_va_loss,
        'best_epoch': best_epoch,
        'epochs': epochs}
    return model, result