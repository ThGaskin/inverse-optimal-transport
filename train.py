# -------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import datetime
import numpy as np
import os
import pickle
import sys
import time
import torch

from ruamel.yaml import YAML

from nn import NeuralNet, Sinkhorn, marginals_and_transport_plan

yaml = YAML(typ="safe")

# Load the configuration
with open(sys.argv[1], "r") as file:
    cfg = yaml.load(file)

# Set default device
device = cfg["device"]

# Load the base paths
BASE_PATH = cfg["BASE_PATH"]

# Load or create the save path, if not running in dry run setting (no data saving)
dry_run = cfg.get("dry_run", False)
if not dry_run:
    save_to_path = cfg["Data_loading"].get("load_from_dir", None)
    if save_to_path is None:
        _date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        note = cfg.get("path_note", None)
        if note:
            _date_time += f"_{note}"
        save_to_path = os.path.expanduser(
            os.path.join(BASE_PATH, f"{cfg.get('OUT_DIR', 'Results')}/{_date_time}")
        )
        os.makedirs(save_to_path)

        # Write the cfg
        with open(f"{save_to_path}/cfg.yaml", "w") as file:
            yaml.dump(cfg, file)
    else:
        save_to_path = os.path.expanduser(save_to_path)

# Update neural network config from any pre-trained configurations
if cfg["Data_loading"].get("load_from_dir", None) is not None:
    with open(f"{cfg['Data_loading']['load_from_dir']}/cfg.yaml", "r") as file:
        nn_cfg = yaml.load(file)
    cfg['NeuralNet'] = nn_cfg['NeuralNet']

# ----------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------
# Path to training data, relative to base path
data_path = cfg["Data_loading"]["data_path"]

# Load args, passed to ``torch.load``
load_args = cfg["Data_loading"].get("load_args", {})

# Load data and masks
training_data = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/training_T.pt"), **load_args
    ).to(device).float()
)
mu = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/training_mu.pt"), **load_args
    ).to(device).float()
)
nu = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/training_nu.pt"), **load_args
    ).to(device).float()
)
T_mask = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/T_mask.pt"),
        **load_args,
    ).to(device).bool()
)
C_mask = (
    torch.load(
        os.path.expanduser(f"{BASE_PATH}/{data_path}/C_mask.pt"),
        **load_args,
    ).to(device).bool()
)
M, N = training_data.shape[1:]

# Initialise the neural network
NN = NeuralNet(
    input_size=M * N,
    output_size=M * N,
    **cfg["NeuralNet"]
).to(device)

# If using a pretrained model, load model and loss time series
if cfg["Data_loading"].get("load_from_dir", None) is not None:
    NN.load_state_dict(
        torch.load(
            f"{cfg['Data_loading']['load_from_dir']}/model_trained.pt",
            map_location=torch.device(device),
            **load_args,
        )
    )
    NN.eval()
    NN.optimizer.load_state_dict(
        torch.load(
            f"{cfg['Data_loading']['load_from_dir']}/optim.pt",
            map_location=torch.device(device),
            **load_args,
        )
    )
    with open(f"{cfg['Data_loading']['load_from_dir']}/loss_dict.pickle", "rb") as file:
        LossDict = pickle.load(file)
    TrainRange = torch.load(f"{cfg['Data_loading']['load_from_dir']}/TrainRange.pt",
                            map_location=torch.device(device),
                            **load_args)
else:
    LossDict = dict(epoch=[], training_loss=[],  regularisation_error = [], prediction_error = [], test_error=[])

    # Training period
    TrainRange = cfg['Training'].get('train_period', dict(range=[0, training_data.shape[0]]))
    if isinstance(TrainRange, dict):
        TrainRange = torch.arange(*TrainRange['range']).long()
    elif isinstance(TrainRange, int):
        TrainRange = torch.multinomial(
            torch.ones(training_data.shape[0])/training_data.shape[0], TrainRange, replacement=False
        ).long()
    elif isinstance(TrainRange, float):
        TrainRange = torch.multinomial(
            torch.ones(training_data.shape[0])/training_data.shape[0], int(TrainRange * training_data.shape[0]), replacement=False
        ).long()
    else:
        TrainRange = torch.tensor(
            TrainRange, device=device
        ).long()
    if not dry_run:
        torch.save(TrainRange, f"{save_to_path}/TrainRange.pt")

# Remaining indices are test
TestRange = [i for i in range(training_data.shape[0]) if i not in TrainRange]

def epoch(_l_dict, *, batch_size, sinkhorn_kwargs, eta):

    # Track the epoch errors
    epoch_training_loss = []
    epoch_accuracy = []
    epoch_regularisation_error = []

    # Initialise the loss
    loss = torch.tensor(0.0, requires_grad=True)

    for j in TrainRange:

        dset = training_data[j]

        # Make a prediction
        _C_pred = NN(dset.reshape(M * N, )).reshape(M, N)

        # Get the marginals from the predicted cost matrix
        m, n = Sinkhorn(
            mu[j],
            nu[j],
            _C_pred,
            **sinkhorn_kwargs,
        )

        _, _, _T_pred = marginals_and_transport_plan(m, n, _C_pred, epsilon=sinkhorn_kwargs["epsilon"])

        # Track the accuracy
        epoch_accuracy.append(torch.masked_select(abs(_T_pred.clone().detach() - dset), T_mask[j]))

        # Training loss = L2 loss on non-zero edges
        training_loss = torch.nn.functional.mse_loss(
            torch.masked_select(_T_pred, T_mask[j]),
            torch.masked_select(dset, T_mask[j])
        )
        epoch_training_loss.append(training_loss.clone().detach().cpu())

        # Regulariser: cost matrix is 1 on zero-flow edges
        regularisation_err = torch.nn.functional.mse_loss(
            torch.masked_select(_C_pred, C_mask),
            torch.ones_like(torch.masked_select(_C_pred, C_mask))
        )
        epoch_regularisation_error.append(regularisation_err.clone().detach().cpu())

        # Sum the loss
        loss = loss + training_loss + eta * regularisation_err

        # Perform a gradient descent step every B iterations
        if j > 0 and (j % batch_size == 0 or j == TrainRange[-1] - 1):
            loss.backward()
            NN.optimizer.step()
            NN.optimizer.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True)

    # Track the accuracy
    _l_dict['prediction_error'].append(torch.cat(epoch_accuracy).mean().item())
    _l_dict['training_loss'].append(np.mean(epoch_training_loss))
    _l_dict['regularisation_error'].append(np.mean(epoch_regularisation_error))

    return _l_dict

def test():
    test_loss = []
    for j in TestRange:

        dset = training_data[j]

        # Make a prediction
        _C_pred = NN(training_data[j].reshape(M * N, )).reshape(M, N).detach()

        # Get the marginals from the predicted cost matrix
        m, n = Sinkhorn(
            mu[j],
            nu[j],
            _C_pred,
            **sinkhorn_kwargs,
        )

        _, _, _T_pred = marginals_and_transport_plan(m, n, _C_pred, epsilon=sinkhorn_kwargs["epsilon"])

        # Track the accuracy
        test_loss.append(
            torch.masked_select(abs(_T_pred.detach() - dset), T_mask[j])
        )
    return torch.mean(torch.cat(test_loss)).item()

# Train
num_epochs = cfg['Training']['N_epochs']
batch_size = cfg['Training']['batch_size']
sinkhorn_kwargs = cfg['Training']['sinkhorn_kwargs']
sinkhorn_kwargs['epsilon'] = torch.tensor(sinkhorn_kwargs['epsilon'], device=device).float()
eta = torch.tensor(cfg['Training']['eta'], device=device).float()
write_every = cfg['Training']['write_every']

# ----------------------------------------------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------------------------------------------
# Print table header
print(
    "{:<10}| {:<15}| {:<15} | {:<15} | {:<15} | {:<5}".format(
        "Epoch", "Training loss", "Regulariser", "Prediction err.", "Test error", "Time [s]"
    )
)
print("—" * 90)

# Train for n epochs
e0 = LossDict['epoch'][-1]+1 if LossDict['epoch'] else 1

for ep in range(e0, e0 + num_epochs):

    t0 = time.time()
    LossDict = epoch(LossDict, batch_size=batch_size, sinkhorn_kwargs=sinkhorn_kwargs, eta=eta)
    # Perform a test, if given
    if TestRange:
        test_err = test()
    else:
        test_err = torch.nan
    dt = time.time() - t0
    LossDict['epoch'].append(ep)
    LossDict['test_error'].append(test_err)

    # Print the table
    _ep_str = f"{LossDict['epoch'][-1]:<10d}"
    print(
        f"{_ep_str}|"
        f"{LossDict['training_loss'][-1]:<15.4f} | "
        f"{LossDict['regularisation_error'][-1]:<15.4f} | "
        f"{LossDict['prediction_error'][-1]:<15.4f} | "
        f"{LossDict['test_error'][-1]:<15.4f} | "
        f"{dt:<5.4f}"
    )

    # Save trained model, initial hidden state (stock), and loss by components
    if not dry_run and (ep % write_every == 0 or (ep-e0) == num_epochs - 1):
        torch.save(NN.state_dict(), f"{save_to_path}/model_trained.pt")
        torch.save(NN.optimizer.state_dict(), f"{save_to_path}/optim.pt")
        with open(f"{save_to_path}/loss_dict.pickle", "wb") as file:
            pickle.dump(
                dict((k, torch.tensor(v).flatten().cpu().tolist()) for k, v in LossDict.items()),
                file,
            )

