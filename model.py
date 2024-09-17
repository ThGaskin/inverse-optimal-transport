import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import xarray as xr

import nn as base

# Check data overwrite is ok
if input("Warning! Running this file will overwrite the existing results saved in 'NN_samples'. "
             "Do you wish to proceed? (Y/N) ") != 'Y':
    exit()

# Load the lookup table, containing coordinates and ISO codes
lookup_table = pd.read_csv("data/lookup_table.csv", index_col=0)

def _get_ISO(c):
    if c == 'Other':
        print(f"Missing ISO: {c}")
        return c
    iso_c = lookup_table.loc[c, 'Alpha-3 code']
    if str(iso_c) == 'nan':
        print(f"Missing ISO: {c}")
        return c
    else:
        return iso_c

DSETS = [
    "Barley_pooled_0.99_2000-2022",
    "Beef_pooled_0.99_2000-2022",
    "Corn_pooled_0.99_2000-2022",
    "Dairy_all_pooled_0.99_2000-2022",
    "Sugar_products_pooled_0.99_2000-2022",
    "Tomatoes_pooled_0.99_2000-2022",
    "Vegetables_pooled_0.99_2000-2022",
    "Soya_beans_pooled_0.99_2000-2022",
]

for DSET_IDX, DSET in enumerate(DSETS):

    # Load in the FAOStat data
    FAO_data = xr.open_dataarray(f"data/FAO_data/{DSET}.nc")

    # Convert the country names to an ISO3 code for easier labelling and selection
    FAO_data = FAO_data.assign_coords({
        "Source": [_get_ISO(c) for c in FAO_data.coords["Source"].data],
        "Destination": [_get_ISO(c) for c in FAO_data.coords["Destination"].data]
    })

    fao_data = FAO_data.sel({"Element": "Quantity, t"}).stack({'idx': ['Year', 'Reporter']}).drop_vars(
        ['Year', 'Reporter']).transpose('idx', ...)

    training_data = torch.from_numpy(
        fao_data.data
    ).float()

    M, N = training_data.shape[1:]

    # Calculate the marginals
    mu, nu = torch.nansum(training_data, dim=-1, keepdim=True), torch.nansum(training_data, dim=1, keepdim=True)

    mask = training_data > 0

    # Get NAN mask
    common_countries = np.intersect1d(fao_data.coords['Source'].data, fao_data.coords['Destination'].data)
    common_idx = torch.tensor([(np.where(fao_data.coords["Source"].data == c)[0].item(),
                                np.where(fao_data.coords["Destination"].data == c)[0].item()) for c in
                               common_countries]).transpose(1, 0)
    common_countries = torch.ones(*mask.shape[1:], dtype=torch.bool)
    common_countries[*common_idx] = False
    training_data = torch.where(training_data >= 0, training_data, 0)
    training_data[:, *common_idx] = 0
    mask = mask | ~common_countries

    # Set up neural network
    NN = base.NeuralNet(
        input_size=M * N,
        output_size=M * N,
        num_layers=5,
        nodes_per_layer=dict(default=60),
        activation_funcs=dict(default='tanh', layer_specific={-1: 'sigmoid'}),
        biases=dict(default=None),
        optimizer='Adam',
        learning_rate=0.002
    )

    # Train
    num_epochs = 10000
    batch_size = 23

    # Kwargs for the Sinkhorn algorithm
    sinkhorn_kwargs = dict(max_iter=100, tolerance=1e-5, epsilon=0.15)

    loss_ts = []
    # Train
    for it in (pbar := tqdm(range(num_epochs))):

        epoch_loss = []
        epoch_accuracy = []
        loss = torch.tensor(0.0, requires_grad=True)
        for j, dset in enumerate(training_data):

            # Make a prediction
            _C_pred = NN(dset.reshape(M * N, )).reshape(M, N)

            # Get the marginals from the predicted cost matrix
            m, n = base.Sinkhorn(
                mu[j],
                nu[j],
                _C_pred,
                **sinkhorn_kwargs,
            )

            _, _, _T_pred = base.marginals_and_transport_plan(m, n, _C_pred, epsilon=sinkhorn_kwargs["epsilon"])

            # Training loss = L2 loss on non-zero edges
            training_loss = torch.nn.functional.mse_loss(_T_pred[mask[j]], dset[mask[j]])

            # Constrain the cost matrix to have column sum = 1
            loss = loss + training_loss + torch.nn.functional.mse_loss(
                torch.where(common_countries, _C_pred, 0).sum(dim=1, keepdim=False), torch.ones(M))

            # Perform a gradient descent step every B iterations
            if j % batch_size == 0 or j == training_data.shape[0] - 1:
                loss.backward()
                NN.optimizer.step()
                NN.optimizer.zero_grad()
                loss = torch.tensor(0.0, requires_grad=True)

            # Track the accuracy
            epoch_loss.append(torch.mean(
                abs(_T_pred.detach()[mask[j]] - dset[mask[j]]) / torch.where(dset[mask[j]] > 0, dset[mask[j]], 1)))

        loss_ts.append(np.mean(epoch_loss).item())
        pbar.set_description(f"Training dset {DSET_IDX+1}/{len(DSETS)}  |  Current mean error: {np.round(loss_ts[-1], 2)}")

    # Number of samples for each year
    N_samples = 1000

    # xr.Dataset containing the data
    _empty = np.zeros((N_samples, len(FAO_data.coords['Year'].data), M, N))
    samples = xr.Dataset(
        data_vars=dict(
            C=(["Sample", "Year", "Source", "Destination"], np.zeros_like(_empty)),
            T=(["Sample", "Year", "Source", "Destination"], np.zeros_like(_empty)),
            T_pred=(["Sample", "Year", "Source", "Destination"], np.zeros_like(_empty))
        ),
        coords={"Sample": np.arange(N_samples),
                "Year": FAO_data.coords["Year"].data,
                "Source": FAO_data.coords["Source"].data,
                "Destination": FAO_data.coords["Destination"].data}
    )

    for j in tqdm(range(N_samples), desc=f"Sampling dset {DSET_IDX+1}/{len(DSETS)}"):

        for year in FAO_data.coords['Year'].data:
            # Pick random entries along the 'Reporter' axis and fill NaN values
            _dset = torch.from_numpy(
                np.apply_along_axis(np.random.choice, 0, FAO_data.sel({"Element": "Quantity, t", "Year": year}).data)
            ).float()

            _C_pred = NN(_dset.reshape(M * N, )).reshape(M, N).detach()

            # Get the marginals from the predicted cost matrix
            m, n = base.Sinkhorn(
                _dset.sum(dim=1, keepdim=True),
                _dset.sum(dim=0, keepdim=True),
                _C_pred,
                **sinkhorn_kwargs,
            )

            _, _, _T_pred = base.marginals_and_transport_plan(m, n, _C_pred, epsilon=sinkhorn_kwargs["epsilon"])

            _dset[_dset == 0] = torch.nan
            _C_pred[_dset == 0] = torch.inf

            samples["C"].loc[{"Year": year, "Sample": j}] = _C_pred
            samples["T"].loc[{"Year": year, "Sample": j}] = _dset
            samples["T_pred"].loc[{"Year": year, "Sample": j}] = _T_pred

    # Save the sample statistics (careful -- this will overwrite the existing datasets)
    xr.Dataset(dict(mean=samples.to_array().mean("Sample"), std=samples.to_array().std("Sample"))).to_netcdf(f"data/NN_Samples/{DSET}_samples_stats.nc")