# Modelling Global Trade with Optimal Transport
### Data and code repository

This repository contains all the code and data required to train a neural network on FAOStat data and plot the results.
Code is presented in Jupyter notebooks. We recommend installing required packages into a virtual environment, as detailed
below. Since the datasets are large, they are stored using git lfs.

---
### Installation
> **_Note_**: The git documentation can be found [here](https://git-scm.com).
- Clone the repository into a location of your choice using `git clone`:

    ```commandline
    git clone https://github.com/ThGaskin/inverse-optimal-transport.git
    ```
    The preferred method is to clone with SSH after having [obtained an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
    – this circumvents having to enter access passwords to push changes to remote.
- Create a virtual environment and install all required packages using
  ```commandline
  pip install -r requirements.txt
- In order to save space,  datasets have been uploaded using [git lfs](https://git-lfs.github.com) (large file
storage). To download, first install lfs via
  ```commandline
  git lfs install
  ```
  This assumes you have the git command line extension installed. Then, from within the repo, do
  ```commandline
  git lfs pull
  ```
  This will pull all the datasets.

### Training and plotting a model
Train a model by running the `model` notebook — all steps are documented there. Plot the results using the `plot` 
notebook.
