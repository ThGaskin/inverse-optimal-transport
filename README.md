# Inverse Optimal Transport
### Inverse Optimal Transport project (Imperial/Warwick)

---
### Installation
> **_Note_**: The git documentation can be found [here](https://git-scm.com).
- Clone the repository into a location of your choice using `git clone`:

    ```commandline
    git clone https://github.com/ThGaskin/inverse-optimal-transport.git
    ```
    The preferred method is to clone with SSH after having [obtained an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
    – this circumvents having to enter access passwords to push changes to remote.
- Create a new branch using the command 
    ```commandline
    git checkout -b name_of_branch
    ```
- Push any changes to a separate branch (not to `main`) using `git add`, `git commit`, `git push`:

    ```commandline
    git add .   # This adds all changes in current directory
    git commit -m "Add a commit message describing the changes"   # Add a short commit message detailing the changes
    git push   # Push to remote
    ```
    Most IDEs (such as VisualCode or PyCharm) have a git interface, avoiding the need to use a terminal.

- Merge changes into the `main` branch by [creating a pull request](https://github.com/ThGaskin/inverse-optimal-transport/compare).
- Add any required packages for a virtual environment to the `requirements.txt` file.
