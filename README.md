# Master Thesis

## Package Management (Anaconda + Poetry)

Using Anaconda it is very simple to install all required packages. 

### Create a new Anaconda environment
You can create an Anaconda environment with a name of your choice, then activate it and install all packages via poetry by executing the following commands:

```
conda create --name my_project_env --file conda-linux-64.lock
conda activate my_project_env
poetry install
```

### Update the environment
Use the following commands to update the environment
```
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-linux-64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```