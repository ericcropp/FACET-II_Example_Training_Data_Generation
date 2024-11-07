To use this repo, clone it using

git clone git@github.com:ericcropp/FACET-II_Example_Training_Data_Generation.git

Then, to get the submodules to load:

git submodule update --init --recursive

Finally, to set up the conda environment on NERSC:

1) Navigate to the directory above
2) conda env create -f environment.yml
3) Follow these instructions: https://docs.nersc.gov/services/jupyter/how-to-guides/#how-to-use-a-conda-environment-as-a-python-kernel
    a) conda activate Multifidelity
    b) python -m ipykernel install --user --name env --display-name MyEnvironment
4) Go to jupyter.nersc.gov, initiate a session, and select a kernel

The components of this library are as follows:
1) Activate_CSR.tao: a file to activate CSR for Bmad, if requested
2) environment.yml: a list of packages needed to recreate the conda environmnet
3) Helper_Functions.py: A list of functions required for the jupyter notebooks to run
4) Training_Data_Example.ipynb: The main notebook for making training data.  