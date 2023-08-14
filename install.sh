#!/bin/bash

echo "Installing master thesis code dependencies"
echo "install FastEMRIWaveforms (few) module..."

read -p "Enter Anaconda environment name: " env_name

git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git

cd FastEMRIWaveforms
bash install.sh env_name=$env_name

echo  "few module installation was succesful"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $env_name

cd ..
conda env update --file environment.yml
