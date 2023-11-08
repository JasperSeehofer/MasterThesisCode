#!/bin/bash

current_dir=$PWD

echo "Install dependencies..."
echo "install FastEMRIWaveforms (few) module..."

read -p "Enter Anaconda environment name: " env_name

cd ..
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git

cd FastEMRIWaveforms
bash install.sh env_name=$env_name

echo  "few module installation was succesful"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $env_name

cd $current_dir
conda env update --name $env_name --file environment.yml
