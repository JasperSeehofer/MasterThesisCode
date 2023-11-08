#!/bin/bash

current_dir = $PWD

cd ..
cd FastEMRIWaveforms

git pull
pip install .

cd current_dir