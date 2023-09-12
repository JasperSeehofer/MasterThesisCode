# Master Thesis

## Package Management (Anaconda)
To create an Anaconda environment for this module one can run the bash script `install.sh`

```
bash install.sh
```

this will ask you to enter an environment name and then install the [few package](https://bhptoolkit.org/FastEMRIWaveforms/html/index.html). For further information see [this chapter](#emri-waveform-python-package). 

### Update the environment
Use the following commands to update the environment
```
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-linux-64.lock
```

### EMRI waveform python package
We use the [python package](https://bhptoolkit.org/FastEMRIWaveforms/html/index.html).