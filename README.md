# Master Thesis

## Package Management (Anaconda)

To create an Anaconda environment for this module one can run the bash script `install.sh`

```bash
bash install.sh
```

this will ask you to enter an environment name and then install the [few package](https://bhptoolkit.org/FastEMRIWaveforms/html/index.html). For further information see [this chapter](#emri-waveform-python-package).

### Update the environment

Use the following commands to update the environment

```terminal
conda env update --name <env-name> --file environment.yml
```

### EMRI waveform python package

We use the [python package](https://bhptoolkit.org/FastEMRIWaveforms/html/index.html).

### Use the package

To use the package activate the created environment, open a terminal and direct to the root directory. Then run

```Python
python -m master_thesis_code
```
