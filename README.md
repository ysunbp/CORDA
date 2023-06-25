# CORDA

This is the github repository for the CORDA model

## Requirements

The setup of the environment is similar to that of [Sudowoodo's](https://github.com/megagonlabs/sudowoodo/tree/main)

* Python 3.7.10
* PyTorch 1.9.0+cu111
* Transformers 4.9.2
* NVIDIA Apex

We can install the required packages with:
```console
$ conda create --name CORDA python=3.7.10
$ conda activate CORDA
$ pip install -r requirements.txt
$ git clone https://github.com/NVIDIA/apex.git
$ pip install -v --disable-pip-version-check --no-cache-dir ./apex
```

## Reproduce the results
