# Inference package for BBHnet

This repo contains the code to run a trained BBHnet on gravitational-wave (GW)
strain file given in `.gwf` format.

Example code to run on LDG-CIT:

```
python nn_inference.py -H /home/tri.nguyen/gcp_data/strain_4k_splits/5/H1/H-H1_GWOSC_O2_4KHZ_R1-1186301952-1024.gwf -L
/home/tri.nguyen/gcp_data/strain_4k_splits/5/L1/L-L1_GWOSC_O2_4KHZ_R1-1186301952-1024.gwf -s models/model_0.pt -o
output-1186301952-1024.h5 -P temp
```
