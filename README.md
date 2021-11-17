# Inference package for BBHnet

This repo contains the code to run a trained BBHnet on gravitational-wave (GW)
strain file given in `.gwf` format.

The script to run BBHnet over a given pair of Hanford and Livingston frame files
is `inference.py`.

Required arguments:
```
  -h, --help            show this help message and exit
  -H H1_FILE, --H1-file H1_FILE
                        Path to H1 strain GWPY frame
  -L L1_FILE, --L1-file L1_FILE
                        Path to L1 strain GWPY frame
  -s STATE, --state STATE
                        Path to NN state dictionary
  -o OUTFILE, --outfile OUTFILE
                        Path to output in HDF5 format
  -P PSD_DIR, --PSD-dir PSD_DIR
                        Path to PSD directory
  -l LOG, --log LOG     Path to output log file
```
Optional arguments:
```
  -dt TIME_SHIFT, --time-shift TIME_SHIFT
                        Time shift of Livingston strain w.r.t Hanford in secs. Default is 0.
```

Below is the example code to run on LDG-CIT. The chosen frame contains a confirmed GW detection, GW170809.

```
python inference.py -H /home/tri.nguyen/gcp_data/strain_4k_splits/5/H1/H-H1_GWOSC_O2_4KHZ_R1-1186301952-1024.gwf -L
/home/tri.nguyen/gcp_data/strain_4k_splits/5/L1/L-L1_GWOSC_O2_4KHZ_R1-1186301952-1024.gwf -s models/model_0.pt -o
output-1186301952-1024.h5 -P temp
```
