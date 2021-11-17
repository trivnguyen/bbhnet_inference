#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import argparse
import logging

import gwpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

# import BBHnet inference package
from bbhnet_inference import utils, plot_utils
from bbhnet_inference.classifiers import cnn_large


# Global variables
SAMPLE_RATE = 1024
OUTPUT_SAMPLE_RATE = 16
NPERSEG = 2048
INPUT_SIZE = int(SAMPLE_RATE * 1)
STEP = int(SAMPLE_RATE / OUTPUT_SAMPLE_RATE)
SHIFT = 40
BATCH_SIZE = 1000
FLAGS = None

# Function to set up print output logger
def set_up_logger(FLAGS):
    ''' Set up logger '''

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # add file handler if enable
    if FLAGS.log is not None:
        file_handler = logging.FileHandler(FLAGS.log, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Function to read and preprocess strain data
def read_preprocess_data(FLAGS):
    ''' read and preprocess data '''

    # Read in strain file in .gwf frame format
    H1_strain = TimeSeries.read(FLAGS.H1_file, 'H1:GWOSC-4KHZ_R1_STRAIN')
    L1_strain = TimeSeries.read(FLAGS.L1_file, 'L1:GWOSC-4KHZ_R1_STRAIN')

    # Replace all NaN values with median, and calculate duty cycle
    H1_dc = (1 - np.isnan(H1_strain).sum() / len(H1_strain))
    L1_dc = (1 - np.isnan(L1_strain).sum() / len(L1_strain))
    H1_strain = np.nan_to_num(H1_strain, H1_strain.median())
    L1_strain = np.nan_to_num(L1_strain, L1_strain.median())

    # Resample strain to 1024 Hz
    H1_strain = H1_strain.resample(SAMPLE_RATE)
    L1_strain = L1_strain.resample(SAMPLE_RATE)

    # Handle PSD: calculate the running psd based on previous frames
    os.makedirs(FLAGS.PSD_dir, exist_ok=True)
    H1_psd_file = os.path.join(FLAGS.PSD_dir, 'H1_psd.h5')
    L1_psd_file = os.path.join(FLAGS.PSD_dir, 'L1_psd.h5')
    _, H1_psd = utils.get_PSD(H1_psd_file, H1_strain,
                              nperseg=NPERSEG, duty_cycle=H1_dc)
    _, L1_psd = utils.get_PSD(L1_psd_file, L1_strain,
                              nperseg=NPERSEG, duty_cycle=H1_dc)
    H1_psd = FrequencySeries(H1_psd, df=SAMPLE_RATE / NPERSEG)
    L1_psd = FrequencySeries(L1_psd, df=SAMPLE_RATE / NPERSEG)

    # Whiten strain using running psd
    H1_strain = H1_strain.whiten(asd=np.sqrt(H1_psd))
    L1_strain = L1_strain.whiten(asd=np.sqrt(L1_psd))

    # Convert whitened strain into a data format that the BBHnet can read
    # we divide the strain into overlapping segments
    # in addition, we also compute the Pearson correlation array
    H1_data = utils.as_stride(H1_strain, INPUT_SIZE, STEP)
    L1_data = utils.as_stride(
        L1_strain, INPUT_SIZE, STEP, shift=int(SAMPLE_RATE * FLAGS.time_shift))
    data = np.stack([H1_data, L1_data], 1)
    correlation = utils.pearson_shift(H1_data, L1_data, shift=SHIFT)

    # also return the GPS time of each data sample
    times = H1_strain.t0.value + np.arange(len(data)) / OUTPUT_SAMPLE_RATE

    return data, correlation, times

# Define argument parser
def parse_args():

    parser = argparse.ArgumentParser()

    # io arguments
    parser.add_argument('-H', '--H1-file', required=True, help='Path to H1 strain GWPY frame')
    parser.add_argument('-L', '--L1-file', required=True, help='Path to L1 strain GWPY frame')
    parser.add_argument('-s', '--state', required=True, help='Path to NN state dictionary')
    parser.add_argument('-o', '--outfile', required=True, help='Path to output in HDF5 format')
    parser.add_argument('-P', '--PSD-dir', required=True, help='Path to PSD directory')
    parser.add_argument('-l', '--log', help='Path to output log file')

    # data arguments
    parser.add_argument('-dt', '--time-shift', type=float, default=0,
                        help='Time shift of Livingston strain w.r.t Hanford [seconds]')

    return parser.parse_args()

if __name__ == "__main__":
    ''' Read in a trained BBHnet and run over given GW frames from Hanford and Livingston '''

    # Parse command line argument and set up output logger
    FLAGS = parse_args()
    logger = set_up_logger(FLAGS)

    # Read in and preprocessing
    logger.info('Read in and preprocess strain')
    data, correlation, times = read_preprocess_data(FLAGS)

    # Import a trained BBHnet
    logger.info('Importing trained BBHnet from {}'.format(FLAGS.state))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = cnn_large.Classifier((2, INPUT_SIZE), corr_dim=SHIFT*2)
    net.load_state_dict(torch.load(FLAGS.state, map_location='cpu'))
    net.to(device)
    net.eval()

    # Because the data may be too large to fit in memory (esp for GPU)
    # we use torch.utils.data.DataLoader to load in the data
    data_loader = DataLoader(list(zip(data, correlation)), batch_size=BATCH_SIZE)

    # Start inference
    output = []
    with torch.no_grad():
        for batch in data_loader:
            # type conversion and move to GPU
            x = batch[0].float().to(device)
            x_corr = batch[1].float().to(device)

            # inference
            output_batch = net(x, x_corr)
            output.append(output_batch.cpu().numpy())
    output = np.concatenate(output)
    output = output.ravel()   # change from [N, 1] to [N, ]

    # Write output file
    logger.info('Writing output into {}'.format(FLAGS.outfile))
    with h5py.File(FLAGS.outfile, 'w') as f:
        f.attrs.update({
            'size': len(output),
            'state': FLAGS.state,
            'H1-file': FLAGS.H1_file,
            'L1-file': FLAGS.L1_file,
        })
        f.create_dataset('out', data=output)
        f.create_dataset('GPSstart', data=times)

