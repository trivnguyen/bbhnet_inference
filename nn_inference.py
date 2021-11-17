#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import argparse
import logging

# set up logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


import numpy as np
import scipy
import scipy.signal as sig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gwpy
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

from inference import utils
from inference import plot_utils
from inference.classifiers import cnn_large

# Define some global variables
RATE = 1024
INPUT_DUR = 1
NPERSEG = int(max(2 * RATE, 2048))
INPUT_SIZE = int(INPUT_DUR * RATE)
STEP = 64
BATCH_SIZE = 256
PEARSON_SHIFT = 40

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Define argument parser
def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # io arguments
    parser.add_argument('--H1-file', required=True, help='Path to H1 strain GWPY frame')
    parser.add_argument('--L1-file', required=True, help='Path to L1 strain GWPY frame')
    parser.add_argument('--state', required=True, help='Path to NN state dictionary')
    parser.add_argument('--outfile', required=True, help='Path to output in HDF5 format')
    parser.add_argument('--PSD-dir', required=True, help='Path to PSD directory')
    parser.add_argument('--log', help='Path to output log file')
    parser.add_argument('--stream-output', action='store_true', help='Enable to stream output to stdout')
    
    # data arguments
    parser.add_argument('--time-shift', type=float, default=0,
                        help='Time shift of Livingston strain w.r.t Hanford [seconds]')
    
    return parser.parse_args()

if __name__ == "__main__":
    ''' Run NN over strain data '''

    params = parse_args()  # parse command line argument
    device = torch.device(DEVICE) # set GPU/CPU device
    
    # add handlers to logger
    if params.stream_output:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if params.log is not None:
        file_handler = logging.FileHandler(params.log, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    
    # Read in strain file in GWPY frame format 
    # check for NaN value, replace NaN value with median of timeseries
    # TODO: figure out a better way
    logger.info('Reading in {}'.format(params.H1_file))
    H1_strain4k = TimeSeries.read(params.H1_file, 'H1:GWOSC-4KHZ_R1_STRAIN')
    H1_duty_cycle = (1 - np.isnan(H1_strain4k).sum() / len(H1_strain4k))
    if np.any(np.isnan(H1_strain4k)):
        logger.warning('NaN found in H1 strain, replacing with median value')
        H1_strain4k = np.nan_to_num(H1_strain4k, H1_strain4k.median())

    logger.info('Reading in {}'.format(params.L1_file))
    L1_strain4k = TimeSeries.read(params.L1_file, 'L1:GWOSC-4KHZ_R1_STRAIN')
    L1_duty_cycle = (1 - np.isnan(L1_strain4k).sum() / len(L1_strain4k))
    if np.any(np.isnan(L1_strain4k)):
        logger.warning('NaN found in L1 strain, replacing with median value')
        L1_strain4k = np.nan_to_num(L1_strain4k, L1_strain4k.median)        

    # Resample strain to 1kHz
    H1_strain1k = H1_strain4k.resample(RATE)
    L1_strain1k = L1_strain4k.resample(RATE)
    
    # Handle PSD
    # create dir to store PSD for Handford and Livingston separated
    os.makedirs(os.path.join(params.PSD_dir, 'H1'), exist_ok=True)
    os.makedirs(os.path.join(params.PSD_dir, 'L1'), exist_ok=True)
    freqs, H1_Pxx = utils.get_PSD(
        os.path.join(params.PSD_dir, 'H1/PSD.h5'), H1_strain1k, nperseg=NPERSEG, duty_cycle=H1_duty_cycle,
        plot=True, plot_dir=os.path.join(params.PSD_dir, 'PSD/H1/plots'))
    freqs, L1_Pxx = utils.get_PSD(
        os.path.join(params.PSD_dir, 'L1/PSD.h5'), L1_strain1k, nperseg=NPERSEG, duty_cycle=L1_duty_cycle,
        plot=True, plot_dir=os.path.join(params.PSD_dir, 'PSD/L1/plots'))

    # Whiten strain
    H1_ASD = FrequencySeries(np.sqrt(H1_Pxx), df=RATE / NPERSEG)
    L1_ASD = FrequencySeries(np.sqrt(L1_Pxx), df=RATE / NPERSEG)
    H1_strain1k_whiten = H1_strain1k.whiten(asd=H1_ASD)
    L1_strain1k_whiten = L1_strain1k.whiten(asd=L1_ASD)

    # Create strain sample for H1 and L1
    logger.info('Creating dataset')
    if params.time_shift != 0:
        logger.info('Shifting Livingston strain by {} secs or {} samples'.format(
            params.time_shift, int(RATE * params.time_shift)))
    
    H1_data = utils.as_stride(H1_strain1k_whiten, INPUT_SIZE, STEP)
    L1_data = utils.as_stride(L1_strain1k_whiten, INPUT_SIZE, STEP, 
                              shift=int(RATE * params.time_shift))
    data = np.stack([H1_data, L1_data], 1)
    # also compute the Pearson correlation coeff
    corr = utils.pearson_shift(H1_data, L1_data, shift=PEARSON_SHIFT)

    # Create dataloader 
    if DEVICE == 'cuda':
        pin_memory = True
    else:
        pin_memory = False
    dataloader = DataLoader(list(zip(data, corr)), batch_size=BATCH_SIZE, 
                            pin_memory=pin_memory, num_workers=4)

    # read in NN state
    logger.info('Reading in NN state from {}'.format(params.state))
    net = cnn_large.Classifier(input_shape=(2, INPUT_SIZE), num_classes=2, 
                               corr_dim=corr.shape[-1])
    net.to(device)
    net.load_state_dict(torch.load(params.state, map_location=device))
    net.eval()

    # start NN inference
    out = []
    with torch.no_grad():
        for i, d in enumerate(dataloader):
            x, xcorr = d
            x = x.float().to(device)
            xcorr = xcorr.float().to(device)

            yhat = net(x, xcorr)
            out.append(yhat.data.cpu().numpy())
    out = np.concatenate(out).ravel()
    GPSstart = H1_strain1k.t0.value + np.arange(len(out)) * STEP / RATE

    logger.info('Writing output into {}'.format(params.outfile))
    with h5py.File(params.outfile, 'w') as f:
        f.attrs.update({
            'size': len(out),
            'state': params.state,
            'H1-file': params.H1_file,
            'L1-file': params.L1_file,
        })
        f.create_dataset('out', data=out, chunks=True)
        f.create_dataset('GPSstart', data=GPSstart, chunks=True)
