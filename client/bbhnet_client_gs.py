#!/usr/bin/env python
# coding: utf-8

import argparse
from functools import partial
import os
import sys
import time
import h5py
import glob
import logging

import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

from google.cloud import storage

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from bbhnet_inference import utils
from bbhnet_inference import plot_utils

# global constant
FLAGS = None
MODEL_NAME = 'bbhnet'
BUCKET_NAME = 'ligo-o2-august-bbhnet'
RATE = 1024
OUT_RATE = 16
STEP = int(RATE / OUT_RATE)
INPUT_DURATION = 1
PEARSON_SHIFT = 40
NPERSEG = int(max(2 * RATE, 2048))
INPUT_SIZE = int(INPUT_DURATION * RATE)
SERVER_IP = '34.123.230.170'


def read_preprocess_data(H1_file, L1_file, out_dir, nptype=np.float32):
    ''' Read input strain and preprocess '''

    # read in strain file in GWPY frame format
    # check for NaN value, replace NaN value with median of timeseries
    # TODO: figure out a better way
    H1_strain4k = TimeSeries.read(H1_file, 'H1:GWOSC-4KHZ_R1_STRAIN')
    H1_duty_cycle = (1 - np.isnan(H1_strain4k).sum() / len(H1_strain4k))
    if np.any(np.isnan(H1_strain4k)):
        logger.warning('NaN found in H1 strain, replacing with median value')
        H1_strain4k = np.nan_to_num(H1_strain4k, H1_strain4k.median())

    L1_strain4k = TimeSeries.read(L1_file, 'L1:GWOSC-4KHZ_R1_STRAIN')
    L1_duty_cycle = (1 - np.isnan(L1_strain4k).sum() / len(L1_strain4k))
    if np.any(np.isnan(L1_strain4k)):
        logger.warning('NaN found in L1 strain, replacing with median value')
        L1_strain4k = np.nan_to_num(L1_strain4k, L1_strain4k.median)

    # resample strain to 1kHz
    H1_strain1k = H1_strain4k.resample(RATE)
    L1_strain1k = L1_strain4k.resample(RATE)

    # handle PSD
    os.makedirs(os.path.join(out_dir, 'PSD/H1'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'PSD/L1'), exist_ok=True)
    freqs, H1_Pxx = utils.get_PSD(
        os.path.join(out_dir, 'PSD/H1/PSD.h5'), H1_strain1k,
        nperseg=NPERSEG, duty_cycle=H1_duty_cycle)

    freqs, L1_Pxx = utils.get_PSD(
        os.path.join(out_dir, 'PSD/L1/PSD.h5'), L1_strain1k,
        nperseg=NPERSEG, duty_cycle=L1_duty_cycle)

    # whiten strain using estimated PSD
    H1_ASD = FrequencySeries(np.sqrt(H1_Pxx), df=RATE / NPERSEG)
    L1_ASD = FrequencySeries(np.sqrt(L1_Pxx), df=RATE / NPERSEG)
    H1_strain1k_whiten = H1_strain1k.whiten(asd=H1_ASD)
    L1_strain1k_whiten = L1_strain1k.whiten(asd=L1_ASD)

    # divide strain time series into shorter, overlapping chunks
    H1_data = utils.as_stride(H1_strain1k_whiten, INPUT_SIZE, STEP)
    L1_data = utils.as_stride(L1_strain1k_whiten, INPUT_SIZE, STEP,
                              shift=int(RATE * FLAGS.time_shift))
    strain_data = np.stack([H1_data, L1_data], 1)

    # compute the correlation
    correlation = utils.pearson_shift(H1_data, L1_data, shift=PEARSON_SHIFT)

    # compute starting GPS time of each sample
    gps_start = H1_strain1k.t0.value + np.arange(len(strain_data)) * STEP / RATE

    # convert all data into type
    strain_data = strain_data.astype(nptype)
    correlation = correlation.astype(nptype)

    # return dataset
    return strain_data, gps_start, correlation


# parse cmd args
def parse_cmd():
    parser = argparse.ArgumentParser()

    # input data args
    parser.add_argument('-dt', '--time-shift', type=float, default=0,
                        help='Time shift of Livingston strain w.r.t Hanford [seconds]')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='Path to output directory')
    parser.add_argument('-i', '--job-id', type=int, required=True,
                        help='Mini job index')

    # triton client args
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        default=False, help='Enable verbose output')
    parser.add_argument('-b', '--batch-size', type=int, required=False, default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-u', '--url', type=str, required=False, default='{}:8001'.format(SERVER_IP),
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-t', '--client-timeout', type=float, required=False, default=None,
                        help='Client timeout in seconds. Default is None.')

    return parser.parse_args()

# define the callback function
def callback(user_data, result, error):
    if error:
        user_data.append(error)
    else:
        user_data.append(result)


if __name__ == '__main__':

    FLAGS = parse_cmd()

    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # set up triton client
    try:
        logger.info('URL: {}'.format(FLAGS.url))
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        logger.info("client creation failed: {}".format(str(e)))
        sys.exit(1)

    if not triton_client.is_model_ready(MODEL_NAME):
        logger.info('model {} is not ready'.format(MODEL_NAME))
        sys.exit(1)

    # set up Google Storage client and bucket
    gs_client = storage.Client()
    bucket = gs_client.get_bucket(BUCKET_NAME)

    H1_blobs = list(bucket.list_blobs(prefix='strain_4k_splits/{}/H1'.format(FLAGS.job_id)))
    L1_blobs = list(bucket.list_blobs(prefix='strain_4k_splits/{}/L1'.format(FLAGS.job_id)))

    # set up output directory and local storage for input strain
    logger.info('Writing output to {}'.format(FLAGS.out_dir))
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.out_dir, 'out'), exist_ok=True)

    # client inputs and outptus
    inputs = [None, None]
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('OUTPUT__0'))

    # iterate over all pairs of Hanford and Livingston
    for i, (H1_blob, L1_blob) in enumerate(zip(H1_blobs, L1_blobs)):

        # download bucket data into local storage temporarily
        H1_file = os.path.join(FLAGS.out_dir, '{}'.format(os.path.basename(H1_blob.name)))
        L1_file = os.path.join(FLAGS.out_dir, '{}'.format(os.path.basename(L1_blob.name)))
        H1_blob.download_to_filename(H1_file)
        L1_blob.download_to_filename(L1_file)

        try:
            # check if GPS times match
            H1_GPS_str = os.path.splitext(os.path.basename(H1_file))[0].split('-')[2:]
            L1_GPS_str = os.path.splitext(os.path.basename(L1_file))[0].split('-')[2:]
            if H1_GPS_str != L1_GPS_str:
                raise ValueError('Hanford and Livingston GPS time do not match')
            GPS_str = H1_GPS_str

            # read in and preprocess input GW strain data
            logger.info('reading and preprocessing frame')
            logger.info('Hanford strain   : {}'.format(H1_file))
            logger.info('Livingston strain: {}'.format(L1_file))
            strain_data, gps_start, correlation = read_preprocess_data(
                H1_file, L1_file, FLAGS.out_dir)

            # list to hold the results of inference
            user_data = []
            output_data = []

            n_batch = int(np.ceil(len(strain_data) / FLAGS.batch_size))
            logger.info('Number of batches: {:d}'.format(n_batch))
            logger.info('Sending request')

            for i_batch in range(n_batch):
                start = i_batch * FLAGS.batch_size
                end = (i_batch + 1) * FLAGS.batch_size

                batched_strain = strain_data[start: end]
                batched_corr = correlation[start: end]

                inputs[0] = grpcclient.InferInput('INPUT__0', batched_strain.shape, 'FP32')
                inputs[1] = grpcclient.InferInput('INPUT__1', batched_corr.shape, 'FP32')
                inputs[0].set_data_from_numpy(batched_strain)
                inputs[1].set_data_from_numpy(batched_corr)

                # Inference call
                try:
                    triton_client.async_infer(
                        model_name=MODEL_NAME,
                        inputs=inputs,
                        callback=partial(callback, user_data),
                        outputs=outputs,
                        client_timeout=FLAGS.client_timeout)
                except InferenceServerException as e:
                    logger.info("inference failed: {}".format(str(e)))
                    sys.exit(1)

                # Wait until the results are available in user_data
                time_out = 20
                while((len(user_data) < i_batch + 1) and time_out > 0):
                    time_out = time_out - 1
                    time.sleep(1)

                # Check for errors
                if type(user_data[i_batch]) == InferenceServerException:
                    logger.info(user_data[i_batch])
                    sys.exit(1)

                # add output data to list
                output_data.append(user_data[i_batch].as_numpy('OUTPUT__0'))

            logger.info("PASS: Async infer")

            # write output data to HDF5 file
            output_data = np.concatenate(output_data)

            # write output data into file
            out_file = os.path.join(FLAGS.out_dir, 'out/out_{}-{}.hdf5'.format(GPS_str[0], GPS_str[1]))
            logger.info('writing output data into {}'.format(out_file))
            with h5py.File(out_file, 'w') as f:
                f.attrs.update({
                    'size': len(output_data),
                    'H1-file': H1_file,
                    'L1-file': L1_file
                })
                f.create_dataset('out', data=output_data)
                f.create_dataset('GPSstart', data=gps_start)
        finally:
            # make sure to delete input data after each inference
            logger.info('Remove input data')
            os.remove(H1_file)
            os.remove(L1_file)

