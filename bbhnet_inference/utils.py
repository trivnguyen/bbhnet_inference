
import os
import h5py
import logging

import numpy as np
import scipy.signal as sig

from . import plot_utils

logger = logging.getLogger(__name__)

def as_stride(x, input_size, step, shift=0):
    ''' Divide input time series into overlapping chunk '''
    
    if shift != 0:
        x = np.roll(x, shift)
    
    noverlap = input_size  - step
    N_sample = (x.shape[-1] - noverlap) // step

    shape = (N_sample, input_size)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    
    return result

def pearson_shift(x, y, shift):
    ''' Calculate the Pearson correlation coefficient between x and y
    for each array shift value from [-shift, shift]. 
    '''
    
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = ((x**2).sum(1) * (y**2).sum(1))**0.5

    corr = np.zeros((x.shape[0], shift * 2))
    for i, s in enumerate(range(-shift, shift)):
        xr = np.roll(x, s, axis=1)
        corr[:, i] = (xr * y).sum(1) / denom
    return corr

# def pearson_shift(x, y, shift):
#     ''' Calculate the Pearson correlation coefficient between x and y
#     for each array shift value from [-shift, shift]. 
#     '''
#     corr = []
#     shift_arr = np.arange(-shift, shift)
#     for s in shift_arr:
#         x_roll = np.roll(x, s, axis=1)        
#         mx = x_roll.mean(1, keepdims=True)
#         my = y.mean(1, keepdims=True)
#         xm, ym = x_roll - mx, y - my
#         r_num = np.sum(xm * ym, axis=1)
#         r_den = np.sqrt(np.sum(xm**2, axis=1) * np.sum(ym**2, axis=1))
#         r = r_num / r_den
#         corr.append(r)
#     corr = np.stack(corr).T
#     return corr

def get_PSD(file, strain, nperseg=256, duty_cycle=1, plot=False, plot_dir=''):
    ''' Function to get PSD '''
    
    # Read in PSD file if exist
    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            prev_GPSstart = f.attrs['GPS-start']
            prev_GPSend = f.attrs['GPS-end']
            prev_T_Pxx = f.attrs['Duration']
            
            if strain.t0.value == prev_GPSend:
                logger.info('Update old PSD with current PSD')
                prev_Pxx = f['Pxx'][:]
            else:
                logger.warning('Gap detected: compute new PSD')
                prev_Pxx = None
    else:
        logger.warning('PSD file not found: compute new PSD')
        prev_Pxx = None
        
    # Calculate current PSD and update
    if prev_Pxx is None:
        freqs, Pxx = sig.welch(strain.value, fs=strain.sample_rate.value, nperseg=nperseg)
        T_Pxx = strain.duration.value * duty_cycle
    else:
        freqs, curr_Pxx = sig.welch(strain.value, fs=strain.sample_rate.value, nperseg=nperseg)
        curr_T_Pxx = strain.duration.value * duty_cycle
        Pxx = (curr_Pxx * curr_T_Pxx + prev_Pxx * prev_T_Pxx) / (curr_T_Pxx + prev_T_Pxx)
        T_Pxx = prev_T_Pxx + curr_T_Pxx

    GPSend = strain.t0.value + strain.duration.value
        
    # Rewrite PSD file
    if prev_Pxx is None:
        with h5py.File(file, 'w') as f:
            f.attrs.update({
                'GPS-start': strain.t0.value,
                'GPS-end': GPSend,
                'Duration': T_Pxx
            })
            f.create_dataset('freqs', data=freqs, chunks=True)
            f.create_dataset('Pxx', data=Pxx, chunks=True)
    else:
        with h5py.File(file, 'a') as f:
            f.attrs.modify('GPS-end', GPSend)
            f.attrs.modify('Duration', T_Pxx)
            f['freqs'][...] = freqs
            f['Pxx'][...] = Pxx
    
    # Plot PSD to file
    if plot:
        if prev_Pxx is None:
            GPSstart = strain.t0.value
        else:
            GPSstart = prev_GPSstart
        os.makedirs(plot_dir, exist_ok=True)
        
        GPSstart = int(GPSstart)
        GPSend = int(GPSend)
        plot_file = os.path.join(plot_dir, '{}-{}.png'.format(GPSstart, GPSend))
        plot_utils.plotPxx(Pxx, freqs, out=plot_file, title='PSD-{}-{}'.format(GPSstart, GPSend))
    
    return freqs, Pxx
