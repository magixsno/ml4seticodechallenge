from __future__ import division

import astropy
import cStringIO
import glob
import ibmseti
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import obspy
import os
import requests
import scipy
import zipfile

from obspy.core import read
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger 
from astropy.convolution import convolve, Box1DKernel
from scipy import ndimage

'''
IF YOU ARE RUNNING A NEWER VERSION OF MATPLOTLIB THIS CODE WILL NOT RUN:

pip install 'matplotlib==1.4.3'

Then run this code with:

python processors/sanjoy_seismogram.py
'''


# Find the data in the zip file
mydatafolder = 'data'
zz = zipfile.ZipFile(os.path.join(mydatafolder, 'basic4.zip'))
basic4list = zz.namelist()
firstfile = basic4list[1400]

# Read data into ibmseti object
aca = ibmseti.compamp.SimCompamp(zz.open(firstfile).read())

# Get the raw complex data
complex_data = aca.complex_data()
complex_data = complex_data.reshape(32, 6144)
cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
spectrogram = np.abs(cpfft)**2

# Create a new empty spectrogram to contain the smooth spedtrogram values
smoothedspectro=np.zeros(np.shape(spectrogram))

# Apply seismogram filtering to data
for i in range(np.shape(spectrogram)[0]):
	background = convolve(spectrogram[i], Box1DKernel(100))
	smoothedspectro[i]=spectrogram[i] - background
	df = 20
	cft = classic_sta_lta(smoothedspectro[i], int(5 * df), int(10 * df))
	indices_ut = np.where(cft>1.7)
	indices_ut = list(indices_ut[0])
	indices_lt = np.where(cft<0.2)
	indices_lt  = list(indices_lt[0])
	indices = indices_ut + indices_lt
	indices_zero = range(0,6144)
	indices_zero = list(set(indices_zero) - set(indices)) #NOTE I (mohit) got rid of the for loop here for speed 
	smoothedspectro[i][indices_zero] = 0
	smoothedspectro[i][indices_ut] = 1
	smoothedspectro[i][indices_lt] = 1
	smoothedspectro[i][:500] = 0

# Clean the signal
smoothedspectro = ndimage.binary_closing(np.asarray(smoothedspectro), structure=np.ones((4,4))).astype(np.int)

# Remove speckles
smoothedspectro = ndimage.binary_opening(np.asarray(smoothedspectro), structure=np.ones((2,2))).astype(np.int)

# Rename
spectrogram = smoothedspectro
    
# Plot spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.get_cmap("binary")
ax.imshow(spectrogram, cmap=cmap,aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0])
ax.set_axis_off()

# Display spectrogram
plt.show()
