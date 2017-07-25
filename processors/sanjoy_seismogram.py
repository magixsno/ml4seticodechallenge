from __future__ import division

import astropy
import cStringIO
import glob
import ibmseti
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

import PIL
from PIL import Image

'''
IF YOU ARE RUNNING A NEWER VERSION OF MATPLOTLIB THIS CODE WILL NOT RUN:

pip install 'matplotlib==1.4.3'

Then run this code with:

python processors/sanjoy_seismogram.py
'''


# Find the data in the zip file
mydatafolder = 'primary_medium'
output_folder = 'data_out/sanjoy_seismogram'
a = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_1.zip'))
b = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_2.zip'))
c = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_3.zip'))
d = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_4.zip'))
e = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_5.zip'))
basic4list = a.namelist() + b.namelist() + c.namelist() + d.namelist() + e.namelist()

count = 0
for uuid in basic4list:
	if count % 100 == 0:
		print str(count/len(datalist)) * 100 + "%"
	count += 1
	if uuid[-3:] != 'dat':
		continue
	# Read data into ibmseti object
        dat_file = open(mydatafolder + "/" + uuid, "rw")
	aca = ibmseti.compamp.SimCompamp(dat_file.read())

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

	# Save as PNG
	filename = uuid + ".png"
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	fig.savefig( os.path.join(output_folder, os.path.basename(filename)))
	plt.close(fig)
