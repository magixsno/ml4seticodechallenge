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


# Find the data in the zip file
mydatafolder = 'primary_medium'
a = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_1.zip'))
b = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_2.zip'))
c = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_3.zip'))
d = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_4.zip'))
e = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_5.zip'))

datalist = a.namelist() + b.namelist() + c.namelist() + d.namelist() + e.namelist()
#firstfile = basic4list[52]
output_folder = 'data_out/jacob_gameoflife'
for uuid in datalist:

	# Read data into ibmseti object
	aca = ibmseti.compamp.SimCompamp(zz.open(firstfile).read())

	# Get the raw complex data
	complex_data = aca.complex_data()
	complex_data = complex_data.reshape(32, 6144)
	complex_data = complex_data * np.hanning(complex_data.shape[1])
	cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
	spectrogram = np.abs(cpfft)**2

	# Create a new empty spectrogram to contain the smooth spectrogram values
	smoothedspectro = np.zeros(np.shape(spectrogram))
	filteredspectro = np.zeros(np.shape(spectrogram))

	# Define parameters
	signalfill = 999999.999
	width = 30

	# Find maximum at each row and fill at maximum +/- width with signal
	for i in range( np.shape( spectrogram )[0] ):
		rowind = np.where( spectrogram[i] == spectrogram[i].max() )
		filteredspectro[i][rowind[0]] = signalfill

		for q in range( 0, width, 1 ):
			filteredspectro[i][rowind[0]-q] = signalfill
			filteredspectro[i][rowind[0]+q] = signalfill

	# Clean up the image
	ndimage.binary_erosion( np.asarray( filteredspectro ), structure=np.ones((100,100))).astype(np.int)
	spectrogram = filteredspectro

	# Plot spectrogram
	fig, ax = plt.subplots(figsize=(10, 5))
	cmap = plt.cm.get_cmap("binary")
	ax.imshow(spectrogram, cmap=cmap,aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0])
	ax.set_axis_off()

	# Display spectrogram
	# plt.show()
	filename = uuid.split('.')[0] + ".png"
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	fig.savefig( os.path.join(output_folder, filename) )
	plt.close(fig)
