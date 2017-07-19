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
mydatafolder = 'data'
zz = zipfile.ZipFile(os.path.join(mydatafolder, 'basic4.zip'))
basic4list = zz.namelist()
firstfile = basic4list[1400]

# Read data into ibmseti object
aca = ibmseti.compamp.SimCompamp(zz.open(firstfile).read())

# Get the raw complex data
complex_data = aca.complex_data()
complex_data = complex_data.reshape(32, 6144)
complex_data = complex_data * np.hanning(complex_data.shape[1])
cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
spectrogram = np.abs(cpfft)**2

# Create a new empty spectrogram to contain the smooth spedtrogram values
smoothedspectro = np.zeros(np.shape(spectrogram))
filteredspectro = np.zeros(np.shape(spectrogram))
signalfill = 999999.999
width = 15
delta = 300
noisetol = 0.5
indtol   = 50

# Apply seismogram filtering to data
for i in range( np.shape( spectrogram )[0] ):
	#background = convolve(spectrogram[i], Box1DKernel(100))
	#smoothedspectro[i] = spectrogram[i] - background
	rowind = np.where( spectrogram[i] == spectrogram[i].max() )
	filteredspectro[i][rowind[0]] = signalfill
	indexes = np.zeros(1)
	indexes[0] = rowind[0]

	for q in range( 0, width, 1 ):
		filteredspectro[i][rowind[0]-q] = signalfill
		filteredspectro[i][rowind[0]+q] = signalfill
		indexes = np.append( rowind[0]-q, indexes )
		indexes = np.append( rowind[0]+q, indexes )

	smoothedspectro[i] = spectrogram[i]
	sortedspectro = np.sort( smoothedspectro[i] )
	maxspec = sortedspectro[ np.shape( sortedspectro )[0] - 1 ]

	for p in range( np.shape( sortedspectro )[0] - 1, np.shape( sortedspectro )[0] - delta, -1 ):
		if( sortedspectro[p] / maxspec >= noisetol ):
			rowind = np.where( smoothedspectro[i] == sortedspectro[p] )
			inddif = indexes - rowind
			if( np.all( inddif < indtol ) ):
				filteredspectro[i][rowind[0]] = signalfill
				indexes = np.append( rowind, indexes )

	blockflag = 0
	whiteflag = 0
	for j in range( np.shape( filteredspectro )[1] ):
		if( ( filteredspectro[i][j] > 0 ) and ( blockflag == 0 ) ):
			blockflag = j
		if( ( filteredspectro[i][j] == 0 ) and ( blockflag > 0 ) and ( whiteflag == 0 ) ):
			whiteflag = j
		if( ( filteredspectro[i][j] > 0 ) and ( whiteflag > 0 ) ):
			for k in range( blockflag, whiteflag, 1 ):
				filteredspectro[i][k] = signalfill
				whiteflag = 0


# Binarize the image
#ndimage.binary_erosion(np.asarray(smoothedspectro), structure=np.ones((100,100))).astype(np.int)
#spectrogram = smoothedspectro
ndimage.binary_erosion( np.asarray( filteredspectro ), structure=np.ones((100,100))).astype(np.int)
spectrogram = filteredspectro

# Plot spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.get_cmap("binary")
ax.imshow(spectrogram, cmap=cmap,aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0])
ax.set_axis_off()

# Display spectrogram
plt.show()
