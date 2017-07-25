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


# Find the data in the zip file
mydatafolder = 'primary_medium'
a = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_1.zip'))
b = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_2.zip'))
c = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_3.zip'))
d = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_4.zip'))
e = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_medium_v3_5.zip'))

datalist = a.namelist() + b.namelist() + c.namelist() + d.namelist() + e.namelist()
output_folder = 'data_out/jacob_fancymax'
#firstfile = basic4list[52]

for uuid in datalist:

	# Read data into ibmseti object
	print uuid[-3:]
	if uuid[-3:] != 'dat':
		continue
	# Read data into ibmseti object
        dat_file = open(mydatafolder + "/" + uuid, "rw")
	aca = ibmseti.compamp.SimCompamp(dat_file.read())

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
	width = 15
	delta = 300
	noisetol = 0.5
	indtol   = 50

	# Find maximum at each row and fill at maximum +/- width with signal
	for i in range( np.shape( spectrogram )[0] ):
		rowind = np.where( spectrogram[i] == spectrogram[i].max() )
		filteredspectro[i][rowind[0]] = signalfill
		indexes = np.zeros(1)
		indexes[0] = rowind[0]

		for q in range( 0, width, 1 ):
			filteredspectro[i][rowind[0]-q] = signalfill
			filteredspectro[i][rowind[0]+q] = signalfill
			indexes = np.append( rowind[0]-q, indexes )
			indexes = np.append( rowind[0]+q, indexes )

		# Sort spectrum to estimate length of signal within each row
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

		# Fill empty spaces between signal blocks
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
	#plt.show()
	# Save as PNG
	filename = uuid.split('.')[0] + ".png"
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	fig.savefig( os.path.join(output_folder, os.path.basename(filename)) )
	plt.close(fig)
