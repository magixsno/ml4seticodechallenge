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


# Game of life python implementation from https://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html#the-game-of-life

def compute_neighbours(Z):
    rows,cols = len(Z), len(Z[0])
    N  = [[0,]*(cols)  for i in range(rows)]
    for x in range(1,cols-1):
        for y in range(1,rows-1):
            N[y][x] = Z[y-1][x-1]+Z[y][x-1]+Z[y+1][x-1] \
                    + Z[y-1][x]            +Z[y+1][x]   \
                    + Z[y-1][x+1]+Z[y][x+1]+Z[y+1][x+1]
    return N

def show(Z):
    for l in Z[1:-1]: print l[1:-1]
    print

def iterate(Z):
    rows,cols = len(Z), len(Z[0])
    N = compute_neighbours(Z)
    for x in range(1,cols-1):
        for y in range(1,rows-1):
            if Z[y][x] == 1 and (N[y][x] < 2 or N[y][x] > 3):
                Z[y][x] = 0
            elif Z[y][x] == 0 and N[y][x] == 3:
                Z[y][x] = 1
    return Z


# Find the data in the zip file
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
    threshold = 1.e-4
    numsquare = 4
    numlife   = 4

    smoothedspectro = spectrogram

    # Normalize, square, and filter the specrogram
    for i in range( 1, numsquare+1, 1 ):
    	smoothedspectro = smoothedspectro / np.average( smoothedspectro )
    	smoothedspectro = smoothedspectro**2
    	smoothedspectro[ smoothedspectro < threshold ] = 0.0

    smoothedspectro[ smoothedspectro > 0.0 ] = 1.0

    # Use Conway's Game of Life as additional "smoothing"
    for j in range( 0, numlife, 1 ):
    	iterate( smoothedspectro )

    smoothedspectro = ndimage.binary_dilation(smoothedspectro, structure=np.ones((5,5))).astype(np.int)

    # Clean up the image
    #ndimage.binary_erosion( np.asarray( filteredspectro ), structure=np.ones((100,100))).astype(np.int)
    #spectrogram = filteredspectro
    #ndimage.binary_erosion( np.asarray( spectrogram ), structure=np.ones((100,100))).astype(np.int)
    #spectrogram = spectrogram

    spectrogram = smoothedspectro

    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("binary")
    ax.imshow(spectrogram, cmap=cmap,aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0])
    ax.set_axis_off()

    # Display spectrogram
    #plt.show()
    filename = uuid.split('.')[0] + ".png"
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	fig.savefig( os.path.join(output_folder, filename) )
	plt.close(fig)
