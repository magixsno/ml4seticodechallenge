import ibmseti
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

'''
#############################################
# Run this file from the ml4seti folder:

python processors/example_processor.py

This creates an example spectrogram, runs a hanning window, then saves the
results to the specified output folder.
#############################################
'''

# Find the data in the zip file
mydatafolder = 'primary_small'
output_folder = 'primary_small/data_out'
zz = zipfile.ZipFile(os.path.join(mydatafolder, 'primary_small_v3.zip'))

# For example, only use the first 100 files
# datalist = zz.namelist()[1:100]
# print(datalist)
# If you want to run this on all the data, use the line below:
datalist = zz.namelist()[1:]
count = 0
# Iterates through all data and saves the png in the output_folder
for uuid in datalist:
	# Read all data into ibmseti object
	aca = ibmseti.compamp.SimCompamp(zz.open(uuid).read())

	# Get the raw complex data
	complex_data = aca.complex_data()

	# Apply a hanning window
	complex_data = complex_data.reshape(32, 6144)
	complex_data = complex_data * np.hanning(complex_data.shape[1])

	# Create spectrogram
	cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
	spectrogram = np.abs(cpfft)**2
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.imshow(np.log(spectrogram), aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0])

	# Save as PNG
	filename = uuid.split('.')[0] + ".png"
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	if(count % 10 == 0):
		print datetime.now()

	fig.savefig( os.path.join(output_folder, filename) )
	plt.close(fig)
	count += 1
