import ibmseti
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

'''
#############################################
# Run this file from the ml4seti folder:

python processors/example_processor.py

#############################################
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

# Apply a hanning window 
complex_data = complex_data.reshape(32, 6144) 
complex_data = complex_data * np.hanning(complex_data.shape[1])

# Create spectrogram
cpfft = np.fft.fftshift( np.fft.fft(complex_data), 1)
spectrogram = np.abs(cpfft)**2
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(np.log(spectrogram), aspect = 0.5*float(spectrogram.shape[1]) / spectrogram.shape[0])

# Display spectrogram
plt.show()
