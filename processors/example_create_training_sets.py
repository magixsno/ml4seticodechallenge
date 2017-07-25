import ibmseti
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
TODO - create file that reads the classifiers and organizes the training sets
using the created images.

'''

# Reads the labels into a dict organized as {uuid: classification_label}
index_file = pd.read_csv(os.path.join(mydatafolder, 'public_list_basic_v2_26may_2017.csv'))
classifications = {}
for index, data in index_file.iterrows():
	classifications[data['UUID']] = data['SIGNAL_CLASSIFICATION']