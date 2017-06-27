# Data

## basic4.zip 

This is a 1.16 Gb file. It currently lives at `https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_basic_v2/basic4.zip`. To get this on your machine use:

```shell
curl -o data/basic4.zip https://dal.objectstorage.open.softlayer.com/v1/AUTH_cdbef52bdf7a449c96936e1071f0a46b/simsignals_basic_v2/basic4.zip
```

The `basic4.zip` contains a list of .dat files. Each of these .dat files is on the order of a few kilobytes. It is just a bunch of bytes. If you try to open this in an editor, it will be gibberish. The best way to read this data is to use the `ibmseti` library. More about that below.

Each .dat file represents a spectrogram. These are uniquely identified by "unique ids" or uuids. This set is already classfied, so we know the label of each spectrogram by its uuid. This data is located in `public_list_basic_v2_26may_2017.csv`. 

## public_list_basic_v2_26may_2017.csv

This is a CSV with the first column representing the uuids and the second column representing the label for the spectrogram.

## ibmseti

This is a library released by seti and ibm that contains built-in functions for easy manipulations and processing of the data set.

Use the command below to install ibmseti on your machine.

```shell
pip install --user --upgrade ibmseti
```

## Reading the data

The following code snippet will access the data:

```
import os
import zipfile
import ibmseti

# Unzip the basic4 zip and read the header for the first spectrogram
zz = zipfile.ZipFile('basic4.zip')
basic4list = zz.namelist()[1:]
firstfile = basic4list[0]
aca = ibmseti.compamp.SimCompamp(zz.open(firstfile).read())
aca.header() # this shows the classification

```

Example output:

```
{u'signal_classification': u'narrowbanddrd',
 u'uuid': u'001b4fbd-bfbc-49e0-83a8-8b3c5b8b303d'}
```

