# Processors

This is where signal processors live. Signal processors take the spectrograms and clean up the data. This includes removing noise, changing the size of the spectrogram, or pruning the data in any way.

## Goal

The basic4 data set has 4 different classifications:

A signal that changes in frequency linearly with time:

![Narrowband](img/narrowband.png "Narrowband")

A signal that changes in frequency quadratically with time:

![Narrowbandddr](img/narrowbandddr.png "Narrowbandddr")

A signal that wanders randomly through close frequencies:

![Squiggle](img/squiggle.png "Squiggle")

No signal found:

![Noise](img/noise.png "Noise")

Therefore, in processing the goal is to amplify this signal whatever methods you can.

## example_spectrogram.py

Here we create a spectrogram using matplotlib. This spectrogram is the raw data. The x-axis is frequency, and the y-axis is time.

```shell
python processors/example_spectrogram.py
``` 

## example_processor.py

Here the generic spectrogram is taken and a hanning window is added to the data. This amplifies the signal and reduces the noise by a noiceable amount. This is really the most basic signal processing you can do.

```shell
python processors/example_processor.py
```