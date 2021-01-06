# pysox
Python wrapper around sox. [Read the Docs here](http://pysox.readthedocs.org).

[![PyPI version](https://badge.fury.io/py/sox.svg)](https://badge.fury.io/py/sox)
[![Documentation Status](https://readthedocs.org/projects/pysox/badge/?version=latest)](https://pysox.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://raw.githubusercontent.com/rabitt/pysox/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg?maxAge=2592000)]()

[![Build Status](https://travis-ci.org/rabitt/pysox.svg?branch=master)](https://travis-ci.org/rabitt/pysox)
[![Coverage Status](https://coveralls.io/repos/github/rabitt/pysox/badge.svg?branch=master)](https://coveralls.io/github/rabitt/pysox?branch=master)

![PySocks](https://s-media-cache-ak0.pinimg.com/736x/62/6f/bc/626fbcae9618eccee1c4c7c947bf9d94.jpg)

This library was presented in the following paper:

[R. M. Bittner](https://github.com/rabitt), [E. J. Humphrey](https://github.com/ejhumphrey) and J. P. Bello, "[pysox: Leveraging the Audio Signal Processing Power of SoX in Python](https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/08/bittner-pysox.pdf)", in Proceedings of the 17th International Society for Music Information Retrieval Conference Late Breaking and Demo Papers, New York City, USA, Aug. 2016.


# Install

This requires that [SoX](http://sox.sourceforge.net/) version 14.4.2 or higher is installed.

To install SoX on Mac with Homebrew:

```brew install sox```

If you want support for `mp3`, `flac`, or `ogg` files, add the following flags:

```brew install sox --with-lame --with-flac --with-libvorbis```

on Linux:

```apt-get install sox```

or install [from source](https://sourceforge.net/projects/sox/files/sox/).



To install the most up-to-date release of this module via PyPi:

```pip install sox```

To install the master branch:

```pip install git+https://github.com/rabitt/pysox.git```

or

```
git clone https://github.com/rabitt/pysox.git
cd pysox
python setup.py install
```


# Tests

If you have a different version of SoX installed, it's recommended that you run
the tests locally to make sure everything behaves as expected, by simply running:

```
pytest
```

# Examples

```python
import sox
# create transformer
tfm = sox.Transformer()
# trim the audio between 5 and 10.5 seconds.
tfm.trim(5, 10.5)
# apply compression
tfm.compand()
# apply a fade in and fade out
tfm.fade(fade_in_len=1.0, fade_out_len=0.5)
# create an output file.
tfm.build_file('path/to/input_audio.wav', 'path/to/output/audio.aiff')
# or equivalently using the legacy API
tfm.build('path/to/input_audio.wav', 'path/to/output/audio.aiff')
# get the output in-memory as a numpy array
# by default the sample rate will be the same as the input file
array_out = tfm.build_array(input_filepath='path/to/input_audio.wav')
# see the applied effects
tfm.effects_log
> ['trim', 'compand', 'fade']

```

Transform in-memory arrays:
```python
import numpy as np
import sox
# sample rate in Hz
sample_rate = 44100
# generate a 1-second sine tone at 440 Hz
y = np.sin(2 * np.pi * 440.0 * np.arange(sample_rate * 1.0) / sample_rate)
# create a transformer
tfm = sox.Transformer()
# shift the pitch up by 2 semitones
tfm.pitch(2)
# transform an in-memory array and return an array
y_out = tfm.build_array(input_array=y, sample_rate_in=sample_rate)
# instead, save output to a file
tfm.build_file(
    input_array=y, sample_rate_in=sample_rate,
    output_filepath='path/to/output.wav'
)
# create an output file with a different sample rate
tfm.set_output_format(rate=8000)
tfm.build_file(
    input_array=y, sample_rate_in=sample_rate,
    output_filepath='path/to/output_8k.wav'
)
```

Concatenate 3 audio files:
```python
import sox
# create combiner
cbn = sox.Combiner()
# pitch shift combined audio up 3 semitones
cbn.pitch(3.0)
# convert output to 8000 Hz stereo
cbn.convert(samplerate=8000, n_channels=2)
# create the output file
cbn.build(
    ['input1.wav', 'input2.wav', 'input3.wav'], 'output.wav', 'concatenate'
)
# the combiner does not currently support array input/output
```

Get file information:
```python
import sox
# get the sample rate
sample_rate = sox.file_info.sample_rate('path/to/file.mp3')
# get the number of samples
n_samples = sox.file_info.num_samples('path/to/file.wav')
# determine if a file is silent
is_silent = sox.file_info.silent('path/to/file.aiff')
# file info doesn't currently support array input
```
