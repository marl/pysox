# pysox
Python wrapper around sox. [Read the Docs here](http://pysox.readthedocs.org).

[![PyPI version](https://badge.fury.io/py/sox.svg)](https://badge.fury.io/py/sox)
[![Documentation Status](https://readthedocs.org/projects/resampy/badge/?version=latest)](http://pysox.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/rabitt/pysox/master/LICENSE.md)
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

on Linux:

```apt-get install sox```

or install [from source](https://sourceforge.net/projects/sox/files/sox/).



To install the most up to date release of this module via PyPi:

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
the tests locally to make sure everything behaves as expected:

```
cd tests
nosetests .
```

# Examples

```python
import sox
# create trasnformer
tfm = sox.Transformer()
# trim the audio between 5 and 10.5 seconds.
tfm.trim(5, 10.5)
# apply compression
tfm.compand()
# apply a fade in and fade out
tfm.fade(fade_in_len=1.0, fade_out_len=0.5)
# create the output file.
tfm.build('path/to/input_audio.wav', 'path/to/output/audio.aiff')
# see the applied effects
tfm.effects_log
> ['trim', 'compand', 'fade']

```

Concatenate 3 audio files:
```python
import sox
# create combiner
cbn = sox.Combiner()
# pitch shift combined audio up 3 semitones
cbn.pitch(3.0)
# convert output to 8000 Hz stereo
cbn.convert(samplerate=8000, channels=2)
# create the output file
cbn.build(
    ['input1.wav', 'input2.wav', 'input3.wav'], output.wav, 'concatenate'
)

```
