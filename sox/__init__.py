#!/usr/bin/env python
""" init method for sox module """
import logging
import os

# Check that SoX is installed and callable
NO_SOX = False
if not len(os.popen('sox -h').readlines()):
    logging.warning("""SoX could not be found!

    If you do not have SoX, proceed here:
     - - - http://sox.sourceforge.net/ - - -

    If you do (or think that you should) have SoX, double-check your
    path variables.
    """)
    NO_SOX = True

from . import file_info
from .combine import Combiner
from .transform import Transformer
from .core import SoxError
from .core import SoxiError
from .version import version as __version__
