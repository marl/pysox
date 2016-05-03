import logging
import os
logging.basicConfig(level=logging.DEBUG)

# Error Message for SoX
__NO_SOX = """SoX could not be found!

    If you do not have SoX, proceed here:
     - - - http://sox.sourceforge.net/ - - -

    If you do (or think that you should) have SoX, double-check your
    path variables.
    """

""" Check that SoX is installed and callable """
if not len(os.popen('sox -h').readlines()):
    logging.warning(__NO_SOX)
    assert False, "SoX assertion failed.\n" + __NO_SOX
