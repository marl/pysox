.. pysox documentation master file, created by
   sphinx-quickstart on Tue May 17 14:11:03 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pysox's documentation!
=================================

pysox is a Python wrapper around the amazing `SoX <http://sox.sourceforge.net/>`_ command line tool.

.. code-block:: python
    # within python
    import sox


Installation
------------

On Linux

.. code-block:: shell

    # optional - if you want support for mp3, flac and ogg files
    $ apt-get install libsox-fmt-all
    # install the sox command line tool
    $ apt-get install sox
    # install pysox
    $ pip install sox

On Mac with `Homebrew <https://brew.sh/>`_

.. code-block:: shell

    # optional - if you want support for mp3, flac and ogg files
    $ brew install sox --with-lame --with-flac --with-libvorbis
    # install the sox command line tool
    $ brew install sox
    # install pysox
    $ pip install sox


Examples
--------
.. toctree::
    :maxdepth: 3

    example

API Reference
=============
.. toctree::
   :maxdepth: 2

   api

Changes
=======
.. toctree::
   :maxdepth: 2

   changes

Contribute
==========
- `Issue Tracker <http://github.com/rabitt/pysox/issues>`_
- `Source Code <http://github.com/rabitt/pysox>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


pysox vs sox
============
The original command line tool is called `SoX <http://sox.sourceforge.net/>`_

This project (the `github repository <http://github.com/rabitt/pysox>`_) is called pysox

The library within python is called `sox`. It can be installed via:

.. code-block:: shell

    $ pip install sox

and imported within Python as

.. code-block:: python

    import sox
