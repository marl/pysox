.. _examples:

Transformer Example
===================

.. code-block:: python
    :linenos:

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


Combiner Example
================

.. code-block:: python
    :linenos:

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


Advanced Usage
==============

The following functions behave differently than the rest in `Transformer`.
the `build` function is not need to be call after them, instead the result
will be output or returned as soon as they are called.

noiseprof / noisered
--------------------

.. code-block:: python
    :linenos:

    import sox
    # create transformer
    tfm = sox.Transformer()
    # create a prof file with noise data
    # you can record an "empty" file contains noise from environment
    # so sox will identify noise data without do harm to what you need
    tfm.noiseprof('noise.wav', 'noise.prof')
    # now noise.prof is in your working directory
    # it is time to fire up noisered and get noise rid
    # amount parameter means how much noise should be remove
    # high amount may cause detail losing
    tfm.noisered('noise.prof', amount=0.3)
    # preview the effect before output
    tfm.preview('sing.wav')
    # create the output
    tfm.build('sing.wav') 

power spectrum
--------------

.. code-block:: python
    :linenos:

    import sox
    # create transformer
    tfm = sox.Transformer()
    # get power spectrum data
    # by default, it analyse sound in channel 1
    power = tfm.power_spectrum('talk.wav')
    # the result is a list contain lists where [0] is frequency
    # and [1] is amplitude
    # you can split amplitude data for further analyse / display for gui
    amp = [pair[1] for pair in power]

stat / stats
------------

Both `stat` and `stats` provides some domain statistical information
about an audio. Here we will show how to get these data, for the meaning
of output information, please read `man sox`.

.. code-block:: python
   :linenos:

   import sox
   # create transformer
   tfm = sox.Transformer()
   # get stat data
   stat_data = tfm.stat('input.wav')
   # now for the stats data
   stats_data = tfm.stat('input.wav')
   type(stat_data)
   > <type 'dict'>
