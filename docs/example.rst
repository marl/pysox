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
