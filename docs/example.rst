.. _examples:

Transformer Example
===================

Transform audio files

.. code-block:: python
    :linenos:

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

Transform in-memory arrays

.. code-block:: python
    :linenos:

    import numpy
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
    cbn.convert(samplerate=8000, n_channels=2)
    # create the output file
    cbn.build(
        ['input1.wav', 'input2.wav', 'input3.wav'], output.wav, 'concatenate'
    )

    # the combiner does not currently support array input/output


File Info Example
=================

.. code-block:: python
    :linenos:

    import sox
    # get the sample rate
    sample_rate = sox.file_info.sample_rate('path/to/file.mp3')
    # get the number of samples
    n_samples = sox.file_info.num_samples('path/to/file.wav')
    # determine if a file is silent
    is_silent = sox.file_info.silent('path/to/file.aiff')

    # file info doesn't currently support array input
