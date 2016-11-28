#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Python wrapper around the SoX library.
This module requires that SoX is installed.
'''

from __future__ import print_function
import logging
import random

from .core import ENCODING_VALS
from .core import enquote_filepath
from .core import is_number
from .core import play
from .core import sox
from .core import SoxError
from .core import VALID_FORMATS

from . import file_info

logging.basicConfig(level=logging.DEBUG)

VERBOSITY_VALS = [0, 1, 2, 3, 4]


class Transformer(object):
    '''Audio file transformer.
    Class which allows multiple effects to be chained to create an output
    file, saved to output_filepath.



    Methods
    -------
    set_globals
        Overwrite the default global arguments.
    build
        Execute the current chain of commands and write output file.

    '''

    def __init__(self):
        '''
        Attributes
        ----------
        input_format : list of str
            Input file format arguments that will be passed to SoX.
        output_format : list of str
            Output file format arguments that will be bassed to SoX.
        effects : list of str
            Effects arguments that will be passed to SoX.
        effects_log : list of str
            Ordered sequence of effects applied.
        globals : list of str
            Global arguments that will be passed to SoX.

        '''
        self.input_format = []
        self.output_format = []

        self.effects = []
        self.effects_log = []

        self.globals = []
        self.set_globals()

    def set_globals(self, dither=False, guard=False, multithread=False,
                    replay_gain=False, verbosity=2):
        '''Sets SoX's global arguments.
        Overwrites any previously set global arguments.
        If this function is not explicity called, globals are set to this
        function's defaults.

        Parameters
        ----------
        dither : bool, default=False
            If True, dithering is applied for low files with low bit rates.
        guard : bool, default=False
            If True, invokes the gain effect to guard against clipping.
        multithread : bool, default=False
            If True, each channel is processed in parallel.
        replay_gain : bool, default=False
            If True, applies replay-gain adjustment to input-files.
        verbosity : int, default=2
            SoX's verbosity level. One of:
                * 0 : No messages are shown at all
                * 1 : Only error messages are shown. These are generated if SoX
                    cannot complete the requested commands.
                * 2 : Warning messages are also shown. These are generated if
                    SoX can complete the requested commands, but not exactly
                    according to the requested command parameters, or if
                    clipping occurs.
                * 3 : Descriptions of SoX’s processing phases are also shown.
                    Useful for seeing exactly how SoX is processing your audio.
                * 4, >4 : Messages to help with debugging SoX are also shown.

        '''
        if not isinstance(dither, bool):
            raise ValueError('dither must be a boolean.')

        if not isinstance(guard, bool):
            raise ValueError('guard must be a boolean.')

        if not isinstance(multithread, bool):
            raise ValueError('multithread must be a boolean.')

        if not isinstance(replay_gain, bool):
            raise ValueError('replay_gain must be a boolean.')

        if verbosity not in VERBOSITY_VALS:
            raise ValueError(
                'Invalid value for VERBOSITY. Must be one {}'.format(
                    VERBOSITY_VALS)
            )

        global_args = []

        if not dither:
            global_args.append('-D')

        if guard:
            global_args.append('-G')

        if multithread:
            global_args.append('--multi-threaded')

        if replay_gain:
            global_args.append('--replay-gain')
            global_args.append('track')

        global_args.append('-V{}'.format(verbosity))

        self.globals = global_args
        return self

    def set_input_format(self, file_type=None, rate=None, bits=None,
                         channels=None, encoding=None, ignore_length=False):
        '''Sets input file format arguments. This is primarily useful when
        dealing with audio files without a file extension. Overwrites any
        previously set input file arguments.

        If this function is not explicity called the input format is inferred
        from the file extension or the file's header.

        Parameters
        ----------
        file_type : str or None, default=None
            The file type of the input audio file. Should be the same as what
            the file extension would be, for ex. 'mp3' or 'wav'.
        rate : float or None, default=None
            The sample rate of the input audio file. If None the sample rate
            is inferred.
        bits : int or None, default=None
            The number of bits per sample. If None, the number of bits per
            sample is inferred.
        channels : int or None, default=None
            The number of channels in the audio file. If None the number of
            channels is inferred.
        encoding : str or None, default=None
            The audio encoding type. Sometimes needed with file-types that
            support more than one encoding type. One of:
                * signed-integer : PCM data stored as signed (‘two’s
                    complement’) integers. Commonly used with a 16 or 24−bit
                    encoding size. A value of 0 represents minimum signal
                    power.
                * unsigned-integer : PCM data stored as unsigned integers.
                    Commonly used with an 8-bit encoding size. A value of 0
                    represents maximum signal power.
                * floating-point : PCM data stored as IEEE 753 single precision
                    (32-bit) or double precision (64-bit) floating-point
                    (‘real’) numbers. A value of 0 represents minimum signal
                    power.
                * a-law : International telephony standard for logarithmic
                    encoding to 8 bits per sample. It has a precision
                    equivalent to roughly 13-bit PCM and is sometimes encoded
                    with reversed bit-ordering.
                * u-law : North American telephony standard for logarithmic
                    encoding to 8 bits per sample. A.k.a. μ-law. It has a
                    precision equivalent to roughly 14-bit PCM and is sometimes
                    encoded with reversed bit-ordering.
                * oki-adpcm : OKI (a.k.a. VOX, Dialogic, or Intel) 4-bit ADPCM;
                    it has a precision equivalent to roughly 12-bit PCM. ADPCM
                    is a form of audio compression that has a good compromise
                    between audio quality and encoding/decoding speed.
                * ima-adpcm : IMA (a.k.a. DVI) 4-bit ADPCM; it has a precision
                    equivalent to roughly 13-bit PCM.
                * ms-adpcm : Microsoft 4-bit ADPCM; it has a precision
                    equivalent to roughly 14-bit PCM.
                * gsm-full-rate : GSM is currently used for the vast majority
                    of the world’s digital wireless telephone calls. It
                    utilises several audio formats with different bit-rates and
                    associated speech quality. SoX has support for GSM’s
                    original 13kbps ‘Full Rate’ audio format. It is usually
                    CPU-intensive to work with GSM audio.
        ignore_length : bool, default=False
            If True, overrides an (incorrect) audio length given in an audio
            file’s header. If this option is given then SoX will keep reading
            audio until it reaches the end of the input file.

        '''
        if file_type not in VALID_FORMATS + [None]:
            raise ValueError(
                'Invalid file_type. Must be one of {}'.format(VALID_FORMATS)
            )

        if not is_number(rate) and rate is not None:
            raise ValueError('rate must be a float or None')

        if rate is not None and rate <= 0:
            raise ValueError('rate must be a positive number')

        if not isinstance(bits, int) and bits is not None:
            raise ValueError('bits must be an int or None')

        if bits is not None and bits <= 0:
            raise ValueError('bits must be a positive number')

        if not isinstance(channels, int) and channels is not None:
            raise ValueError('channels must be an int or None')

        if channels is not None and channels <= 0:
            raise ValueError('channels must be a positive number')

        if encoding not in ENCODING_VALS + [None]:
            raise ValueError(
                'Invalid encoding. Must be one of {}'.format(ENCODING_VALS)
            )

        if not isinstance(ignore_length, bool):
            raise ValueError('ignore_length must be a boolean')

        input_format = []

        if file_type is not None:
            input_format.extend(['-t', '{}'.format(file_type)])

        if rate is not None:
            input_format.extend(['-r', '{}'.format(rate)])

        if bits is not None:
            input_format.extend(['-b', '{}'.format(bits)])

        if channels is not None:
            input_format.extend(['-c', '{}'.format(channels)])

        if encoding is not None:
            input_format.extend(['-e', '{}'.format(encoding)])

        if ignore_length:
            input_format.append('--ignore-length')

        self.input_format = input_format
        return self

    def set_output_format(self, file_type=None, rate=None, bits=None,
                          channels=None, encoding=None, comments=None,
                          append_comments=True):
        '''Sets output file format arguments. These arguments will overwrite
        any format related arguments supplied by other effects (e.g. rate).

        If this function is not explicity called the output format is inferred
        from the file extension or the file's header.

        Parameters
        ----------
        file_type : str or None, default=None
            The file type of the output audio file. Should be the same as what
            the file extension would be, for ex. 'mp3' or 'wav'.
        rate : float or None, default=None
            The sample rate of the output audio file. If None the sample rate
            is inferred.
        bits : int or None, default=None
            The number of bits per sample. If None, the number of bits per
            sample is inferred.
        channels : int or None, default=None
            The number of channels in the audio file. If None the number of
            channels is inferred.
        encoding : str or None, default=None
            The audio encoding type. Sometimes needed with file-types that
            support more than one encoding type. One of:
                * signed-integer : PCM data stored as signed (‘two’s
                    complement’) integers. Commonly used with a 16 or 24−bit
                    encoding size. A value of 0 represents minimum signal
                    power.
                * unsigned-integer : PCM data stored as unsigned integers.
                    Commonly used with an 8-bit encoding size. A value of 0
                    represents maximum signal power.
                * floating-point : PCM data stored as IEEE 753 single precision
                    (32-bit) or double precision (64-bit) floating-point
                    (‘real’) numbers. A value of 0 represents minimum signal
                    power.
                * a-law : International telephony standard for logarithmic
                    encoding to 8 bits per sample. It has a precision
                    equivalent to roughly 13-bit PCM and is sometimes encoded
                    with reversed bit-ordering.
                * u-law : North American telephony standard for logarithmic
                    encoding to 8 bits per sample. A.k.a. μ-law. It has a
                    precision equivalent to roughly 14-bit PCM and is sometimes
                    encoded with reversed bit-ordering.
                * oki-adpcm : OKI (a.k.a. VOX, Dialogic, or Intel) 4-bit ADPCM;
                    it has a precision equivalent to roughly 12-bit PCM. ADPCM
                    is a form of audio compression that has a good compromise
                    between audio quality and encoding/decoding speed.
                * ima-adpcm : IMA (a.k.a. DVI) 4-bit ADPCM; it has a precision
                    equivalent to roughly 13-bit PCM.
                * ms-adpcm : Microsoft 4-bit ADPCM; it has a precision
                    equivalent to roughly 14-bit PCM.
                * gsm-full-rate : GSM is currently used for the vast majority
                    of the world’s digital wireless telephone calls. It
                    utilises several audio formats with different bit-rates and
                    associated speech quality. SoX has support for GSM’s
                    original 13kbps ‘Full Rate’ audio format. It is usually
                    CPU-intensive to work with GSM audio.
        comments : str or None, default=None
            If not None, the string is added as a comment in the header of the
            output audio file. If None, no comments are added.
        append_comments : bool, default=True
            If True, comment strings are appended to SoX's default comments. If
            False, the supplied comment replaces the existing comment.

        '''
        if file_type not in VALID_FORMATS + [None]:
            raise ValueError(
                'Invalid file_type. Must be one of {}'.format(VALID_FORMATS)
            )

        if not is_number(rate) and rate is not None:
            raise ValueError('rate must be a float or None')

        if rate is not None and rate <= 0:
            raise ValueError('rate must be a positive number')

        if not isinstance(bits, int) and bits is not None:
            raise ValueError('bits must be an int or None')

        if bits is not None and bits <= 0:
            raise ValueError('bits must be a positive number')

        if not isinstance(channels, int) and channels is not None:
            raise ValueError('channels must be an int or None')

        if channels is not None and channels <= 0:
            raise ValueError('channels must be a positive number')

        if encoding not in ENCODING_VALS + [None]:
            raise ValueError(
                'Invalid encoding. Must be one of {}'.format(ENCODING_VALS)
            )

        if comments is not None and not isinstance(comments, str):
            raise ValueError('comments must be a string or None')

        if not isinstance(append_comments, bool):
            raise ValueError('append_comments must be a boolean')

        output_format = []

        if file_type is not None:
            output_format.extend(['-t', '{}'.format(file_type)])

        if rate is not None:
            output_format.extend(['-r', '{}'.format(rate)])

        if bits is not None:
            output_format.extend(['-b', '{}'.format(bits)])

        if channels is not None:
            output_format.extend(['-c', '{}'.format(channels)])

        if encoding is not None:
            output_format.extend(['-e', '{}'.format(encoding)])

        if comments is not None:
            if append_comments:
                output_format.extend(['--add-comment', comments])
            else:
                output_format.extend(['--comment', comments])

        self.output_format = output_format
        return self

    def build(self, input_filepath, output_filepath):
        '''Builds the output_file by executing the current set of commands.

        Parameters
        ----------
        input_filepath : str
            Path to input audio file.
        output_filepath : str
            Path to desired output file. If a file already exists at the given
            path, the file will be overwritten.

        '''
        file_info.validate_input_file(input_filepath)
        file_info.validate_output_file(output_filepath)

        if input_filepath == output_filepath:
            raise ValueError(
                "input_filepath must be different from output_filepath."
            )

        args = []
        args.extend(self.globals)
        args.extend(self.input_format)
        args.append(enquote_filepath(input_filepath))
        args.extend(self.output_format)
        args.append(enquote_filepath(output_filepath))
        args.extend(self.effects)

        status, out, err = sox(args)

        if status != 0:
            raise SoxError(
                "Stdout: {}\nStderr: {}".format(out, err)
            )
        else:
            logging.info(
                "Created %s with effects: %s",
                output_filepath,
                " ".join(self.effects_log)
            )
            if out is not None:
                logging.info("[SoX] {}".format(out))
            return True

    def preview(self, input_filepath):
        '''Play a preview of the output with the current set of effects
        '''
        args = ["play", "--no-show-progress"]
        args.extend(self.globals)
        args.extend(self.input_format)
        args.append(input_filepath)
        args.extend(self.effects)

        play(args)

    def allpass(self, frequency, width_q=2.0):
        '''Apply a two-pole all-pass filter. An all-pass filter changes the
        audio’s frequency to phase relationship without changing its frequency
        to amplitude relationship. The filter is described in detail in at
        http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters
        ----------
        frequency : float
            The filter's center frequency in Hz.
        width_q : float, default=2.0
            The filter's width as a Q-factor.

        See Also
        --------
        equalizer, highpass, lowpass, sinc

        '''
        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(width_q) or width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'allpass', '{}'.format(frequency), '{}q'.format(width_q)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('allpass')
        return self

    def bandpass(self, frequency, width_q=2.0, constant_skirt=False):
        '''Apply a two-pole Butterworth band-pass filter with the given central
        frequency, and (3dB-point) band-width. The filter rolls off at 6dB per
        octave (20dB per decade) and is described in detail in
        http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters
        ----------
        frequency : float
            The filter's center frequency in Hz.
        width_q : float, default=2.0
            The filter's width as a Q-factor.
        constant_skirt : bool, default=False
            If True, selects constant skirt gain (peak gain = width_q).
            If False, selects constant 0dB peak gain.

        See Also
        --------
        bandreject, sinc

        '''
        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(width_q) or width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if not isinstance(constant_skirt, bool):
            raise ValueError("constant_skirt must be a boolean.")

        effect_args = ['bandpass']

        if constant_skirt:
            effect_args.append('-c')

        effect_args.extend(['{}'.format(frequency), '{}q'.format(width_q)])

        self.effects.extend(effect_args)
        self.effects_log.append('bandpass')
        return self

    def bandreject(self, frequency, width_q=2.0):
        '''Apply a two-pole Butterworth band-reject filter with the given
        central frequency, and (3dB-point) band-width. The filter rolls off at
        6dB per octave (20dB per decade) and is described in detail in
        http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters
        ----------
        frequency : float
            The filter's center frequency in Hz.
        width_q : float, default=2.0
            The filter's width as a Q-factor.
        constant_skirt : bool, default=False
            If True, selects constant skirt gain (peak gain = width_q).
            If False, selects constant 0dB peak gain.

        See Also
        --------
        bandreject, sinc

        '''
        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(width_q) or width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'bandreject', '{}'.format(frequency), '{}q'.format(width_q)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('bandreject')
        return self

    def bass(self, gain_db, frequency=100.0, slope=0.5):
        '''Boost or cut the bass (lower) frequencies of the audio using a
        two-pole shelving filter with a response similar to that of a standard
        hi-fi’s tone-controls. This is also known as shelving equalisation.

        The filters are described in detail in
        http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters
        ----------
        gain_db : float
            The gain at 0 Hz.
            For a large cut use -20, for a large boost use 20.
        frequency : float, default=100.0
            The filter's cutoff frequency in Hz.
        slope : float, default=0.5
            The steepness of the filter's shelf transition.
            For a gentle slope use 0.3, and use 1.0 for a steep slope.

        See Also
        --------
        treble, equalizer

        '''
        if not is_number(gain_db):
            raise ValueError("gain_db must be a number")

        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(slope) or slope <= 0 or slope > 1.0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'bass', '{}'.format(gain_db), '{}'.format(frequency),
            '{}s'.format(slope)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('bass')
        return self

    def bend(self, n_bends, start_times, end_times, cents, frame_rate=25,
             oversample_rate=16):
        '''Changes pitch by specified amounts at specified times.
        The pitch-bending algorithm utilises the Discrete Fourier Transform
        (DFT) at a particular frame rate and over-sampling rate.

        Parameters
        ----------
        n_bends : int
            The number of intervals to pitch shift
        start_times : list of floats
            A list of absolute start times (in seconds), in order
        end_times : list of floats
            A list of absolute end times (in seconds) in order.
            [start_time, end_time] intervals may not overlap!
        cents : list of floats
            A list of pitch shifts in cents. A positive value shifts the pitch
            up, a negative value shifts the pitch down.
        frame_rate : int, default=25
            The number of DFT frames to process per second, between 10 and 80
        oversample_rate: int, default=16
            The number of frames to over sample per second, between 4 and 32

        See Also
        --------
        pitch

        '''
        if not isinstance(n_bends, int) or n_bends < 1:
            raise ValueError("n_bends must be a positive integer.")

        if not isinstance(start_times, list) or len(start_times) != n_bends:
            raise ValueError("start_times must be a list of length n_bends.")

        if any([(not is_number(p) or p <= 0) for p in start_times]):
            raise ValueError("start_times must be positive floats.")

        if sorted(start_times) != start_times:
            raise ValueError("start_times must be in increasing order.")

        if not isinstance(end_times, list) or len(end_times) != n_bends:
            raise ValueError("end_times must be a list of length n_bends.")

        if any([(not is_number(p) or p <= 0) for p in end_times]):
            raise ValueError("end_times must be positive floats.")

        if sorted(end_times) != end_times:
            raise ValueError("end_times must be in increasing order.")

        if any([e <= s for s, e in zip(start_times, end_times)]):
            raise ValueError(
                "end_times must be element-wise greater than start_times."
            )

        if any([e > s for s, e in zip(start_times[1:], end_times[:-1])]):
            raise ValueError(
                "[start_time, end_time] intervals must be non-overlapping."
            )

        if not isinstance(cents, list) or len(cents) != n_bends:
            raise ValueError("cents must be a list of length n_bends.")

        if any([not is_number(p) for p in cents]):
            raise ValueError("elements of cents must be floats.")

        if (not isinstance(frame_rate, int) or
                frame_rate < 10 or frame_rate > 80):
            raise ValueError("frame_rate must be an integer between 10 and 80")

        if (not isinstance(oversample_rate, int) or
                oversample_rate < 4 or oversample_rate > 32):
            raise ValueError(
                "oversample_rate must be an integer between 4 and 32."
            )

        effect_args = [
            'bend',
            '-f', '{}'.format(frame_rate),
            '-o', '{}'.format(oversample_rate)
        ]

        last = 0
        for i in range(n_bends):
            t_start = round(start_times[i] - last, 2)
            t_end = round(end_times[i] - start_times[i], 2)
            effect_args.append(
                '{},{},{}'.format(t_start, cents[i], t_end)
            )
            last = end_times[i]

        self.effects.extend(effect_args)
        self.effects_log.append('bend')
        return self


    def biquad(self, b, a):
        '''Apply a biquad IIR filter with the given coefficients.

        Parameters
        ----------
        b : list of floats
            Numerator coefficients. Must be length 3
        a : list of floats
            Denominator coefficients. Must be length 3

        See Also
        --------
        fir, treble, bass, equalizer

        '''
        if not isinstance(b, list):
            raise ValueError('b must be a list.')

        if not isinstance(a, list):
            raise ValueError('a must be a list.')

        if len(b) != 3:
            raise ValueError('b must be a length 3 list.')

        if len(a) != 3:
            raise ValueError('a must be a length 3 list.')

        if not all([is_number(b_val) for b_val in b]):
            raise ValueError('all elements of b must be numbers.')

        if not all([is_number(a_val) for a_val in a]):
            raise ValueError('all elements of a must be numbers.')

        effect_args = [
            'biquad', '{}'.format(b[0]), '{}'.format(b[1]), '{}'.format(b[2]),
            '{}'.format(a[0]), '{}'.format(a[1]), '{}'.format(a[2])
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('biquad')
        return self

    def channels(self, n_channels):
        '''Change the number of channels in the audio signal. If decreasing the
        number of channels it mixes channels together, if increasing the number
        of channels it duplicates.

        Note: This overrides arguments used in the convert effect!

        Parameters
        ----------
        n_channels : int
            Desired number of channels.

        See Also
        --------
        convert

        '''
        if not isinstance(n_channels, int) or n_channels <= 0:
            raise ValueError('n_channels must be a positive integer.')

        effect_args = ['channels', '{}'.format(n_channels)]

        self.effects.extend(effect_args)
        self.effects_log.append('channels')
        return self

    def chorus(self, gain_in=0.5, gain_out=0.9, n_voices=3, delays=None,
               decays=None, speeds=None, depths=None, shapes=None):
        '''Add a chorus effect to the audio. This can makeasingle vocal sound
        like a chorus, but can also be applied to instrumentation.

        Chorus resembles an echo effect with a short delay, but whereas with
        echo the delay is constant, with chorus, it is varied using sinusoidal
        or triangular modulation. The modulation depth defines the range the
        modulated delay is played before or after the delay. Hence the delayed
        sound will sound slower or faster, that is the delayed sound tuned
        around the original one, like in a chorus where some vocals are
        slightly off key.

        Parameters
        ----------
        gain_in : float, default=0.3
            The time in seconds over which the instantaneous level of the input
            signal is averaged to determine increases in volume.
        gain_out : float, default=0.8
            The time in seconds over which the instantaneous level of the input
            signal is averaged to determine decreases in volume.
        n_voices : int, default=3
            The number of voices in the chorus effect.
        delays : list of floats > 20 or None, default=None
            If a list, the list of delays (in miliseconds) of length n_voices.
            If None, the individual delay parameters are chosen automatically
            to be between 40 and 60 miliseconds.
        decays : list of floats or None, default=None
            If a list, the list of decays (as a fraction of gain_in) of length
            n_voices.
            If None, the individual decay parameters are chosen automatically
            to be between 0.3 and 0.4.
        speeds : list of floats or None, default=None
            If a list, the list of modulation speeds (in Hz) of length n_voices
            If None, the individual speed parameters are chosen automatically
            to be between 0.25 and 0.4 Hz.
        depths : list of floats or None, default=None
            If a list, the list of depths (in miliseconds) of length n_voices.
            If None, the individual delay parameters are chosen automatically
            to be between 1 and 3 miliseconds.
        shapes : list of 's' or 't' or None, deault=None
            If a list, the list of modulation shapes - 's' for sinusoidal or
            't' for triangular - of length n_voices.
            If None, the individual shapes are chosen automatically.

        '''
        if not is_number(gain_in) or gain_in <= 0 or gain_in > 1:
            raise ValueError("gain_in must be a number between 0 and 1.")
        if not is_number(gain_out) or gain_out <= 0 or gain_out > 1:
            raise ValueError("gain_out must be a number between 0 and 1.")
        if not isinstance(n_voices, int) or n_voices <= 0:
            raise ValueError("n_voices must be a positive integer.")

        # validate delays
        if not (delays is None or isinstance(delays, list)):
            raise ValueError("delays must be a list or None")
        if delays is not None:
            if len(delays) != n_voices:
                raise ValueError("the length of delays must equal n_voices")
            if any((not is_number(p) or p < 20) for p in delays):
                raise ValueError("the elements of delays must be numbers > 20")
        else:
            delays = [random.uniform(40, 60) for _ in range(n_voices)]

        # validate decays
        if not (decays is None or isinstance(decays, list)):
            raise ValueError("decays must be a list or None")
        if decays is not None:
            if len(decays) != n_voices:
                raise ValueError("the length of decays must equal n_voices")
            if any((not is_number(p) or p <= 0 or p > 1) for p in decays):
                raise ValueError(
                    "the elements of decays must be between 0 and 1"
                )
        else:
            decays = [random.uniform(0.3, 0.4) for _ in range(n_voices)]

        # validate speeds
        if not (speeds is None or isinstance(speeds, list)):
            raise ValueError("speeds must be a list or None")
        if speeds is not None:
            if len(speeds) != n_voices:
                raise ValueError("the length of speeds must equal n_voices")
            if any((not is_number(p) or p <= 0) for p in speeds):
                raise ValueError("the elements of speeds must be numbers > 0")
        else:
            speeds = [random.uniform(0.25, 0.4) for _ in range(n_voices)]

        # validate depths
        if not (depths is None or isinstance(depths, list)):
            raise ValueError("depths must be a list or None")
        if depths is not None:
            if len(depths) != n_voices:
                raise ValueError("the length of depths must equal n_voices")
            if any((not is_number(p) or p <= 0) for p in depths):
                raise ValueError("the elements of depths must be numbers > 0")
        else:
            depths = [random.uniform(1.0, 3.0) for _ in range(n_voices)]

        # validate shapes
        if not (shapes is None or isinstance(shapes, list)):
            raise ValueError("shapes must be a list or None")
        if shapes is not None:
            if len(shapes) != n_voices:
                raise ValueError("the length of shapes must equal n_voices")
            if any((p not in ['t', 's']) for p in shapes):
                raise ValueError("the elements of shapes must be 's' or 't'")
        else:
            shapes = [random.choice(['t', 's']) for _ in range(n_voices)]

        effect_args = ['chorus', '{}'.format(gain_in), '{}'.format(gain_out)]

        for i in range(n_voices):
            effect_args.extend([
                '{}'.format(delays[i]),
                '{}'.format(decays[i]),
                '{}'.format(speeds[i]),
                '{}'.format(depths[i]),
                '-{}'.format(shapes[i])
            ])

        self.effects.extend(effect_args)
        self.effects_log.append('chorus')
        return self

    def compand(self, attack_time=0.3, decay_time=0.8, soft_knee_db=6.0,
                tf_points=[(-70, -70), (-60, -20), (0, 0)]):
        '''Compand (compress or expand) the dynamic range of the audio.

        Parameters
        ----------
        attack_time : float, default=0.3
            The time in seconds over which the instantaneous level of the input
            signal is averaged to determine increases in volume.
        decay_time : float, default=0.8
            The time in seconds over which the instantaneous level of the input
            signal is averaged to determine decreases in volume.
        soft_knee_db : float or None, default=6.0
            The ammount (in dB) for which the points at where adjacent line
            segments on the transfer function meet will be rounded.
            If None, no soft_knee is applied.
        tf_points : list of tuples
            Transfer function points as a list of tuples corresponding to
            points in (dB, dB) defining the compander's transfer function.

        See Also
        --------
        mcompand, contrast
        '''
        if not is_number(attack_time) or attack_time <= 0:
            raise ValueError("attack_time must be a positive number.")

        if not is_number(decay_time) or decay_time <= 0:
            raise ValueError("decay_time must be a positive number.")

        if attack_time > decay_time:
            logging.warning(
                "attack_time is larger than decay_time.\n"
                "For most situations, attack_time should be shorter than "
                "decay time because the human ear is more sensitive to sudden "
                "loud music than sudden soft music."
            )

        if not (is_number(soft_knee_db) or soft_knee_db is None):
            raise ValueError("soft_knee_db must be a number or None.")

        if not isinstance(tf_points, list):
            raise TypeError("tf_points must be a list.")
        if len(tf_points) == 0:
            raise ValueError("tf_points must have at least one point.")
        if any(not isinstance(pair, tuple) for pair in tf_points):
            raise ValueError("elements of tf_points must be pairs")
        if any(len(pair) != 2 for pair in tf_points):
            raise ValueError("Tuples in tf_points must be length 2")
        if any(not (is_number(p[0]) and is_number(p[1])) for p in tf_points):
            raise ValueError("Tuples in tf_points must be pairs of numbers.")
        if any((p[0] > 0 or p[1] > 0) for p in tf_points):
            raise ValueError("Tuple values in tf_points must be <= 0 (dB).")
        if len(tf_points) > len(set([p[0] for p in tf_points])):
            raise ValueError("Found duplicate x-value in tf_points.")

        tf_points = sorted(
            tf_points,
            key=lambda tf_points: tf_points[0]
        )
        transfer_list = []
        for point in tf_points:
            transfer_list.extend([
                "{}".format(point[0]), "{}".format(point[1])
            ])

        effect_args = [
            'compand',
            "{},{}".format(attack_time, decay_time)
        ]

        if soft_knee_db is not None:
            effect_args.append(
                "{}:{}".format(soft_knee_db, ",".join(transfer_list))
            )
        else:
            effect_args.append(",".join(transfer_list))

        self.effects.extend(effect_args)
        self.effects_log.append('compand')
        return self

    def contrast(self, amount=75):
        '''Comparable with compression, this effect modifies an audio signal to
        make it sound louder.

        Parameters
        ----------
        amount : float
            Amount of enhancement between 0 and 100.

        See Also
        --------
        compand, mcompand

        '''
        if not is_number(amount) or amount < 0 or amount > 100:
            raise ValueError('amount must be a number between 0 and 100.')

        effect_args = ['contrast', '{}'.format(amount)]

        self.effects.extend(effect_args)
        self.effects_log.append('contrast')
        return self

    def convert(self, samplerate=None, n_channels=None, bitdepth=None):
        '''Converts output audio to the specified format.

        Parameters
        ----------
        samplerate : float, default=None
            Desired samplerate. If None, defaults to the same as input.
        n_channels : int, default=None
            Desired number of channels. If None, defaults to the same as input.
        bitdepth : int, default=None
            Desired bitdepth. If None, defaults to the same as input.

        See Also
        --------
        rate

        '''
        bitdepths = [8, 16, 24, 32, 64]
        if bitdepth is not None:
            if bitdepth not in bitdepths:
                raise ValueError(
                    "bitdepth must be one of {}.".format(str(bitdepths))
                )
            self.output_format.extend(['-b', '{}'.format(bitdepth)])
        if n_channels is not None:
            if not isinstance(n_channels, int) or n_channels <= 0:
                raise ValueError(
                    "n_channels must be a positive integer."
                )
            self.output_format.extend(['-c', '{}'.format(n_channels)])
        if samplerate is not None:
            if not is_number(samplerate) or samplerate <= 0:
                raise ValueError("samplerate must be a positive number.")
            self.rate(samplerate)
        return self

    def dcshift(self, shift=0.0):
        '''Apply a DC shift to the audio.

        Parameters
        ----------
        shift : float
            Amount to shift audio between -2 and 2. (Audio is between -1 and 1)

        See Also
        --------
        highpass

        '''
        if not is_number(shift) or shift < -2 or shift > 2:
            raise ValueError('shift must be a number between -2 and 2.')

        effect_args = ['dcshift', '{}'.format(shift)]

        self.effects.extend(effect_args)
        self.effects_log.append('dcshift')
        return self

    def deemph(self):
        '''Apply Compact Disc (IEC 60908) de-emphasis (a treble attenuation
        shelving filter). Pre-emphasis was applied in the mastering of some
        CDs issued in the early 1980s. These included many classical music
        albums, as well as now sought-after issues of albums by The Beatles,
        Pink Floyd and others. Pre-emphasis should be removed at playback time
        by a de-emphasis filter in the playback device. However, not all modern
        CD players have this filter, and very few PC CD drives have it; playing
        pre-emphasised audio without the correct de-emphasis filter results in
        audio that sounds harsh and is far from what its creators intended.

        The de-emphasis filter is implemented as a biquad and requires the
        input audio sample rate to be either 44.1kHz or 48kHz. Maximum
        deviation from the ideal response is only 0.06dB (up to 20kHz).

        See Also
        --------
        bass, treble
        '''
        effect_args = ['deemph']

        self.effects.extend(effect_args)
        self.effects_log.append('deemph')
        return self

    def delay(self, positions):
        '''Delay one or more audio channels such that they start at the given
        positions.

        Parameters
        ----------
        positions: list of floats
            List of times (in seconds) to delay each audio channel.
            If fewer positions are given than the number of channels, the
            remaining channels will be unaffected.

        '''
        if not isinstance(positions, list):
            raise ValueError("positions must be a a list of numbers")

        if not all((is_number(p) and p >= 0) for p in positions):
            raise ValueError("positions must be positive nubmers")

        effect_args = ['delay']
        effect_args.extend(['{}'.format(p) for p in positions])

        self.effects.extend(effect_args)
        self.effects_log.append('delay')
        return self

    def downsample(self, factor=2):
        '''Downsample the signal by an integer factor. Only the first out of
        each factor samples is retained, the others are discarded.

        No decimation filter is applied. If the input is not a properly
        bandlimited baseband signal, aliasing will occur. This may be desirable
        e.g., for frequency translation.

        For a general resampling effect with anti-aliasing, see rate.

        Parameters
        ----------
        factor : int, default=2
            Downsampling factor.

        See Also
        --------
        rate, upsample

        '''
        if not isinstance(factor, int) or factor < 1:
            raise ValueError('factor must be a positive integer.')

        effect_args = ['downsample', '{}'.format(factor)]

        self.effects.extend(effect_args)
        self.effects_log.append('downsample')
        return self

    def earwax(self):
        '''Makes audio easier to listen to on headphones. Adds ‘cues’ to 44.1kHz
        stereo audio so that when listened to on headphones the stereo image is
        moved from inside your head (standard for headphones) to outside and in
        front of the listener (standard for speakers).

        Warning: Will only work properly on 44.1kHz stereo audio!

        '''
        effect_args = ['earwax']

        self.effects.extend(effect_args)
        self.effects_log.append('earwax')
        return self

    def echo(self, gain_in=0.8, gain_out=0.9, n_echos=1, delays=[60],
             decays=[0.4]):
        '''Add echoing to the audio.

        Echoes are reflected sound and can occur naturally amongst mountains
        (and sometimes large buildings) when talking or shouting; digital echo
        effects emulate this behav- iour and are often used to help fill out
        the sound of a single instrument or vocal. The time differ- ence
        between the original signal and the reflection is the 'delay' (time),
        and the loudness of the reflected signal is the 'decay'. Multiple
        echoes can have different delays and decays.

        Parameters
        ----------
        gain_in : float, default=0.8
            Input volume, between 0 and 1
        gain_out : float, default=0.9
            Output volume, between 0 and 1
        n_echos : int, default=1
            Number of reflections
        delays : list, default=[60]
            List of delays in miliseconds
        decays : list, default=[0.4]
            List of decays, relative to gain in between 0 and 1

        See Also
        --------
        echos, reverb, chorus
        '''
        if not is_number(gain_in) or gain_in <= 0 or gain_in > 1:
            raise ValueError("gain_in must be a number between 0 and 1.")

        if not is_number(gain_out) or gain_out <= 0 or gain_out > 1:
            raise ValueError("gain_out must be a number between 0 and 1.")

        if not isinstance(n_echos, int) or n_echos <= 0:
            raise ValueError("n_echos must be a positive integer.")

        # validate delays
        if not isinstance(delays, list):
            raise ValueError("delays must be a list")

        if len(delays) != n_echos:
            raise ValueError("the length of delays must equal n_echos")

        if any((not is_number(p) or p <= 0) for p in delays):
            raise ValueError("the elements of delays must be numbers > 0")

        # validate decays
        if not isinstance(decays, list):
            raise ValueError("decays must be a list")

        if len(decays) != n_echos:
            raise ValueError("the length of decays must equal n_echos")
        if any((not is_number(p) or p <= 0 or p > 1) for p in decays):
            raise ValueError(
                "the elements of decays must be between 0 and 1"
            )

        effect_args = ['echo', '{}'.format(gain_in), '{}'.format(gain_out)]

        for i in range(n_echos):
            effect_args.extend([
                '{}'.format(delays[i]),
                '{}'.format(decays[i])
            ])

        self.effects.extend(effect_args)
        self.effects_log.append('echo')
        return self

    def echos(self, gain_in=0.8, gain_out=0.9, n_echos=1, delays=[60],
              decays=[0.4]):
        '''Add a sequence of echoes to the audio.

        Like the echo effect, echos stand for ‘ECHO in Sequel’, that is the
        first echos takes the input, the second the input and the first echos,
        the third the input and the first and the second echos, ... and so on.
        Care should be taken using many echos; a single echos has the same
        effect as a single echo.

        Parameters
        ----------
        gain_in : float, default=0.8
            Input volume, between 0 and 1
        gain_out : float, default=0.9
            Output volume, between 0 and 1
        n_echos : int, default=1
            Number of reflections
        delays : list, default=[60]
            List of delays in miliseconds
        decays : list, default=[0.4]
            List of decays, relative to gain in between 0 and 1

        See Also
        --------
        echo, reverb, chorus
        '''
        if not is_number(gain_in) or gain_in <= 0 or gain_in > 1:
            raise ValueError("gain_in must be a number between 0 and 1.")

        if not is_number(gain_out) or gain_out <= 0 or gain_out > 1:
            raise ValueError("gain_out must be a number between 0 and 1.")

        if not isinstance(n_echos, int) or n_echos <= 0:
            raise ValueError("n_echos must be a positive integer.")

        # validate delays
        if not isinstance(delays, list):
            raise ValueError("delays must be a list")

        if len(delays) != n_echos:
            raise ValueError("the length of delays must equal n_echos")

        if any((not is_number(p) or p <= 0) for p in delays):
            raise ValueError("the elements of delays must be numbers > 0")

        # validate decays
        if not isinstance(decays, list):
            raise ValueError("decays must be a list")

        if len(decays) != n_echos:
            raise ValueError("the length of decays must equal n_echos")
        if any((not is_number(p) or p <= 0 or p > 1) for p in decays):
            raise ValueError(
                "the elements of decays must be between 0 and 1"
            )

        effect_args = ['echos', '{}'.format(gain_in), '{}'.format(gain_out)]

        for i in range(n_echos):
            effect_args.extend([
                '{}'.format(delays[i]),
                '{}'.format(decays[i])
            ])

        self.effects.extend(effect_args)
        self.effects_log.append('echos')
        return self

    def equalizer(self, frequency, width_q, gain_db):
        '''Apply a two-pole peaking equalisation (EQ) filter to boost or
        reduce around a given frequency.
        This effect can be applied multiple times to produce complex EQ curves.

        Parameters
        ----------
        frequency : float
            The filter's central frequency in Hz.
        width_q : float
            The filter's width as a Q-factor.
        gain_db : float
            The filter's gain in dB.

        See Also
        --------
        bass, treble

        '''
        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(width_q) or width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if not is_number(gain_db):
            raise ValueError("gain_db must be a number.")

        effect_args = [
            'equalizer',
            '{}'.format(frequency),
            '{}q'.format(width_q),
            '{}'.format(gain_db)
        ]
        self.effects.extend(effect_args)
        self.effects_log.append('equalizer')
        return self

    def fade(self, fade_in_len=0.0, fade_out_len=0.0, fade_shape='q'):
        '''Add a fade in and/or fade out to an audio file.
        Default fade shape is 1/4 sine wave.

        Parameters
        ----------
        fade_in_len : float, default=0.0
            Length of fade-in (seconds). If fade_in_len = 0,
            no fade in is applied.
        fade_out_len : float, defaut=0.0
            Length of fade-out (seconds). If fade_out_len = 0,
            no fade in is applied.
        fade_shape : str, default='q'
            Shape of fade. Must be one of
             * 'q' for quarter sine (default),
             * 'h' for half sine,
             * 't' for linear,
             * 'l' for logarithmic
             * 'p' for inverted parabola.

        See Also
        --------
        splice

        '''
        fade_shapes = ['q', 'h', 't', 'l', 'p']
        if fade_shape not in fade_shapes:
            raise ValueError(
                "Fade shape must be one of {}".format(" ".join(fade_shapes))
            )
        if not is_number(fade_in_len) or fade_in_len < 0:
            raise ValueError("fade_in_len must be a nonnegative number.")
        if not is_number(fade_out_len) or fade_out_len < 0:
            raise ValueError("fade_out_len must be a nonnegative number.")

        effect_args = []

        if fade_in_len > 0:
            effect_args.extend([
                'fade', str(fade_shape), str(fade_in_len)
            ])

        if fade_out_len > 0:
            effect_args.extend([
                'reverse', 'fade', str(fade_shape),
                str(fade_out_len), 'reverse'
            ])

        if len(effect_args) > 0:
            self.effects.extend(effect_args)
            self.effects_log.append('fade')

        return self

    def fir(self, coefficients):
        '''Use SoX’s FFT convolution engine with given FIR filter coefficients.

        Parameters
        ----------
        coefficients : list
            fir filter coefficients

        '''
        if not isinstance(coefficients, list):
            raise ValueError("coefficients must be a list.")

        if not all([is_number(c) for c in coefficients]):
            raise ValueError("coefficients must be numbers.")

        effect_args = ['fir']
        effect_args.extend(['{}'.format(c) for c in coefficients])

        self.effects.extend(effect_args)
        self.effects_log.append('fir')

        return self

    def flanger(self, delay=0, depth=2, regen=0, width=71, speed=0.5,
                shape='sine', phase=25, interp='linear'):
        '''Apply a flanging effect to the audio.

        Parameters
        ----------
        delay : float, default=0
            Base delay (in miliseconds) between 0 and 30.
        depth : float, default=2
            Added swept delay (in miliseconds) between 0 and 10.
        regen : float, default=0
            Percentage regeneration between -95 and 95.
        width : float, default=71,
            Percentage of delayed signal mixed with original between 0 and 100.
        speed : float, default=0.5
            Sweeps per second (in Hz) between 0.1 and 10.
        shape : 'sine' or 'triangle', default='sine'
            Swept wave shape
        phase : float, default=25
            Swept wave percentage phase-shift for multi-channel flange between
            0 and 100. 0 = 100 = same phase on each channel
        interp : 'linear' or 'quadratic', default='linear'
            Digital delay-line interpolation type.

        See Also
        --------
        tremolo
        '''
        if not is_number(delay) or delay < 0 or delay > 30:
            raise ValueError("delay must be a number between 0 and 30.")
        if not is_number(depth) or depth < 0 or depth > 10:
            raise ValueError("depth must be a number between 0 and 10.")
        if not is_number(regen) or regen < -95 or regen > 95:
            raise ValueError("regen must be a number between -95 and 95.")
        if not is_number(width) or width < 0 or width > 100:
            raise ValueError("width must be a number between 0 and 100.")
        if not is_number(speed) or speed < 0.1 or speed > 10:
            raise ValueError("speed must be a number between 0.1 and 10.")
        if shape not in ['sine', 'triangle']:
            raise ValueError("shape must be one of 'sine' or 'triangle'.")
        if not is_number(phase) or phase < 0 or phase > 100:
            raise ValueError("phase must be a number between 0 and 100.")
        if interp not in ['linear', 'quadratic']:
            raise ValueError("interp must be one of 'linear' or 'quadratic'.")

        effect_args = [
            'flanger',
            '{}'.format(delay),
            '{}'.format(depth),
            '{}'.format(regen),
            '{}'.format(width),
            '{}'.format(speed),
            '{}'.format(shape),
            '{}'.format(phase),
            '{}'.format(interp)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('flanger')

        return self

    def gain(self, gain_db=0.0, normalize=True, limiter=False, balance=None):
        '''Apply amplification or attenuation to the audio signal.

        Parameters
        ----------
        gain_db : float, default=0.0
            Gain adjustment in decibels (dB).
        normalize : bool, default=True
            If True, audio is normalized to gain_db relative to full scale.
            If False, simply adjusts the audio power level by gain_db.
        limiter : bool, default=False
            If True, a simple limiter is invoked to prevent clipping.
        balance : str or None, default=None
            Balance gain across channels. Can be one of:
             * None applies no balancing (default)
             * 'e' applies gain to all channels other than that with the
                highest peak level, such that all channels attain the same
                peak level
             * 'B' applies gain to all channels other than that with the
                highest RMS level, such that all channels attain the same
                RMS level
             * 'b' applies gain with clipping protection to all channels other
                than that with the highest RMS level, such that all channels
                attain the same RMS level
            If normalize=True, 'B' and 'b' are equivalent.

        See Also
        --------
        loudness

        '''
        if not is_number(gain_db):
            raise ValueError("gain_db must be a number.")

        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a boolean.")

        if not isinstance(limiter, bool):
            raise ValueError("limiter must be a boolean.")

        if balance not in [None, 'e', 'B', 'b']:
            raise ValueError("balance must be one of None, 'e', 'B', or 'b'.")

        effect_args = ['gain']

        if balance is not None:
            effect_args.append('-{}'.format(balance))

        if normalize:
            effect_args.append('-n')

        if limiter:
            effect_args.append('-l')

        effect_args.append('{}'.format(gain_db))
        self.effects.extend(effect_args)
        self.effects_log.append('gain')

        return self

    def highpass(self, frequency, width_q=0.707, n_poles=2):
        '''Apply a high-pass filter with 3dB point frequency. The filter can be
        either single-pole or double-pole. The filters roll off at 6dB per pole
        per octave (20dB per pole per decade).

        Parameters
        ----------
        frequency : float
            The filter's cutoff frequency in Hz.
        width_q : float, default=0.707
            The filter's width as a Q-factor. Applies only when n_poles=2.
            The default gives a Butterworth response.
        n_poles : int, default=2
            The number of poles in the filter. Must be either 1 or 2

        See Also
        --------
        lowpass, equalizer, sinc, allpass

        '''
        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(width_q) or width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if n_poles not in [1, 2]:
            raise ValueError("n_poles must be 1 or 2.")

        effect_args = [
            'highpass', '-{}'.format(n_poles), '{}'.format(frequency)
        ]

        if n_poles == 2:
            effect_args.append('{}q'.format(width_q))

        self.effects.extend(effect_args)
        self.effects_log.append('highpass')

        return self

    def lowpass(self, frequency, width_q=0.707, n_poles=2):
        '''Apply a low-pass filter with 3dB point frequency. The filter can be
        either single-pole or double-pole. The filters roll off at 6dB per pole
        per octave (20dB per pole per decade).

        Parameters
        ----------
        frequency : float
            The filter's cutoff frequency in Hz.
        width_q : float, default=0.707
            The filter's width as a Q-factor. Applies only when n_poles=2.
            The default gives a Butterworth response.
        n_poles : int, default=2
            The number of poles in the filter. Must be either 1 or 2

        See Also
        --------
        highpass, equalizer, sinc, allpass

        '''
        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(width_q) or width_q <= 0:
            raise ValueError("width_q must be a positive number.")

        if n_poles not in [1, 2]:
            raise ValueError("n_poles must be 1 or 2.")

        effect_args = [
            'lowpass', '-{}'.format(n_poles), '{}'.format(frequency)
        ]

        if n_poles == 2:
            effect_args.append('{}q'.format(width_q))

        self.effects.extend(effect_args)
        self.effects_log.append('lowpass')

        return self

    def hilbert(self, num_taps=None):
        '''Apply an odd-tap Hilbert transform filter, phase-shifting the signal
        by 90 degrees. This is used in many matrix coding schemes and for
        analytic signal generation. The process is often written as a
        multiplication by i (or j), the imaginary unit. An odd-tap Hilbert
        transform filter has a bandpass characteristic, attenuating the lowest
        and highest frequencies.

        Parameters
        ----------
        num_taps : int or None, default=None
            Number of filter taps - must be odd. If none, it is chosen to have
            a cutoff frequency of about 75 Hz.

        '''
        if num_taps is not None and not isinstance(num_taps, int):
            raise ValueError("num taps must be None or an odd integer.")

        if num_taps is not None and num_taps % 2 == 0:
            raise ValueError("num_taps must an odd integer.")

        effect_args = ['hilbert']

        if num_taps is not None:
            effect_args.extend(['-n', '{}'.format(num_taps)])

        self.effects.extend(effect_args)
        self.effects_log.append('hilbert')

        return self

    def loudness(self, gain_db=-10.0, reference_level=65.0):
        '''Loudness control. Similar to the gain effect, but provides
        equalisation for the human auditory system.

        The gain is adjusted by gain_db and the signal is equalised according
        to ISO 226 w.r.t. reference_level.

        Parameters
        ----------
        gain_db : float, default=-10.0
            Loudness adjustment amount (in dB)
        reference_level : float, default=65.0
            Reference level (in dB) according to which the signal is equalized.
            Must be between 50 and 75 (dB)

        See Also
        --------
        gain

        '''
        if not is_number(gain_db):
            raise ValueError('gain_db must be a number.')

        if not is_number(reference_level):
            raise ValueError('reference_level must be a number')

        if reference_level > 75 or reference_level < 50:
            raise ValueError('reference_level must be between 50 and 75')

        effect_args = [
            'loudness',
            '{}'.format(gain_db),
            '{}'.format(reference_level)
        ]
        self.effects.extend(effect_args)
        self.effects_log.append('loudness')

        return self

    def mcompand(self, n_bands=2, crossover_frequencies=[1600],
                 attack_time=[0.005, 0.000625], decay_time=[0.1, 0.0125],
                 soft_knee_db=[6.0, None],
                 tf_points=[[(-47, -40), (-34, -34), (-17, -33), (0, 0)],
                 [(-47, -40), (-34, -34), (-15, -33), (0, 0)]]):

        '''The multi-band compander is similar to the single-band compander but
        the audio is first divided into bands using Linkwitz-Riley cross-over
        filters and a separately specifiable compander run on each band.

        When used with n_bands=1, this effect is identical to compand.
        When using n_bands > 1, the first set of arguments applies a single
        band compander, and each subsequent set of arugments is applied on
        each of the crossover frequencies.

        Parameters
        ----------
        n_bands : int, default=2
            The number of bands.
        crossover_frequencies : list of float, default=[1600]
            A list of crossover frequencies in Hz of length n_bands-1.
            The first band is always the full spectrum, followed by the bands
            specified by crossover_frequencies.
        attack_time : list of float, default=[0.005, 0.000625]
            A list of length n_bands, where each element is the time in seconds
            over which the instantaneous level of the input signal is averaged
            to determine increases in volume over the current band.
        decay_time : list of float, default=[0.1, 0.0125]
            A list of length n_bands, where each element is the time in seconds
            over which the instantaneous level of the input signal is averaged
            to determine decreases in volume over the current band.
        soft_knee_db : list of float or None, default=[6.0, None]
            A list of length n_bands, where each element is the ammount (in dB)
            for which the points at where adjacent line segments on the
            transfer function meet will be rounded over the current band.
            If None, no soft_knee is applied.
        tf_points : list of list of tuples, default=[
                [(-47, -40), (-34, -34), (-17, -33), (0, 0)],
                [(-47, -40), (-34, -34), (-15, -33), (0, 0)]]
            A list of length n_bands, where each element is the transfer
            function points as a list of tuples corresponding to points in
            (dB, dB) defining the compander's transfer function over the
            current band.

        See Also
        --------
        compand, contrast

        '''
        if not isinstance(n_bands, int) or n_bands < 1:
            raise ValueError("n_bands must be a positive integer.")

        if (not isinstance(crossover_frequencies, list) or
                len(crossover_frequencies) != n_bands - 1):
            raise ValueError(
                "crossover_frequences must be a list of length n_bands - 1"
            )

        if any([not is_number(f) or f < 0 for f in crossover_frequencies]):
            raise ValueError(
                "crossover_frequencies elements must be positive floats."
            )

        if not isinstance(attack_time, list) or len(attack_time) != n_bands:
            raise ValueError("attack_time must be a list of length n_bands")

        if any([not is_number(a) or a <= 0 for a in attack_time]):
            raise ValueError("attack_time elements must be positive numbers.")

        if not isinstance(decay_time, list) or len(decay_time) != n_bands:
            raise ValueError("decay_time must be a list of length n_bands")

        if any([not is_number(d) or d <= 0 for d in decay_time]):
            raise ValueError("decay_time elements must be positive numbers.")

        if any([a > d for a, d in zip(attack_time, decay_time)]):
            logging.warning(
                "Elements of attack_time are larger than decay_time.\n"
                "For most situations, attack_time should be shorter than "
                "decay time because the human ear is more sensitive to sudden "
                "loud music than sudden soft music."
            )

        if not isinstance(soft_knee_db, list) or len(soft_knee_db) != n_bands:
            raise ValueError("soft_knee_db must be a list of length n_bands.")

        if any([(not is_number(d) and d is not None) for d in soft_knee_db]):
            raise ValueError(
                "elements of soft_knee_db must be a number or None."
            )

        if not isinstance(tf_points, list) or len(tf_points) != n_bands:
            raise ValueError("tf_points must be a list of length n_bands.")

        if any([not isinstance(t, list) or len(t) == 0 for t in tf_points]):
            raise ValueError(
                "tf_points must be a list with at least one point."
            )

        for tfp in tf_points:
            if any(not isinstance(pair, tuple) for pair in tfp):
                raise ValueError("elements of tf_points lists must be pairs")
            if any(len(pair) != 2 for pair in tfp):
                raise ValueError("Tuples in tf_points lists must be length 2")
            if any(not (is_number(p[0]) and is_number(p[1])) for p in tfp):
                raise ValueError(
                    "Tuples in tf_points lists must be pairs of numbers."
                )
            if any((p[0] > 0 or p[1] > 0) for p in tfp):
                raise ValueError(
                    "Tuple values in tf_points lists must be <= 0 (dB)."
                )
            if len(tf_points) > len(set([p[0] for p in tfp])):
                raise ValueError("Found duplicate x-value in tf_points list.")

        effect_args = ['mcompand']

        for i in range(n_bands):

            if i > 0:
                effect_args.append('{}'.format(crossover_frequencies[i - 1]))

            intermed_args = ["{},{}".format(attack_time[i], decay_time[i])]

            tf_points_band = tf_points[i]
            tf_points_band = sorted(
                tf_points_band,
                key=lambda tf_points_band: tf_points_band[0]
            )
            transfer_list = []
            for point in tf_points_band:
                transfer_list.extend([
                    "{}".format(point[0]), "{}".format(point[1])
                ])

            if soft_knee_db[i] is not None:
                intermed_args.append(
                    "{}:{}".format(soft_knee_db[i], ",".join(transfer_list))
                )
            else:
                intermed_args.append(",".join(transfer_list))

            effect_args.append('"{}"'.format(' '.join(intermed_args)))

        self.effects.extend(effect_args)
        self.effects_log.append('mcompand')
        return self

    def norm(self, db_level=-3.0):
        '''Normalize an audio file to a particular db level.
        This behaves identically to the gain effect with normalize=True.

        Parameters
        ----------
        db_level : float, default=-3.0
            Output volume (db)

        See Also
        --------
        gain, loudness

        '''
        if not is_number(db_level):
            raise ValueError('db_level must be a number.')

        effect_args = [
            'norm',
            '{}'.format(db_level)
        ]
        self.effects.extend(effect_args)
        self.effects_log.append('norm')

        return self

    def oops(self):
        '''Out Of Phase Stereo effect. Mixes stereo to twin-mono where each
        mono channel contains the difference between the left and right stereo
        channels. This is sometimes known as the 'karaoke' effect as it often
        has the effect of removing most or all of the vocals from a recording.

        '''
        effect_args = ['oops']
        self.effects.extend(effect_args)
        self.effects_log.append('oops')

        return self

    def overdrive(self, gain_db=20.0, colour=20.0):
        '''Apply non-linear distortion.

        Parameters
        ----------
        gain_db : float, default=20
            Controls the amount of distortion (dB).
        colour : float, default=20
            Controls the amount of even harmonic content in the output (dB).

        '''
        if not is_number(gain_db):
            raise ValueError('db_level must be a number.')

        if not is_number(colour):
            raise ValueError('colour must be a number.')

        effect_args = [
            'overdrive',
            '{}'.format(gain_db),
            '{}'.format(colour)
        ]
        self.effects.extend(effect_args)
        self.effects_log.append('overdrive')

        return self

    def pad(self, start_duration=0.0, end_duration=0.0):
        '''Add silence to the beginning or end of a file.
        Calling this with the default arguments has no effect.

        Parameters
        ----------
        start_duration : float
            Number of seconds of silence to add to beginning.
        end_duration : float
            Number of seconds of silence to add to end.

        See Also
        --------
        delay

        '''
        if not is_number(start_duration) or start_duration < 0:
            raise ValueError("Start duration must be a positive number.")

        if not is_number(end_duration) or end_duration < 0:
            raise ValueError("End duration must be positive.")

        effect_args = [
            'pad',
            '{}'.format(start_duration),
            '{}'.format(end_duration)
        ]
        self.effects.extend(effect_args)
        self.effects_log.append('pad')

        return self

    def phaser(self, gain_in=0.8, gain_out=0.74, delay=3, decay=0.4, speed=0.5,
               modulation_shape='sinusoidal'):
        '''Apply a phasing effect to the audio.

        Parameters
        ----------
        gain_in : float, default=0.8
            Input volume between 0 and 1
        gain_out: float, default=0.74
            Output volume between 0 and 1
        delay : float, default=3
            Delay in miliseconds between 0 and 5
        decay : float, default=0.4
            Decay relative to gain_in, between 0.1 and 0.5.
        speed : float, default=0.5
            Modulation speed in Hz, between 0.1 and 2
        modulation_shape : str, defaul='sinusoidal'
            Modulation shpae. One of 'sinusoidal' or 'triangular'

        See Also
        --------
        flanger, tremolo
        '''
        if not is_number(gain_in) or gain_in <= 0 or gain_in > 1:
            raise ValueError("gain_in must be a number between 0 and 1.")

        if not is_number(gain_out) or gain_out <= 0 or gain_out > 1:
            raise ValueError("gain_out must be a number between 0 and 1.")

        if not is_number(delay) or delay <= 0 or delay > 5:
            raise ValueError("delay must be a positive number.")

        if not is_number(decay) or decay < 0.1 or decay > 0.5:
            raise ValueError("decay must be a number between 0.1 and 0.5.")

        if not is_number(speed) or speed < 0.1 or speed > 2:
            raise ValueError("speed must be a positive number.")

        if modulation_shape not in ['sinusoidal', 'triangular']:
            raise ValueError(
                "modulation_shape must be one of 'sinusoidal', 'triangular'."
            )

        effect_args = [
            'phaser',
            '{}'.format(gain_in),
            '{}'.format(gain_out),
            '{}'.format(delay),
            '{}'.format(decay),
            '{}'.format(speed)
        ]

        if modulation_shape == 'sinusoidal':
            effect_args.append('-s')
        elif modulation_shape == 'triangular':
            effect_args.append('-t')

        self.effects.extend(effect_args)
        self.effects_log.append('phaser')

        return self

    def pitch(self, n_semitones, quick=False):
        '''Pitch shift the audio without changing the tempo.

        This effect uses the WSOLA algorithm. The audio is chopped up into
        segments which are then shifted in the time domain and overlapped
        (cross-faded) at points where their waveforms are most similar as
        determined by measurement of least squares.

        Parameters
        ----------
        n_semitones : float
            The number of semitones to shift. Can be positive or negative.
        quick : bool, default=False
            If True, this effect will run faster but with lower sound quality.

        See Also
        --------
        bend, speed, tempo

        '''
        if not is_number(n_semitones):
            raise ValueError("n_semitones must be a positive number")

        if n_semitones < -12 or n_semitones > 12:
            logging.warning(
                "Using an extreme pitch shift. "
                "Quality of results will be poor"
            )

        if not isinstance(quick, bool):
            raise ValueError("quick must be a boolean.")

        effect_args = ['pitch']

        if quick:
            effect_args.append('-q')

        effect_args.append('{}'.format(n_semitones * 100.))

        self.effects.extend(effect_args)
        self.effects_log.append('pitch')

        return self

    def rate(self, samplerate, quality='h'):
        '''Change the audio sampling rate (i.e. resample the audio) to any
        given `samplerate`. Better the resampling quality = slower runtime.

        Parameters
        ----------
        samplerate : float
            Desired sample rate.
        quality : str
            Resampling quality. One of:
             * q : Quick - very low quality,
             * l : Low,
             * m : Medium,
             * h : High (default),
             * v : Very high
        silence_threshold : float
            Silence threshold as percentage of maximum sample amplitude.
        min_silence_duration : float
            The minimum ammount of time in seconds required for a region to be
            considered non-silent.
        buffer_around_silence : bool
            If True, leaves a buffer of min_silence_duration around removed
            silent regions.

        See Also
        --------
        upsample, downsample, convert

        '''
        quality_vals = ['q', 'l', 'm', 'h', 'v']
        if not is_number(samplerate) or samplerate <= 0:
            raise ValueError("Samplerate must be a positive number.")

        if quality not in quality_vals:
            raise ValueError(
                "Quality must be one of {}.".format(' '.join(quality_vals))
            )

        effect_args = [
            'rate',
            '-{}'.format(quality),
            '{}'.format(samplerate)
        ]
        self.effects.extend(effect_args)
        self.effects_log.append('rate')

        return self

    def remix(self, remix_dictionary=None, num_output_channels=None):
        '''Remix the channels of an audio file.

        Note: volume options are not yet implemented

        Parameters
        ----------
        remix_dictionary : dict or None
            Dictionary mapping output channel to list of input channel(s).
            Empty lists indicate the corresponding output channel should be
            empty. If None, mixes all channels down to a single mono file.
        num_output_channels : int or None
            The number of channels in the output file. If None, the number of
            output channels is equal to the largest key in remix_dictionary.
            If remix_dictionary is None, this variable is ignored.

        Examples
        --------
        Remix a 4-channel input file. The output file will have
        input channel 2 in channel 1, a mixdown of input channels 1 an 3 in
        channel 2, an empty channel 3, and a copy of input channel 4 in
        channel 4.

        >>> import sox
        >>> tfm = sox.Transformer()
        >>> remix_dictionary = {1: [2], 2: [1, 3], 4: [4]}
        >>> tfm.remix(remix_dictionary)

        '''
        if not (isinstance(remix_dictionary, dict) or remix_dictionary is None):
            raise ValueError("remix_dictionary must be a dictionary or None.")

        if remix_dictionary is not None:

            if not all([isinstance(i, int) and i > 0 for i
                        in remix_dictionary.keys()]):
                raise ValueError(
                    "remix dictionary must have positive integer keys."
                )

            if not all([isinstance(v, list) for v
                        in remix_dictionary.values()]):
                raise ValueError("remix dictionary values must be lists.")

            for v_list in remix_dictionary.values():
                if not all([isinstance(v, int) and v > 0 for v in v_list]):
                    raise ValueError(
                        "elements of remix dictionary values must "
                        "be positive integers"
                    )

        if not ((isinstance(num_output_channels, int) and
                 num_output_channels > 0) or num_output_channels is None):
            raise ValueError(
                "num_output_channels must be a positive integer or None."
            )

        effect_args = ['remix']
        if remix_dictionary is None:
            effect_args.append('-')
        else:
            if num_output_channels is None:
                num_output_channels = max(remix_dictionary.keys())

            for channel in range(1, num_output_channels + 1):
                if channel in remix_dictionary.keys():
                    out_channel = ','.join(
                        [str(i) for i in remix_dictionary[channel]]
                    )
                else:
                    out_channel = '0'

                effect_args.append(out_channel)

        self.effects.extend(effect_args)
        self.effects_log.append('remix')

        return self

    def repeat(self, count=1):
        '''Repeat the entire audio count times.

        Parameters
        ----------
        count : int, default=1
            The number of times to repeat the audio.

        '''
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a postive integer.")

        effect_args = ['repeat', '{}'.format(count)]
        self.effects.extend(effect_args)
        self.effects_log.append('repeat')

    def reverb(self, reverberance=50, high_freq_damping=50, room_scale=100,
               stereo_depth=100, pre_delay=0, wet_gain=0, wet_only=False):
        '''Add reverberation to the audio using the ‘freeverb’ algorithm.
        A reverberation effect is sometimes desirable for concert halls that
        are too small or contain so many people that the hall’s natural
        reverberance is diminished. Applying a small amount of stereo reverb
        to a (dry) mono signal will usually make it sound more natural.

        Parameters
        ----------
        reverberance : float, default=50
            Percentage of reverberance
        high_freq_damping : float, default=50
            Percentage of high-frequency damping.
        room_scale : float, default=100
            Scale of the room as a percentage.
        stereo_depth : float, default=100
            Stereo depth as a percentage.
        pre_delay : float, default=0
            Pre-delay in milliseconds.
        wet_gain : float, default=0
            Amount of wet gain in dB
        wet_only : bool, default=False
            If True, only outputs the wet signal.

        See Also
        --------
        echo

        '''

        if (not is_number(reverberance) or reverberance < 0 or
                reverberance > 100):
            raise ValueError("reverberance must be between 0 and 100")

        if (not is_number(high_freq_damping) or high_freq_damping < 0 or
                high_freq_damping > 100):
            raise ValueError("high_freq_damping must be between 0 and 100")

        if (not is_number(room_scale) or room_scale < 0 or
                room_scale > 100):
            raise ValueError("room_scale must be between 0 and 100")

        if (not is_number(stereo_depth) or stereo_depth < 0 or
                stereo_depth > 100):
            raise ValueError("stereo_depth must be between 0 and 100")

        if not is_number(pre_delay) or pre_delay < 0:
            raise ValueError("pre_delay must be a positive number")

        if not is_number(wet_gain):
            raise ValueError("wet_gain must be a number")

        if not isinstance(wet_only, bool):
            raise ValueError("wet_only must be a boolean.")

        effect_args = ['reverb']

        if wet_only:
            effect_args.append('-w')

        effect_args.extend([
            '{}'.format(reverberance),
            '{}'.format(high_freq_damping),
            '{}'.format(room_scale),
            '{}'.format(stereo_depth),
            '{}'.format(pre_delay),
            '{}'.format(wet_gain)
        ])

        self.effects.extend(effect_args)
        self.effects_log.append('reverb')

        return self

    def reverse(self):
        '''Reverse the audio completely
        '''
        effect_args = ['reverse']
        self.effects.extend(effect_args)
        self.effects_log.append('reverse')

        return self

    def silence(self, location=0, silence_threshold=0.1,
                min_silence_duration=0.1, buffer_around_silence=False):
        '''Removes silent regions from an audio file.

        Parameters
        ----------
        location : int, default=0
            Where to remove silence. One of:
             * 0 to remove silence throughout the file (default),
             * 1 to remove silence from the beginning,
             * -1 to remove silence from the end,
        silence_threshold : float, default=0.1
            Silence threshold as percentage of maximum sample amplitude.
            Must be between 0 and 100.
        min_silence_duration : float, default=0.1
            The minimum ammount of time in seconds required for a region to be
            considered non-silent.
        buffer_around_silence : bool, default=False
            If True, leaves a buffer of min_silence_duration around removed
            silent regions.

        See Also
        --------
        vad

        '''
        if location not in [-1, 0, 1]:
            raise ValueError("location must be one of -1, 0, 1.")

        if not is_number(silence_threshold) or silence_threshold < 0:
            raise ValueError(
                "silence_threshold must be a number between 0 and 100"
            )
        elif silence_threshold >= 100:
            raise ValueError(
                "silence_threshold must be a number between 0 and 100"
            )

        if not is_number(min_silence_duration) or min_silence_duration <= 0:
            raise ValueError(
                "min_silence_duration must be a positive number."
            )

        if not isinstance(buffer_around_silence, bool):
            raise ValueError("buffer_around_silence must be a boolean.")

        effect_args = []

        if location == -1:
            effect_args.append('reverse')

        if buffer_around_silence:
            effect_args.extend(['silence', '-l'])
        else:
            effect_args.append('silence')

        effect_args.extend([
            '1',
            '{}'.format(min_silence_duration),
            '{}%'.format(silence_threshold)
        ])

        if location == 0:
            effect_args.extend([
                '-1',
                '{}'.format(min_silence_duration),
                '{}%'.format(silence_threshold)
            ])

        if location == -1:
            effect_args.append('reverse')

        self.effects.extend(effect_args)
        self.effects_log.append('silence')

        return self

    def sinc(self, filter_type='high', cutoff_freq=3000,
             stop_band_attenuation=120, transition_bw=None,
             phase_response=None):
        '''Apply a sinc kaiser-windowed low-pass, high-pass, band-pass, or
        band-reject filter to the signal.

        Parameters
        ----------
        filter_type : str, default='high'
            Type of filter. One of:
                - 'high' for a high-pass filter
                - 'low' for a low-pass filter
                - 'pass' for a band-pass filter
                - 'reject' for a band-reject filter
        cutoff_freq : float or list, default=3000
            A scalar or length 2 list indicating the filter's critical
            frequencies. The critical frequencies are given in Hz and must be
            positive. For a high-pass or low-pass filter, cutoff_freq
            must be a scalar. For a band-pass or band-reject filter, it must be
            a length 2 list.
        stop_band_attenuation : float, default=120
            The stop band attenuation in dB
        transition_bw : float, list or None, default=None
            The transition band-width in Hz.
            If None, sox's default of 5% of the total bandwith is used.
            If a float, the given transition bandwith is used for both the
            upper and lower bands (if applicable).
            If a list, the first argument is used for the lower band and the
            second for the upper band.
        phase_response : float or None
            The filter's phase response between 0 (minimum) and 100 (maximum).
            If None, sox's default phase repsonse is used.

        See Also
        --------
        band, bandpass, bandreject, highpass, lowpass
        '''
        filter_types = ['high', 'low', 'pass', 'reject']
        if filter_type not in filter_types:
            raise ValueError(
                "filter_type must be one of {}".format(', '.join(filter_types))
            )

        if not (is_number(cutoff_freq) or isinstance(cutoff_freq, list)):
            raise ValueError("cutoff_freq must be a number or a list")

        if filter_type in ['high', 'low'] and isinstance(cutoff_freq, list):
            raise ValueError(
                "For filter types 'high' and 'low', "
                "cutoff_freq must be a float, not a list"
            )

        if filter_type in ['pass', 'reject'] and is_number(cutoff_freq):
            raise ValueError(
                "For filter types 'pass' and 'reject', "
                "cutoff_freq must be a list, not a float"
            )

        if is_number(cutoff_freq) and cutoff_freq <= 0:
            raise ValueError("cutoff_freq must be a postive number")

        if isinstance(cutoff_freq, list):
            if len(cutoff_freq) != 2:
                raise ValueError(
                    "If cutoff_freq is a list it may only have 2 elements."
                )

            if any([not is_number(f) or f <= 0 for f in cutoff_freq]):
                raise ValueError(
                    "elements of cutoff_freq must be positive numbers"
                )

            cutoff_freq = sorted(cutoff_freq)

        if not is_number(stop_band_attenuation) or stop_band_attenuation < 0:
            raise ValueError("stop_band_attenuation must be a positive number")

        if not (is_number(transition_bw) or
                isinstance(transition_bw, list) or transition_bw is None):
            raise ValueError("transition_bw must be a number, a list or None.")

        if filter_type in ['high', 'low'] and isinstance(transition_bw, list):
            raise ValueError(
                "For filter types 'high' and 'low', "
                "transition_bw must be a float, not a list"
            )

        if is_number(transition_bw) and transition_bw <= 0:
            raise ValueError("transition_bw must be a postive number")

        if isinstance(transition_bw, list):
            if any([not is_number(f) or f <= 0 for f in transition_bw]):
                raise ValueError(
                    "elements of transition_bw must be positive numbers"
                )
            if len(transition_bw) != 2:
                raise ValueError(
                    "If transition_bw is a list it may only have 2 elements."
                )

        if phase_response is not None and not is_number(phase_response):
            raise ValueError("phase_response must be a number or None.")

        if (is_number(phase_response) and
                (phase_response < 0 or phase_response > 100)):
            raise ValueError("phase response must be between 0 and 100")

        effect_args = ['sinc']
        effect_args.extend(['-a', '{}'.format(stop_band_attenuation)])

        if phase_response is not None:
            effect_args.extend(['-p', '{}'.format(phase_response)])

        if filter_type == 'high':
            if transition_bw is not None:
                effect_args.extend(['-t', '{}'.format(transition_bw)])
            effect_args.append('{}'.format(cutoff_freq))
        elif filter_type == 'low':
            effect_args.append('-{}'.format(cutoff_freq))
            if transition_bw is not None:
                effect_args.extend(['-t', '{}'.format(transition_bw)])
        else:
            if is_number(transition_bw):
                effect_args.extend(['-t', '{}'.format(transition_bw)])
            elif isinstance(transition_bw, list):
                effect_args.extend(['-t', '{}'.format(transition_bw[0])])

        if filter_type == 'pass':
            effect_args.append('{}-{}'.format(cutoff_freq[0], cutoff_freq[1]))
        elif filter_type == 'reject':
            effect_args.append('{}-{}'.format(cutoff_freq[1], cutoff_freq[0]))

        if isinstance(transition_bw, list):
            effect_args.extend(['-t', '{}'.format(transition_bw[1])])

        self.effects.extend(effect_args)
        self.effects_log.append('sinc')
        return self

    def speed(self, factor):
        '''Adjust the audio speed (pitch and tempo together).

        Technically, the speed effect only changes the sample rate information,
        leaving the samples themselves untouched. The rate effect is invoked
        automatically to resample to the output sample rate, using its default
        quality/speed. For higher quality or higher speed resampling, in
        addition to the speed effect, specify the rate effect with the desired
        quality option.

        Parameters
        ----------
        factor : float
            The ratio of the new speed to the old speed.
            For ex. 1.1 speeds up the audio by 10%; 0.9 slows it down by 10%.
            Note - this argument is the inverse of what is passed to the sox
            stretch effect for consistency with speed.

        See Also
        --------
        rate, tempo, pitch
        '''
        if not is_number(factor) or factor <= 0:
            raise ValueError("factor must be a positive number")

        if factor < 0.5 or factor > 2:
            logging.warning(
                "Using an extreme factor. Quality of results will be poor"
            )

        effect_args = ['speed', '{}'.format(factor)]

        self.effects.extend(effect_args)
        self.effects_log.append('speed')

        return self

    def swap(self):
        '''Swap stereo channels. If the input is not stereo, pairs of channels
        are swapped, and a possible odd last channel passed through.

        E.g., for seven channels, the output order will be 2, 1, 4, 3, 6, 5, 7.

        See Also
        ----------
        remix

        '''
        effect_args = ['swap']
        self.effects.extend(effect_args)
        self.effects_log.append('swap')

        return self

    def stretch(self, factor, window=20):
        '''Change the audio duration (but not its pitch).
        **Unless factor is close to 1, use the tempo effect instead.**

        This effect is broadly equivalent to the tempo effect with search set
        to zero, so in general, its results are comparatively poor; it is
        retained as it can sometimes out-perform tempo for small factors.

        Parameters
        ----------
        factor : float
            The ratio of the new tempo to the old tempo.
            For ex. 1.1 speeds up the tempo by 10%; 0.9 slows it down by 10%.
            Note - this argument is the inverse of what is passed to the sox
            stretch effect for consistency with tempo.
        window : float, default=20
            Window size in miliseconds

        See Also
        --------
        tempo, speed, pitch

        '''
        if not is_number(factor) or factor <= 0:
            raise ValueError("factor must be a positive number")

        if factor < 0.5 or factor > 2:
            logging.warning(
                "Using an extreme time stretching factor. "
                "Quality of results will be poor"
            )

        if abs(factor - 1.0) > 0.1:
            logging.warning(
                "For this stretch factor, "
                "the tempo effect has better performance."
            )

        if not is_number(window) or window <= 0:
            raise ValueError(
                "window must be a positive number."
            )

        effect_args = ['stretch', '{}'.format(factor), '{}'.format(window)]

        self.effects.extend(effect_args)
        self.effects_log.append('stretch')

        return self

    def tempo(self, factor, audio_type=None, quick=False):
        '''Time stretch audio without changing pitch.

        This effect uses the WSOLA algorithm. The audio is chopped up into
        segments which are then shifted in the time domain and overlapped
        (cross-faded) at points where their waveforms are most similar as
        determined by measurement of least squares.

        Parameters
        ----------
        factor : float
            The ratio of new tempo to the old tempo.
            For ex. 1.1 speeds up the tempo by 10%; 0.9 slows it down by 10%.
        audio_type : str
            Type of audio, which optimizes algorithm parameters. One of:
             * m : Music,
             * s : Speech,
             * l : Linear (useful when factor is close to 1),
        quick : bool, default=False
            If True, this effect will run faster but with lower sound quality.

        See Also
        --------
        stretch, speed, pitch

        '''
        if not is_number(factor) or factor <= 0:
            raise ValueError("factor must be a positive number")

        if factor < 0.5 or factor > 2:
            logging.warning(
                "Using an extreme time stretching factor. "
                "Quality of results will be poor"
            )

        if abs(factor - 1.0) <= 0.1:
            logging.warning(
                "For this stretch factor, "
                "the stretch effect has better performance."
            )

        if audio_type not in [None, 'm', 's', 'l']:
            raise ValueError(
                "audio_type must be one of None, 'm', 's', or 'l'."
            )

        if not isinstance(quick, bool):
            raise ValueError("quick must be a boolean.")

        effect_args = ['tempo']

        if quick:
            effect_args.append('-q')

        if audio_type is not None:
            effect_args.append('-{}'.format(audio_type))

        effect_args.append('{}'.format(factor))

        self.effects.extend(effect_args)
        self.effects_log.append('tempo')

        return self

    def treble(self, gain_db, frequency=3000.0, slope=0.5):
        '''Boost or cut the treble (lower) frequencies of the audio using a
        two-pole shelving filter with a response similar to that of a standard
        hi-fi’s tone-controls. This is also known as shelving equalisation.

        The filters are described in detail in
        http://musicdsp.org/files/Audio-EQ-Cookbook.txt

        Parameters
        ----------
        gain_db : float
            The gain at the Nyquist frequency.
            For a large cut use -20, for a large boost use 20.
        frequency : float, default=100.0
            The filter's cutoff frequency in Hz.
        slope : float, default=0.5
            The steepness of the filter's shelf transition.
            For a gentle slope use 0.3, and use 1.0 for a steep slope.

        See Also
        --------
        bass, equalizer

        '''
        if not is_number(gain_db):
            raise ValueError("gain_db must be a number")

        if not is_number(frequency) or frequency <= 0:
            raise ValueError("frequency must be a positive number.")

        if not is_number(slope) or slope <= 0 or slope > 1.0:
            raise ValueError("width_q must be a positive number.")

        effect_args = [
            'treble', '{}'.format(gain_db), '{}'.format(frequency),
            '{}s'.format(slope)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('treble')

        return self

    def tremolo(self, speed=6.0, depth=40.0):
        '''Apply a tremolo (low frequency amplitude modulation) effect to the
        audio. The tremolo frequency in Hz is giv en by speed, and the depth
        as a percentage by depth (default 40).

        Parameters
        ----------
        speed : float
            Tremolo speed in Hz.
        depth : float
            Tremolo depth as a percentage of the total amplitude.

        See Also
        --------
        flanger

        Examples
        --------
        >>> tfm = sox.Transformer()

        For a growl-type effect

        >>> tfm.tremolo(speed=100.0)
        '''
        if not is_number(speed) or speed <= 0:
            raise ValueError("speed must be a positive number.")
        if not is_number(depth) or depth <= 0 or depth > 100:
            raise ValueError("depth must be a positive number less than 100.")

        effect_args = [
            'tremolo',
            '{}'.format(speed),
            '{}'.format(depth)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('tremolo')

        return self

    def trim(self, start_time, end_time):
        '''Excerpt a clip from an audio file, given a start and end time.

        Parameters
        ----------
        start_time : float
            Start time of the clip (seconds)
        end_time : float
            End time of the clip (seconds)

        '''
        if not is_number(start_time) or start_time < 0:
            raise ValueError("start_time must be a positive number.")
        if not is_number(end_time) or end_time < 0:
            raise ValueError("end_time must be a positive number.")
        if start_time >= end_time:
            raise ValueError("start_time must be smaller than end_time.")

        effect_args = [
            'trim',
            '{}'.format(start_time),
            '{}'.format(end_time - start_time)
        ]

        self.effects.extend(effect_args)
        self.effects_log.append('trim')

        return self

    def upsample(self, factor=2):
        '''Upsample the signal by an integer factor: zero-value samples are
        inserted between each pair of input samples. As a result, the original
        spectrum is replicated into the new frequency space (imaging) and
        attenuated. The upsample effect is typically used in combination with
        filtering effects.

        Parameters
        ----------
        factor : int, default=2
            Integer upsampling factor.

        See Also
        --------
        rate, downsample

        '''
        if not isinstance(factor, int) or factor < 1:
            raise ValueError('factor must be a positive integer.')

        effect_args = ['upsample', '{}'.format(factor)]

        self.effects.extend(effect_args)
        self.effects_log.append('upsample')

        return self

    def vad(self, location=1, normalize=True, activity_threshold=7.0,
            min_activity_duration=0.25, initial_search_buffer=1.0,
            max_gap=0.25, initial_pad=0.0):
        '''Voice Activity Detector. Attempts to trim silence and quiet
        background sounds from the ends of recordings of speech. The algorithm
        currently uses a simple cepstral power measurement to detect voice, so
        may be fooled by other things, especially music.

        The effect can trim only from the front of the audio, so in order to
        trim from the back, the reverse effect must also be used.

        Parameters
        ----------
        location : 1 or -1, default=1
            If 1, trims silence from the beginning
            If -1, trims silence from the end
        normalize : bool, default=True
            If true, normalizes audio before processing.
        activity_threshold : float, default=7.0
            The measurement level used to trigger activity detection. This may
            need to be cahnged depending on the noise level, signal level, and
            other characteristics of the input audio.
        min_activity_duration : float, default=0.25
            The time constant (in seconds) used to help ignore short bursts of
            sound.
        initial_search_buffer : float, default=1.0
            The amount of audio (in seconds) to search for quieter/shorter
            bursts of audio to include prior to the detected trigger point.
        max_gap : float, default=0.25
            The allowed gap (in seconds) between quiteter/shorter bursts of
            audio to include prior to the detected trigger point
        initial_pad : float, default=0.0
            The amount of audio (in seconds) to preserve before the trigger
            point and any found quieter/shorter bursts.

        See Also
        --------
        silence

        Examples
        --------
        >>> tfm = sox.Transformer()

        Remove silence from the beginning of speech

        >>> tfm.vad(initial_pad=0.3)

        Remove silence from the end of speech

        >>> tfm.vad(location=-1, initial_pad=0.2)

        '''
        if location not in [-1, 1]:
            raise ValueError("location must be -1 or 1.")
        if not isinstance(normalize, bool):
            raise ValueError("normalize muse be a boolean.")
        if not is_number(activity_threshold):
            raise ValueError("activity_threshold must be a number.")
        if not is_number(min_activity_duration) or min_activity_duration < 0:
            raise ValueError("min_activity_duration must be a positive number")
        if not is_number(initial_search_buffer) or initial_search_buffer < 0:
            raise ValueError("initial_search_buffer must be a positive number")
        if not is_number(max_gap) or max_gap < 0:
            raise ValueError("max_gap must be a positive number.")
        if not is_number(initial_pad) or initial_pad < 0:
            raise ValueError("initial_pad must be a positive number.")

        effect_args = []

        if normalize:
            effect_args.append('norm')

        if location == -1:
            effect_args.append('reverse')

        effect_args.extend([
            'vad',
            '-t', '{}'.format(activity_threshold),
            '-T', '{}'.format(min_activity_duration),
            '-s', '{}'.format(initial_search_buffer),
            '-g', '{}'.format(max_gap),
            '-p', '{}'.format(initial_pad)
        ])

        if location == -1:
            effect_args.append('reverse')

        self.effects.extend(effect_args)
        self.effects_log.append('vad')

        return self
