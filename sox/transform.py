#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Python wrapper around the SoX library.
This module requires that SoX is installed.
'''

from __future__ import print_function
import logging

from .core import is_number
from .core import play
from .core import sox
from .core import SoxError

from . import file_info

logging.basicConfig(level=logging.DEBUG)

VERBOSITY_VALS = [0, 1, 2, 3, 4]


class Transformer(object):
    '''Audio file transformer.
    Class which allows multiple effects to be chained to create an output
    file, saved to output_filepath.

    Parameters
    ----------
    input_filepath : str
        Path to input audio file.
    output_filepath : str
        Path to desired output file. If a file already exists at the given
        path, the file will be overwritten.

    Attributes
    ----------
    input_filepath : str
        Path to input audio file.
    output_filepath : str
        Path where the output file will be written.
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

    Methods
    -------
    set_globals
        Overwrite the default global arguments.
    build
        Execute the current chain of commands and write output file.

    '''

    def __init__(self, input_filepath, output_filepath):
        file_info.validate_input_file(input_filepath)
        file_info.validate_output_file(output_filepath)

        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

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

    def build(self):
        '''Builds the output_file by executing the current set of commands.
        '''
        args = []
        args.extend(self.globals)
        args.extend(self.input_format)
        args.append(self.input_filepath)
        args.extend(self.output_format)
        args.append(self.output_filepath)
        args.extend(self.effects)

        status = sox(args)

        if status is False:
            raise SoxError
        else:
            logging.info(
                "Created %s with effects: %s",
                self.output_filepath,
                " ".join(self.effects_log)
            )
            return True

    def preview(self):
        '''Play a preview of the output with the current set of effects
        '''
        args = ["play", "--no-show-progress"]
        args.extend(self.globals)
        args.extend(self.input_format)
        args.append(self.input_filepath)
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

    def bend(self):
        raise NotImplementedError

    def biquad(self):
        raise NotImplementedError

    def channels(self):
        raise NotImplementedError

    def chorus(self):
        raise NotImplementedError

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

    def contrast(self):
        raise NotImplementedError

    def convert(self, samplerate=None, channels=None, bitdepth=None):
        '''Converts output audio to the specified format.

        Parameters
        ----------
        samplerate : float, default=None
            Desired samplerate. If None, defaults to the same as input.
        channels : int, default=None
            Desired channels. If None, defaults to the same as input.
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
        if channels is not None:
            if not isinstance(channels, int) or channels <= 0:
                raise ValueError(
                    "channels must be a positive integer."
                )
            self.output_format.extend(['-c', '{}'.format(channels)])
        if samplerate is not None:
            if not is_number(samplerate) or samplerate <= 0:
                raise ValueError("samplerate must be a positive number.")
            self.rate(samplerate)

    def dcshift(self):
        raise NotImplementedError

    def deemph(self):
        raise NotImplementedError

    def delay(self):
        raise NotImplementedError

    def dither(self):
        raise NotImplementedError

    def downsample(self):
        raise NotImplementedError

    def earwax(self):
        raise NotImplementedError

    def echo(self):
        raise NotImplementedError

    def echos(self):
        raise NotImplementedError

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

    def fir(self):
        raise NotImplementedError

    def flanger(self):
        raise NotImplementedError

    def gain(self, gain_db=0.0, normalize=True, limiter=False, balance=None):
        '''Apply amplification or attenuation to the audio signal.

        Parameters
        ----------
        gain_db : float, default=0.0
            Target gain in decibels (dB).
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
        norm, loudness

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

    def hilbert(self):
        raise NotImplementedError

    def loudness(self, gain_db=-10.0, reference_level=65.0):
        '''Loudness control. Similar to the gain effect, but provides
        equalisation for the human auditory system.

        The gain is adjusted by gain_db and the signal equalised according to
        ISO 226 w.r.t. reference_level.

        Parameters
        ----------
        gain_db : float, default=-10.0
            Output loudness (in dB)
        reference_level : float, default=65.0
            Reference level (in dB) according to which the signal is equalized.
            Must be between 50 and 75 (dB)

        See Also
        --------
        gain, loudness

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

    def mcompand(self):
        raise NotImplementedError

    def noisered(self):
        raise NotImplementedError

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

    def oops(self):
        raise NotImplementedError

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

    def phaser(self):
        raise NotImplementedError

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

    def remix(self):
        raise NotImplementedError

    def repeat(self):
        raise NotImplementedError

    def reverb(self, reverberance=50, high_freq_damping=50, room_scale=100,
               stereo_depth=100, pre_delay=0, wet_gain=0, wet_only=False):
        '''Add reverberation to the audio using the ‘freeverb’ algorithm. 
        A reverberation effect is sometimes desirable for concert halls that are
        too small or contain so many people that the hall’s natural reverberance
        is diminished. Applying a small amount of stereo reverb to a (dry) mono
        signal will usually make it sound more natural.

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

    def reverse(self):
        '''Reverse the audio completely
        '''
        effect_args = ['reverse']
        self.effects.extend(effect_args)
        self.effects_log.append('reverse')

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

    def sinc(self):
        raise NotImplementedError

    def speed(self):
        raise NotImplementedError

    def splice(self):
        raise NotImplementedError

    def swap(self):
        raise NotImplementedError

    def stretch(self):
        raise NotImplementedError

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

    def tremolo(self):
        raise NotImplementedError

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

    def upsample(self):
        raise NotImplementedError

    def vad(self):
        raise NotImplementedError
