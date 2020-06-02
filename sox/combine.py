#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Python wrapper around the SoX library.
This module requires that SoX is installed.
'''

from __future__ import print_function

from . import file_info
from . import core
from .log import logger
from .core import ENCODING_VALS
from .core import is_number
from .core import sox
from .core import play
from .core import SoxError
from .core import SoxiError
from .core import VALID_FORMATS

from .transform import Transformer


COMBINE_VALS = [
    'concatenate', 'merge', 'mix', 'mix-power', 'multiply'
]


class Combiner(Transformer):
    '''Audio file combiner.
    Class which allows multiple files to be combined to create an output
    file, saved to output_filepath.

    Inherits all methods from the Transformer class, thus any effects can be
    applied after combining.
    '''

    def __init__(self):
        super(Combiner, self).__init__()

    def build(self, input_filepath_list, output_filepath, combine_type,
              input_volumes=None):
        '''Builds the output_file by executing the current set of commands.

        Parameters
        ----------
        input_filepath_list : list of str
            List of paths to input audio files.
        output_filepath : str
            Path to desired output file. If a file already exists at the given
            path, the file will be overwritten.
        combine_type : str
            Input file combining method. One of the following values:
                * concatenate : combine input files by concatenating in the
                    order given.
                * merge : combine input files by stacking each input file into
                    a new channel of the output file.
                * mix : combine input files by summing samples in corresponding
                    channels.
                * mix-power : combine input files with volume adjustments such
                    that the output volume is roughly equivlent to one of the
                    input signals.
                * multiply : combine input files by multiplying samples in
                    corresponding samples.
        input_volumes : list of float, default=None
            List of volumes to be applied upon combining input files. Volumes
            are applied to the input files in order.
            If None, input files will be combined at their original volumes.

        Returns
        -------
        status : bool
            True on success.

        '''
        file_info.validate_input_file_list(input_filepath_list)
        file_info.validate_output_file(output_filepath)
        _validate_combine_type(combine_type)
        _validate_volumes(input_volumes)

        input_format_list = _build_input_format_list(
            input_filepath_list, input_volumes,
            self.input_format
        )

        try:
            _validate_file_formats(input_filepath_list, combine_type)
        except SoxiError:
            logger.warning("unable to validate file formats.")

        args = []
        args.extend(self.globals)
        args.extend(['--combine', combine_type])

        input_args = _build_input_args(input_filepath_list, input_format_list)
        args.extend(input_args)

        args.extend(self._output_format_args(self.output_format))
        args.append(output_filepath)
        args.extend(self.effects)

        status, out, err = sox(args)

        if status != 0:
            raise SoxError(
                "Stdout: {}\nStderr: {}".format(out, err)
            )
        else:
            logger.info(
                "Created %s with combiner %s and  effects: %s",
                output_filepath,
                combine_type,
                " ".join(self.effects_log)
            )
            if out is not None:
                logger.info("[SoX] {}".format(out))
            return True

    def preview(self, input_filepath_list, combine_type, input_volumes=None):
        '''Play a preview of the output with the current set of effects

        Parameters
        ----------
        input_filepath_list : list of str
            List of paths to input audio files.
        combine_type : str
            Input file combining method. One of the following values:
                * concatenate : combine input files by concatenating in the
                    order given.
                * merge : combine input files by stacking each input file into
                    a new channel of the output file.
                * mix : combine input files by summing samples in corresponding
                    channels.
                * mix-power : combine input files with volume adjustments such
                    that the output volume is roughly equivlent to one of the
                    input signals.
                * multiply : combine input files by multiplying samples in
                    corresponding samples.
        input_volumes : list of float, default=None
            List of volumes to be applied upon combining input files. Volumes
            are applied to the input files in order.
            If None, input files will be combined at their original volumes.

        '''
        args = ["play", "--no-show-progress"]
        args.extend(self.globals)
        args.extend(['--combine', combine_type])

        input_format_list = _build_input_format_list(
            input_filepath_list, input_volumes, self.input_format
        )
        input_args = _build_input_args(input_filepath_list, input_format_list)
        args.extend(input_args)
        args.extend(self.effects)

        play(args)

    def set_input_format(self, file_type=None, rate=None, bits=None,
                         channels=None, encoding=None, ignore_length=None):
        '''Sets input file format arguments. This is primarily useful when
        dealing with audio files without a file extension. Overwrites any
        previously set input file arguments.

        If this function is not explicity called the input format is inferred
        from the file extension or the file's header.

        Parameters
        ----------
        file_type : list of str or None, default=None
            The file type of the input audio file. Should be the same as what
            the file extension would be, for ex. 'mp3' or 'wav'.
        rate : list of float or None, default=None
            The sample rate of the input audio file. If None the sample rate
            is inferred.
        bits : list of int or None, default=None
            The number of bits per sample. If None, the number of bits per
            sample is inferred.
        channels : list of int or None, default=None
            The number of channels in the audio file. If None the number of
            channels is inferred.
        encoding : list of str or None, default=None
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
        ignore_length : list of bool or None, default=None
            If True, overrides an (incorrect) audio length given in an audio
            file’s header. If this option is given then SoX will keep reading
            audio until it reaches the end of the input file.

        '''
        if file_type is not None and not isinstance(file_type, list):
            raise ValueError("file_type must be a list or None.")

        if file_type is not None:
            if not all([f in VALID_FORMATS for f in file_type]):
                raise ValueError(
                    'file_type elements '
                    'must be one of {}'.format(VALID_FORMATS)
                )
        else:
            file_type = []

        if rate is not None and not isinstance(rate, list):
            raise ValueError("rate must be a list or None.")

        if rate is not None:
            if not all([is_number(r) and r > 0 for r in rate]):
                raise ValueError('rate elements must be positive floats.')
        else:
            rate = []

        if bits is not None and not isinstance(bits, list):
            raise ValueError("bits must be a list or None.")

        if bits is not None:
            if not all([isinstance(b, int) and b > 0 for b in bits]):
                raise ValueError('bit elements must be positive ints.')
        else:
            bits = []

        if channels is not None and not isinstance(channels, list):
            raise ValueError("channels must be a list or None.")

        if channels is not None:
            if not all([isinstance(c, int) and c > 0 for c in channels]):
                raise ValueError('channel elements must be positive ints.')
        else:
            channels = []

        if encoding is not None and not isinstance(encoding, list):
            raise ValueError("encoding must be a list or None.")

        if encoding is not None:
            if not all([e in ENCODING_VALS for e in encoding]):
                raise ValueError(
                    'elements of encoding must '
                    'be one of {}'.format(ENCODING_VALS)
                )
        else:
            encoding = []

        if ignore_length is not None and not isinstance(ignore_length, list):
            raise ValueError("ignore_length must be a list or None.")

        if ignore_length is not None:
            if not all([isinstance(l, bool) for l in ignore_length]):
                raise ValueError("ignore_length elements must be booleans.")
        else:
            ignore_length = []

        max_input_arg_len = max([
            len(file_type), len(rate), len(bits), len(channels),
            len(encoding), len(ignore_length)
        ])

        input_format = []
        for _ in range(max_input_arg_len):
            input_format.append([])

        for i, f in enumerate(file_type):
            input_format[i].extend(['-t', '{}'.format(f)])

        for i, r in enumerate(rate):
            input_format[i].extend(['-r', '{}'.format(r)])

        for i, b in enumerate(bits):
            input_format[i].extend(['-b', '{}'.format(b)])

        for i, c in enumerate(channels):
            input_format[i].extend(['-c', '{}'.format(c)])

        for i, e in enumerate(encoding):
            input_format[i].extend(['-e', '{}'.format(e)])

        for i, l in enumerate(ignore_length):
            if l is True:
                input_format[i].append('--ignore-length')

        self.input_format = input_format
        return self


def _validate_file_formats(input_filepath_list, combine_type):
    '''Validate that combine method can be performed with given files.
    Raises IOError if input file formats are incompatible.
    '''
    _validate_sample_rates(input_filepath_list, combine_type)

    if combine_type == 'concatenate':
        _validate_num_channels(input_filepath_list, combine_type)


def _validate_sample_rates(input_filepath_list, combine_type):
    ''' Check if files in input file list have the same sample rate
    '''
    sample_rates = [
        file_info.sample_rate(f) for f in input_filepath_list
    ]
    if not core.all_equal(sample_rates):
        raise IOError(
            "Input files do not have the same sample rate. The {} combine "
            "type requires that all files have the same sample rate"
            .format(combine_type)
        )


def _validate_num_channels(input_filepath_list, combine_type):
    ''' Check if files in input file list have the same number of channels
    '''
    channels = [
        file_info.channels(f) for f in input_filepath_list
    ]
    if not core.all_equal(channels):
        raise IOError(
            "Input files do not have the same number of channels. The "
            "{} combine type requires that all files have the same "
            "number of channels"
            .format(combine_type)
        )


def _build_input_format_list(input_filepath_list, input_volumes=None,
                             input_format=None):
    '''Set input formats given input_volumes.

    Parameters
    ----------
    input_filepath_list : list of str
        List of input files
    input_volumes : list of float, default=None
        List of volumes to be applied upon combining input files. Volumes
        are applied to the input files in order.
        If None, input files will be combined at their original volumes.
    input_format : list of lists, default=None
        List of input formats to be applied to each input file. Formatting
        arguments are applied to the input files in order.
        If None, the input formats will be inferred from the file header.

    '''
    n_inputs = len(input_filepath_list)
    input_format_list = []
    for _ in range(n_inputs):
        input_format_list.append([])

    # Adjust length of input_volumes list
    if input_volumes is None:
        vols = [1] * n_inputs
    else:
        n_volumes = len(input_volumes)
        if n_volumes < n_inputs:
            logger.warning(
                'Volumes were only specified for %s out of %s files.'
                'The last %s files will remain at their original volumes.',
                n_volumes, n_inputs, n_inputs - n_volumes
            )
            vols = input_volumes + [1] * (n_inputs - n_volumes)
        elif n_volumes > n_inputs:
            logger.warning(
                '%s volumes were specified but only %s input files exist.'
                'The last %s volumes will be ignored.',
                n_volumes, n_inputs, n_volumes - n_inputs
            )
            vols = input_volumes[:n_inputs]
        else:
            vols = [v for v in input_volumes]

    # Adjust length of input_format list
    if input_format is None:
        fmts = [[] for _ in range(n_inputs)]
    else:
        n_fmts = len(input_format)
        if n_fmts < n_inputs:
            logger.warning(
                'Input formats were only specified for %s out of %s files.'
                'The last %s files will remain unformatted.',
                n_fmts, n_inputs, n_inputs - n_fmts
            )
            fmts = [f for f in input_format]
            fmts.extend([[] for _ in range(n_inputs - n_fmts)])
        elif n_fmts > n_inputs:
            logger.warning(
                '%s Input formats were specified but only %s input files exist'
                '. The last %s formats will be ignored.',
                n_fmts, n_inputs, n_fmts - n_inputs
            )
            fmts = input_format[:n_inputs]
        else:
            fmts = [f for f in input_format]

    for i, (vol, fmt) in enumerate(zip(vols, fmts)):
        input_format_list[i].extend(['-v', '{}'.format(vol)])
        input_format_list[i].extend(fmt)

    return input_format_list


def _build_input_args(input_filepath_list, input_format_list):
    ''' Builds input arguments by stitching input filepaths and input
    formats together.
    '''
    if len(input_format_list) != len(input_filepath_list):
        raise ValueError(
            "input_format_list & input_filepath_list are not the same size"
        )

    input_args = []
    zipped = zip(input_filepath_list, input_format_list)
    for input_file, input_fmt in zipped:
        input_args.extend(input_fmt)
        input_args.append(input_file)

    return input_args


def _validate_combine_type(combine_type):
    '''Check that the combine_type is valid.

    Parameters
    ----------
    combine_type : str
        Combine type.

    '''
    if combine_type not in COMBINE_VALS:
        raise ValueError(
            'Invalid value for combine_type. Must be one of {}'.format(
                COMBINE_VALS)
        )


def _validate_volumes(input_volumes):
    '''Check input_volumes contains a valid list of volumes.

    Parameters
    ----------
    input_volumes : list
        list of volume values. Castable to numbers.

    '''
    if not (input_volumes is None or isinstance(input_volumes, list)):
        raise TypeError("input_volumes must be None or a list.")

    if isinstance(input_volumes, list):
        for vol in input_volumes:
            if not core.is_number(vol):
                raise ValueError(
                    "Elements of input_volumes must be numbers: found {}"
                    .format(vol)
                )
