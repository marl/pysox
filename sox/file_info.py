''' Audio file info computed by soxi.
'''
import logging
import os

from .core import VALID_FORMATS
from .core import soxi
from .core import SoxError


def bitrate(input_filepath):
    '''
    Number of bits per sample (0 if not applicable).

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    bitrate : int
        number of bits per sample
        returns 0 if not applicable
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'b')
    if output == '0':
        logging.warning("Bitrate unavailable for %s", input_filepath)
    return int(output)


def channels(input_filepath):
    '''
    Show number of channels.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    channels : int
        number of channels
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'c')
    return int(output)


def comments(input_filepath):
    '''
    Show file comments (annotations) if available.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    comments : str
        File comments from header.
        If no comments are present, returns an empty string.
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'a')
    return str(output)


def duration(input_filepath):
    '''
    Show duration in seconds (0 if unavailable).

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    duration : float
        Duration of audio file in seconds.
        If unavailable or empty, returns 0.
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'D')
    if output == '0':
        logging.warning("Duration unavailable for %s", input_filepath)

    return float(output)


def encoding(input_filepath):
    '''
    Show the name of the audio encoding.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    encoding : str
        audio encoding type
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'e')
    return str(output)


def file_type(input_filepath):
    '''
    Show detected file-type.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    file_type : str
        file format type (ex. 'wav')
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 't')
    return str(output)


def num_samples(input_filepath):
    '''
    Show number of samples (0 if unavailable).

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    n_samples : int
        total number of samples in audio file.
        Returns 0 if empty or unavailable
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 's')
    if output == '0':
        logging.warning("Number of samples unavailable for %s", input_filepath)
    return int(output)


def sample_rate(input_filepath):
    '''
    Show sample-rate.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    samplerate : float
        number of samples/second
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'r')
    return float(output)


def validate_input_file(input_filepath):
    '''Input file validation function. Checks that file exists and can be
    processed by SoX.

    Parameters
    ----------
    input_filepath : str
        The input filepath.

    '''
    if not os.path.exists(input_filepath):
        raise IOError(
            "input_filepath {} does not exist.".format(input_filepath)
        )
    ext = file_extension(input_filepath)
    if ext not in VALID_FORMATS:
        logging.info("Valid formats: %s", " ".join(VALID_FORMATS))
        raise SoxError(
            "This install of SoX cannot process .{} files.".format(ext)
        )


def validate_input_file_list(input_filepath_list):
    '''Input file list validation function. Checks that object is a list and
    contains valid filepaths that can be processed by SoX.

    Parameters
    ----------
    input_filepath_list : list
        A list of filepaths.

    '''
    if not isinstance(input_filepath_list, list):
        raise TypeError("input_filepath_list must be a list.")
    elif len(input_filepath_list) < 2:
        raise ValueError("input_filepath_list must have at least 2 files.")

    for input_filepath in input_filepath_list:
        validate_input_file(input_filepath)


def validate_output_file(output_filepath):
    '''Output file validation function. Checks that file can be written, and
    has a valid file extension. Throws a warning if the path already exists,
    as it will be overwritten on build.

    Parameters
    ----------
    output_filepath : str
        The output filepath.

    Returns:
    --------
    output_filepath : str
        The output filepath.

    '''

    nowrite_conditions = [
        bool(os.path.dirname(output_filepath)),
        not os.access(os.path.dirname(output_filepath), os.W_OK)]

    if all(nowrite_conditions):
        raise IOError(
            "SoX cannot write to output_filepath {}".format(output_filepath)
        )

    ext = file_extension(output_filepath)
    if ext not in VALID_FORMATS:
        logging.info("Valid formats: %s", " ".join(VALID_FORMATS))
        raise SoxError(
            "This install of SoX cannot process .{} files.".format(ext)
        )

    if os.path.exists(output_filepath):
        logging.warning(
            'output_file: %s already exists and will be overwritten on build',
            output_filepath
        )


def file_extension(filepath):
    '''Get the extension of a filepath.

    Parameters
    ----------
    filepath : str
        File path.

    '''
    return os.path.splitext(filepath)[1][1:]
