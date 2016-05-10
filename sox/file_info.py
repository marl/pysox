""" Audio file info computed by soxi.
"""
import logging
import subprocess
from subprocess import CalledProcessError

from .core import validate_input_file

SOXI_ARGS = ['b', 'c', 'a', 'D', 'e', 't', 's', 'r']


def soxi(filepath, argument):
    """ Base call to Soxi.

    Parameters
    ----------
    filepath : str
        Path to audio file.
    argument : str
        Argument to pass to Soxi.

    Returns
    -------
    shell_output : str
        command line output of Soxi

    """
    validate_input_file(filepath)

    if argument not in SOXI_ARGS:
        raise ValueError("Invalid argument '{}' to Soxi".format(argument))

    args = ['soxi']
    args.append("-{}".format(argument))
    args.append(filepath)

    try:
        shell_output = subprocess.check_output(
            " ".join(args),
            shell=True
        )
    except CalledProcessError as cpe:
        logging.info("Soxi error message: {}".format(cpe.output))
        raise SoxiError("Soxi failed with exit code {}".format(cpe.returncode))

    return str(shell_output).strip('\n')


class SoxiError(Exception):
    """Exception to be raised when SoXi exits with non-zero status.
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def bitrate(input_filepath):
    """
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
    """
    output = soxi(input_filepath, 'b')
    if output == '0':
        logging.warning("Bitrate unavailable for %s", input_filepath)
    return int(output)


def channels(input_filepath):
    """
    Show number of channels.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    channels : int
        number of channels
    """
    output = soxi(input_filepath, 'c')
    return int(output)


def comments(input_filepath):
    """
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
    """
    output = soxi(input_filepath, 'a')
    return str(output)


def duration(input_filepath):
    """
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
    """
    output = soxi(input_filepath, 'D')
    if output == '0':
        logging.warning("Duration unavailable for %s", input_filepath)

    return float(output)


def encoding(input_filepath):
    """
    Show the name of the audio encoding.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    encoding : str
        audio encoding type
    """
    output = soxi(input_filepath, 'e')
    return str(output)


def file_type(input_filepath):
    """
    Show detected file-type.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    file_type : str
        file format type (ex. 'wav')
    """
    output = soxi(input_filepath, 't')
    return str(output)


def num_samples(input_filepath):
    """
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
    """
    output = soxi(input_filepath, 's')
    if output == '0':
        logging.warning("Number of samples unavailable for %s", input_filepath)
    return int(output)


def sample_rate(input_filepath):
    """
    Show sample-rate.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    samplerate : float
        number of samples/second
    """
    output = soxi(input_filepath, 'r')
    return float(output)
