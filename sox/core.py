'''Base module for calling SoX '''
import logging
import subprocess
from subprocess import CalledProcessError

from . import NO_SOX

SOXI_ARGS = ['b', 'c', 'a', 'D', 'e', 't', 's', 'r']

ENCODING_VALS = [
    'signed-integer', 'unsigned-integer', 'floating-point', 'a-law', 'u-law',
    'oki-adpcm', 'ima-adpcm', 'ms-adpcm', 'gsm-full-rate'
]


def enquote_filepath(fpath):
    """Wrap a filepath in double-quotes to protect difficult characters.
    """
    if ' ' in fpath:
        fpath = '"{}"'.format(fpath.strip("'").strip('"'))
    return fpath


def sox(args):
    '''Pass an argument list to SoX.

    Parameters
    ----------
    args : iterable
        Argument list for SoX. The first item can, but does not
        need to, be 'sox'.

    Returns:
    --------
    status : bool
        True on success.

    '''
    if args[0].lower() != "sox":
        args.insert(0, "sox")
    else:
        args[0] = "sox"

    try:
        command = ' '.join(args)
        logging.info("Executing: %s", command)

        process_handle = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True
        )

        out, err = process_handle.communicate()
        out = out.decode("utf-8")
        err = err.decode("utf-8")

        status = process_handle.returncode
        return status, out, err

    except OSError as error_msg:
        logging.error("OSError: SoX failed! %s", error_msg)
    except TypeError as error_msg:
        logging.error("TypeError: %s", error_msg)
    return 1, None, None


class SoxError(Exception):
    '''Exception to be raised when SoX exits with non-zero status.
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def _get_valid_formats():
    ''' Calls SoX help for a lists of audio formats available with the current
    install of SoX.

    Returns:
    --------
    formats : list
        List of audio file extensions that SoX can process.

    '''
    if NO_SOX:
        return []

    shell_output = subprocess.check_output(
        'sox -h | grep "AUDIO FILE FORMATS"',
        shell=True
    )
    formats = str(shell_output).strip('\n').split(' ')[3:]
    return formats


VALID_FORMATS = _get_valid_formats()


def soxi(filepath, argument):
    ''' Base call to Soxi.

    Parameters
    ----------
    filepath : str
        Path to audio file.

    argument : str
        Argument to pass to Soxi.

    Returns
    -------
    shell_output : str
        Command line output of Soxi
    '''

    if argument not in SOXI_ARGS:
        raise ValueError("Invalid argument '{}' to Soxi".format(argument))

    args = ['soxi']
    args.append("-{}".format(argument))
    args.append(enquote_filepath(filepath))

    try:
        shell_output = subprocess.check_output(
            " ".join(args),
            shell=True, stderr=subprocess.PIPE
        )
    except CalledProcessError as cpe:
        logging.info("Soxi error message: {}".format(cpe.output))
        raise SoxiError("Soxi failed with exit code {}".format(cpe.returncode))

    shell_output = shell_output.decode("utf-8")

    return str(shell_output).strip('\n')


def play(args):
    '''Pass an argument list to play.

    Parameters
    ----------
    args : iterable
        Argument list for play. The first item can, but does not
        need to, be 'play'.

    Returns:
    --------
    status : bool
        True on success.

    '''
    if args[0].lower() != "play":
        args.insert(0, "play")
    else:
        args[0] = "play"

    try:
        logging.info("Executing: %s", " ".join(args))
        process_handle = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        status = process_handle.wait()
        if process_handle.stderr is not None:
            logging.info(process_handle.stderr)

        if status == 0:
            return True
        else:
            logging.info("Play returned with error code %s", status)
            return False
    except OSError as error_msg:
        logging.error("OSError: Play failed! %s", error_msg)
    except TypeError as error_msg:
        logging.error("TypeError: %s", error_msg)
    return False


class SoxiError(Exception):
    '''Exception to be raised when SoXi exits with non-zero status.
    '''

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def is_number(var):
    '''Check if variable is a numeric value.

    Parameters
    ----------
    var : object

    Returns:
    --------
    bool
        True if var is numeric, False otherwise.
    '''
    try:
        float(var)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def all_equal(list_of_things):
    '''Check if a list contains identical elements.

    Parameters
    ----------
    list_of_things : list
        list of objects

    Returns:
    --------
    bool
        True if all list elements are the same.
    '''
    return len(set(list_of_things)) <= 1
