"""Base module for calling SoX """
import logging
import subprocess
import os


def sox(args):
    """Pass an argument list to SoX.

    Parameters
    ----------
    args : iterable
        Argument list for SoX. The first item can, but does not
        need to, be 'sox'.

    Returns:
    --------
    status : bool
        True on success.

    """
    if args[0].lower() != "sox":
        args.insert(0, "sox")
    else:
        args[0] = "sox"

    try:
        logging.info("Executing: %s", " ".join(args))
        process_handle = subprocess.Popen(args, stderr=subprocess.PIPE)
        status = process_handle.wait()
        if process_handle.stdout is not None:
            logging.info(process_handle.stdout)
        return status == 0
    except OSError as error_msg:
        logging.error("OSError: SoX failed! %s", error_msg)
    except TypeError as error_msg:
        logging.error("TypeError: %s", error_msg)
    return False


class SoxError(Exception):
    """Exception to be raised when SoX exits with non-zero status.
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def _get_valid_formats():
    """ Calls SoX help for a lists of audio formats available with the current
    install of SoX.

    Returns:
    --------
    formats : list
        List of audio file extensions that SoX can process.

    """
    shell_output = subprocess.check_output(
        "sox -h | grep 'AUDIO FILE FORMATS'",
        shell=True
    )
    formats = str(shell_output).strip('\n').split(' ')[3:]
    return formats


VALID_FORMATS = _get_valid_formats()


def _file_extension(filepath):
    """Get the extension of a filepath.

    Parameters
    ----------
    filepath : str
        File path.

    Returns:
    --------
    extension : str
        File extension.

    """
    return os.path.splitext(filepath)[1][1:]


def set_input_file(input_filepath):
    """Input file validation function. Checks that file exists and can be
    processed by SoX.

    Parameters
    ----------
    input_filepath : str
        The input filepath.

    Returns:
    --------
    input_filepath : str
        The input filepath.

    """
    if not os.path.exists(input_filepath):
        raise IOError(
            "input_filepath {} does not exist.".format(input_filepath)
        )
    ext = _file_extension(input_filepath)
    if ext not in VALID_FORMATS:
        logging.info("Valid formats: %s", " ".join(VALID_FORMATS))
        raise SoxError(
            "This install of SoX cannot process .{} files.".format(ext)
        )

    return input_filepath


def set_output_file(output_filepath):
    """Output file validation function. Checks that file can be written, and
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

    """
    if not os.access(os.path.dirname(output_filepath), os.W_OK):
        raise IOError(
            "SoX cannot write to output_filepath {}".format(output_filepath)
        )

    ext = _file_extension(output_filepath)
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

    return output_filepath


def is_number(var):
    """Check if variable is a numeric value.

    Parameters
    ----------
    var : object

    Returns:
    --------
    bool
        True if var is numeric, False otherwise.
    """
    try:
        float(var)
        return True
    except ValueError:
        return False
    except TypeError:
        return False
