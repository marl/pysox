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

        process_handle = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = process_handle.communicate()
        status = process_handle.returncode
        if out is not None:
            logging.info(out)
        if status == 0:
            return True
        else:
            logging.info("SoX returned with error code %s", status)
            logging.info(out)
            logging.info(err)
            return False
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


def file_info(input_file):
    """ Get audio file information including file format, bit rate,
        and sample rate.

    Parameters
    ----------
    input_file : str
        Path to audio file.

    Returns
    -------
    ret_dict : dictionary
        Dictionary containing file information.
            - Bit Rate
            - Channels
            - Duration
            - File Size
            - Precision
            - Sample Encoding
            - Sample Rate
    """
    if os.path.exists(input_file):
        ret_dict = {}
        soxi_out = subprocess.check_output(["soxi", input_file]).split('\n')
        for line in soxi_out:
            if len(line) > 0:
                separator = line.find(':')
                key = line[:separator].strip()
                if key == 'Input File':
                    continue
                value = line[separator + 1:].strip()
                ret_dict[key] = _soxi_parse(key, value)
        return ret_dict
    else:
        raise IOError("{} does not exist.".format(input_file))


def _soxi_parse(key, value):
    """ Helper function for file_info. Parses key value pairs returned by soxi.
    """
    ret_value = None
    try:
        if 'Duration' == key:
            ret_value = [x.strip() for x in value.split('=')]
            # parse time into seconds
            hours, minutes, seconds = ret_value[0].split(':')
            secs_duration = float(seconds) + (int(minutes) * 60.) + \
                (int(hours) * 3600.)
            # parse samples
            samples_duration = int(ret_value[1].split(' ')[0])
            # parse sectors
            ret_value = {"seconds": secs_duration, "samples": samples_duration}
        elif key in ['Channels', 'Sample Rate']:
            ret_value = int(value)
        elif key in ['Precision']:
            ret_value = int(value.split('-')[0])
        elif key in ['File Size']:
            ret_value = float(value.strip('k'))
        elif key in ['Bit Rate']:
            ret_value = float(value.strip('M'))
        else:
            ret_value = value
    except ValueError:
        ret_value = value
    return ret_value


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

    """
    return os.path.splitext(filepath)[1][1:]


def validate_input_file(input_filepath):
    """Input file validation function. Checks that file exists and can be
    processed by SoX.

    Parameters
    ----------
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


def validate_input_file_list(input_filepath_list):
    """Input file list validation function. Checks that object is a list and
    contains valid filepaths that can be processed by SoX.

    Parameters
    ----------
    input_filepath_list : list
        A list of filepaths.

    """
    if not isinstance(input_filepath_list, list):
        raise TypeError("input_filepath_list must be a list.")

    for input_filepath in input_filepath_list:
        validate_input_file(input_filepath)


def validate_output_file(output_filepath):
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


def validate_volumes(input_volumes):
    """Check input_volumes contains a valid list of volumes.

    Parameters
    ----------
    input_volumes : list
        list of volume values. Castable to numbers.

    """
    if not (input_volumes is None or isinstance(input_volumes, list)):
        raise TypeError("input_volumes must be None or a list.")

    if isinstance(input_volumes, list):
        for vol in input_volumes:
            if not is_number(vol):
                raise ValueError(
                    "Elements of input_volumes must be numbers: found {}"
                    .format(vol)
                )


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
