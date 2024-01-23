''' Audio file info computed by soxi.
'''
import os
from numbers import Number
from pathlib import Path
from typing import List, Dict
from typing import Optional, Union

from .core import VALID_FORMATS
from .core import sox
from .core import soxi
from .log import logger


def bitdepth(input_filepath: Union[str, Path]) -> Optional[int]:
    '''
    Number of bits per sample, or None if not applicable.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    bitdepth : int or None
        Number of bits per sample.
        Returns None if not applicable.
    '''

    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'b')
    if output == '0':
        logger.warning("Bit depth unavailable for %s", input_filepath)
        return None
    return int(output)


def bitrate(input_filepath: Union[str, Path]) -> Optional[float]:
    '''
    Bit rate averaged over the whole file.
    Expressed in bytes per second (bps), or None if not applicable.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    bitrate : float or None
        Bit rate, expressed in bytes per second.
        Returns None if not applicable.
    '''

    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'B')
    # The characters below stand for kilo, Mega, Giga, etc.
    greek_prefixes = '\0kMGTPEZY'
    if output == "0":
        logger.warning("Bit rate unavailable for %s", input_filepath)
        return None
    elif output[-1] in greek_prefixes:
        multiplier = 1000.0**(greek_prefixes.index(output[-1]))
        return float(output[:-1])*multiplier
    else:
        return float(output[:-1])


def channels(input_filepath: Union[str, Path]) -> int:
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


def comments(input_filepath: Union[str, Path]) -> str:
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


def duration(input_filepath: Union[str, Path]) -> Optional[float]:
    '''
    Show duration in seconds, or None if not available.

    Parameters
    ----------
    input_filepath : str
        Path to audio file.

    Returns
    -------
    duration : float or None
        Duration of audio file in seconds.
        If unavailable or empty, returns None.
    '''
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 'D')
    if float(output) == 0.0:
        logger.warning("Duration unavailable for %s", input_filepath)
        return None
    return float(output)


def encoding(input_filepath: Union[str, Path]) -> str:
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


def file_type(input_filepath: Union[str, Path]) -> str:
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


def num_samples(input_filepath: Union[str, Path]) -> Optional[int]:
    '''
    Show number of samples, or None if unavailable.

    Parameters
    ----------
    input_filepath : path-like (str or pathlib.Path)
        Path to audio file.

    Returns
    -------
    n_samples : int or None
        total number of samples in audio file.
        Returns None if empty or unavailable.
    '''
    input_filepath = str(input_filepath)
    validate_input_file(input_filepath)
    output = soxi(input_filepath, 's')
    if output == '0':
        logger.warning("Number of samples unavailable for %s", input_filepath)
        return None
    return int(output)


def sample_rate(input_filepath: Union[str, Path]) -> float:
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


def silent(input_filepath: Union[str, Path], threshold:float = 0.001) -> bool:
    '''
    Determine if an input file is silent.

    Parameters
    ----------
    input_filepath : str
        The input filepath.
    threshold : float
        Threshold for determining silence

    Returns
    -------
    is_silent : bool
        True if file is determined silent.
    '''
    validate_input_file(input_filepath)
    stat_dictionary = stat(input_filepath)
    mean_norm = stat_dictionary['Mean    norm']
    if mean_norm is not float('nan'):
        if mean_norm >= threshold:
            return False
        else:
            return True
    else:
        return True


def validate_input_file(input_filepath: Union[str, Path]) -> None:
    '''Input file validation function. Checks that file exists and can be
    processed by SoX.

    Parameters
    ----------
    input_filepath : path-like (str or pathlib.Path)
        The input filepath.

    '''
    input_filepath = Path(input_filepath)
    if not input_filepath.exists():
        raise OSError(
            f"input_filepath {input_filepath} does not exist."
        )
    ext = file_extension(input_filepath)
    if ext not in VALID_FORMATS:
        logger.info("Valid formats: %s", " ".join(VALID_FORMATS))
        logger.warning(
            f"This install of SoX cannot process .{ext} files."
        )


def validate_input_file_list(input_filepath_list: List[Union[str, Path]]) -> None:
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


def validate_output_file(output_filepath: Union[str, Path]) -> None:
    '''Output file validation function. Checks that file can be written, and
    has a valid file extension. Throws a warning if the path already exists,
    as it will be overwritten on build.

    Parameters
    ----------
    output_filepath : path-like (str or pathlib.Path)
        The output filepath.

    '''
    # This function enforces use of the path as a string, because
    # os.access has no analog in pathlib.
    output_filepath = str(output_filepath)

    if output_filepath == '-n':
        return

    nowrite_conditions = [
        bool(os.path.dirname(output_filepath)) or
             not os.access(os.getcwd(), os.W_OK),
        not os.access(os.path.dirname(output_filepath), os.W_OK)]

    if all(nowrite_conditions):
        raise OSError(
            f"SoX cannot write to output_filepath {output_filepath}"
        )

    ext = file_extension(output_filepath)
    if ext not in VALID_FORMATS:
        logger.info("Valid formats: %s", " ".join(VALID_FORMATS))
        logger.warning(
            f"This install of SoX cannot process .{ext} files."
        )

    if os.path.exists(output_filepath):
        logger.warning(
            'output_file: %s already exists and will be overwritten on build',
            output_filepath
        )


def file_extension(filepath: Union[str, Path]) -> str:
    '''Get the extension of a filepath.

    Parameters
    ----------
    filepath : path-like (str or pathlib.Path)
        File path.

    Returns
    -------
    extension : str
        The file's extension
    '''
    return Path(filepath).suffix[1:].lower()


def info(filepath: Union[str, Path]) -> Dict[str, Union[str, Number]]:
    '''Get a dictionary of file information

    Parameters
    ----------
    filepath : str
        File path.

    Returns
    -------
    info_dictionary : dict
        Dictionary of file information. Fields are:
            * channels
            * sample_rate
            * bitdepth
            * bitrate
            * duration
            * num_samples
            * encoding
            * silent
    '''
    info_dictionary = {
        'channels': channels(filepath),
        'sample_rate': sample_rate(filepath),
        'bitdepth': bitdepth(filepath),
        'bitrate': bitrate(filepath),
        'duration': duration(filepath),
        'num_samples': num_samples(filepath),
        'encoding': encoding(filepath),
        'silent': silent(filepath)
    }
    return info_dictionary


def stat(filepath: Union[str, Path]) -> Dict[str, Optional[float]]:
    '''Returns a dictionary of audio statistics.

    Parameters
    ----------
    filepath : str
        File path.

    Returns
    -------
    stat_dictionary : dict
        Dictionary of audio statistics.
    '''
    stat_output = _stat_call(filepath)
    stat_dictionary = _parse_stat(stat_output)
    return stat_dictionary


def _stat_call(filepath: Union[str, Path]) -> str:
    '''Call sox's stat function.

    Parameters
    ----------
    filepath : str
        File path.

    Returns
    -------
    stat_output : str
        Sox output from stderr.
    '''
    validate_input_file(filepath)
    args = ['sox', filepath, '-n', 'stat']
    _, _, stat_output = sox(args)
    return stat_output


def _parse_stat(stat_output: str) -> Dict[str, Optional[float]]:
    '''Parse the string output from sox's stat function

    Parameters
    ----------
    stat_output : str
        Sox output from stderr.

    Returns
    -------
    stat_dictionary : dict
        Dictionary of audio statistics.
    '''
    lines = stat_output.split('\n')
    stat_dict = {}
    for line in lines:
        split_line = line.split(':')
        if len(split_line) == 2:
            key = split_line[0]
            val = split_line[1].strip(' ')
            try:
                val = float(val)
            except ValueError:
                val = None
            stat_dict[key] = val

    return stat_dict
