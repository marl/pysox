from pathlib import Path
import os
import unittest

from sox import file_info
from sox.core import SoxError


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SPACEY_FILE = relpath("data/annoying filename (derp).wav")
INPUT_FILE = relpath('data/input.wav')
INPUT_FILE2 = relpath('data/input.aiff')
INPUT_FILE3 = relpath('data/input.WAV')
EMPTY_FILE = relpath('data/empty.wav')
INPUT_FILE_INVALID = relpath('data/input.xyz')
OUTPUT_FILE = relpath('data/output.wav')
SILENT_FILE = relpath('data/silence.wav')


class TestCarriageReturnStrip(unittest.TestCase):

    def test_carriage_return_strip(self):
        def part_of_file_info_bitrate(output):
            # The characters below stand for kilo, Mega, Giga, etc.
            greek_prefixes = '\0kMGTPEZY'
            # Don't need the 'if output == "0":' branch here
            if output[-1] in greek_prefixes:
                multiplier = 1000.0**(greek_prefixes.index(output[-1]))
                return float(output[:-1])*multiplier
            else:
                return float(output[:-1])
        # Simulate a shell output with carriage return
        shell_output = '256k\n\r'
        # When only '\n' is stripped
        output = shell_output.strip('\n')
        with self.assertRaises(ValueError):
            part_of_file_info_bitrate(output)
        # When '\n\r' is stripped
        output = shell_output.strip('\n\r')
        actual = part_of_file_info_bitrate(output)
        expected = 256000.0
        self.assertIsInstance(actual, float)
        self.assertEqual(expected, actual)


class TestBitrate(unittest.TestCase):

    def test_wav(self):
        actual = file_info.bitrate(INPUT_FILE)
        expected = 706000.0
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.bitrate(Path(INPUT_FILE))
        expected = 706000.0
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.bitrate(INPUT_FILE2)
        expected = 768000.0
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.bitrate(EMPTY_FILE)
        expected = None
        self.assertEqual(expected, actual)



class TestBitdepth(unittest.TestCase):

    def test_wav(self):
        actual = file_info.bitdepth(INPUT_FILE)
        expected = 16
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.bitdepth(Path(INPUT_FILE))
        expected = 16
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.bitdepth(INPUT_FILE2)
        expected = 32
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.bitdepth(INPUT_FILE)
        expected = 16
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.bitdepth(INPUT_FILE2)
        expected = 32
        self.assertEqual(expected, actual)


class TestChannels(unittest.TestCase):

    def test_wav(self):
        actual = file_info.channels(INPUT_FILE)
        expected = 1
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.channels(Path(INPUT_FILE))
        expected = 1
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.channels(INPUT_FILE2)
        expected = 3
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.channels(EMPTY_FILE)
        expected = 1
        self.assertEqual(expected, actual)


class TestComments(unittest.TestCase):

    def test_wav(self):
        actual = file_info.comments(INPUT_FILE)
        expected = ""
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.comments(Path(INPUT_FILE))
        expected = ""
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.comments(INPUT_FILE2)
        expected = "Processed by SoX"
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.comments(EMPTY_FILE)
        expected = ""
        self.assertEqual(expected, actual)


class TestDuration(unittest.TestCase):

    def test_wav(self):
        actual = file_info.duration(INPUT_FILE)
        expected = 10.0
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.duration(Path(INPUT_FILE))
        expected = 10.0
        self.assertEqual(expected, actual)

    def test_spacey_wav(self):
        actual = file_info.duration(SPACEY_FILE)
        expected = 10.0
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.duration(INPUT_FILE2)
        expected = 10.0
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.duration(EMPTY_FILE)
        expected = None
        self.assertEqual(expected, actual)


class TestEncoding(unittest.TestCase):

    def test_wav(self):
        actual = file_info.encoding(INPUT_FILE)
        expected = "Signed Integer PCM"
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.encoding(Path(INPUT_FILE))
        expected = "Signed Integer PCM"
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.encoding(INPUT_FILE2)
        expected = "Signed Integer PCM"
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.encoding(EMPTY_FILE)
        expected = "Signed Integer PCM"
        self.assertEqual(expected, actual)


class TestFileType(unittest.TestCase):

    def test_wav(self):
        actual = file_info.file_type(INPUT_FILE)
        expected = "wav"
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.file_type(Path(INPUT_FILE))
        expected = "wav"
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.file_type(INPUT_FILE2)
        expected = "aiff"
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.file_type(EMPTY_FILE)
        expected = "wav"
        self.assertEqual(expected, actual)


class TestNumSamples(unittest.TestCase):

    def test_wav(self):
        actual = file_info.num_samples(INPUT_FILE)
        expected = 441000
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.num_samples(Path(INPUT_FILE))
        expected = 441000
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.num_samples(INPUT_FILE2)
        expected = 80000
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.num_samples(EMPTY_FILE)
        expected = None
        self.assertEqual(expected, actual)


class TestSampleRate(unittest.TestCase):

    def test_wav(self):
        actual = file_info.sample_rate(INPUT_FILE)
        expected = 44100
        self.assertEqual(expected, actual)

    def test_wav_pathlib(self):
        actual = file_info.sample_rate(Path(INPUT_FILE))
        expected = 44100
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.sample_rate(INPUT_FILE2)
        expected = 8000
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.sample_rate(EMPTY_FILE)
        expected = 44100
        self.assertEqual(expected, actual)


class TestSilent(unittest.TestCase):

    def test_nonsilent(self):
        actual = file_info.silent(INPUT_FILE)
        expected = False
        self.assertEqual(expected, actual)

    def test_nonsilent_pathlib(self):
        actual = file_info.silent(Path(INPUT_FILE))
        expected = False
        self.assertEqual(expected, actual)

    def test_silent(self):
        actual = file_info.silent(SILENT_FILE)
        expected = True
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.silent(EMPTY_FILE)
        expected = True
        self.assertEqual(expected, actual)


class TestFileExtension(unittest.TestCase):

    def test_ext1(self):
        actual = file_info.file_extension('simplefile.xyz')
        expected = 'xyz'
        self.assertEqual(expected, actual)

    def test_ext1_pathlib(self):
        actual = file_info.file_extension(Path('simplefile.xyz'))
        expected = 'xyz'
        self.assertEqual(expected, actual)

    def test_ext2(self):
        actual = file_info.file_extension('less.simple.file.xyz')
        expected = 'xyz'
        self.assertEqual(expected, actual)

    def test_ext3(self):
        actual = file_info.file_extension('longext.asdf')
        expected = 'asdf'
        self.assertEqual(expected, actual)

    def test_ext4(self):
        actual = file_info.file_extension('this/has/a/path/file.123')
        expected = '123'
        self.assertEqual(expected, actual)

    def test_ext5(self):
        actual = file_info.file_extension('this.is/a/weird.path/file.x23zya')
        expected = 'x23zya'
        self.assertEqual(expected, actual)

    def test_ext6(self):
        actual = file_info.file_extension('simplefile.MP3')
        expected = 'mp3'
        self.assertEqual(expected, actual)


class TestInfo(unittest.TestCase):

    def test_dictionary(self):
        for use_pathlib in [False, True]:
            with self.subTest():
                input_file = Path(INPUT_FILE) if use_pathlib else INPUT_FILE

                actual = file_info.info(input_file)
                expected = {
                    'channels': 1,
                    'sample_rate': 44100.0,
                    'bitdepth': 16,
                    'bitrate': 706000.0,
                    'duration': 10.0,
                    'num_samples': 441000,
                    'encoding': 'Signed Integer PCM',
                    'silent': False
                }
                self.assertEqual(expected, actual)


class TestValidateInputFile(unittest.TestCase):

    def test_valid(self):
        actual = file_info.validate_input_file(INPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)

    def test_valid_pathlib(self):
        actual = file_info.validate_input_file(Path(INPUT_FILE))
        expected = None
        self.assertEqual(expected, actual)

    def test_valid_wspaces(self):
        actual = file_info.validate_input_file(SPACEY_FILE)
        expected = None
        self.assertEqual(expected, actual)

    def test_nonexistent(self):
        with self.assertRaises(IOError):
            file_info.validate_input_file('data/asdfasdfasdf.wav')

    def test_invalid_format(self):
        actual = file_info.validate_input_file(INPUT_FILE_INVALID)
        expected = None
        self.assertEqual(expected, actual)


class TestValidateInputFileList(unittest.TestCase):

    def test_valid(self):
        actual = file_info.validate_input_file_list([INPUT_FILE, INPUT_FILE])
        expected = None
        self.assertEqual(expected, actual)

    def test_nonlist(self):
        with self.assertRaises(TypeError):
            file_info.validate_input_file_list(INPUT_FILE)

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            file_info.validate_input_file_list([])

    def test_len_one_list(self):
        with self.assertRaises(ValueError):
            file_info.validate_input_file_list([INPUT_FILE])

    def test_nonexistent(self):
        with self.assertRaises(IOError):
            file_info.validate_input_file_list(
                ['data/asdfasdfasdf.wav', INPUT_FILE]
            )

    def test_invalid_format(self):
        actual = file_info.validate_input_file_list(
            [INPUT_FILE_INVALID, INPUT_FILE]
        )
        expected = None
        self.assertEqual(expected, actual)


class TestValidateOutputFile(unittest.TestCase):

    def test_valid(self):
        actual = file_info.validate_output_file(OUTPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)

    def test_not_writeable(self):
        with self.assertRaises(IOError):
            file_info.validate_output_file('notafolder/output.wav')

    def test_invalid_format(self):
        actual = file_info.validate_output_file('output.xyz')
        expected = None
        self.assertEqual(expected, actual)

    def test_file_exists(self):
        actual = file_info.validate_output_file(INPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)


class TestStat(unittest.TestCase):

    def test_silent_file(self):
        expected = {
            'Samples read': 627456,
            'Length (seconds)': 14.228027,
            'Scaled by': 2147483647.0,
            'Maximum amplitude': 0.010895,
            'Minimum amplitude': -0.004883,
            'Midline amplitude': 0.003006,
            'Mean    norm': 0.000137,
            'Mean    amplitude': -0.000062,
            'RMS     amplitude': 0.000200,
            'Maximum delta': 0.015778,
            'Minimum delta': 0.000000,
            'Mean    delta': 0.000096,
            'RMS     delta': 0.000124,
            'Rough   frequency': 4349,
            'Volume adjustment': 91.787
        }
        actual = file_info.stat(SILENT_FILE)
        self.assertEqual(expected, actual)


class TestStatCall(unittest.TestCase):

    def test_stat_call(self):
        expected = ('Samples read:            627456\nLength (seconds):'
                    '     14.228027\nScaled by:         2147483647.0\nMax'
                    'imum amplitude:     0.010895\nMinimum amplitude:    '
                    '-0.004883\nMidline amplitude:     0.003006\nMean    '
                    'norm:          0.000137\nMean    amplitude:    -0.000'
                    '062\nRMS     amplitude:     0.000200\nMaximum delta:  '
                    '       0.015778\nMinimum delta:         0.000000\nMean'
                    '    delta:         0.000096\nRMS     delta:         '
                    '0.000124\nRough   frequency:         4349\nVolume '
                    'adjustment:       91.787\n')
        actual = file_info._stat_call(SILENT_FILE)
        self.assertEqual(expected, actual)


class TestParseStat(unittest.TestCase):

    def test_empty(self):
        stat_output = ''
        expected = {}
        actual = file_info._parse_stat(stat_output)
        self.assertEqual(expected, actual)

    def test_simple(self):
        stat_output = 'Blorg: 1.2345\nPlombus:   -0.0001\nMrs.   Pancakes: a'
        expected = {
            'Blorg': 1.2345,
            'Plombus': -0.0001,
            'Mrs.   Pancakes': None
        }
        actual = file_info._parse_stat(stat_output)
        self.assertEqual(expected, actual)

    def test_real_output(self):
        stat_output = ('Samples read:            627456\nLength (seconds):'
                       '     14.228027\nScaled by:         2147483647.0\nMax'
                       'imum amplitude:     0.010895\nMinimum amplitude:    '
                       '-0.004883\nMidline amplitude:     0.003006\nMean    '
                       'norm:          0.000137\nMean    amplitude:    -0.000'
                       '062\nRMS     amplitude:     0.000200\nMaximum delta:  '
                       '       0.015778\nMinimum delta:         0.000000\nMean'
                       '    delta:         0.000096\nRMS     delta:         '
                       '0.000124\nRough   frequency:         4349\nVolume '
                       'adjustment:       91.787\n')
        expected = {
            'Samples read': 627456,
            'Length (seconds)': 14.228027,
            'Scaled by': 2147483647.0,
            'Maximum amplitude': 0.010895,
            'Minimum amplitude': -0.004883,
            'Midline amplitude': 0.003006,
            'Mean    norm': 0.000137,
            'Mean    amplitude': -0.000062,
            'RMS     amplitude': 0.000200,
            'Maximum delta': 0.015778,
            'Minimum delta': 0.000000,
            'Mean    delta': 0.000096,
            'RMS     delta': 0.000124,
            'Rough   frequency': 4349,
            'Volume adjustment': 91.787
        }
        actual = file_info._parse_stat(stat_output)
        self.assertEqual(expected, actual)

