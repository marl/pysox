import unittest
import os

from sox import file_info
from sox.core import SoxError


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SPACEY_FILE = relpath("data/annoying filename (derp).wav")
INPUT_FILE = relpath('data/input.wav')
INPUT_FILE2 = relpath('data/input.aiff')
EMPTY_FILE = relpath('data/empty.wav')
INPUT_FILE_INVALID = relpath('data/input.xyz')
OUTPUT_FILE = relpath('data/output.wav')


class TestBitrate(unittest.TestCase):

    def test_wav(self):
        actual = file_info.bitrate(INPUT_FILE)
        expected = 16
        self.assertEqual(expected, actual)

    def test_aiff(self):
        actual = file_info.bitrate(INPUT_FILE2)
        expected = 32
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.bitrate(EMPTY_FILE)
        expected = 16
        self.assertEqual(expected, actual)


class TestChannels(unittest.TestCase):

    def test_wav(self):
        actual = file_info.channels(INPUT_FILE)
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
        expected = 0
        self.assertEqual(expected, actual)


class TestEncoding(unittest.TestCase):

    def test_wav(self):
        actual = file_info.encoding(INPUT_FILE)
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

    def test_aiff(self):
        actual = file_info.num_samples(INPUT_FILE2)
        expected = 80000
        self.assertEqual(expected, actual)

    def test_empty(self):
        actual = file_info.num_samples(EMPTY_FILE)
        expected = 0
        self.assertEqual(expected, actual)


class TestSampleRate(unittest.TestCase):

    def test_wav(self):
        actual = file_info.sample_rate(INPUT_FILE)
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


class TestFileExtension(unittest.TestCase):

    def test_ext1(self):
        actual = file_info.file_extension('simplefile.xyz')
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


class TestValidateInputFile(unittest.TestCase):

    def test_valid(self):
        actual = file_info.validate_input_file(INPUT_FILE)
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
        with self.assertRaises(SoxError):
            file_info.validate_input_file(INPUT_FILE_INVALID)


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
        with self.assertRaises(SoxError):
            file_info.validate_input_file_list(
                [INPUT_FILE_INVALID, INPUT_FILE]
            )


class TestValidateOutputFile(unittest.TestCase):

    def test_valid(self):
        actual = file_info.validate_output_file(OUTPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)

    def test_not_writeable(self):
        with self.assertRaises(IOError):
            file_info.validate_output_file('notafolder/output.wav')

    def test_invalid_format(self):
        with self.assertRaises(SoxError):
            file_info.validate_output_file('output.xyz')

        with self.assertRaises(SoxError):
            file_info.validate_output_file('./output.xyz')

    def test_file_exists(self):
        actual = file_info.validate_output_file(INPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)
