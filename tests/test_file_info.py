import unittest

from sox import file_info
from sox.file_info import SoxiError
from sox.core import SoxError

INPUT_FILE = 'data/input.wav'
INPUT_FILE2 = 'data/input.aiff'
EMPTY_FILE = 'data/empty.wav'
BREAK_SOXI_FILE = 'data/empty.aiff'
INPUT_FILE_INVALID = 'data/input.xyz'

class TestSoxi(unittest.TestCase):

    def test_base_case(self):
        actual = file_info.soxi(INPUT_FILE, 's')
        expected = '441000'
        self.assertEqual(expected, actual)

    def test_invalid_argument(self):
        with self.assertRaises(ValueError):
            file_info.soxi(INPUT_FILE, None)

    def test_nonexistent_file(self):
        with self.assertRaises(IOError):
            file_info.soxi('data/asdf.wav', 's')

    def test_invalid_filetype(self):
        with self.assertRaises(SoxError):
            file_info.soxi(INPUT_FILE_INVALID, 's')

    def test_soxi_error(self):
        with self.assertRaises(SoxiError):
            file_info.soxi(BREAK_SOXI_FILE, 's')


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

