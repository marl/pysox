import unittest

from sox import core
from sox.core import SoxError

INPUT_FILE = 'data/input.wav'
INPUT_FILE_INVALID = 'data/input.xyz'
OUTPUT_FILE = 'data/output.wav'


class TestSox(unittest.TestCase):

    def test_base_case(self):
        args = ['sox', INPUT_FILE, OUTPUT_FILE]
        expected = True
        actual = core.sox(args)
        self.assertEqual(expected, actual)

    def test_base_case2(self):
        args = [INPUT_FILE, OUTPUT_FILE]
        expected = True
        actual = core.sox(args)
        self.assertEqual(expected, actual)

    def test_sox_fail_bad_args(self):
        args = ['-asdf']
        expected = False
        actual = core.sox(args)
        self.assertEqual(expected, actual)

    def test_sox_fail_bad_files(self):
        args = ['asdf.wav', 'flululu.wav']
        expected = False
        actual = core.sox(args)
        self.assertEqual(expected, actual)

    def test_sox_fail_bad_ext(self):
        args = ['input.wav', 'output.xyz']
        expected = False
        actual = core.sox(args)
        self.assertEqual(expected, actual)


class TestGetValidFormats(unittest.TestCase):

    def setUp(self):
        self.formats = core._get_valid_formats()

    def test_wav(self):
        self.assertIn('wav', self.formats)

    def test_aiff(self):
        self.assertIn('aiff', self.formats)

    def test_notin(self):
        self.assertNotIn('AUDIO', self.formats)
        self.assertNotIn('FILE', self.formats)
        self.assertNotIn('FORMATS', self.formats)
        self.assertNotIn('AUDIO FILE FORMATS', self.formats)


class TestValidFormats(unittest.TestCase):

    def test_wav(self):
        self.assertIn('wav', core.VALID_FORMATS)

    def test_aiff(self):
        self.assertIn('aiff', core.VALID_FORMATS)

    def test_notin(self):
        self.assertNotIn('AUDIO', core.VALID_FORMATS)
        self.assertNotIn('FILE', core.VALID_FORMATS)
        self.assertNotIn('FORMATS', core.VALID_FORMATS)
        self.assertNotIn('AUDIO FILE FORMATS', core.VALID_FORMATS)


class TestFileExtension(unittest.TestCase):

    def test_ext1(self):
        actual = core._file_extension('simplefile.xyz')
        expected = 'xyz'
        self.assertEqual(expected, actual)

    def test_ext2(self):
        actual = core._file_extension('less.simple.file.xyz')
        expected = 'xyz'
        self.assertEqual(expected, actual)

    def test_ext3(self):
        actual = core._file_extension('longext.asdf')
        expected = 'asdf'
        self.assertEqual(expected, actual)

    def test_ext4(self):
        actual = core._file_extension('this/has/a/path/file.123')
        expected = '123'
        self.assertEqual(expected, actual)

    def test_ext5(self):
        actual = core._file_extension('this.is/a/weird.path/file.x23zya')
        expected = 'x23zya'
        self.assertEqual(expected, actual)


class TestValidateInputFile(unittest.TestCase):

    def test_valid(self):
        actual = core.validate_input_file(INPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)

    def test_nonexistent(self):
        with self.assertRaises(IOError):
            core.validate_input_file('data/asdfasdfasdf.wav')

    def test_invalid_format(self):
        with self.assertRaises(SoxError):
            core.validate_input_file(INPUT_FILE_INVALID)


class TestValidateInputFileList(unittest.TestCase):

    def test_valid(self):
        actual = core.validate_input_file_list([INPUT_FILE, INPUT_FILE])
        expected = None
        self.assertEqual(expected, actual)

    def test_nonexistent(self):
        with self.assertRaises(IOError):
            core.validate_input_file_list(['data/asdfasdfasdf.wav', INPUT_FILE])

    def test_invalid_format(self):
        with self.assertRaises(SoxError):
            core.validate_input_file_list([INPUT_FILE_INVALID, INPUT_FILE])


class TestValidateOutputFile(unittest.TestCase):

    def test_valid(self):
        actual = core.validate_output_file(OUTPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)

    def test_not_writeable(self):
        with self.assertRaises(IOError):
            core.validate_output_file('data/notafolder/output.wav')

    def test_invalid_format(self):
        with self.assertRaises(SoxError):
            core.validate_output_file('data/output.xyz')

    def test_file_exists(self):
        actual = core.validate_output_file(INPUT_FILE)
        expected = None
        self.assertEqual(expected, actual)


class TestValidateVolumes(unittest.TestCase):

    def test_valid_none(self):
        actual = core.validate_volumes(None)
        expected = None
        self.assertEqual(expected, actual)

    def test_valid_list(self):
        actual = core.validate_volumes([1, 0.1, 3])
        expected = None
        self.assertEqual(expected, actual)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            core.validate_volumes(1)

    def test_invalid_vol(self):
        with self.assertRaises(ValueError):
            core.validate_volumes([1.1, 'z', -0.5, 2])


class TestIsNumber(unittest.TestCase):

    def test_numeric(self):
        actual = core.is_number(0)
        expected = True
        self.assertEqual(expected, actual)

    def test_numeric2(self):
        actual = core.is_number(1.213215)
        expected = True
        self.assertEqual(expected, actual)

    def test_numeric3(self):
        actual = core.is_number(-100)
        expected = True
        self.assertEqual(expected, actual)

    def test_numeric4(self):
        actual = core.is_number('13.54')
        expected = True
        self.assertEqual(expected, actual)

    def test_nonnumeric(self):
        actual = core.is_number('a')
        expected = False
        self.assertEqual(expected, actual)

    def test_nonnumeric2(self):
        actual = core.is_number('-f')
        expected = False
        self.assertEqual(expected, actual)

    def test_nonnumeric3(self):
        actual = core.is_number([1])
        expected = False
        self.assertEqual(expected, actual)
