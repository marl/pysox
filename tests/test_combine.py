import unittest
import os

from sox import combine
from sox.core import SoxError


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


INPUT_WAV = relpath('data/input.wav')
INPUT_WAV2 = relpath('data/input2.wav')
INPUT_WAV3 = relpath('data/input3.wav')
INPUT_AIFF = relpath('data/input.aiff')
INPUT_FILE_INVALID = relpath('data/input.xyz')
OUTPUT_FILE = relpath('data/output.wav')


def new_combiner():
    return combine.Combiner()


class TestCombineDefault(unittest.TestCase):
    def setUp(self):
        self.cbn = new_combiner()

    def test_globals(self):
        expected = ['-D', '-V2']
        actual = self.cbn.globals
        self.assertEqual(expected, actual)

    def test_output_format(self):
        expected = []
        actual = self.cbn.output_format
        self.assertEqual(expected, actual)

    def test_effects(self):
        expected = []
        actual = self.cbn.effects
        self.assertEqual(expected, actual)

    def test_effects_log(self):
        expected = []
        actual = self.cbn.effects_log
        self.assertEqual(expected, actual)

    def test_build(self):
        expected_result = True
        actual_result = self.cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'concatenate'
        )
        self.assertEqual(expected_result, actual_result)

    def test_failed_build(self):
        cbn = new_combiner()
        with self.assertRaises(SoxError):
            cbn.build(
                [INPUT_FILE_INVALID, INPUT_WAV], OUTPUT_FILE, 'concatenate'
            )


class TestCombineTypes(unittest.TestCase):

    def setUp(self):
        self.cbn = new_combiner()

    def test_concatenate(self):
        expected = True
        actual = self.cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'concatenate'
        )
        self.assertEqual(expected, actual)

    def test_merge(self):
        expected = True
        actual = self.cbn.build([INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'merge')
        self.assertEqual(expected, actual)

    def test_mix(self):
        expected = True
        actual = self.cbn.build([INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix')
        self.assertEqual(expected, actual)

    def test_mixpower(self):
        expected = True
        actual = self.cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix-power'
        )
        self.assertEqual(expected, actual)

    def test_multiply(self):
        expected = True
        actual = self.cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'multiply'
        )
        self.assertEqual(expected, actual)


class TestValidateFileFormats(unittest.TestCase):

    def test_different_samplerates(self):
        with self.assertRaises(IOError):
            combine._validate_file_formats([INPUT_WAV, INPUT_WAV2], 'mix')

    def test_different_num_channels(self):
        with self.assertRaises(IOError):
            combine._validate_file_formats(
                [INPUT_WAV, INPUT_WAV3], 'concatenate'
            )


class TestValidateSampleRates(unittest.TestCase):

    def test_different_samplerates(self):
        with self.assertRaises(IOError):
            combine._validate_sample_rates([INPUT_WAV, INPUT_WAV2], 'mix')

    def test_same_samplerates(self):
        expected = None
        actual = combine._validate_sample_rates([INPUT_WAV, INPUT_WAV], 'mix')
        self.assertEqual(expected, actual)


class TestValidateNumChannels(unittest.TestCase):

    def test_different_numchannels(self):
        with self.assertRaises(IOError):
            combine._validate_num_channels([INPUT_WAV, INPUT_WAV3], 'mix')

    def test_same_numchannels(self):
        expected = None
        actual = combine._validate_num_channels([INPUT_WAV, INPUT_WAV], 'mix')
        self.assertEqual(expected, actual)


class TestBuildInputFormatList(unittest.TestCase):

    def test_none(self):
        expected = [['-v', '1'], ['-v', '1']]
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], None
        )
        self.assertEqual(expected, actual)

    def test_equal_num(self):
        expected = [['-v', '0.5'], ['-v', '1.1']]
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], [0.5, 1.1]
        )
        self.assertEqual(expected, actual)

    def test_greater_num(self):
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], [0.5, 1.1, 3]
        )
        expected = [['-v', '0.5'], ['-v', '1.1']]
        self.assertEqual(expected, actual)

    def test_lesser_num(self):
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV, INPUT_WAV], [0.5, 1.1]
        )
        expected = [['-v', '0.5'], ['-v', '1.1'], ['-v', '1']]
        self.assertEqual(expected, actual)


class TestBuildInputArgs(unittest.TestCase):

    def test_unequal_length(self):
        with self.assertRaises(ValueError):
            combine._build_input_args([INPUT_WAV, INPUT_WAV], [['-v', '1']])

    def test_basic(self):
        expected = ['-v', '1', INPUT_WAV, '-v', '1', INPUT_WAV]
        actual = combine._build_input_args(
            [INPUT_WAV, INPUT_WAV], [['-v', '1'], ['-v', '1']]
        )
        self.assertEqual(expected, actual)


class TestValidateCombineType(unittest.TestCase):

    def test_valid(self):
        actual = combine._validate_combine_type('mix')
        expected = None
        self.assertEqual(expected, actual)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            combine._validate_combine_type('combine')


class TestValidateVolumes(unittest.TestCase):

    def test_valid_none(self):
        actual = combine._validate_volumes(None)
        expected = None
        self.assertEqual(expected, actual)

    def test_valid_list(self):
        actual = combine._validate_volumes([1, 0.1, 3])
        expected = None
        self.assertEqual(expected, actual)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            combine._validate_volumes(1)

    def test_invalid_vol(self):
        with self.assertRaises(ValueError):
            combine._validate_volumes([1.1, 'z', -0.5, 2])
