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
        expected = {}
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

    def test_build_with_vols(self):
        expected_result = True
        actual_result = self.cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix',
            input_volumes=[0.5, 2]
        )
        self.assertEqual(expected_result, actual_result)

    def test_failed_build(self):
        cbn = new_combiner()
        with self.assertRaises(SoxError):
            cbn.build(
                [INPUT_FILE_INVALID, INPUT_WAV], OUTPUT_FILE, 'concatenate'
            )

    def test_build_with_output_format(self):
        expected_result = True
        cbn = new_combiner()
        cbn.set_output_format(rate=8000)
        actual_result = self.cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'concatenate'
        )
        self.assertEqual(expected_result, actual_result)


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


class TestSetInputFormat(unittest.TestCase):

    def test_none(self):
        cbn = new_combiner()
        cbn.set_input_format()
        expected = []
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_file_type(self):
        cbn = new_combiner()
        cbn.set_input_format(file_type=['wav', 'aiff'])
        expected = [['-t', 'wav'], ['-t', 'aiff']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_invalid_file_type(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(file_type='wav')

    def test_invalid_file_type_val(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(file_type=['xyz', 'wav'])

    def test_rate(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[2000, 44100, 22050])
        expected = [['-r', '2000'], ['-r', '44100'], ['-r', '22050']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_invalid_rate(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(rate=2000)

    def test_invalid_rate_val(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(rate=[-2, 'a'])

    def test_bits(self):
        cbn = new_combiner()
        cbn.set_input_format(bits=[16])
        expected = [['-b', '16']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_invalid_bits(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(bits=32)

    def test_invalid_bits_val(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(bits=[0])

    def test_channels(self):
        cbn = new_combiner()
        cbn.set_input_format(channels=[1, 2, 3])
        expected = [['-c', '1'], ['-c', '2'], ['-c', '3']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_invalid_channels(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(channels='x')

    def test_invalid_channels_val(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(channels=[1.5, 2, 3])

    def test_encoding(self):
        cbn = new_combiner()
        cbn.set_input_format(encoding=['floating-point', 'oki-adpcm'])
        expected = [['-e', 'floating-point'], ['-e', 'oki-adpcm']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_invalid_encoding(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(encoding='wav')

    def test_invalid_encoding_val(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(encoding=['xyz', 'wav'])

    def test_ignore_length(self):
        cbn = new_combiner()
        cbn.set_input_format(ignore_length=[True, False, True])
        expected = [['--ignore-length'], [], ['--ignore-length']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_invalid_ignore_length(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(ignore_length=1)

    def test_invalid_ignore_length_val(self):
        cbn = new_combiner()
        with self.assertRaises(ValueError):
            cbn.set_input_format(ignore_length=[False, True, 3])

    def test_multiple_same_len(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 2000], bits=[32, 8])
        expected = [['-r', '44100', '-b', '32'], ['-r', '2000', '-b', '8']]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_multiple_different_len(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 2000], bits=[32, 8, 16])
        expected = [
            ['-r', '44100', '-b', '32'],
            ['-r', '2000', '-b', '8'],
            ['-b', '16']
        ]
        actual = cbn.input_format
        self.assertEqual(expected, actual)

    def test_build_same_len(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 44100], channels=[1, 1])
        actual = cbn.build([INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix')
        expected = True
        self.assertEqual(expected, actual)

    def test_build_same_len_vol(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 44100], channels=[1, 1])
        actual = cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix', input_volumes=[1, 2]
        )
        expected = True
        self.assertEqual(expected, actual)

    def test_build_greater_len(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 44100, 44100], channels=[1, 1])
        actual = cbn.build([INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix')
        expected = True
        self.assertEqual(expected, actual)

    def test_build_greater_len_vol(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 44100, 44100], channels=[1, 1])
        actual = cbn.build(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix', input_volumes=[1, 2]
        )
        expected = True
        self.assertEqual(expected, actual)

    def test_build_lesser_len(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 44100], channels=[1, 1])
        actual = cbn.build(
            [INPUT_WAV, INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix'
        )
        expected = True
        self.assertEqual(expected, actual)

    def test_build_lesser_len_vol(self):
        cbn = new_combiner()
        cbn.set_input_format(rate=[44100, 44100], channels=[1, 1])
        actual = cbn.build(
            [INPUT_WAV, INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix',
            input_volumes=[1, 2]
        )
        expected = True
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
            [INPUT_WAV, INPUT_WAV], None, None
        )
        self.assertEqual(expected, actual)

    def test_equal_num_vol(self):
        expected = [['-v', '0.5'], ['-v', '1.1']]
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], [0.5, 1.1], None
        )
        self.assertEqual(expected, actual)

    def test_greater_num_vol(self):
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], [0.5, 1.1, 3], None
        )
        expected = [['-v', '0.5'], ['-v', '1.1']]
        self.assertEqual(expected, actual)

    def test_lesser_num_vol(self):
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV, INPUT_WAV], [0.5, 1.1], None
        )
        expected = [['-v', '0.5'], ['-v', '1.1'], ['-v', '1']]
        self.assertEqual(expected, actual)

    def test_equal_num_fmt(self):
        expected = [['-v', '1', '-t', 'wav'], ['-v', '1', '-t', 'aiff']]
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], None, [['-t', 'wav'], ['-t', 'aiff']]
        )
        self.assertEqual(expected, actual)

    def test_greater_num_fmt(self):
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV], None,
            [['-t', 'wav'], ['-t', 'aiff'], ['-t', 'wav']]
        )
        expected = [['-v', '1', '-t', 'wav'], ['-v', '1', '-t', 'aiff']]
        self.assertEqual(expected, actual)

    def test_lesser_num_fmt(self):
        actual = combine._build_input_format_list(
            [INPUT_WAV, INPUT_WAV, INPUT_WAV], None,
            [['-t', 'wav'], ['-t', 'aiff']]
        )
        expected = [
            ['-v', '1', '-t', 'wav'], ['-v', '1', '-t', 'aiff'], ['-v', '1']
        ]
        self.assertEqual(expected, actual)


class TestCombinePreview(unittest.TestCase):
    def setUp(self):
        self.cbn = new_combiner()
        self.cbn.trim(0, 0.1)

    def test_valid(self):
        expected = None
        actual = self.cbn.preview([INPUT_WAV, INPUT_WAV], 'mix')
        self.assertEqual(expected, actual)

    def test_valid_vol(self):
        expected = None
        actual = self.cbn.preview([INPUT_WAV, INPUT_WAV], 'mix', [1.0, 0.5])
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
