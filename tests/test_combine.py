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


def new_combiner(combiner='concatenate'):
    return combine.Combiner(
        [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, combiner
    )


class TestCombineDefault(unittest.TestCase):
    def setUp(self):
        self.cbn = new_combiner()

    def test_globals(self):
        expected = ['-D', '-V2']
        actual = self.cbn.globals
        self.assertEqual(expected, actual)

    def test_input_filepath_list(self):
        expected = [INPUT_WAV, INPUT_WAV]
        actual = self.cbn.input_filepath_list
        self.assertEqual(expected, actual)

    def test_output_filepath(self):
        expected = OUTPUT_FILE
        actual = self.cbn.output_filepath
        self.assertEqual(expected, actual)

    def test_input_format_list(self):
        expected = [['-v', '1'], ['-v', '1']]
        actual = self.cbn.input_format_list
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

    def test_combine(self):
        expected = ['--combine', 'concatenate']
        actual = self.cbn.combine
        self.assertEqual(expected, actual)

    def test_input_args_pre_build(self):
        expected = []
        actual = self.cbn.input_args
        self.assertEqual(expected, actual)

    def test_build(self):
        expected_result = True
        actual_result = self.cbn.build()
        self.assertEqual(expected_result, actual_result)

        expected = ['-v', '1', INPUT_WAV, '-v', '1', INPUT_WAV]
        actual = self.cbn.input_args
        self.assertEqual(expected, actual)

    def test_failed_build(self):
        cbn = new_combiner()
        cbn.input_filepath_list[0] = INPUT_FILE_INVALID
        with self.assertRaises(SoxError):
            cbn.build()


class TestCombineTypes(unittest.TestCase):

    def test_concatenate(self):
        cbn = new_combiner('concatenate')
        expected = True
        actual = cbn.build()
        self.assertEqual(expected, actual)

    def test_merge(self):
        cbn = new_combiner('merge')
        expected = True
        actual = cbn.build()
        self.assertEqual(expected, actual)

    def test_mix(self):
        cbn = new_combiner('mix')
        expected = True
        actual = cbn.build()
        self.assertEqual(expected, actual)

    def test_mixpower(self):
        cbn = new_combiner('mix-power')
        expected = True
        actual = cbn.build()
        self.assertEqual(expected, actual)

    def test_multiply(self):
        cbn = new_combiner('multiply')
        expected = True
        actual = cbn.build()
        self.assertEqual(expected, actual)


class TestValidateFileFormats(unittest.TestCase):

    def test_different_samplerates(self):
        with self.assertRaises(IOError):
            combine.Combiner(
                [INPUT_WAV, INPUT_WAV2], OUTPUT_FILE, 'mix'
            )

    def test_different_num_channels(self):
        with self.assertRaises(IOError):
            combine.Combiner(
                [INPUT_WAV, INPUT_WAV3], OUTPUT_FILE, 'concatenate'
            )


class TestSetInputFormatList(unittest.TestCase):

    def test_none(self):
        cbn = combine.Combiner(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix'
        )
        expected = [['-v', '1'], ['-v', '1']]
        actual = cbn.input_format_list
        self.assertEqual(expected, actual)

        expected_res = True
        actual_res = cbn.build()
        self.assertEqual(expected_res, actual_res)

        expected_input_args = [
            '-v', '1', INPUT_WAV, '-v', '1', INPUT_WAV
        ]
        actual_input_args = cbn.input_args
        self.assertEqual(expected_input_args, actual_input_args)

    def test_equal_num(self):
        cbn = combine.Combiner(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix',
            [0.5, 1.1]
        )
        expected = [['-v', '0.5'], ['-v', '1.1']]
        actual = cbn.input_format_list
        self.assertEqual(expected, actual)

        expected_res = True
        actual_res = cbn.build()
        self.assertEqual(expected_res, actual_res)

        expected_input_args = [
            '-v', '0.5', INPUT_WAV, '-v', '1.1', INPUT_WAV
        ]
        actual_input_args = cbn.input_args
        self.assertEqual(expected_input_args, actual_input_args)

    def test_greater_num(self):
        cbn = combine.Combiner(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix',
            [0.5, 1.1, 3]
        )
        expected = [['-v', '0.5'], ['-v', '1.1']]
        actual = cbn.input_format_list
        self.assertEqual(expected, actual)

        expected_res = True
        actual_res = cbn.build()
        self.assertEqual(expected_res, actual_res)

        expected_input_args = [
            '-v', '0.5', INPUT_WAV, '-v', '1.1', INPUT_WAV
        ]
        actual_input_args = cbn.input_args
        self.assertEqual(expected_input_args, actual_input_args)

    def test_lesser_num(self):
        cbn = combine.Combiner(
            [INPUT_WAV, INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix',
            [0.5, 1.1]
        )
        expected = [['-v', '0.5'], ['-v', '1.1'], ['-v', '1']]
        actual = cbn.input_format_list
        self.assertEqual(expected, actual)

        expected_res = True
        actual_res = cbn.build()
        self.assertEqual(expected_res, actual_res)

        expected_input_args = [
            '-v', '0.5', INPUT_WAV, '-v', '1.1', INPUT_WAV,
            '-v', '1', INPUT_WAV
        ]
        actual_input_args = cbn.input_args
        self.assertEqual(expected_input_args, actual_input_args)

    def test_unequal_args(self):
        cbn = combine.Combiner(
            [INPUT_WAV, INPUT_WAV], OUTPUT_FILE, 'mix'
        )
        cbn.input_format_list = cbn.input_filepath_list[0]
        with self.assertRaises(ValueError):
            cbn.build()


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
