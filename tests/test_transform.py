import unittest
import os

from sox import transform, file_info
from sox.core import SoxError
import soundfile as sf
import numpy as np


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SPACEY_FILE = relpath("data/annoying filename (derp).wav")
INPUT_FILE = relpath('data/input.wav')
INPUT_FILE4 = relpath('data/input4.wav')
OUTPUT_FILE = relpath('data/output.wav')
OUTPUT_FILE_ALT = relpath('data/output_alt.wav')
NOISE_PROF_FILE = relpath('data/noise.prof')


def new_transformer():
    return transform.Transformer()


def tfm_assert_array_to_file_output(input_file, output_file, tfm,
                                    dtype_in='int16', dtype_out='int16',
                                    test_file_out=True, skip_array_tests=False,
                                    **kwargs):
    input_array, rate = sf.read(input_file, dtype=dtype_in)
    actual_output, _ = sf.read(output_file, dtype=dtype_out)

    if not skip_array_tests:
        # array in, array out
        est_array = tfm.build_array(
            input_array=input_array, sample_rate_in=rate, **kwargs
        )
        assert np.allclose(actual_output, est_array.astype(dtype_out))

        # file in, array out
        est_array = tfm.build_array(
            input_filepath=input_file, sample_rate_in=rate, **kwargs
        )
        assert np.allclose(actual_output, est_array.astype(dtype_out))

    # array in, file out
    if test_file_out:
        tfm.build(input_array=input_array, output_filepath=OUTPUT_FILE_ALT,
                  sample_rate_in=rate, **kwargs)
        est_array, _ = sf.read(OUTPUT_FILE_ALT, dtype=dtype_out)
        assert np.allclose(actual_output, est_array.astype(dtype_out))
        # test alias
        tfm.build_file(input_array=input_array, output_filepath=OUTPUT_FILE_ALT,
                       sample_rate_in=rate, **kwargs)


class TestTransformDefault(unittest.TestCase):
    def setUp(self):
        self.transformer = transform.Transformer()

    def test_globals(self):
        expected = ['-D', '-V2']
        actual = self.transformer.globals
        self.assertEqual(expected, actual)

    def test_input_format(self):
        expected = {}
        actual = self.transformer.input_format
        self.assertEqual(expected, actual)

    def test_output_format(self):
        expected = {}
        actual = self.transformer.output_format
        self.assertEqual(expected, actual)

    def test_effects(self):
        expected = []
        actual = self.transformer.effects
        self.assertEqual(expected, actual)

    def test_effects_log(self):
        expected = []
        actual = self.transformer.effects_log
        self.assertEqual(expected, actual)


class TestTransformSetGlobals(unittest.TestCase):

    def setUp(self):
        self.tfm = new_transformer()

    def test_defaults(self):
        actual = self.tfm.globals
        expected = ['-D', '-V2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_dither(self):
        self.tfm.set_globals(dither=True)
        actual = self.tfm.globals
        expected = ['-V2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_dither_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_globals(dither=None)

    def test_guard(self):
        self.tfm.set_globals(guard=True)
        actual = self.tfm.globals
        expected = ['-D', '-G', '-V2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_guard_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_globals(guard='-G')

    def test_multithread(self):
        self.tfm.set_globals(multithread=True)
        actual = self.tfm.globals
        expected = ['-D', '--multi-threaded', '-V2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_multithread_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_globals(multithread='a')

    def test_replay_gain(self):
        self.tfm.set_globals(replay_gain=True)
        actual = self.tfm.globals
        expected = ['-D', '--replay-gain', 'track', '-V2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_replay_gain_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_globals(replay_gain='track')

    def test_verbosity(self):
        self.tfm.set_globals(verbosity=0)
        actual = self.tfm.globals
        expected = ['-D', '-V0']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_verbosity_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_globals(verbosity='debug')


class TestTransformSetInputFormat(unittest.TestCase):

    def setUp(self):
        self.tfm = new_transformer()

    def test_defaults(self):
        actual = self.tfm.input_format
        expected = {}
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = []
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type(self):
        self.tfm.set_input_format(file_type='wav')
        actual = self.tfm.input_format
        expected = {
            'file_type': 'wav',
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': None,
            'ignore_length': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['-t', 'wav']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(file_type='blurg')
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'file_type': 'blurg'})

    def test_rate(self):
        self.tfm.set_input_format(rate=44100)
        actual = self.tfm.input_format
        expected = {
            'file_type': None,
            'rate': 44100,
            'bits': None,
            'channels': None,
            'encoding': None,
            'ignore_length': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['-r', '44100.000000']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_rate_scinotation(self):
        self.tfm.set_input_format(rate=1.0e3)
        actual = self.tfm.input_format
        expected = {
            'file_type': None,
            'rate': 1.0e3,
            'bits': None,
            'channels': None,
            'encoding': None,
            'ignore_length': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['-r', '1000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_rate_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(rate='a')
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'rate': 'a'})

    def test_rate_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(rate=0)
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'rate': 0})

    def test_bits(self):
        self.tfm.set_input_format(bits=32)
        actual = self.tfm.input_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': 32,
            'channels': None,
            'encoding': None,
            'ignore_length': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['-b', '32']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_bits_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(bits='a')
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'bits': 'a'})

    def test_bits_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(bits=-4)
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'bits': -4})

    def test_channels(self):
        self.tfm.set_input_format(channels=2)
        actual = self.tfm.input_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': 2,
            'encoding': None,
            'ignore_length': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['-c', '2']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_channels_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(channels='a')
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'channels': 'a'})

    def test_channels_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(channels=-2)
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'channels': -2})

    def test_encoding(self):
        self.tfm.set_input_format(encoding='signed-integer')
        actual = self.tfm.input_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': 'signed-integer',
            'ignore_length': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['-e', 'signed-integer']
        self.assertEqual(expected_args, actual_args)
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_encoding_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(encoding='16-bit-signed-integer')
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'encoding': '16-bit-signed-integer'})

    def test_ignore_length(self):
        self.tfm.set_input_format(ignore_length=True)
        actual = self.tfm.input_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': None,
            'ignore_length': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._input_format_args(self.tfm.input_format)
        expected_args = ['--ignore-length']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_ignore_length_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(ignore_length=None)
        with self.assertRaises(ValueError):
            self.tfm._input_format_args({'ignore_length': None})


class TestTransformSetOutputFormat(unittest.TestCase):

    def setUp(self):
        self.tfm = new_transformer()

    def test_defaults(self):
        actual = self.tfm.output_format
        expected = {}
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = []
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type(self):
        self.tfm.set_output_format(file_type='wav')
        actual = self.tfm.output_format
        expected = {
            'file_type': 'wav',
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': None,
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-t', 'wav']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type_null_output(self):
        self.tfm.set_output_format(file_type='wav')
        actual = self.tfm.output_format
        expected = {
            'file_type': 'wav',
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': None,
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-t', 'wav']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, '-n')
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(file_type='blurg')
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'file_type': 'blurg'})

    def test_rate(self):
        self.tfm.set_output_format(rate=44100)
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': 44100,
            'bits': None,
            'channels': None,
            'encoding': None,
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-r', '44100.000000']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_rate_scinotation(self):
        self.tfm.set_output_format(rate=1.0e3)
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': 1.0e3,
            'bits': None,
            'channels': None,
            'encoding': None,
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-r', '1000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_rate_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(rate='a')
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'rate': 'a'})

    def test_rate_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(rate=0)
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'rate': 0})

    def test_bits(self):
        self.tfm.set_output_format(bits=32)
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': 32,
            'channels': None,
            'encoding': None,
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-b', '32']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_bits_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(bits='a')
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'bits': 'a'})

    def test_bits_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(bits=-4)
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'bits': -4})

    def test_channels(self):
        self.tfm.set_output_format(channels=2)
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': 2,
            'encoding': None,
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-c', '2']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_channels_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(channels='a')
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'channels': 'a'})

    def test_channels_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(channels=-2)
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'channels': -2})

    def test_encoding(self):
        self.tfm.set_output_format(encoding='signed-integer')
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': 'signed-integer',
            'comments': None,
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['-e', 'signed-integer']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_encoding_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(encoding='16-bit-signed-integer')
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'encoding': '16-bit-signed-integer'})

    def test_comments(self):
        self.tfm.set_output_format(comments='asdf')
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': None,
            'comments': 'asdf',
            'append_comments': True
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['--add-comment', 'asdf']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_comments_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(comments=2)
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'comments': 2})

    def test_append_comments(self):
        self.tfm.set_output_format(comments='asdf', append_comments=False)
        actual = self.tfm.output_format
        expected = {
            'file_type': None,
            'rate': None,
            'bits': None,
            'channels': None,
            'encoding': None,
            'comments': 'asdf',
            'append_comments': False
        }
        self.assertEqual(expected, actual)

        actual_args = self.tfm._output_format_args(self.tfm.output_format)
        expected_args = ['--comment', 'asdf']
        self.assertEqual(expected_args, actual_args)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_append_comments_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(append_comments=None)
        with self.assertRaises(ValueError):
            self.tfm._output_format_args({'append_comments': None})

class TestTransformerBuild(unittest.TestCase):
    def setUp(self):
        self.tfm = new_transformer()

    def test_valid(self):
        status = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        self.assertTrue(status)

    def test_valid_spacey(self):
        status = self.tfm.build(SPACEY_FILE, OUTPUT_FILE)
        self.assertTrue(status)

    def test_null_output(self):
        status = self.tfm.build(INPUT_FILE, '-n')
        self.assertTrue(status)

    def test_extra_arg(self):
        status = self.tfm.build(INPUT_FILE, OUTPUT_FILE, extra_args=['norm'])
        self.assertTrue(status)

    def test_invalid(self):
        with self.assertRaises(IOError):
            self.tfm.build('blah/asdf.wav', OUTPUT_FILE)

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.tfm.build({'not a string or numpy array'}, OUTPUT_FILE)

    def test_input_output_equal(self):
        with self.assertRaises(ValueError):
            self.tfm.build(INPUT_FILE, INPUT_FILE)

    def test_extra_args_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.build(INPUT_FILE, OUTPUT_FILE, extra_args=0)

    def test_failed_sox(self):
        self.tfm.effects = ['channels', '-1']
        with self.assertRaises(SoxError):
            self.tfm.build(INPUT_FILE, OUTPUT_FILE)

    def test_two_inputs(self):
        with self.assertRaises(ValueError):
            self.tfm.build(
                input_filepath=INPUT_FILE, input_array=np.array([1, 2]))

    def test_input_array_not_array(self):
        with self.assertRaises(TypeError):
            self.tfm.build(input_array=[1, 2, 3])

    def test_input_array_missing_sr(self):
        with self.assertRaises(ValueError):
            self.tfm.build(input_array=np.array([1, 2, 3]))

    def test_no_output_file(self):
        with self.assertRaises(ValueError):
            self.tfm.build(INPUT_FILE)

    def test_no_input(self):
        with self.assertRaises(ValueError):
            self.tfm.build()


class TestTransformerBuildArray(unittest.TestCase):
    def setUp(self):
        self.tfm = new_transformer()

    def test_valid(self):
        arr_out = self.tfm.build_array(INPUT_FILE)
        self.assertTrue(isinstance(arr_out, np.ndarray))

    def test_valid_spacey(self):
        arr_out = self.tfm.build_array(SPACEY_FILE)
        self.assertTrue(isinstance(arr_out, np.ndarray))

    def test_extra_arg(self):
        arr_out = self.tfm.build_array(INPUT_FILE, extra_args=['norm'])
        self.assertTrue(isinstance(arr_out, np.ndarray))

    def test_invalid(self):
        with self.assertRaises(IOError):
            self.tfm.build_array('blah/asdf.wav')

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.tfm.build_array({'not a string or numpy array'})

    def test_extra_args_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.build_array(INPUT_FILE, extra_args=0)

    def test_failed_sox(self):
        self.tfm.effects = ['channels', '-1']
        with self.assertRaises(SoxError):
            self.tfm.build_array(INPUT_FILE)

    def test_two_inputs(self):
        with self.assertRaises(ValueError):
            self.tfm.build_array(
                input_filepath=INPUT_FILE, input_array=np.array([1, 2]))

    def test_input_array_not_array(self):
        with self.assertRaises(TypeError):
            self.tfm.build_array(input_array=[1, 2, 3])

    def test_input_array_missing_sr(self):
        with self.assertRaises(ValueError):
            self.tfm.build_array(input_array=np.array([1, 2, 3]))

    def test_no_input(self):
        with self.assertRaises(ValueError):
            self.tfm.build_array()

    def test_bits8(self):
        self.tfm.set_output_format(bits=8)
        arr_out = self.tfm.build_array(INPUT_FILE)
        self.assertEqual(arr_out.dtype, np.int8)

    def test_bits16(self):
        self.tfm.set_output_format(bits=16)
        arr_out = self.tfm.build_array(INPUT_FILE)
        self.assertEqual(arr_out.dtype, np.int16)

    def test_bits32(self):
        self.tfm.set_output_format(bits=32)
        arr_out = self.tfm.build_array(INPUT_FILE)
        self.assertEqual(arr_out.dtype, np.float32)

    def test_bits64(self):
        self.tfm.set_output_format(bits=64)
        arr_out = self.tfm.build_array(INPUT_FILE)
        self.assertEqual(arr_out.dtype, np.float64)

    def test_bits_invalid(self):
        self.tfm.set_output_format(bits=17)
        with self.assertRaises(ValueError):
            self.tfm.build_array(INPUT_FILE)


class TestTransformerClearEffects(unittest.TestCase):

    def test_clear(self):
        tfm = new_transformer()
        tfm.clear_effects()

        expected = []
        actual = tfm.effects
        self.assertEqual(expected, actual)

        actual = tfm.effects_log
        self.assertEqual(expected, actual)


class TestTransformerPreview(unittest.TestCase):
    def setUp(self):
        self.tfm = new_transformer()
        self.tfm.trim(0, 0.1)

    def test_valid(self):
        expected = None
        actual = self.tfm.preview(INPUT_FILE)
        self.assertEqual(expected, actual)


class TestTransformerAllpass(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.allpass(500)

        actual_args = tfm.effects
        expected_args = ['allpass', '500.000000', '2.000000q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['allpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.allpass(0)

    def test_width_q_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.allpass(500.0, width_q='a')


class TestTransformerBandpass(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.bandpass(500.0)

        actual_args = tfm.effects
        expected_args = ['bandpass', '500.000000', '2.000000q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bandpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_constant_skirt(self):
        tfm = new_transformer()
        tfm.bandpass(500.0, constant_skirt=True)

        actual_args = tfm.effects
        expected_args = ['bandpass', '-c', '500.000000', '2.000000q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bandpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bandpass(0)

    def test_width_q_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bandpass(500.0, width_q='a')

    def test_constant_skirt_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bandpass(500.0, constant_skirt=0)


class TestTransformerBandreject(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.bandreject(500.0)

        actual_args = tfm.effects
        expected_args = ['bandreject', '500.000000', '2.000000q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bandreject']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bandreject(0)

    def test_width_q_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bandreject(500.0, width_q='a')


class TestTransformerBass(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.bass(-20.0)

        actual_args = tfm.effects
        expected_args = ['bass', '-20.000000', '100.000000', '0.500000s']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bass('x')

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bass(20.0, frequency=0)

    def test_slope_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bass(5.0, slope=0)


class TestTransformerBend(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                 end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

        actual_args = tfm.effects
        expected_args = [
            'bend',
            '-f', '25',
            '-o', '16',
            '0.350000,180.000000,0.250000',
            '0.150000,740.000000,0.530000',
            '0.000000,-540.000000,0.300000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bend']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_bends_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=1.5, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_start_times_invalid_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=1.2,
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_start_times_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_start_times_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[-1.2, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_start_times_invalid_order(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[2.5, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_end_times_invalid_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28], end_times=None,
                     cents=[180, 740, -540])

    def test_end_times_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28], end_times=[],
                     cents=[180, 740, -540])

    def test_end_times_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 'x'], cents=[180, 740, -540])

    def test_end_times_invalid_order(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[1.58, 1.28, 0.6], cents=[180, 740, -540])

    def test_start_greater_end(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.68],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_overlapping_intervals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.2],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540])

    def test_cents_invalid_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=180)

    def test_cents_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[180])

    def test_cents_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[None, None, None])

    def test_frame_rate_valid(self):
        tfm = new_transformer()
        tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                 end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540],
                 frame_rate=50)

        actual_args = tfm.effects
        expected_args = [
            'bend',
            '-f', '50',
            '-o', '16',
            '0.350000,180.000000,0.250000',
            '0.150000,740.000000,0.530000',
            '0.000000,-540.000000,0.300000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frame_rate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540],
                     frame_rate=0)

    def test_oversample_rate_valid(self):
        tfm = new_transformer()
        tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                 end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540],
                 oversample_rate=31)

        actual_args = tfm.effects
        expected_args = [
            'bend',
            '-f', '25',
            '-o', '31',
            '0.350000,180.000000,0.250000',
            '0.150000,740.000000,0.530000',
            '0.000000,-540.000000,0.300000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_oversample_rate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.bend(n_bends=3, start_times=[0.35, 0.75, 1.28],
                     end_times=[0.6, 1.28, 1.58], cents=[180, 740, -540],
                     oversample_rate=5.5)


class TestTransformerBiquad(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.biquad([0, 0, 0], [1, 0, 0])

        actual_args = tfm.effects
        expected_args = [
            'biquad', '0.000000', '0.000000', '0.000000',
            '1.000000', '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['biquad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_b_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.biquad('a', [1, 0, 0])

    def test_a_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.biquad([0, 0, 0], 1)

    def test_b_wrong_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.biquad([0, 0, 0, 0], [1, 0, 0])

    def test_a_wrong_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.biquad([0, 0, 0], [1, 0, 0, 0])

    def test_b_non_num(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.biquad([0, 0, 'a'], [1, 0, 0])

    def test_a_non_num(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.biquad([0, 0, 0], [1, None, 0])


class TestTransformerChannels(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.channels(3)

        actual_args = tfm.effects
        expected_args = ['channels', '3']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['channels']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=3)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=3)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_invalid_nchannels(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.channels(1.2)


class TestTransformerChorus(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.chorus()

        # check only the first 3 args - the rest are randomized
        actual_args = tfm.effects[:3]
        expected_args = ['chorus', '0.5', '0.9']
        self.assertEqual(expected_args, actual_args)

        self.assertGreaterEqual(float(tfm.effects[3]), 40.0)
        self.assertLessEqual(float(tfm.effects[3]), 60.0)
        self.assertGreaterEqual(float(tfm.effects[4]), 0.3)
        self.assertLessEqual(float(tfm.effects[4]), 0.4)
        self.assertGreaterEqual(float(tfm.effects[5]), 0.25)
        self.assertLessEqual(float(tfm.effects[5]), 0.4)
        self.assertGreaterEqual(float(tfm.effects[6]), 1.0)
        self.assertLessEqual(float(tfm.effects[6]), 3.0)
        self.assertIn(tfm.effects[7], ['-s', '-t'])

        actual_log = tfm.effects_log
        expected_log = ['chorus']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_explicit_args(self):
        tfm = new_transformer()
        tfm.chorus(
            n_voices=1, delays=[50.0], decays=[0.32], speeds=[0.25],
            depths=[2.0], shapes=['t']
        )

        # check only the first 3 args - the rest are randomized
        actual_args = tfm.effects
        expected_args = [
            'chorus', '0.5', '0.9', '50.000000',
            '0.320000', '0.250000', '2.000000', '-t'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['chorus']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_gain_in(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(gain_in=0)

    def test_invalid_gain_out(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(gain_out=1.1)

    def test_invalid_n_voices(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(n_voices=0)

    def test_invalid_delays(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(delays=40)

    def test_invalid_delays_wronglen(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(delays=[40, 60])

    def test_invalid_delays_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(delays=[40, 10, 60])

    def test_invalid_decays(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(decays=0.4)

    def test_invalid_decays_wronglen(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(decays=[0.2, 0.6])

    def test_invalid_decays_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(decays=['a', 'b', 'c'])

    def test_invalid_speeds(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(speeds=0.4)

    def test_invalid_speeds_wronglen(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(speeds=[0.2, 0.6])

    def test_invalid_speeds_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(speeds=[0.2, 0.2, 0])

    def test_invalid_depths(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(depths=12)

    def test_invalid_depths_wronglen(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(depths=[])

    def test_invalid_depths_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(depths=[0.0, 0.0, 0.0])

    def test_invalid_shapes(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(shapes='s')

    def test_invalid_shapes_wronglen(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(shapes=['s', 's'])

    def test_invalid_shapes_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.chorus(shapes=['s', 's', 'c'])


class TestTransformerContrast(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.contrast()

        actual_args = tfm.effects
        expected_args = ['contrast', '75.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['contrast']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_amount_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.contrast(amount='a')

    def test_invalid_amount_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.contrast(amount=-1)

    def test_invalid_amount_big(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.contrast(amount=101)


class TestTransformerCompand(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.compand()

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.300000,0.800000',
            '6.000000:-70.000000,-70.000000,-60.000000,' +
            '-20.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['compand']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_attack_time_valid(self):
        tfm = new_transformer()
        tfm.compand(attack_time=0.5)

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.500000,0.800000',
            '6.000000:-70.000000,-70.000000,-60.000000,' +
            '-20.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_attack_time_invalid_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(attack_time=-1)

    def test_attack_time_invalid_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(attack_time=None)

    def test_decay_time_valid(self):
        tfm = new_transformer()
        tfm.compand(decay_time=0.5)

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.300000,0.500000',
            '6.000000:-70.000000,-70.000000,-60.000000,' +
            '-20.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_decay_time_invalid_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(decay_time=0.0)

    def test_decay_time_invalid_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(decay_time='a')

    def test_attack_bigger_decay(self):
        tfm = new_transformer()
        tfm.compand(attack_time=1.0, decay_time=0.5)

        actual_args = tfm.effects
        expected_args = [
            'compand', '1.000000,0.500000',
            '6.000000:-70.000000,-70.000000,-60.000000,' +
            '-20.000000,0.000000,0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_soft_knee_valid(self):
        tfm = new_transformer()
        tfm.compand(soft_knee_db=-2)

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.300000,0.800000',
            '-2.000000:-70.000000,-70.000000,-60.000000,' +
            '-20.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_soft_knee_none(self):
        tfm = new_transformer()
        tfm.compand(soft_knee_db=None)

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.300000,0.800000',
            '-70.000000,-70.000000,-60.000000,-20.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_soft_knee_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(soft_knee_db='s')

    def test_tf_points_valid(self):
        tfm = new_transformer()
        tfm.compand(tf_points=[(0, -4), (-70, -60), (-60, -20), (-40, -40)])

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.300000,0.800000',
            '6.000000:-70.000000,-60.000000,-60.000000,-20.000000,' +
            '-40.000000,-40.000000,0.000000,-4.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_tf_points_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(TypeError):
            tfm.compand(tf_points=(0, 0))

    def test_tf_points_empty(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[])

    def test_tf_points_nontuples(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[[-70, -70], [-60, -20]])

    def test_tf_points_tup_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[(0, -2), (-60, -20), (-70, -70, 0)])

    def test_tf_points_tup_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[(0, -2), ('a', -20)])

    def test_tf_points_tup_nonnum2(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[('a', 'b'), ('c', 'd')])

    def test_tf_points_tup_positive(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[(0, 2), (40, -20)])

    def test_tf_points_tup_dups(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(tf_points=[(0, -2), (0, -20)])


class TestTransformerConvert(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.convert()

        actual_args = tfm.output_format
        expected_args = {}
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_samplerate_valid(self):
        tfm = new_transformer()
        tfm.convert(samplerate=8000)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '8000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['rate']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', rate=8000)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', rate=8000)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_samplerate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(samplerate=0)

    def test_channels_valid(self):
        tfm = new_transformer()
        tfm.convert(n_channels=3)

        actual = tfm.output_format
        expected = {'channels': 3}
        self.assertEqual(expected, actual)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=3)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=3)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_channels_invalid1(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(n_channels=0)

    def test_channels_invalid2(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(n_channels=1.5)

    def test_bitdepth_valid(self):
        tfm = new_transformer()
        tfm.convert(bitdepth=8)

        actual = tfm.output_format
        expected = {'bits': 8}
        self.assertEqual(expected, actual)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        ## pysndfile doesn't support int8
        # tfm.set_output_format(file_type='raw', bits=8)
        # tfm_assert_array_to_file_output(
        #     INPUT_FILE, OUTPUT_FILE, tfm, dtype_out='int8', test_file_out=False)
        #
        # tfm.set_output_format(file_type='wav', bits=8)
        # tfm_assert_array_to_file_output(
        #     INPUT_FILE, OUTPUT_FILE, tfm, dtype_out='int8',
        #     skip_array_tests=True
        # )

    def test_bitdepth_valid2(self):
        tfm = new_transformer()
        tfm.convert(bitdepth=16)

        actual = tfm.output_format
        expected = {'bits': 16}
        self.assertEqual(expected, actual)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', bits=16)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', bits=16)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_bitdepth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(bitdepth=17)


class TestTransformerDcshift(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.dcshift()

        actual_args = tfm.effects
        expected_args = ['dcshift', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['dcshift']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_shift_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.dcshift(shift='a')

    def test_invalid_shift_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.dcshift(shift=-3)

    def test_invalid_shift_big(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.dcshift(shift=5)


class TestTransformerDeemph(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.deemph()

        actual_args = tfm.effects
        expected_args = ['deemph']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['deemph']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)


class TestTransformerDelay(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.delay([1.0])

        actual_args = tfm.effects
        expected_args = ['delay', '1.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['delay']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_default_three_channel(self):
        tfm = new_transformer()
        tfm.delay([0.0, 1.0])

        actual_args = tfm.effects
        expected_args = ['delay', '0.000000', '1.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['delay']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE4, OUTPUT_FILE, tfm)

    def test_invalid_position_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.delay(1.0)

    def test_invalid_position_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.delay([-1.0, 1.0])


class TestTransformerDownsample(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.downsample()

        actual_args = tfm.effects
        expected_args = ['downsample', '2']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['downsample']
        self.assertEqual(expected_log, actual_log)

    @unittest.skip("Tests pass on local machine and fail on remote.")
    def test_default_build(self):
        tfm = new_transformer()
        tfm.downsample()

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_factor_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.downsample(factor='a')

    def test_invalid_factor_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.downsample(factor=0)


class TestTransformerEarwax(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()

        tfm.channels(2)
        tfm.earwax()

        actual_args = tfm.effects
        expected_args = ['channels', '2', 'earwax']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['channels', 'earwax']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=2)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=2)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)


class TestTransformerEcho(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.echo()

        actual_args = tfm.effects
        expected_args = ['echo', '0.800000', '0.900000', '60', '0.4']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['echo']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_in_valid(self):
        tfm = new_transformer()
        tfm.echo(gain_in=1.0)

        actual_args = tfm.effects
        expected_args = ['echo', '1.000000', '0.900000', '60', '0.4']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_in_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(gain_in=0)

    def test_gain_out_valid(self):
        tfm = new_transformer()
        tfm.echo(gain_out=1.0)

        actual_args = tfm.effects
        expected_args = ['echo', '0.800000', '1.000000', '60', '0.4']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_out_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(gain_out=None)

    def test_n_echos_valid(self):
        tfm = new_transformer()
        tfm.echo(n_echos=2, delays=[60, 60], decays=[0.4, 0.4])

        actual_args = tfm.effects
        expected_args = [
            'echo', '0.800000', '0.900000', '60', '0.4', '60', '0.4'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_echos_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(n_echos=1.2)

    def test_delays_valid(self):
        tfm = new_transformer()
        tfm.echo(n_echos=2, delays=[1, 600], decays=[0.4, 0.4])

        actual_args = tfm.effects
        expected_args = [
            'echo', '0.800000', '0.900000', '1', '0.4', '600', '0.4'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_delays_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(delays=0)

    def test_delays_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(delays=[6, 60])

    def test_delays_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(delays=[0])

    def test_decays_valid(self):
        tfm = new_transformer()
        tfm.echo(n_echos=2, delays=[60, 60], decays=[0.1, 1.0])

        actual_args = tfm.effects
        expected_args = [
            'echo', '0.800000', '0.900000', '60', '0.1', '60', '1.0'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_decays_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(decays=None)

    def test_decays_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(decays=[0.1, 0.6])

    def test_decays_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echo(decays=[2])


class TestTransformerEchos(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.echos()

        actual_args = tfm.effects
        expected_args = [
            'echos', '0.800000', '0.900000', '60.000000', '0.400000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['echos']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_in_valid(self):
        tfm = new_transformer()
        tfm.echos(gain_in=1.0)

        actual_args = tfm.effects
        expected_args = [
            'echos', '1.000000', '0.900000', '60.000000', '0.400000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_in_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(gain_in=0)

    def test_gain_out_valid(self):
        tfm = new_transformer()
        tfm.echos(gain_out=1.0)

        actual_args = tfm.effects
        expected_args = [
            'echos', '0.800000', '1.000000', '60.000000', '0.400000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_out_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(gain_out=None)

    def test_n_echos_valid(self):
        tfm = new_transformer()
        tfm.echos(n_echos=2, delays=[60, 60], decays=[0.4, 0.4])

        actual_args = tfm.effects
        expected_args = [
            'echos', '0.800000', '0.900000', '60.000000', '0.400000',
            '60.000000', '0.400000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_echos_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(n_echos=1.2)

    def test_delays_valid(self):
        tfm = new_transformer()
        tfm.echos(n_echos=2, delays=[1, 600], decays=[0.4, 0.4])

        actual_args = tfm.effects
        expected_args = [
            'echos', '0.800000', '0.900000', '1.000000', '0.400000',
            '600.000000', '0.400000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_delays_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(delays=0)

    def test_delays_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(delays=[6, 60])

    def test_delays_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(delays=[0])

    def test_decays_valid(self):
        tfm = new_transformer()
        tfm.echos(n_echos=2, delays=[60, 60], decays=[0.1, 1.0])

        actual_args = tfm.effects
        expected_args = [
            'echos', '0.800000', '0.900000', '60.000000', '0.100000',
            '60.000000', '1.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_decays_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(decays=None)

    def test_decays_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(decays=[0.1, 0.6])

    def test_decays_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.echos(decays=[2])


class TestTransformerEqualizer(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.equalizer(500.0, 2, 3)

        actual_args = tfm.effects
        expected_args = ['equalizer', '500.000000', '2.000000q', '3.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['equalizer']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.equalizer(-20, 2, 3)

    def test_width_q_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.equalizer(500.0, 0, 3)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.equalizer(500.0, 0.5, None)


class TestTransformerFade(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.fade(fade_in_len=0.5)

        actual_args = tfm.effects
        expected_args = ['fade', 'q', '0.500000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['fade']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_fade_in_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_in_len=1.2)

        actual_args = tfm.effects
        expected_args = ['fade', 'q', '1.200000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_fade_in_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fade(fade_in_len=-1)

    def test_fade_out_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_out_len=3)

        actual_args = tfm.effects
        expected_args = ['reverse', 'fade', 'q', '3.000000', 'reverse']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_fade_out_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fade(fade_out_len='q')

    def test_fade_shape_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_shape='p', fade_in_len=1.5)

        actual_args = tfm.effects
        expected_args = ['fade', 'p', '1.500000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_fade_shape_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fade(fade_shape='x')


class TestTransformerFir(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.fir([0.0195, -0.082, 0.234, 0.891, -0.145, 0.043])

        actual_args = tfm.effects
        expected_args = [
            'fir', '0.019500', '-0.082000', '0.234000', '0.891000',
            '-0.145000', '0.043000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['fir']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_coeffs_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fir(0.0195)

    def test_invalid_coeffs_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fir(['a', 'b', 'c'])


class TestTransformerFlanger(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.flanger()

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '0.000000', '71.000000',
            '0.500000', 'sine', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['flanger']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_delay_valid(self):
        tfm = new_transformer()
        tfm.flanger(delay=10)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '10.000000', '2.000000', '0.000000', '71.000000',
            '0.500000', 'sine', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_delay_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(delay=31)

    def test_flanger_depth_valid(self):
        tfm = new_transformer()
        tfm.flanger(depth=0)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '0.000000', '0.000000', '71.000000',
            '0.500000', 'sine', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_depth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(depth=None)

    def test_flanger_regen_valid(self):
        tfm = new_transformer()
        tfm.flanger(regen=-95)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '-95.000000', '71.000000',
            '0.500000', 'sine', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_regen_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(regen=100)

    def test_flanger_width_valid(self):
        tfm = new_transformer()
        tfm.flanger(width=0)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '0.000000', '0.000000',
            '0.500000', 'sine', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_width_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(width='z')

    def test_flanger_speed_valid(self):
        tfm = new_transformer()
        tfm.flanger(speed=10)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '0.000000', '71.000000',
            '10.000000', 'sine', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_speed_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(speed=0.0)

    def test_flanger_shape_valid(self):
        tfm = new_transformer()
        tfm.flanger(shape='triangle')

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '0.000000', '71.000000',
            '0.500000', 'triangle', '25.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_shape_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(shape='square')

    def test_flanger_phase_valid(self):
        tfm = new_transformer()
        tfm.flanger(phase=95)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '0.000000', '71.000000',
            '0.500000', 'sine', '95.000000', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_phase_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(phase=-1)

    def test_flanger_interp_valid(self):
        tfm = new_transformer()
        tfm.flanger(interp='quadratic')

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0.000000', '2.000000', '0.000000', '71.000000',
            '0.500000', 'sine', '25.000000', 'quadratic'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_flanger_interp_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(interp='cubic')


class TestTransformerGain(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.gain()

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['gain']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.gain(gain_db=6)

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '6.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(gain_db=None)

    def test_normalize_valid(self):
        tfm = new_transformer()
        tfm.gain(normalize=False)

        actual_args = tfm.effects
        expected_args = ['gain', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_normalize_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(normalize='6')

    def test_limiter_valid(self):
        tfm = new_transformer()
        tfm.gain(limiter=True)

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '-l', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_limiter_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(limiter='0')

    def test_balance_valid(self):
        tfm = new_transformer()
        tfm.gain(balance='B')

        actual_args = tfm.effects
        expected_args = ['gain', '-B', '-n', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_balance_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(balance='h')


class TestTransformerHighpass(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.highpass(1000.0)

        actual_args = tfm.effects
        expected_args = ['highpass', '-2', '1000.000000', '0.707000q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['highpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_one_pole(self):
        tfm = new_transformer()
        tfm.highpass(1000.0, n_poles=1)

        actual_args = tfm.effects
        expected_args = ['highpass', '-1', '1000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['highpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.highpass(-20)

    def test_width_q_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.highpass(1000.0, width_q=0.0)

    def test_n_poles_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.highpass(1000.0, n_poles=3)


class TestTransformerHilbert(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.hilbert()

        actual_args = tfm.effects
        expected_args = ['hilbert']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['hilbert']
        self.assertEqual(expected_log, actual_log)

    @unittest.skip("Tests pass on local machine and fail on remote.")
    def test_default_build(self):
        tfm = new_transformer()
        tfm.hilbert()
        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_num_taps_valid(self):
        tfm = new_transformer()
        tfm.hilbert(num_taps=17)

        actual_args = tfm.effects
        expected_args = ['hilbert', '-n', '17']
        self.assertEqual(expected_args, actual_args)

    @unittest.skip("Tests pass on local machine and fail on remote.")
    def test_num_taps_valid_build(self):
        tfm = new_transformer()
        tfm.hilbert(num_taps=17)
        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_num_taps_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.hilbert(num_taps=2.2)

    def test_num_taps_invalid_even(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.hilbert(num_taps=2)


class TestTransformerLowpass(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.lowpass(1000.0)

        actual_args = tfm.effects
        expected_args = ['lowpass', '-2', '1000.000000', '0.707000q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['lowpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_one_pole(self):
        tfm = new_transformer()
        tfm.lowpass(1000.0, n_poles=1)

        actual_args = tfm.effects
        expected_args = ['lowpass', '-1', '1000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['lowpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.lowpass(-20)

    def test_width_q_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.lowpass(1000.0, width_q=0.0)

    def test_n_poles_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.lowpass(1000.0, n_poles=3)


class TestTransformerLoudness(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.loudness()

        actual_args = tfm.effects
        expected_args = ['loudness', '-10.000000', '65.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['loudness']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.loudness(gain_db=0)

        actual_args = tfm.effects
        expected_args = ['loudness', '0.000000', '65.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.loudness(gain_db='0dB')

    def test_reference_level_valid(self):
        tfm = new_transformer()
        tfm.loudness(reference_level=50)

        actual_args = tfm.effects
        expected_args = ['loudness', '-10.000000', '50.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_reference_level_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.loudness(reference_level=None)

    def test_reference_level_oorange(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.loudness(reference_level=15.0)


class TestTransformerMcompand(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.mcompand()

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 6.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000',
            '1600.000000',
            '0.000625,0.012500 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-15.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['mcompand']
        self.assertEqual(expected_log, actual_log)
        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_bands_valid(self):
        tfm = new_transformer()
        tfm.mcompand(
            n_bands=1, crossover_frequencies=[],
            attack_time=[0.005], decay_time=[0.1],
            soft_knee_db=[6.0],
            tf_points=[[(-47, -40), (-34, -34), (-17, -33), (0, 0)]],
            gain=[None]
        )

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 6.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_bands_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(n_bands=0)

    def test_crossover_frequencies_valid(self):
        tfm = new_transformer()
        tfm.mcompand(crossover_frequencies=[100])

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 6.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000',
            '100.000000',
            '0.000625,0.012500 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-15.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_crossover_frequencies_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(crossover_frequencies=[100, 200])

    def test_crossover_frequencies_invalid_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(crossover_frequencies=[-200])

    def test_attack_time_valid(self):
        tfm = new_transformer()
        tfm.mcompand(attack_time=[0.5, 0.0625])

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.500000,0.100000 6.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000',
            '1600.000000',
            '0.062500,0.012500 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-15.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_attack_time_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(attack_time=0.5)

    def test_attack_time_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(attack_time=[0.5])

    def test_attack_time_invalid_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(attack_time=[-1, 0.5])

    def test_attack_time_invalid_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(attack_time=[None, 'a'])

    def test_decay_time_valid(self):
        tfm = new_transformer()
        tfm.mcompand(decay_time=[0.001, 0.5])

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.001000 6.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000',
            '1600.000000',
            '0.000625,0.500000 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-15.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_decay_time_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(decay_time=0.5)

    def test_decay_time_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(decay_time=[0.5])

    def test_decay_time_invalid_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(decay_time=[0.5, 0.0])

    def test_decay_time_invalid_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(decay_time=['a', 'b'])

    def test_soft_knee_valid(self):
        tfm = new_transformer()
        tfm.mcompand(soft_knee_db=[-2, -5])

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 -2.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000',
            '1600.000000',
            '0.000625,0.012500 -5.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-15.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_soft_knee_none(self):
        tfm = new_transformer()
        tfm.mcompand(soft_knee_db=[None, None])

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-17.000000,-33.000000,0.000000,0.000000',
            '1600.000000',
            '0.000625,0.012500 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-15.000000,-33.000000,0.000000,0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_soft_knee_db_invalid_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(soft_knee_db=0.5)

    def test_soft_knee_db_invalid_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(soft_knee_db=[6])

    def test_soft_knee_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(soft_knee_db=['s', -5])

    def test_tf_points_valid(self):
        tfm = new_transformer()
        tfm.mcompand(
            tf_points=[
                [(0, -4), (-70, -60), (-60, -20), (-40, -40)],
                [(0, -4), (-70, -60)]
            ]
        )

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 6.000000:-70.000000,-60.000000,-60.000000,'
            '-20.000000,-40.000000,-40.000000,0.000000,-4.000000',
            '1600.000000',
            '0.000625,0.012500 -70.000000,-60.000000,0.000000,-4.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_tf_points_wrong_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[[(0, 0)]])

    def test_tf_points_elt_nonlist(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[(0, 0), [(-30, -40)]])

    def test_tf_points_elt_empty(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[[], []])

    def test_tf_points_elt_nontuples(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(
                tf_points=[[[-70, -70], [-60, -20]], [(0, -4), (-70, -60)]]
            )

    def test_tf_points_elt_tup_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(
                tf_points=[[(0, -2), (-60, -20), (-70, -70, 0)], [(0, 0)]]
            )

    def test_tf_points_elt_tup_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[[(0, -2), ('a', -20)], [(0, 0)]])

    def test_tf_points_tup_nonnum2(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[[('a', 'b'), ('c', 'd')], [(0, 0)]])

    def test_tf_points_tup_positive(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[[(0, 2), (40, -20)], [(0, 0)]])

    def test_tf_points_tup_dups(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(tf_points=[[(0, -2), (0, -20)], [(0, 0)]])

    def test_gain_valid(self):
        tfm = new_transformer()
        tfm.mcompand(gain=[3.0, -1.0])

        actual_args = tfm.effects
        expected_args = [
            'mcompand',
            '0.005000,0.100000 6.000000:-47.000000,-40.000000,-34.000000,'
            '-34.000000,-17.000000,-33.000000,0.000000,0.000000 3.000000',
            '1600.000000',
            '0.000625,0.012500 -47.000000,-40.000000,-34.000000,-34.000000,'
            '-15.000000,-33.000000,0.000000,0.000000 -1.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['mcompand']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_len_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(gain=[-2])

    def test_gain_values_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.mcompand(gain=['a', None])


class TestTransformerNoiseprof(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        save_path = os.path.join(os.getcwd(), 'noise.prof')
        actual_res = tfm.noiseprof(INPUT_FILE, save_path)
        expected_res = None
        self.assertEqual(expected_res, actual_res)
        self.assertTrue(os.path.exists(save_path))

        # this should not update the effects
        actual_args = tfm.effects
        expected_args = []
        self.assertEqual(expected_args, actual_args)

        # this should not update the effects log
        actual_log = tfm.effects_log
        expected_log = []
        self.assertEqual(expected_log, actual_log)

    def test_noise_prof_invalid_dir(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.noiseprof(INPUT_FILE, os.getcwd())

    def test_noise_prof_invalid_write(self):
        tfm = new_transformer()
        with self.assertRaises(IOError):
            tfm.noiseprof(INPUT_FILE, '/usr/noise.prof')

    def test_noise_prof_invalid_cwd(self):
        tfm = new_transformer()
        _cwd = os.getcwd()
        os.chdir('/')
        with self.assertRaises(IOError):
            tfm.noiseprof(INPUT_FILE, 'noise.prof')
        os.chdir(_cwd)


class TestTransformerNoisered(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.noisered(NOISE_PROF_FILE)

        actual_args = tfm.effects
        expected_args = [
            'noisered',
            NOISE_PROF_FILE,
            '0.500000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['noisered']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_noise_prof_valid(self):
        tfm = new_transformer()
        tfm.noisered(NOISE_PROF_FILE)

        actual_args = tfm.effects
        expected_args = [
            'noisered',
            NOISE_PROF_FILE,
            '0.500000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_noise_prof_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(IOError):
            tfm.noisered('/usr/noise.prof')

    def test_amount_valid(self):
        tfm = new_transformer()
        tfm.noisered(
            NOISE_PROF_FILE, 0.7
        )

        actual_args = tfm.effects
        expected_args = [
            'noisered', NOISE_PROF_FILE, '0.700000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_amount_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.noisered(NOISE_PROF_FILE, 1.3)


class TestTransformerNorm(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.norm()

        actual_args = tfm.effects
        expected_args = ['norm', '-3.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['norm']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_db_level_valid(self):
        tfm = new_transformer()
        tfm.norm(db_level=0)

        actual_args = tfm.effects
        expected_args = ['norm', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_db_level_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.norm(db_level='-2dB')


class TestTransformerOops(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.oops()

        actual_args = tfm.effects
        expected_args = ['oops']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['oops']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE4, OUTPUT_FILE, tfm)


class TestTransformerOverdrive(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.overdrive()

        actual_args = tfm.effects
        expected_args = ['overdrive', '20.000000', '20.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['overdrive']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.overdrive(gain_db=2)

        actual_args = tfm.effects
        expected_args = ['overdrive', '2.000000', '20.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.overdrive(gain_db='-2dB')

    def test_colour_valid(self):
        tfm = new_transformer()
        tfm.overdrive(colour=0)

        actual_args = tfm.effects
        expected_args = ['overdrive', '20.000000', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_colour_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.overdrive(colour=None)


class TestTransformerPad(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.pad()

        actual_args = tfm.effects
        expected_args = ['pad', '0.000000', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['pad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_start_duration_valid(self):
        tfm = new_transformer()
        tfm.pad(start_duration=3)

        actual_args = tfm.effects
        expected_args = ['pad', '3.000000', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_start_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pad(start_duration=-1)

    def test_end_duration_valid(self):
        tfm = new_transformer()
        tfm.pad(end_duration=0.2)

        actual_args = tfm.effects
        expected_args = ['pad', '0.000000', '0.200000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_end_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pad(end_duration='foo')


class TestTransformerPhaser(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.phaser()

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.800000', '0.740000', '3.000000', '0.400000',
            '0.500000', '-s'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['phaser']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_in_valid(self):
        tfm = new_transformer()
        tfm.phaser(gain_in=0.5)

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.500000', '0.740000', '3.000000', '0.400000',
            '0.500000', '-s']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_in_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.phaser(gain_in=0)

    def test_gain_out_valid(self):
        tfm = new_transformer()
        tfm.phaser(gain_out=1.0)

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.800000', '1.000000', '3.000000', '0.400000',
            '0.500000', '-s']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_out_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.phaser(gain_out=1.1)

    def test_delay_valid(self):
        tfm = new_transformer()
        tfm.phaser(delay=5)

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.800000', '0.740000', '5.000000', '0.400000',
            '0.500000', '-s'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_delay_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.phaser(delay=None)

    def test_decay_valid(self):
        tfm = new_transformer()
        tfm.phaser(decay=0.1)

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.800000', '0.740000', '3.000000', '0.100000',
            '0.500000', '-s'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_decay_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.phaser(decay=0.0)

    def test_speed_valid(self):
        tfm = new_transformer()
        tfm.phaser(speed=2)

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.800000', '0.740000', '3.000000', '0.400000',
            '2.000000', '-s'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_speed_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.phaser(speed=-1)

    def test_modulation_shape_valid(self):
        tfm = new_transformer()
        tfm.phaser(modulation_shape='triangular')

        actual_args = tfm.effects
        expected_args = [
            'phaser', '0.800000', '0.740000', '3.000000', '0.400000',
            '0.500000', '-t'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_modulation_shape_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.phaser(modulation_shape='square')


class TestTransformerPitch(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.pitch(0.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '0.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['pitch']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_semitones_valid(self):
        tfm = new_transformer()
        tfm.pitch(-3.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '-300.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_semitones_warning(self):
        tfm = new_transformer()
        tfm.pitch(13.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '1300.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_n_semitones_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pitch('a')

    def test_quick_valid(self):
        tfm = new_transformer()
        tfm.pitch(1.0, quick=True)

        actual_args = tfm.effects
        expected_args = ['pitch', '-q', '100.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_quick_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pitch(1.0, quick=1)


class TestTransformerRate(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.rate(48000)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '48000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['rate']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', rate=48000)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', rate=48000)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_samplerate_valid(self):
        tfm = new_transformer()
        tfm.rate(1000.5)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '1000.500000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', rate=1000.5)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', rate=1000.5)
        tfm_assert_array_to_file_output(
            INPUT_FILE, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_samplerate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.rate(0)

    def test_quality_valid(self):
        tfm = new_transformer()
        tfm.rate(44100.0, quality='q')

        actual_args = tfm.effects
        expected_args = ['rate', '-q', '44100.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_quality_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.rate(44100.0, quality='foo')


class TestTransformerRemix(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.remix()

        actual_args = tfm.effects
        expected_args = ['remix', '-']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=1)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=1)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_remix_dictionary_valid(self):
        tfm = new_transformer()
        tfm.remix(remix_dictionary={1: [1, 2], 3: [1]})

        actual_args = tfm.effects
        expected_args = ['remix', '1,2', '0', '1']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=3)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=3)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_num_channels_valid(self):
        tfm = new_transformer()
        tfm.remix(remix_dictionary={1: [1, 2], 3: [1]}, num_output_channels=4)

        actual_args = tfm.effects
        expected_args = ['remix', '1,2', '0', '1', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=4)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=4)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_remix_dictionary_none(self):
        tfm = new_transformer()
        tfm.remix(remix_dictionary=None, num_output_channels=7)

        actual_args = tfm.effects
        expected_args = ['remix', '-']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

        tfm.set_output_format(file_type='raw', channels=1)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, test_file_out=False)

        tfm.set_output_format(file_type='wav', channels=1)
        tfm_assert_array_to_file_output(
            INPUT_FILE4, OUTPUT_FILE, tfm, skip_array_tests=True)

    def test_remix_dict_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.remix(remix_dictionary=3)

    def test_remix_dict_invalid2(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.remix(remix_dictionary={1: [1], 2.5: [2, 4]})

    def test_remix_dict_invalid3(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.remix(remix_dictionary={1: 1, 2: [2]})

    def test_remix_dict_invalid4(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.remix(remix_dictionary={1: [1], 3: [0, 1]})

    def test_num_output_channels_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.remix(num_output_channels=0)


class TestTransformerRepeat(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.repeat()

        actual_args = tfm.effects
        expected_args = ['repeat', '1']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['repeat']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_count_valid(self):
        tfm = new_transformer()
        tfm.repeat(count=2)

        actual_args = tfm.effects
        expected_args = ['repeat', '2']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_count_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.repeat(count=0)

    def test_count_invalid_fmt(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.repeat(count=None)


class TestTransformerReverb(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.reverb()

        actual_args = tfm.effects
        expected_args = [
            'reverb', '50.000000', '50.000000', '100.000000', '100.000000',
            '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['reverb']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_reverberance_valid(self):
        tfm = new_transformer()
        tfm.reverb(reverberance=90)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '90.000000', '50.000000', '100.000000', '100.000000',
            '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_reverberance_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(reverberance=150)

    def test_high_freq_damping_valid(self):
        tfm = new_transformer()
        tfm.reverb(high_freq_damping=10)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '50.000000', '10.000000', '100.000000', '100.000000',
            '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_high_freq_damping_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(high_freq_damping='a')

    def test_room_scale_valid(self):
        tfm = new_transformer()
        tfm.reverb(room_scale=10)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '50.000000', '50.000000', '10.000000', '100.000000',
            '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_room_scale_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(room_scale=-100)

    def test_stereo_depth_valid(self):
        tfm = new_transformer()
        tfm.reverb(stereo_depth=50)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '50.000000', '50.000000', '100.000000', '50.000000',
            '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_stereo_depth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(stereo_depth=101)

    def test_pre_delay_valid(self):
        tfm = new_transformer()
        tfm.reverb(pre_delay=10)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '50.000000', '50.000000', '100.000000', '100.000000',
            '10.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_pre_delay_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(pre_delay=-1)

    def test_wet_gain_valid(self):
        tfm = new_transformer()
        tfm.reverb(wet_gain=5)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '50.000000', '50.000000', '100.000000', '100.000000',
            '0.000000', '5.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_wet_gain_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(wet_gain='z')

    def test_wet_only_valid(self):
        tfm = new_transformer()
        tfm.reverb(wet_only=True)

        actual_args = tfm.effects
        expected_args = [
            'reverb', '-w', '50.000000', '50.000000', '100.000000',
            '100.000000', '0.000000', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_wet_only_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(wet_only=6)


class TestTransformerReverse(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.reverse()

        actual_args = tfm.effects
        expected_args = ['reverse']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['reverse']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)


class TestTransformerSilence(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.silence()

        actual_args = tfm.effects
        expected_args = [
            'silence', '1', '0.100000', '0.100000%',
            '-1', '0.100000', '0.100000%'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['silence']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_location_beginning(self):
        tfm = new_transformer()
        tfm.silence(location=1)

        actual_args = tfm.effects
        expected_args = ['silence', '1', '0.100000', '0.100000%']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_location_end(self):
        tfm = new_transformer()
        tfm.silence(location=-1)

        actual_args = tfm.effects
        expected_args = [
            'reverse', 'silence', '1', '0.100000', '0.100000%', 'reverse'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_location_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(location=2)

    def test_silence_threshold_valid(self):
        tfm = new_transformer()
        tfm.silence(silence_threshold=10.5)

        actual_args = tfm.effects
        expected_args = [
            'silence', '1', '0.100000', '10.500000%',
            '-1', '0.100000', '10.500000%'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_silence_threshold_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(silence_threshold=101)

    def test_silence_threshold_invalid2(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(silence_threshold=-0.1)

    def test_min_silence_duration_valid(self):
        tfm = new_transformer()
        tfm.silence(min_silence_duration=2)

        actual_args = tfm.effects
        expected_args = [
            'silence', '1', '2.000000', '0.100000%',
            '-1', '2.000000', '0.100000%']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_min_silence_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(min_silence_duration='a')

    def test_buffer_around_silence_valid(self):
        tfm = new_transformer()
        tfm.silence(buffer_around_silence=True)

        actual_args = tfm.effects
        expected_args = [
            'silence', '-l', '1', '0.100000', '0.100000%',
            '-1', '0.100000', '0.100000%'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_buffer_around_silence_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(buffer_around_silence=0)


class TestTransformerSinc(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.sinc()

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '3000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_filter_type_valid_low(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='low')

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '-3000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_filter_type_valid_pass(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='pass', cutoff_freq=[3000, 4000])

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '3000.000000-4000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_filter_type_valid_reject(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='reject', cutoff_freq=[3000, 4000])

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '4000.000000-3000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_filter_type_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(filter_type='stop')

    def test_cutoff_freq_valid_float(self):
        tfm = new_transformer()
        tfm.sinc(cutoff_freq=300.4)

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '300.400000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_cutoff_freq_valid_list(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='pass', cutoff_freq=[300.4, 1000])

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '300.400000-1000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_cutoff_freq_valid_unordered(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='pass', cutoff_freq=[1000, 300.4])

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '120.000000', '300.400000-1000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_cutoff_freq_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(cutoff_freq=None)

    def test_cutoff_freq_invalid_high(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(cutoff_freq=[1000, 300.4])

    def test_cutoff_freq_invalid_reject(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(filter_type='reject', cutoff_freq=1000.0)

    def test_cutoff_freq_invalid_number(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(cutoff_freq=-1000)

    def test_cutoff_freq_invalid_list_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(filter_type='pass', cutoff_freq=[1000, 2000, 3000])

    def test_cutoff_freq_invalid_list(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(filter_type='pass', cutoff_freq=['a', 'b'])

    def test_stop_band_attenuation_valid(self):
        tfm = new_transformer()
        tfm.sinc(stop_band_attenuation=60)

        actual_args = tfm.effects
        expected_args = ['sinc', '-a', '60.000000', '3000.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_stop_band_attenuation_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(stop_band_attenuation=-3)

    def test_transition_bw_valid_high(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='high', transition_bw=100)

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-t', '100.000000', '3000.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_transition_bw_valid_low(self):
        tfm = new_transformer()
        tfm.sinc(filter_type='low', transition_bw=100)

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-3000.000000', '-t', '100.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_transition_bw_valid_pass_float(self):
        tfm = new_transformer()
        tfm.sinc(
            filter_type='pass', cutoff_freq=[3000, 4000], transition_bw=100
        )

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-t', '100.000000',
            '3000.000000-4000.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_transition_bw_valid_pass_list(self):
        tfm = new_transformer()
        tfm.sinc(
            filter_type='pass', cutoff_freq=[3000, 4000],
            transition_bw=[100, 200]
        )

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-t', '100.000000',
            '3000.000000-4000.000000', '-t', '200.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_transition_bw_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(transition_bw='a')

    def test_transition_bw_invalid_low(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(filter_type='low', transition_bw=[100, 200])

    def test_transition_bw_invalid_float(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(transition_bw=-100)

    def test_transition_bw_invalid_list_elt(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(
                filter_type='pass', cutoff_freq=[3000, 4000],
                transition_bw=[100, 'z']
            )

    def test_transition_bw_linvalid_list_len(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(
                filter_type='reject', cutoff_freq=[3000, 4000],
                transition_bw=[100, 200, 300]
            )

    def test_phase_response_valid_low(self):
        tfm = new_transformer()
        tfm.sinc(phase_response=0)

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-p', '0.000000', '3000.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_phase_response_valid_mid(self):
        tfm = new_transformer()
        tfm.sinc(phase_response=25)

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-p', '25.000000', '3000.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_phase_response_valid_high(self):
        tfm = new_transformer()
        tfm.sinc(phase_response=100)

        actual_args = tfm.effects
        expected_args = [
            'sinc', '-a', '120.000000', '-p', '100.000000', '3000.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['sinc']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_phase_response_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(phase_response='z')

    def test_phase_response_invalid_large(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(phase_response=101)

    def test_phase_response_invalid_small(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.sinc(phase_response=-1)


class TestTransformerSpeed(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.speed(1.5)

        actual_args = tfm.effects
        expected_args = ['speed', '1.500000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['speed']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_valid(self):
        tfm = new_transformer()
        tfm.speed(0.7)

        actual_args = tfm.effects
        expected_args = ['speed', '0.700000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_valid_extreme(self):
        tfm = new_transformer()
        tfm.speed(2.5)

        actual_args = tfm.effects
        expected_args = ['speed', '2.500000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.speed(-1)


class TestTransformerStat(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        actual = tfm.stat(INPUT_FILE)
        expected = {
            'Length (seconds)': '10.000000',
            'Minimum amplitude': '-0.264252',
            'RMS amplitude': '0.053924',
            'Volume adjustment': '3.521',
            'Mean norm': '0.042598',
            'Minimum delta': '0.000000',
            'Samples read': '441000',
            'Maximum amplitude': '0.284027',
            'Mean amplitude': '0.000014',
            'RMS delta': '0.006387',
            'Midline amplitude': '0.009888',
            'Maximum delta': '0.112427',
            'Mean delta': '0.004495',
            'Scaled by': '2147483647.0',
            'Rough frequency': '831'
        }

        self.assertEqual(expected, actual)

    def test_scale(self):
        tfm = new_transformer()
        actual = tfm.stat(INPUT_FILE, scale=2147483647.0 / 2.0)
        expected = {
            'Length (seconds)': '10.000000',
            'Minimum amplitude': '-0.528503',
            'RMS amplitude': '0.107848',
            'Volume adjustment': '3.521',
            'Mean norm': '0.085196',
            'Minimum delta': '0.000000',
            'Samples read': '441000',
            'Maximum amplitude': '0.568054',
            'Mean amplitude': '0.000028',
            'RMS delta': '0.012773',
            'Midline amplitude': '0.019775',
            'Maximum delta': '0.224854',
            'Mean delta': '0.008991',
            'Scaled by': '1073741823.5',
            'Rough frequency': '831'
        }

        self.assertEqual(expected, actual)

    def test_scale_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.stat(INPUT_FILE, 'asdf')

    def test_rms(self):
        tfm = new_transformer()
        actual = tfm.stat(INPUT_FILE, rms=True)
        expected = {
            'Length (seconds)': '10.000000',
            'Minimum amplitude': '-4.900466',
            'RMS amplitude': '1.000000',
            'Volume adjustment': '3.521',
            'Mean norm': '0.789962',
            'Minimum delta': '0.000000',
            'Samples read': '441000',
            'Maximum amplitude': '5.267194',
            'Mean amplitude': '0.000256',
            'RMS delta': '0.118437',
            'Midline amplitude': '0.183364',
            'Maximum delta': '2.084919',
            'Mean delta': '0.083366',
            'Rough frequency': '831',
            'Scaled by rms': '0.053924'
        }

        self.assertEqual(expected, actual)

    def test_multichannel(self):
        tfm = new_transformer()
        actual = tfm.stat(INPUT_FILE4)
        expected = {
            'Length (seconds)': '10.000000',
            'Minimum amplitude': '-0.264252',
            'RMS amplitude': '0.053924',
            'Volume adjustment': '3.521',
            'Mean norm': '0.042598',
            'Minimum delta': '0.000000',
            'Samples read': '441000',
            'Maximum amplitude': '0.284027',
            'Mean amplitude': '0.000014',
            'RMS delta': '0.006387',
            'Midline amplitude': '0.009888',
            'Maximum delta': '0.112427',
            'Mean delta': '0.004495',
            'Scaled by': '2147483647.0',
            'Rough frequency': '831'
        }
        self.assertEqual(expected, actual)


class TestTransformerPowerSpectrum(unittest.TestCase):

    def test_valid(self):
        tfm = new_transformer()
        actual = tfm.power_spectrum(INPUT_FILE)

        expected_len = 221184
        expected_first = [0.0, 0.016436]
        expected_last = [22039.234375, 0.000697]

        self.assertEqual(expected_len, len(actual))
        self.assertEqual(expected_first, actual[0])
        self.assertEqual(expected_last, actual[-1])

    def test_multichannel(self):
        tfm = new_transformer()
        actual = tfm.power_spectrum(INPUT_FILE4)
        expected_len = 221184
        expected_first = [0.0, 0.016436]
        expected_last = [22039.234375, 0.000697]

        self.assertEqual(expected_len, len(actual))
        self.assertEqual(expected_first, actual[0])
        self.assertEqual(expected_last, actual[-1])


class TestTransformerStats(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        actual = tfm.stats(INPUT_FILE)
        expected = {
            'Max level': '0.284027',
            'Crest factor': '5.27',
            'RMS Tr dB': '-78.27',
            'Num samples': '441k',
            'DC offset': '0.000014',
            'Length s': '10.000',
            'Bit-depth': '15/16',
            'Flat factor': '0.00',
            'Pk lev dB': '-10.93',
            'Min level': '-0.264252',
            'RMS Pk dB': '-21.75',
            'Window s': '0.050',
            'Scale max': '1.000000',
            'RMS lev dB': '-25.36',
            'Pk count': '2'
        }

        self.assertEqual(expected, actual)

    def test_multichannel(self):
        tfm = new_transformer()
        actual = tfm.stats(INPUT_FILE4)
        expected = {
            'Max level': '0.284027',
            'Crest factor': '5.27',
            'RMS Tr dB': '-78.27',
            'Num samples': '441k',
            'DC offset': '0.000014',
            'Length s': '10.000',
            'Bit-depth': '15/16',
            'Flat factor': '0.00',
            'Pk lev dB': '-10.93',
            'Min level': '-0.264252',
            'RMS Pk dB': '-21.75',
            'Window s': '0.050',
            'Scale max': '1.000000',
            'RMS lev dB': '-25.36',
            'Pk count': '2'
        }
        self.assertEqual(expected, actual)


class TestTransformerSwap(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.swap()

        actual_args = tfm.effects
        expected_args = ['swap']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['swap']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE4, OUTPUT_FILE, tfm)


class TestTransformerStretch(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.stretch(1.1)

        actual_args = tfm.effects
        expected_args = ['stretch', '1.100000', '20.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['stretch']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_valid(self):
        tfm = new_transformer()
        tfm.stretch(0.7)

        actual_args = tfm.effects
        expected_args = ['stretch', '0.700000', '20.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_extreme(self):
        tfm = new_transformer()
        tfm.stretch(0.2)

        actual_args = tfm.effects
        expected_args = ['stretch', '0.200000', '20.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.stretch(-1)

    def test_window_valid(self):
        tfm = new_transformer()
        tfm.stretch(0.99, window=10)

        actual_args = tfm.effects
        expected_args = ['stretch', '0.990000', '10.000000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_window_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.stretch(0.99, window=0)


class TestTransformerTempo(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.tempo(1.1)

        actual_args = tfm.effects
        expected_args = ['tempo', '1.100000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['tempo']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_valid(self):
        tfm = new_transformer()
        tfm.tempo(0.9)

        actual_args = tfm.effects
        expected_args = ['tempo', '0.900000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_warning(self):
        tfm = new_transformer()
        tfm.tempo(0.1)

        actual_args = tfm.effects
        expected_args = ['tempo', '0.100000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_factor_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tempo(-1.1)

    def test_audio_type_valid(self):
        tfm = new_transformer()
        tfm.tempo(1.5, audio_type='m')

        actual_args = tfm.effects
        expected_args = ['tempo', '-m', '1.500000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_audio_type_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tempo(1.5, audio_type=1)

    def test_quick_valid(self):
        tfm = new_transformer()
        tfm.tempo(1.5, quick=True)

        actual_args = tfm.effects
        expected_args = ['tempo', '-q', '1.500000']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_quick_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tempo(1.5, quick=1)


class TestTransformerTreble(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.treble(20.0)

        actual_args = tfm.effects
        expected_args = ['treble', '20.000000', '3000.000000', '0.500000s']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['treble']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.treble('x')

    def test_frequency_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.treble(-20.0, frequency=0)

    def test_slope_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.treble(-20, slope=0)


class TestTransformerTremolo(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.tremolo()

        actual_args = tfm.effects
        expected_args = ['tremolo', '6.000000', '40.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['tremolo']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_speed_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tremolo(speed=0)

    def test_depth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tremolo(depth=101)


class TestTransformerTrim(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.trim(5.0)

        actual_args = tfm.effects
        expected_args = ['trim', '5.000000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['trim']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_end_time(self):
        tfm = new_transformer()
        tfm.trim(0, 8.5)

        actual_args = tfm.effects
        expected_args = ['trim', '0.000000', '8.500000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['trim']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_start_time(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.trim(-1, 8)

    def test_invalid_end_time(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.trim(1, -8)

    def test_invalid_time_pair(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.trim(8, 2)


class TestTransformerUpsample(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.upsample()

        actual_args = tfm.effects
        expected_args = ['upsample', '2']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['upsample']
        self.assertEqual(expected_log, actual_log)

    @unittest.skip("Tests pass on local machine and fail on remote.")
    def test_default_build(self):
        tfm = new_transformer()
        tfm.upsample()
        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_factor_nonnum(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.upsample(factor='a')

    def test_invalid_factor_decimal(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.upsample(factor=1.5)

    def test_invalid_factor_neg(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.upsample(factor=0)


class TestTransformerVad(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.vad()

        actual_args = tfm.effects
        expected_args = [
            'norm', 'vad',
            '-t', '7.000000',
            '-T', '0.250000',
            '-s', '1.000000',
            '-g', '0.250000',
            '-p', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_end_location(self):
        tfm = new_transformer()
        tfm.vad(location=-1)

        actual_args = tfm.effects
        expected_args = [
            'norm',
            'reverse',
            'vad',
            '-t', '7.000000',
            '-T', '0.250000',
            '-s', '1.000000',
            '-g', '0.250000',
            '-p', '0.000000',
            'reverse'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_no_normalize(self):
        tfm = new_transformer()
        tfm.vad(normalize=False)

        actual_args = tfm.effects
        expected_args = [
            'vad',
            '-t', '7.000000',
            '-T', '0.250000',
            '-s', '1.000000',
            '-g', '0.250000',
            '-p', '0.000000'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_location(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(location=0)

    def test_invalid_normalize(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(normalize=0)

    def test_invalid_activity_threshold(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(activity_threshold='a')

    def test_invalid_min_activity_duration(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(min_activity_duration=-2)

    def test_invalid_initial_search_buffer(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(initial_search_buffer=-1)

    def test_invalid_max_gap(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(max_gap=-1)

    def test_invalid_initial_pad(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vad(initial_pad=-1)


class TestTransformerVol(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.vol(0.8)

        actual_args = tfm.effects
        expected_args = ['vol', '0.800000', 'amplitude']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_limiter_gain(self):
        tfm = new_transformer()
        tfm.vol(1.8, limiter_gain=0.02)

        actual_args = tfm.effects
        expected_args = ['vol', '1.800000', 'amplitude', '0.020000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_limiter_gain_vol_down(self):
        tfm = new_transformer()
        tfm.vol(0.8, limiter_gain=0.02)

        actual_args = tfm.effects
        expected_args = ['vol', '0.800000', 'amplitude']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_limiter_gain_vol_down_db(self):
        tfm = new_transformer()
        tfm.vol(-2.0, gain_type='db', limiter_gain=0.05)

        actual_args = tfm.effects
        expected_args = ['vol', '-2.000000', 'dB']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_limiter_gain_vol_up_db(self):
        tfm = new_transformer()
        tfm.vol(2.0, gain_type='db', limiter_gain=0.05)

        actual_args = tfm.effects
        expected_args = ['vol', '2.000000', 'dB', '0.050000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_type_power(self):
        tfm = new_transformer()
        tfm.vol(0.8, gain_type='power')

        actual_args = tfm.effects
        expected_args = ['vol', '0.800000', 'power']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_gain_type_db(self):
        tfm = new_transformer()
        tfm.vol(0.8, gain_type='db')

        actual_args = tfm.effects
        expected_args = ['vol', '0.800000', 'dB']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vol']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)
        tfm_assert_array_to_file_output(INPUT_FILE, OUTPUT_FILE, tfm)

    def test_invalid_gain(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vol('a')

    def test_invalid_gain_power(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vol(-0.5, gain_type='power')

    def test_invalid_gain_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vol(1.5, gain_type='asdf')

    def test_invalid_limiter_gain(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.vol(0.2, limiter_gain=-0.2)
