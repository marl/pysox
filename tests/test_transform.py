import unittest
import os

from sox import transform
from sox.core import SoxError


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SPACEY_FILE = relpath("data/annoying filename (derp).wav")
INPUT_FILE = relpath('data/input.wav')
INPUT_FILE4 = relpath('data/input4.wav')
OUTPUT_FILE = relpath('data/output.wav')


def new_transformer():
    return transform.Transformer()


class TestTransformDefault(unittest.TestCase):
    def setUp(self):
        self.transformer = transform.Transformer()

    def test_globals(self):
        expected = ['-D', '-V2']
        actual = self.transformer.globals
        self.assertEqual(expected, actual)

    def test_input_format(self):
        expected = []
        actual = self.transformer.input_format
        self.assertEqual(expected, actual)

    def test_output_format(self):
        expected = []
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
        expected = []
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type(self):
        self.tfm.set_input_format(file_type='wav')
        actual = self.tfm.input_format
        expected = ['-t', 'wav']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(file_type='blurg')

    def test_rate(self):
        self.tfm.set_input_format(rate=44100)
        actual = self.tfm.input_format
        expected = ['-r', '44100']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_rate_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(rate='a')

    def test_rate_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(rate=0)

    def test_bits(self):
        self.tfm.set_input_format(bits=32)
        actual = self.tfm.input_format
        expected = ['-b', '32']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_bits_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(bits='a')

    def test_bits_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(bits=-4)

    def test_channels(self):
        self.tfm.set_input_format(channels=2)
        actual = self.tfm.input_format
        expected = ['-c', '2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_channels_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(channels='a')

    def test_channels_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(channels=-2)

    def test_encoding(self):
        self.tfm.set_input_format(encoding='signed-integer')
        actual = self.tfm.input_format
        expected = ['-e', 'signed-integer']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_encoding_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(encoding='16-bit-signed-integer')

    def test_ignore_length(self):
        self.tfm.set_input_format(ignore_length=True)
        actual = self.tfm.input_format
        expected = ['--ignore-length']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_ignore_length_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_input_format(ignore_length=None)


class TestTransformSetOutputFormat(unittest.TestCase):

    def setUp(self):
        self.tfm = new_transformer()

    def test_defaults(self):
        actual = self.tfm.output_format
        expected = []
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type(self):
        self.tfm.set_output_format(file_type='wav')
        actual = self.tfm.output_format
        expected = ['-t', 'wav']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_file_type_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(file_type='blurg')

    def test_rate(self):
        self.tfm.set_output_format(rate=44100)
        actual = self.tfm.output_format
        expected = ['-r', '44100']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_rate_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(rate='a')

    def test_rate_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(rate=0)

    def test_bits(self):
        self.tfm.set_output_format(bits=32)
        actual = self.tfm.output_format
        expected = ['-b', '32']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_bits_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(bits='a')

    def test_bits_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(bits=-4)

    def test_channels(self):
        self.tfm.set_output_format(channels=2)
        actual = self.tfm.output_format
        expected = ['-c', '2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_channels_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(channels='a')

    def test_channels_invalid2(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(channels=-2)

    def test_encoding(self):
        self.tfm.set_output_format(encoding='signed-integer')
        actual = self.tfm.output_format
        expected = ['-e', 'signed-integer']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_encoding_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(encoding='16-bit-signed-integer')

    def test_comments(self):
        self.tfm.set_output_format(comments='asdf')
        actual = self.tfm.output_format
        expected = ['--add-comment', 'asdf']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_comments_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(comments=2)

    def test_append_comments(self):
        self.tfm.set_output_format(comments='asdf', append_comments=False)
        actual = self.tfm.output_format
        expected = ['--comment', 'asdf']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_append_comments_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_output_format(append_comments=None)


class TestTransformerBuild(unittest.TestCase):
    def setUp(self):
        self.tfm = new_transformer()

    def test_valid(self):
        status = self.tfm.build(INPUT_FILE, OUTPUT_FILE)
        self.assertTrue(status)

    def test_valid_spacey(self):
        status = self.tfm.build(SPACEY_FILE, OUTPUT_FILE)
        self.assertTrue(status)

    def test_invalid(self):
        with self.assertRaises(IOError):
            self.tfm.build('blah/asdf.wav', OUTPUT_FILE)


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
        tfm.allpass(500.0)

        actual_args = tfm.effects
        expected_args = ['allpass', '500.0', '2.0q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['allpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['bandpass', '500.0', '2.0q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bandpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_constant_skirt(self):
        tfm = new_transformer()
        tfm.bandpass(500.0, constant_skirt=True)

        actual_args = tfm.effects
        expected_args = ['bandpass', '-c', '500.0', '2.0q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bandpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['bandreject', '500.0', '2.0q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bandreject']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['bass', '-20.0', '100.0', '0.5s']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['bass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


class TestTransformerBiquad(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.biquad([0, 0, 0], [1, 0, 0])

        actual_args = tfm.effects
        expected_args = ['biquad', '0', '0', '0', '1', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['biquad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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

    def test_explicit_args(self):
        tfm = new_transformer()
        tfm.chorus(
            n_voices=1, delays=[50.0], decays=[0.32], speeds=[0.25],
            depths=[2.0], shapes=['t']
        )

        # check only the first 3 args - the rest are randomized
        actual_args = tfm.effects
        expected_args = [
            'chorus', '0.5', '0.9', '50.0', '0.32', '0.25', '2.0', '-t'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['chorus']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['contrast', '75']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['contrast']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['compand', '0.3,0.8', '6.0:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['compand']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_attack_time_valid(self):
        tfm = new_transformer()
        tfm.compand(attack_time=0.5)

        actual_args = tfm.effects
        expected_args = ['compand', '0.5,0.8', '6.0:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['compand', '0.3,0.5', '6.0:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['compand', '1.0,0.5', '6.0:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_soft_knee_valid(self):
        tfm = new_transformer()
        tfm.compand(soft_knee_db=-2)

        actual_args = tfm.effects
        expected_args = ['compand', '0.3,0.8', '-2:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_soft_knee_none(self):
        tfm = new_transformer()
        tfm.compand(soft_knee_db=None)

        actual_args = tfm.effects
        expected_args = ['compand', '0.3,0.8', '-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_soft_knee_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.compand(soft_knee_db='s')

    def test_tf_points_valid(self):
        tfm = new_transformer()
        tfm.compand(tf_points=[(0, -4), (-70, -60), (-60, -20), (-40, -40)])

        actual_args = tfm.effects
        expected_args = [
            'compand', '0.3,0.8', '6.0:-70,-60,-60,-20,-40,-40,0,-4'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = []
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_samplerate_valid(self):
        tfm = new_transformer()
        tfm.convert(samplerate=8000)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '8000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['rate']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_samplerate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(samplerate=0)

    def test_channels_valid(self):
        tfm = new_transformer()
        tfm.convert(n_channels=3)

        actual_args = tfm.output_format
        expected_args = ['-c', '3']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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

        actual_args = tfm.output_format
        expected_args = ['-b', '8']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_bitdepth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(bitdepth=17)


class TestTransformerDcshift(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.dcshift()

        actual_args = tfm.effects
        expected_args = ['dcshift', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['dcshift']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


class TestTransformerDelay(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.delay([1.0])

        actual_args = tfm.effects
        expected_args = ['delay', '1.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['delay']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_default_three_channel(self):
        tfm = new_transformer()
        tfm.delay([0.0, 1.0])

        actual_args = tfm.effects
        expected_args = ['delay', '0.0', '1.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['delay']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE4, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_invalid_position_type(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.delay(1.0)

    def test_invalid_position_vals(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.delay([-1.0, 1.0])


@unittest.skip("Tests pass on local machine and fail on remote.")
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

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


class TestTransformerEqualizer(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.equalizer(500.0, 2, 3)

        actual_args = tfm.effects
        expected_args = ['equalizer', '500.0', '2q', '3']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['equalizer']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['fade', 'q', '0.5']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['fade']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_fade_in_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_in_len=1.2)

        actual_args = tfm.effects
        expected_args = ['fade', 'q', '1.2']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_fade_in_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fade(fade_in_len=-1)

    def test_fade_out_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_out_len=3)

        actual_args = tfm.effects
        expected_args = ['reverse', 'fade', 'q', '3', 'reverse']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_fade_out_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fade(fade_out_len='q')

    def test_fade_shape_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_shape='p', fade_in_len=1.5)

        actual_args = tfm.effects
        expected_args = ['fade', 'p', '1.5']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
            'fir', '0.0195', '-0.082', '0.234', '0.891', '-0.145', '0.043'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['fir']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
            'flanger', '0', '2', '0', '71', '0.5', 'sine', '25', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['flanger']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_delay_valid(self):
        tfm = new_transformer()
        tfm.flanger(delay=10)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '10', '2', '0', '71', '0.5', 'sine', '25', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_delay_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(delay=31)

    def test_flanger_depth_valid(self):
        tfm = new_transformer()
        tfm.flanger(depth=0)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0', '0', '0', '71', '0.5', 'sine', '25', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_depth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(depth=None)

    def test_flanger_regen_valid(self):
        tfm = new_transformer()
        tfm.flanger(regen=-95)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0', '2', '-95', '71', '0.5', 'sine', '25', 'linear'
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
            'flanger', '0', '2', '0', '0', '0.5', 'sine', '25', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_width_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(width='z')

    def test_flanger_speed_valid(self):
        tfm = new_transformer()
        tfm.flanger(speed=10)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0', '2', '0', '71', '10', 'sine', '25', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_speed_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(speed=0.0)

    def test_flanger_shape_valid(self):
        tfm = new_transformer()
        tfm.flanger(shape='triangle')

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0', '2', '0', '71', '0.5', 'triangle', '25', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_shape_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(shape='square')

    def test_flanger_phase_valid(self):
        tfm = new_transformer()
        tfm.flanger(phase=95)

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0', '2', '0', '71', '0.5', 'sine', '95', 'linear'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_phase_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(phase=-1)

    def test_flanger_interp_valid(self):
        tfm = new_transformer()
        tfm.flanger(interp='quadratic')

        actual_args = tfm.effects
        expected_args = [
            'flanger', '0', '2', '0', '71', '0.5', 'sine', '25', 'quadratic'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_flanger_interp_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.flanger(interp='cubic')


class TestTransformerGain(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.gain()

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['gain']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.gain(gain_db=6)

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '6']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(gain_db=None)

    def test_normalize_valid(self):
        tfm = new_transformer()
        tfm.gain(normalize=False)

        actual_args = tfm.effects
        expected_args = ['gain', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_normalize_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(normalize='6')

    def test_limiter_valid(self):
        tfm = new_transformer()
        tfm.gain(limiter=True)

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '-l', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_limiter_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(limiter='0')

    def test_balance_valid(self):
        tfm = new_transformer()
        tfm.gain(balance='B')

        actual_args = tfm.effects
        expected_args = ['gain', '-B', '-n', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_balance_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.gain(balance='h')


class TestTransformerHighpass(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.highpass(1000.0)

        actual_args = tfm.effects
        expected_args = ['highpass', '-2', '1000.0', '0.707q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['highpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_one_pole(self):
        tfm = new_transformer()
        tfm.highpass(1000.0, n_poles=1)

        actual_args = tfm.effects
        expected_args = ['highpass', '-1', '1000.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['highpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


@unittest.skip("Tests pass on local machine and fail on remote.")
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

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_num_taps_valid(self):
        tfm = new_transformer()
        tfm.hilbert(num_taps=17)

        actual_args = tfm.effects
        expected_args = ['hilbert', '-n', '17']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['lowpass', '-2', '1000.0', '0.707q']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['lowpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_one_pole(self):
        tfm = new_transformer()
        tfm.lowpass(1000.0, n_poles=1)

        actual_args = tfm.effects
        expected_args = ['lowpass', '-1', '1000.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['lowpass']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['loudness', '-10.0', '65.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['loudness']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.loudness(gain_db=0)

        actual_args = tfm.effects
        expected_args = ['loudness', '0', '65.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.loudness(gain_db='0dB')

    def test_reference_level_valid(self):
        tfm = new_transformer()
        tfm.loudness(reference_level=50)

        actual_args = tfm.effects
        expected_args = ['loudness', '-10.0', '50']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_reference_level_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.loudness(reference_level=None)

    def test_reference_level_oorange(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.loudness(reference_level=15.0)


class TestTransformerNorm(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.norm()

        actual_args = tfm.effects
        expected_args = ['norm', '-3.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['norm']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_db_level_valid(self):
        tfm = new_transformer()
        tfm.norm(db_level=0)

        actual_args = tfm.effects
        expected_args = ['norm', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


class TestTransformerOverdrive(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.overdrive()

        actual_args = tfm.effects
        expected_args = ['overdrive', '20.0', '20.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['overdrive']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.overdrive(gain_db=2)

        actual_args = tfm.effects
        expected_args = ['overdrive', '2', '20.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.overdrive(gain_db='-2dB')

    def test_colour_valid(self):
        tfm = new_transformer()
        tfm.overdrive(colour=0)

        actual_args = tfm.effects
        expected_args = ['overdrive', '20.0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_colour_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.overdrive(colour=None)


class TestTransformerPad(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.pad()

        actual_args = tfm.effects
        expected_args = ['pad', '0.0', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['pad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_start_duration_valid(self):
        tfm = new_transformer()
        tfm.pad(start_duration=3)

        actual_args = tfm.effects
        expected_args = ['pad', '3', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_start_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pad(start_duration=-1)

    def test_end_duration_valid(self):
        tfm = new_transformer()
        tfm.pad(end_duration=0.2)

        actual_args = tfm.effects
        expected_args = ['pad', '0.0', '0.2']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_end_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pad(end_duration='foo')


class TestTransformerPitch(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.pitch(0.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['pitch']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_n_semitones_valid(self):
        tfm = new_transformer()
        tfm.pitch(-3.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '-300.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_n_semitones_warning(self):
        tfm = new_transformer()
        tfm.pitch(13.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '1300.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_n_semitones_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pitch('a')

    def test_quick_valid(self):
        tfm = new_transformer()
        tfm.pitch(1.0, quick=True)

        actual_args = tfm.effects
        expected_args = ['pitch', '-q', '100.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_quick_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pitch(1.0, quick=1)


class TestTransformerRate(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.rate(48000)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '48000']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['rate']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_samplerate_valid(self):
        tfm = new_transformer()
        tfm.rate(1000.5)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '1000.5']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_samplerate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.rate(0)

    def test_quality_valid(self):
        tfm = new_transformer()
        tfm.rate(44100.0, quality='q')

        actual_args = tfm.effects
        expected_args = ['rate', '-q', '44100.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_quality_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.rate(44100.0, quality='foo')


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

    def test_count_valid(self):
        tfm = new_transformer()
        tfm.repeat(count=2)

        actual_args = tfm.effects
        expected_args = ['repeat', '2']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['reverb', '50', '50', '100', '100', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['reverb']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_reverberance_valid(self):
        tfm = new_transformer()
        tfm.reverb(reverberance=90)

        actual_args = tfm.effects
        expected_args = ['reverb', '90', '50', '100', '100', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_reverberance_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(reverberance=150)

    def test_high_freq_damping_valid(self):
        tfm = new_transformer()
        tfm.reverb(high_freq_damping=10)

        actual_args = tfm.effects
        expected_args = ['reverb', '50', '10', '100', '100', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_high_freq_damping_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(high_freq_damping='a')

    def test_room_scale_valid(self):
        tfm = new_transformer()
        tfm.reverb(room_scale=10)

        actual_args = tfm.effects
        expected_args = ['reverb', '50', '50', '10', '100', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_room_scale_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(room_scale=-100)

    def test_stereo_depth_valid(self):
        tfm = new_transformer()
        tfm.reverb(stereo_depth=50)

        actual_args = tfm.effects
        expected_args = ['reverb', '50', '50', '100', '50', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_stereo_depth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(stereo_depth=101)

    def test_pre_delay_valid(self):
        tfm = new_transformer()
        tfm.reverb(pre_delay=10)

        actual_args = tfm.effects
        expected_args = ['reverb', '50', '50', '100', '100', '10', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_pre_delay_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(pre_delay=-1)

    def test_wet_gain_valid(self):
        tfm = new_transformer()
        tfm.reverb(wet_gain=5)

        actual_args = tfm.effects
        expected_args = ['reverb', '50', '50', '100', '100', '0', '5']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_wet_gain_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.reverb(wet_gain='z')

    def test_wet_only_valid(self):
        tfm = new_transformer()
        tfm.reverb(wet_only=True)

        actual_args = tfm.effects
        expected_args = ['reverb', '-w', '50', '50', '100', '100', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


class TestTransformerSilence(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.silence()

        actual_args = tfm.effects
        expected_args = ['silence', '1', '0.1', '0.1%', '-1', '0.1', '0.1%']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['silence']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_location_beginning(self):
        tfm = new_transformer()
        tfm.silence(location=1)

        actual_args = tfm.effects
        expected_args = ['silence', '1', '0.1', '0.1%']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_location_end(self):
        tfm = new_transformer()
        tfm.silence(location=-1)

        actual_args = tfm.effects
        expected_args = ['reverse', 'silence', '1', '0.1', '0.1%', 'reverse']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_location_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(location=2)

    def test_silence_threshold_valid(self):
        tfm = new_transformer()
        tfm.silence(silence_threshold=10.5)

        actual_args = tfm.effects
        expected_args = ['silence', '1', '0.1', '10.5%', '-1', '0.1', '10.5%']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['silence', '1', '2', '0.1%', '-1', '2', '0.1%']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_min_silence_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(min_silence_duration='a')

    def test_buffer_around_silence_valid(self):
        tfm = new_transformer()
        tfm.silence(buffer_around_silence=True)

        actual_args = tfm.effects
        expected_args = [
            'silence', '-l', '1', '0.1', '0.1%', '-1', '0.1', '0.1%'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_buffer_around_silence_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(buffer_around_silence=0)


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


class TestTransformerTempo(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.tempo(1.1)

        actual_args = tfm.effects
        expected_args = ['tempo', '1.1']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['tempo']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_factor_valid(self):
        tfm = new_transformer()
        tfm.tempo(0.9)

        actual_args = tfm.effects
        expected_args = ['tempo', '0.9']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_factor_warning(self):
        tfm = new_transformer()
        tfm.tempo(0.1)

        actual_args = tfm.effects
        expected_args = ['tempo', '0.1']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_factor_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tempo(-1.1)

    def test_audio_type_valid(self):
        tfm = new_transformer()
        tfm.tempo(1.5, audio_type='m')

        actual_args = tfm.effects
        expected_args = ['tempo', '-m', '1.5']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_audio_type_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tempo(1.5, audio_type=1)

    def test_quick_valid(self):
        tfm = new_transformer()
        tfm.tempo(1.5, quick=True)

        actual_args = tfm.effects
        expected_args = ['tempo', '-q', '1.5']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_quick_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.tempo(1.5, quick=1)


class TestTransformerTreble(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.treble(20.0)

        actual_args = tfm.effects
        expected_args = ['treble', '20.0', '3000.0', '0.5s']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['treble']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        expected_args = ['tremolo', '6.0', '40.0']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['tremolo']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
        tfm.trim(0, 8.5)

        actual_args = tfm.effects
        expected_args = ['trim', '0', '8.5']
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['trim']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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


@unittest.skip("Tests pass on local machine and fail on remote.")
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

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
            '-t', '7.0',
            '-T', '0.25',
            '-s', '1.0',
            '-g', '0.25',
            '-p', '0.0'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_end_location(self):
        tfm = new_transformer()
        tfm.vad(location=-1)

        actual_args = tfm.effects
        expected_args = [
            'norm',
            'reverse',
            'vad',
            '-t', '7.0',
            '-T', '0.25',
            '-s', '1.0',
            '-g', '0.25',
            '-p', '0.0',
            'reverse'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_no_normalize(self):
        tfm = new_transformer()
        tfm.vad(normalize=False)

        actual_args = tfm.effects
        expected_args = [
            'vad',
            '-t', '7.0',
            '-T', '0.25',
            '-s', '1.0',
            '-g', '0.25',
            '-p', '0.0'
        ]
        self.assertEqual(expected_args, actual_args)

        actual_log = tfm.effects_log
        expected_log = ['vad']
        self.assertEqual(expected_log, actual_log)

        actual_res = tfm.build(INPUT_FILE, OUTPUT_FILE)
        expected_res = True
        self.assertEqual(expected_res, actual_res)

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
