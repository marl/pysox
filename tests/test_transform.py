import unittest
import os

from sox import transform
from sox.core import SoxError


def relpath(f):
    return os.path.join(os.path.dirname(__file__), f)


SPACEY_FILE = relpath("data/annoying filename (derp).wav")
INPUT_FILE = relpath('data/input.wav')
OUTPUT_FILE = relpath('data/output.wav')


def new_transformer():
    return transform.Transformer(
        INPUT_FILE, OUTPUT_FILE
    )


class TestTransformDefault(unittest.TestCase):
    def setUp(self):
        self.input_filepath = INPUT_FILE
        self.output_filepath = OUTPUT_FILE
        self.transformer = transform.Transformer(
            self.input_filepath, self.output_filepath
        )

    def test_globals(self):
        expected = ['-D', '-V2']
        actual = self.transformer.globals
        self.assertEqual(expected, actual)

    def test_input_filepath(self):
        expected = INPUT_FILE
        actual = self.transformer.input_filepath
        self.assertEqual(expected, actual)

    def test_output_filepath(self):
        expected = OUTPUT_FILE
        actual = self.transformer.output_filepath
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

        actual_result = self.tfm.build()
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_dither(self):
        self.tfm.set_globals(dither=True)
        actual = self.tfm.globals
        expected = ['-V2']
        self.assertEqual(expected, actual)

        actual_result = self.tfm.build()
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

        actual_result = self.tfm.build()
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

        actual_result = self.tfm.build()
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

        actual_result = self.tfm.build()
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

        actual_result = self.tfm.build()
        expected_result = True
        self.assertEqual(expected_result, actual_result)

    def test_verbosity_invalid(self):
        with self.assertRaises(ValueError):
            self.tfm.set_globals(verbosity='debug')


class TestTransformerBuild(unittest.TestCase):
    def setUp(self):
        self.transformer_valid = transform.Transformer(
            INPUT_FILE, OUTPUT_FILE
        )

        self.transformer_invalid = transform.Transformer(
            INPUT_FILE, OUTPUT_FILE
        )
        self.transformer_invalid.input_filepath = 'blah/asdf.wav'

    def test_valid(self):
        status = self.transformer_valid.build()
        self.assertTrue(status)

    def test_invalid(self):
        with self.assertRaises(SoxError):
            self.transformer_invalid.build()


class TestTransformerBuildSpacey(unittest.TestCase):
    def setUp(self):
        self.transformer_valid = transform.Transformer(
            SPACEY_FILE, OUTPUT_FILE
        )

        self.transformer_invalid = transform.Transformer(
            SPACEY_FILE, OUTPUT_FILE
        )
        self.transformer_invalid.input_filepath = 'blah/asdf.wav'

    def test_valid(self):
        status = self.transformer_valid.build()
        self.assertTrue(status)

    def test_invalid(self):
        with self.assertRaises(SoxError):
            self.transformer_invalid.build()


class TestTransformerPreview(unittest.TestCase):
    def setUp(self):
        self.transformer_valid = transform.Transformer(
            INPUT_FILE, OUTPUT_FILE
        )
        self.transformer_valid.trim(0, 0.1)

    def test_valid(self):
        expected = None
        actual = self.transformer_valid.preview()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_attack_time_valid(self):
        tfm = new_transformer()
        tfm.compand(attack_time=0.5)

        actual_args = tfm.effects
        expected_args = ['compand', '0.5,0.8', '6.0:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_soft_knee_valid(self):
        tfm = new_transformer()
        tfm.compand(soft_knee_db=-2)

        actual_args = tfm.effects
        expected_args = ['compand', '0.3,0.8', '-2:-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_soft_knee_none(self):
        tfm = new_transformer()
        tfm.compand(soft_knee_db=None)

        actual_args = tfm.effects
        expected_args = ['compand', '0.3,0.8', '-70,-70,-60,-20,0,0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_samplerate_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(samplerate=0)

    def test_channels_valid(self):
        tfm = new_transformer()
        tfm.convert(channels=3)

        actual_args = tfm.output_format
        expected_args = ['-c', '3']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_channels_invalid1(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(channels=0)

    def test_channels_invalid2(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(channels=1.5)

    def test_bitdepth_valid(self):
        tfm = new_transformer()
        tfm.convert(bitdepth=8)

        actual_args = tfm.output_format
        expected_args = ['-b', '8']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_bitdepth_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.convert(bitdepth=17)


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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_fade_in_valid(self):
        tfm = new_transformer()
        tfm.fade(fade_in_len=1.2)

        actual_args = tfm.effects
        expected_args = ['fade', 'q', '1.2']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_fade_shape_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.fade(fade_shape='x')


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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.gain(gain_db=6)

        actual_args = tfm.effects
        expected_args = ['gain', '-n', '6']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.loudness(gain_db=0)

        actual_args = tfm.effects
        expected_args = ['loudness', '0', '65.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_db_level_valid(self):
        tfm = new_transformer()
        tfm.norm(db_level=0)

        actual_args = tfm.effects
        expected_args = ['norm', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_db_level_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.norm(db_level='-2dB')


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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_gain_db_valid(self):
        tfm = new_transformer()
        tfm.overdrive(gain_db=2)

        actual_args = tfm.effects
        expected_args = ['overdrive', '2', '20.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_start_duration_valid(self):
        tfm = new_transformer()
        tfm.pad(start_duration=3)

        actual_args = tfm.effects
        expected_args = ['pad', '3', '0.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_n_semitones_valid(self):
        tfm = new_transformer()
        tfm.pitch(-3.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '-300.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_n_semitones_warning(self):
        tfm = new_transformer()
        tfm.pitch(13.0)

        actual_args = tfm.effects
        expected_args = ['pitch', '1300.0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_samplerate_valid(self):
        tfm = new_transformer()
        tfm.rate(1000.5)

        actual_args = tfm.effects
        expected_args = ['rate', '-h', '1000.5']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_quality_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.rate(44100.0, quality='foo')


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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_reverberance_valid(self):
        tfm = new_transformer()
        tfm.reverb(reverberance=90)

        actual_args = tfm.effects
        expected_args = ['reverb', '90', '50', '100', '100', '0', '0']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_location_beginning(self):
        tfm = new_transformer()
        tfm.silence(location=1)

        actual_args = tfm.effects
        expected_args = ['silence', '1', '0.1', '0.1%']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_location_end(self):
        tfm = new_transformer()
        tfm.silence(location=-1)

        actual_args = tfm.effects
        expected_args = ['reverse', 'silence', '1', '0.1', '0.1%', 'reverse']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_buffer_around_silence_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.silence(buffer_around_silence=0)


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

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_factor_valid(self):
        tfm = new_transformer()
        tfm.tempo(0.9)

        actual_args = tfm.effects
        expected_args = ['tempo', '0.9']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_factor_warning(self):
        tfm = new_transformer()
        tfm.tempo(0.1)

        actual_args = tfm.effects
        expected_args = ['tempo', '0.1']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

        actual_res = tfm.build()
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

