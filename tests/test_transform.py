import unittest

from sox import transform
from sox.core import SoxError

INPUT_FILE = 'data/input.wav'
OUTPUT_FILE = 'data/output.wav'


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
        expected = ['-D', '-V4']
        actual = self.transformer.globals
        self.assertEqual(expected, actual)

    def test_input_filepath(self):
        expected = 'data/input.wav'
        actual = self.transformer.input_filepath
        self.assertEqual(expected, actual)

    def test_output_filepath(self):
        expected = 'data/output.wav'
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


class TestTransformerBuild(unittest.TestCase):
    def setUp(self):
        self.transformer_valid = transform.Transformer(
            INPUT_FILE, OUTPUT_FILE
        )

        self.transformer_invalid = transform.Transformer(
            INPUT_FILE, OUTPUT_FILE
        )
        self.transformer_invalid.input_filepath = 'data/asdf.wav'

    def test_valid(self):
        status = self.transformer_valid.build()
        self.assertTrue(status)

    def test_invalid(self):
        with self.assertRaises(SoxError):
            self.transformer_invalid.build()


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


class TestTransformerNorm(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.norm()

        actual_args = tfm.effects
        expected_args = ['norm', '-3']
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


class TestTransformerPad(unittest.TestCase):

    def test_default(self):
        tfm = new_transformer()
        tfm.pad()

        actual_args = tfm.effects
        expected_args = ['pad', '0', '0']
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
        expected_args = ['pad', '3', '0']
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
        expected_args = ['pad', '0', '0.2']
        self.assertEqual(expected_args, actual_args)

        actual_res = tfm.build()
        expected_res = True
        self.assertEqual(expected_res, actual_res)

    def test_end_duration_invalid(self):
        tfm = new_transformer()
        with self.assertRaises(ValueError):
            tfm.pad(end_duration='foo')


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

