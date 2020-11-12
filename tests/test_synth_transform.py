import numpy as np
import soundfile as sf
import tempfile
import unittest

from sox import transform


class TestSynthTransformerDefault(unittest.TestCase):

    def setUp(self):
        self.tfm = transform.SynthTransformer()

    def test_globals(self):
        expected = ['-D', '-V2']
        actual = self.tfm.globals
        self.assertEqual(expected, actual)

    def test_input_format(self):
        expected = {'channels': 1}
        actual = self.tfm.input_format
        self.assertEqual(expected, actual)

    def test_output_format(self):
        expected = {}
        actual = self.tfm.output_format
        self.assertEqual(expected, actual)

    def test_effects(self):
        expected = ['synth']
        actual = self.tfm.effects
        self.assertEqual(expected, actual)

    def test_effects_log(self):
        expected = []
        actual = self.tfm.effects_log
        self.assertEqual(expected, actual)

    def test_invalid_build_args(self):
        with self.assertRaises(ValueError):
            self.tfm.build_array(input_filepath='some_path')

        with self.assertRaises(ValueError):
            sig = np.zeros(16000)
            self.tfm.build_array(input_array=sig)


class TestSynthTransformerInvalidMethods(unittest.TestCase):

    def setUp(self):
        self.tfm = transform.SynthTransformer()

    def test_set_input_format(self):
        with self.assertRaises(AttributeError):
            self.tfm.set_input_format(file_type='wav')


class TestSynthTransformerNoise(unittest.TestCase):

    def setUp(self):
        self.tfm = transform.SynthTransformer()

    def test_invalid_noise_type(self):
        with self.assertRaises(ValueError):
            self.tfm._add_noise('invalid-noise-type', 3)

    def test_invalid_times(self):
        noise_type = self.tfm._valid_noise_types[0]
        with self.assertRaises(ValueError):
            self.tfm._add_noise(noise_type, -1)

        with self.assertRaises(ValueError):
            self.tfm._add_noise(noise_type, 0)

    def test_build_array(self):
        sr = 16000
        dur = 3
        self.tfm.whitenoise(length=dur)
        got = self.tfm.build_array(sample_rate_in=sr)
        expected = sr * dur
        self.assertEqual(expected, got.size)

    def test_build_file(self):
        sr = 16000
        dur = 3
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            self.tfm.whitenoise(length=dur)
            self.tfm.build_file(output_filepath=f.name, sample_rate_in=sr)
            got, got_sr = sf.read(f.name)

            expected = sr * dur
            self.assertEqual(expected, got.size)
            self.assertEqual(sr, got_sr)

    def test_pinknoise(self):
        sr = 16000
        dur = 3
        self.tfm.pinknoise(length=dur)
        got = self.tfm.build_array(sample_rate_in=sr)
        expected = sr * dur
        self.assertEqual(expected, got.size)

    def test_brownnoise(self):
        sr = 16000
        dur = 3
        self.tfm.brownnoise(length=dur)
        got = self.tfm.build_array(sample_rate_in=sr)
        expected = sr * dur
        self.assertEqual(expected, got.size)

    def test_tpdfnoise(self):
        sr = 16000
        dur = 3
        self.tfm.tpdfnoise(length=dur)
        got = self.tfm.build_array(sample_rate_in=sr)
        expected = sr * dur
        self.assertEqual(expected, got.size)
