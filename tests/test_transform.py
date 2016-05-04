import unittest

from sox import transform
from sox.core import SoxError

INPUT_FILE = 'data/input.wav'
OUTPUT_FILE = 'data/output.wav'


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
