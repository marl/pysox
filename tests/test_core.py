import unittest

from sox import core
from sox.core import SoxError

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
