import numpy as np
import pathlib
import unittest

from synpivimage import io

__this_dir__ = pathlib.Path(__file__).parent


class TestIO(unittest.TestCase):

    def setUp(self) -> None:
        self.filenames = []

    def tearDown(self) -> None:
        """delete created files"""
        for filename in self.filenames:
            pathlib.Path(filename).unlink(missing_ok=True)

    def test_imwrite_imread_16(self):
        im16 = np.random.randint(0, 2 ** 16 - 1, (16, 16), dtype=np.uint16)
        filename = io.imwrite(__this_dir__ / 'im16.tiff',
                              im16)
        self.filenames.append(filename)
        self.assertTrue(filename.exists())

        with self.assertRaises(FileNotFoundError):
            _ = io.imread('invalid.tiff', flags=-1)
        im16load = io.imread(filename, flags=-1)
        np.testing.assert_array_equal(im16, im16load)

        filename = io.imwrite16(
            __this_dir__ / 'im16.tiff',
            overwrite=True,
            img=im16
        )
        im16load = io.imread(filename, flags=-1)
        np.testing.assert_array_equal(im16, im16load)
        im16load = io.imread16(filename)
        np.testing.assert_array_equal(im16, im16load)

    def test_imwrite_imread_8(self):
        im8 = np.random.randint(0, 2 ** 8 - 1, (16, 16), dtype=np.uint8)
        filename = io.imwrite(__this_dir__ / 'im8.tiff',
                              im8)
        self.filenames.append(filename)
        self.assertTrue(filename.exists())

        with self.assertRaises(FileNotFoundError):
            _ = io.imread('invalid.tiff', flags=-1)
        im8load = io.imread(filename, flags=-1)
        np.testing.assert_array_equal(im8, im8load)

        filename = io.imwrite8(
            __this_dir__ / 'im8.tiff',
            overwrite=True,
            img=im8
        )
        im8load = io.imread(filename, flags=-1)
        np.testing.assert_array_equal(im8, im8load)
        im8load = io.imread8(filename)
        np.testing.assert_array_equal(im8, im8load)
