import pathlib
import unittest

from synpivimage.laser import Laser


class TestComponents(unittest.TestCase):

    def test_save_component(self):
        cam = Laser(shape_factor=2, width=0.1)
        filename = cam.save('laser.json')
        self.assertEqual(filename.suffix, '.json')
        self.assertTrue(filename.exists())
        filename.unlink()
        filename = cam.save('laser')
        self.assertTrue(filename.exists())
        self.assertEqual(filename.suffix, '.json')

    def tearDown(self) -> None:
        """clean up"""
        pathlib.Path('laser.json').unlink(missing_ok=True)
