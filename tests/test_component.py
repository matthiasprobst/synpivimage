import unittest

from synpivimage.laser import Laser


class TestComponents(unittest.TestCase):

    def test_save_component(self):
        cam = Laser(shape_factor=2, width=0.1)
        filename = cam.save_json('laser.json')
        self.assertEqual(filename.suffix, '.json')
        self.assertTrue(filename.exists())
        filename.unlink()
        filename = cam.save_json('laser')
        self.assertTrue(filename.exists())
        self.assertEqual(filename.suffix, '.json')

        filename.unlink(missing_ok=True)
