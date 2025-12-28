import json
import pathlib
import unittest

import synpivimage

__this_dir__ = pathlib.Path(__file__).parent


class TestVersion(unittest.TestCase):
    def test_version(self):
        this_version = "x.x.x"
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject_filename = __this_dir__ / "../pyproject.toml"
        with open(pyproject_filename, "rb") as f:
            pyproject_data = tomllib.load(f)
            this_version = pyproject_data["project"]["version"]
        self.assertEqual(synpivimage.__version__, this_version)

    def test_codemeta(self):
        """checking if the version in codemeta.json is the same as the one of the toolbox"""

        codemeta_filename = __this_dir__ / "../codemeta.json"
        with open(codemeta_filename, "r") as f:
            codemeta = json.load(f)

        self.assertEqual(codemeta["version"], synpivimage.__version__)
