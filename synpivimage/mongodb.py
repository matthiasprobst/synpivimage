import pathlib
import pymongo
from dataclasses import dataclass
from h5rdmtoolbox.h5database import H5Files
from tqdm import tqdm
from typing import List

from ._version import __version__

sft_dict = dict(name='synpivimage',
                version=__version__,
                url='https://github.com/MatthiasProbst/synpivimage',
                description='Tool to build synthetic Particle Image Velocimetry (PIV) images')


@dataclass
class SynPivImageSet:
    """Interface class to HDF5 files containing synpivimag-data stored in a directory
    Input is the root directory."""

    set_dir: pathlib.Path
    collection: pymongo.collection.Collection = None

    def __post_init__(self):
        self.set_dir = pathlib.Path(self.set_dir)

    def collect_hdf_files(self) -> List[pathlib.Path]:
        """Collect HDF files"""
        return sorted(self.set_dir.rglob('*.hdf'))

    @property
    def files(self) -> H5Files:
        """Return H5Files"""
        _h5files = sorted(self.collect_hdf_files())
        return H5Files(*_h5files)

    def insert(self, collection: pymongo.collection.Collection = None) -> None:
        """Insert data into collection"""
        if collection is None and self.collection is None:
            raise ValueError('A collection must be provided')
        if collection is None:
            collection = self.collection
        else:
            self.collection = collection
        print('Writing data to mongoDB. This may take a while if your files are big ... ')
        _h5files = sorted(self.collect_hdf_files())
        with self.files as h5:
            h5grps = list(h5.values())
            nfiles = len(h5grps)
            for grp in tqdm(h5grps, total=nfiles):
                grp.images.mongo.insert(0, collection,
                                        dims=(grp['particle_density'],),
                                        additional_fields={'software': sft_dict})
