import abc
import pathlib
import shutil
import warnings
from typing import Literal
from typing import Union, Optional

import cv2
import numpy as np
from ontolutils import query
from ontolutils.classes.utils import split_URIRef
from pivmetalib.pivmeta import VirtualLaser, VirtualCamera

from .camera import Camera
from .laser import Laser
from .particles import Particles

Format = Literal['json', 'json-ld']


class Writer(abc.ABC):
    """Abstract base class for writing images and metadata to disk."""

    @abc.abstractmethod
    def writeA(self, index: int, img: np.ndarray, particles: Particles = None) -> pathlib.Path:
        """Write image A"""

    @abc.abstractmethod
    def writeB(self, index: int, img: np.ndarray, particles: Particles = None) -> pathlib.Path:
        """Write image B"""


class Imwriter(Writer):
    """Context manager for writing images and metadata to a folder.

    Example:
    --------
    with Imwriter('case_name', image_dir='path/to/folder', camera=camera, laser=laser) as imwriter:
        imwriter.writeA(imgA, particles=particlesA)
        imwriter.writeB(imgB, particles=particlesB)
    """

    def __init__(self,
                 case_name: str,
                 image_dir: Optional[Union[str, pathlib.Path]] = None,
                 suffix: str = '.tif',
                 overwrite: bool = False,
                 camera: Camera = None,
                 laser: Laser = None):
        self.case_name = case_name
        self.image_dir = image_dir
        self.suffix = suffix
        self.overwrite = overwrite
        self.camera = camera
        self.laser = laser
        self.img_filenames = []
        self.particle_filenames = []
        self._enabled = False

    def __enter__(self):
        self.img_filenames = []
        self.particle_filenames = []

        if self.image_dir is None:
            image_dir = pathlib.Path.cwd() / self.case_name
        else:
            image_dir = pathlib.Path(self.image_dir) / self.case_name

        if image_dir.exists() and not self.overwrite:
            raise FileExistsError(f"Directory {image_dir} exists and overwrite is False")
        if image_dir.exists() and self.overwrite:
            shutil.rmtree(image_dir)

        self.image_dir = image_dir

        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / 'imgs').mkdir(parents=True, exist_ok=True)

        if self.camera:
            self.camera.save_jsonld(image_dir / 'camera.json')
        if self.laser:
            self.laser.save_jsonld(image_dir / 'laser.json')

        self._enabled = True
        return self

    def write(self, index: int, img: np.ndarray, ab: Optional[Literal['A', 'B']] = None,
              particles: Particles = None) -> pathlib.Path:
        """Write image

        Parameters
        ----------
        index : int
            Image index. Used to create the filename (e.g. img_000001.tif)
        img: np.ndarray
            Image array (2D)
        ab: Optional[Literal['A', 'B']]
            Image type (A or B) or None, then no A/B-suffix is added
        particles: Particles
            Particle object used to generate img

        Returns
        -------
        img_filename : pathlib.Path
            The filename of the image
        """
        if not self._enabled:
            raise ValueError('Imwriter is not enabled')
        if ab is None:
            ab = ''
        img_filename = self.image_dir / 'imgs' / f'img_{index:06d}{ab}{self.suffix}'
        cv2.imwrite(str(img_filename), np.asarray(img))

        if particles:
            particle_dir = (self.image_dir / 'particles')
            particle_dir.mkdir(parents=True, exist_ok=True)
            particle_filename = particle_dir / f'particles_{index:06d}{ab}.json'
            particles.save_jsonld(particle_dir / f'particles_{index:06d}{ab}.json')
            self.particle_filenames.append(particle_filename)

        self.img_filenames.append(img_filename)
        return img_filename

    def writeA(self, index: int, img: np.ndarray, particles: Particles = None) -> pathlib.Path:
        """Write image A

        Parameters
        ----------
        index : int
            Image index. Used to create the filename (e.g. img_000001A.tif)
        img: np.ndarray
            Image A array (2D)
        particles: Particles
            Particle object used to generate img A

        Returns
        -------
        img_filename : pathlib.Path
            The filename of the image
        """
        return self.write(index, img, 'A', particles)

    def writeB(self, index: int, img: np.ndarray, particles: Particles = None) -> pathlib.Path:
        """Write image B

        Parameters
        ----------
        index : int
            Image index. Used to create the filename (e.g. img_000001B.tif)
        img: np.ndarray
            Image A array (2D)
        particles: Particles
            Particle object used to generate img A

        Returns
        -------
        img_filename : pathlib.Path
            The filename of the image
        """
        return self.write(index, img, 'B', particles)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._enabled = False


class HDF5Writer(Writer):
    """Context manager for writing images and metadata to a folder.

    Example:
    --------
    >>> with synpivimage.HDF5Writer(filename='single_img.hdf',
    >>>                             camera=cam,
    >>>                             laser=laser,
    >>>                             overwrite=True) as h5:
    >>>     h5.writeA(img, particles=part)
    >>>     h5.writeA(img, particles=part)
    """

    def __init__(self,
                 filename: Union[str, pathlib.Path],
                 *,
                 n_images: int,
                 overwrite: bool = False,
                 camera: Camera = None,
                 laser: Laser = None):
        """
        Save images and metadata to a HDF5 file.

        Parameters
        ----------
        filename : Union[str, pathlib.Path]
            The filename of the HDF5 file
        n_images : int
            Number of images to save. Needed to pre-allocate space in the HDF5 file
        overwrite : bool
            Overwrite the file if it exists
        camera : Camera
            Camera object
        laser : Laser
            Laser object

        """
        self.filename = pathlib.Path(filename)
        self.overwrite = overwrite
        self.camera = camera
        self.laser = laser
        self._h5 = None
        self._n_images = n_images

    def __enter__(self, n_imgs: Optional[int] = None):
        if self.filename.exists() and not self.overwrite:
            raise FileExistsError(f"File {self.filename} exists and overwrite is False")
        if self.filename.exists() and self.overwrite:
            self.filename.unlink()
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5Writer")
        self._h5 = h5py.File(self.filename, 'w')
        return self

    def _get_dsimgA(self) -> "h5py.Dataset":
        """Get or create the dataset for image A"""
        ds_nameA = "images/img_A"
        if ds_nameA in self._h5:
            return self._h5[ds_nameA]
        return self._h5.create_dataset(ds_nameA,
                                       shape=(self._n_images, self.camera.ny, self.camera.nx),
                                       maxshape=(self._n_images, self.camera.ny, self.camera.nx),
                                       dtype='uint16')

    def _get_dsimgB(self) -> "h5py.Dataset":
        """Get or create the dataset for image A"""
        ds_nameA = "images/img_B"
        if ds_nameA in self._h5:
            return self._h5[ds_nameA]
        return self._h5.create_dataset(ds_nameA,
                                       shape=(self._n_images, self.camera.ny, self.camera.nx),
                                       maxshape=(self._n_images, self.camera.ny, self.camera.nx),
                                       dtype='uint16')

    def _write_particles(self, index: int, ab: str, particles):
        """Write particles to HDF. data to write:
        x,y,z,size,source_intensity,max_image_photons,image_electrons,image_quantized_electrons,flag

        """
        particle_group = f'particles/{ab}'
        n = len(particles)
        if particle_group not in self._h5:
            gr = self._h5.create_group(particle_group)
            ds_shape = (self._n_images, n)
            ds = gr.create_dataset('x', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.x
            ds = gr.create_dataset('y', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.y
            ds = gr.create_dataset('z', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.z
            ds = gr.create_dataset('size', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.size
            ds = gr.create_dataset('source_intensity', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.source_intensity
            ds = gr.create_dataset('max_image_photons', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.max_image_photons
            ds = gr.create_dataset('image_electrons', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.image_electrons
            ds = gr.create_dataset('image_quantized_electrons', shape=ds_shape, maxshape=(None, n), dtype='float32')
            ds[0, :] = particles.image_quantized_electrons
            ds = gr.create_dataset('flag', shape=ds_shape, maxshape=(None, n), dtype='uint8')
            ds[0, :] = particles.flag
            ds.resize((ds.shape[0] + 1, *ds.shape[1:]))

            return

        ds = self._h5[particle_group]['x']
        curr_shape = ds.shape

        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.x

        ds = self._h5[particle_group]['y']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.y

        ds = self._h5[particle_group]['z']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.z

        ds = self._h5[particle_group]['size']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.size

        ds = self._h5[particle_group]['source_intensity']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.source_intensity

        ds = self._h5[particle_group]['max_image_photons']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.max_image_photons

        ds = self._h5[particle_group]['image_electrons']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.image_electrons

        ds = self._h5[particle_group]['image_quantized_electrons']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.image_quantized_electrons

        ds = self._h5[particle_group]['flag']
        if self._n_images is None:
            ds.resize((curr_shape[0] + 1, *curr_shape[1:]))
        ds[index, :] = particles.flag

    def writeA(self, index: int, img: np.ndarray, particles: Particles = None):
        """Write image A

        Parameters
        ----------
        index: int
            Image index
        img: np.ndarray
            Image A array
        particles: Particles
            Particle object used to generate img A
        """
        ds = self._get_dsimgA()
        if index >= ds.shape[0]:
            raise KeyError(f'Image index {index} is out of range. Only {ds.shape[0]} images are expected, thus '
                           f'index should be in the range [0, {ds.shape[0] - 1}]')
        ds[index, ...] = img

        if particles is not None:
            self._write_particles(index, 'A', particles)

    def writeB(self, index: int, img: np.ndarray, particles: Particles = None):
        """Write image B

        Parameters
        ----------
        index: int
            Image index
        img: np.ndarray
            Image B array
        particles: Particles
            Particle object used to generate img B
        """
        ds = self._get_dsimgB()
        if index >= ds.shape[0]:
            raise KeyError(f'Image index {index} is out of range. Only {ds.shape[0]} images are expected, thus '
                           f'index should be in the range [0, {ds.shape[0] - 1}]')
        ds[index, ...] = img

        if particles is not None:
            self._write_particles(index, 'B', particles)

    def __exit__(self, exc_type, exc_val, exc_tb):
        ds = self._get_dsimgA()

        n_imgs = ds.shape[0]
        img_idx_ds = ds.parent.create_dataset('image_index', data=np.arange(n_imgs))
        img_idx_ds.make_scale()

        ds.dims[0].attach_scale(img_idx_ds)
        nx_ds = ds.parent.create_dataset('ix', data=np.arange(self.camera.nx))
        ds.attrs['long_name'] = 'x pixel coordinate'
        ds.attrs['standard_name'] = 'x_pixel_coordinate'  # https://matthiasprobst.github.io/pivmeta/#x_pixel_coordinate
        nx_ds.make_scale()
        ds.dims[2].attach_scale(nx_ds)

        ny_ds = ds.parent.create_dataset('iy', data=np.arange(self.camera.ny))
        ny_ds.make_scale()
        ds.attrs['long_name'] = 'y pixel coordinate'
        ds.attrs['standard_name'] = 'y_pixel_coordinate'  # https://matthiasprobst.github.io/pivmeta/#x_pixel_coordinate
        ds.dims[1].attach_scale(ny_ds)

        for cmp_name, cmp, onto_model in zip(('laser', 'camera'),
                                             (self.laser, self.camera),
                                             (VirtualLaser, VirtualCamera)):

            if cmp:
                laser_jsonld = cmp.model_dump_jsonld()

                model = None
                models = query(onto_model, data=laser_jsonld)
                try:
                    model = models[0]
                except KeyError:
                    warnings.warn(f'Could not write laser attributes because metadata could not be extracted '
                                  f'from JSON-LD str: {laser_jsonld}')
                if model is None:
                    continue

                comp_gr = self._h5.create_group(cmp_name)
                for k, v in cmp.model_dump().items():
                    ds = comp_gr.create_dataset(k, data=v, dtype='float32')
                    for p in model.hasParameter:
                        if p.label != k:
                            continue

                        for model_field in p.model_dump():
                            if model_field == 'hasNumericalValue':
                                continue

                            attr_val = getattr(p, model_field)
                            if attr_val is None:
                                continue

                            if isinstance(attr_val, (int, float)):
                                ds.attrs[str(model_field)] = attr_val
                            else:
                                ns, n = split_URIRef(attr_val)
                                if n == 'UNITLESS':
                                    ds.attrs[model_field] = '-'
                                else:
                                    if model_field == "hasStandardName":
                                        ds.attrs[str(model_field)] = n.standardName
                                    else:
                                        ds.attrs[str(model_field)] = n

        self._h5.close()
        self._h5 = None
