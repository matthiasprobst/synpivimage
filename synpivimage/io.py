import pathlib
import shutil
from typing import Literal
from typing import Union, Optional

import cv2
import numpy as np

from .camera import Camera
from .laser import Laser
from .particles import Particles

Format = Literal['json', 'json-ld']


# def parse_filename(filename: Union[str, pathlib.Path],
#                    overwrite: bool) -> pathlib.Path:
#     filename = pathlib.Path(filename)
#
#     if filename.exists():
#         if not overwrite:
#             raise FileExistsError(f"File {filename} exists and overwrite is False")
#         else:
#             filename.unlink()
#     return filename


# def build_meta_dict(**kwargs) -> dict:
#     """generates a dictionary from the keyword arguments"""
#     meta = {}
#
#     for k, v in kwargs.items():
#         if v is not None:
#             if hasattr(v, 'model_dump_jsonld'):
#                 meta[k] = v.model_dump_jsonld()
#             else:
#                 meta[k] = v
#     meta['software_info'] = get_package_meta()
#     return meta


# def hdfwrite(group: h5py.Group,
#              img: Union[np.ndarray, List[np.ndarray]],
#              camera: Camera,
#              laser: Laser,
#              particles: Union[Particles, List[Particles]],
#              img_dataset_name='img',
#              bit_depth=16):
#     warnings.warn("hdfwrite is deprecated, use hdfwrite16 or hdfwrite8 instead", DeprecationWarning)
#     return _hdfwrite(group, img, camera, laser, particles, img_dataset_name=img_dataset_name, bit_depth=bit_depth)


# def hdfwrite16(group: h5py.Group,
#                img: Union[np.ndarray, List[np.ndarray]],
#                camera: Camera,
#                laser: Laser,
#                particles: Union[Particles, List[Particles]],
#                img_dataset_name='img'):
#     return _hdfwrite(group, img, camera, laser, particles, img_dataset_name=img_dataset_name, bit_depth=16)
#
#
# def hdfwrite8(group: h5py.Group,
#               img: Union[np.ndarray, List[np.ndarray]],
#               camera: Camera,
#               laser: Laser,
#               particles: Union[Particles, List[Particles]],
#               img_dataset_name='img'):
#     return _hdfwrite(group, img, camera, laser, particles, img_dataset_name=img_dataset_name, bit_depth=8)


# def _hdfwrite(group: h5py.Group,
#               img: Union[np.ndarray, List[np.ndarray]],
#               camera: Camera,
#               laser: Laser,
#               particles: Union[Particles, List[Particles]],
#               img_dataset_name,
#               bit_depth):
#     """Write to open HDF5 file"""
#     if bit_depth == 8:
#         _dtype = 'uint8'
#     elif bit_depth == 16:
#         _dtype = 'uint16'
#     else:
#         raise ValueError("bit_depth must be 8 or 16")
#     group.attrs['software_info'] = json.dumps(get_package_meta())
#     if isinstance(img, (list, tuple)):
#         assert isinstance(particles, (list, tuple)), "particles must be a list or tuple if img is a list or tuple"
#         assert len(img) == len(particles), "img and particles must have the same length"
#         n_imgs = len(img)
#         img_shape = img[0].shape
#         imgs = [np.asarray(i) for i in img]
#     else:
#         n_imgs = 1
#         img_shape = img.shape
#         imgs = [np.asarray(img)]
#         particles = [particles, ]
#     imgds = group.create_dataset(img_dataset_name, shape=(n_imgs, *img_shape), dtype=_dtype)
#     for i, im in enumerate(imgs):
#         imgds[i, ...] = im
#     metadata = group.create_group('virtual_piv_setup')
#
#     def _write_component(_component, _group):
#         for k, v in _component.model_dump().items():
#             _group.create_dataset(k, data=v)
#
#     ref_len = particles[0].x.size
#     equal_length_particles = all(ref_len == p.x.size for p in particles)
#     if equal_length_particles:
#         laser_grp = metadata.create_group(particles[0].__class__.__name__)
#
#         for ic, (k, v) in enumerate(particles[0].model_dump().items()):
#             ds = laser_grp.create_dataset(k, shape=(n_imgs, ref_len))
#             ds[0, :] = v
#
#         for i, p in enumerate(particles):
#             for ic, (k, v) in enumerate(p.model_dump().items()):
#                 laser_grp[k][i, :] = v
#
#         iterator = (camera, laser,)
#     else:
#         iterator = (camera, laser, particles[0])
#     for component in iterator:
#         if component:
#             if isinstance(component, (list, tuple)):
#                 for ic, c in enumerate(component):
#                     comp_grp = metadata.create_group(c.__class__.__name__ + f'_{ic}')
#                     _write_component(c, comp_grp)
#             else:
#                 comp_grp = metadata.create_group(component.__class__.__name__)
#                 _write_component(component, comp_grp)


class Imwriter:
    """Context manager for writing images and metadata to a folder.

    Example:
    --------
    with Imwriter('case_name', image_dir='path/to/folder', camera=camera, laser=laser) as imwriter:
        imwriter.writeA(imgA, particles=particlesA)
        imwriter.writeB(imgB, particles=particlesB)
    """

    def __init__(self, case_name: str,
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
        self._img_idx = 0
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
        (image_dir / 'particles').mkdir(parents=True, exist_ok=True)

        if self.camera:
            self.camera.save_jsonld(image_dir / 'camera.json')
        if self.laser:
            self.laser.save_jsonld(image_dir / 'laser.json')

        self._enabled = True
        self._img_idx = 0
        return self

    def write(self, img: np.ndarray, ab: Optional[Literal['A', 'B']] = None, particles: Particles = None):
        """Write an image to a file. Calls cv2.imwrite."""
        if not self._enabled:
            raise ValueError('Imwriter is not enabled')
        if ab is None:
            ab = ''
        img_filename = self.image_dir / 'imgs' / f'img_{self._img_idx:06d}{ab}{self.suffix}'
        cv2.imwrite(str(img_filename), np.asarray(img))

        if particles:
            particle_filename = self.image_dir / 'particles' / f'particles_{self._img_idx:06d}{ab}.json'
            particles.save_jsonld(self.image_dir / 'particles' / f'particles_{self._img_idx:06d}{ab}.json')

        self.img_filenames.append(img_filename)
        self.particle_filenames.append(particle_filename)
        self._img_idx += 1
        return img_filename

    def writeA(self, imgA: np.ndarray, particles: Particles = None):
        """Write image A"""
        return self.write(imgA, 'A', particles)

    def writeB(self, imgB: np.ndarray, particles: Particles = None):
        """Write image B"""
        return self.write(imgB, 'B', particles)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Done')
        # we might want to rename the filenames
        n_filenames = len(self.img_filenames)
        n_digits = len(str(n_filenames))
        for i, fn in enumerate(self.img_filenames):
            ab = fn.stem[-1]
            new_img_dn = fn.with_name(f'img_{i:0{n_digits}d}{ab}{self.suffix}')
            fn.rename(new_img_dn)
            self.img_filenames[i] = new_img_dn

        for i, fn in enumerate(self.particle_filenames):
            ab = fn.stem[-1]
            new_part_dn = fn.with_name(f'particles_{i:0{n_digits}d}{ab}.json')
            fn.rename(new_part_dn)
            self.particle_filenames[i] = new_part_dn

        self._enabled = False


class HDF5Writer:
    def __init__(self,
                 filename: Union[str, pathlib.Path],
                 image_dataset_name: str = 'images',
                 particle_dataset_name: str = 'particles',
                 overwrite: bool = False,
                 camera: Camera = None,
                 laser: Laser = None):
        """
        Save images and metadata to a HDF5 file.

        Parameters
        ----------
        filename : Union[str, pathlib.Path]
            The filename of the HDF5 file

        """
        self.filename = pathlib.Path(filename)
        self.image_dataset_name = image_dataset_name
        self.particle_dataset_name = particle_dataset_name
        self.overwrite = overwrite
        self.camera = camera
        self.laser = laser
        self._h5 = None
        self._image_index_a = 0
        self._image_index_b = 0

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

    def _get_dsA(self):
        ds_nameA = f"{self.particle_dataset_name}_A"
        if ds_nameA in self._h5:
            return self._h5[ds_nameA]
        return self._h5.create_dataset(ds_nameA,
                                       shape=(1, self.camera.ny, self.camera.nx),
                                       maxshape=(None, None, None),
                                       dtype='uint16')

    def _get_dsB(self):
        ds_nameA = f"{self.particle_dataset_name}_B"
        if ds_nameA in self._h5:
            return self._h5[ds_nameA]
        return self._h5.create_dataset(ds_nameA,
                                       shape=(1, self.camera.ny, self.camera.nx),
                                       maxshape=(None, None, None),
                                       dtype='uint16')

    def writeA(self, imgA: np.ndarray, particles: Particles = None):
        """Write image A"""
        dsname = f"{self.particle_dataset_name}_A"
        if dsname not in self._h5:
            ds = self._get_dsA()
        else:
            ds = self._h5[dsname]
            ds.resize((self._image_index_a + 1, *ds.shape[1:]))
        ds[self._image_index_a, ...] = imgA
        self._image_index_a += 1

    def writeB(self, imgB: np.ndarray, particles: Particles = None):
        """Write image B"""
        dsname = f"{self.particle_dataset_name}_A"
        if dsname not in self._h5:
            ds = self._create_dataset_a()
        else:
            ds = self._h5[dsname]
            ds.resize((self._image_index_a + 1, *ds.shape[1:]))
        ds[self._image_index_a, ...] = imgB
        self._image_index_a += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        ds = self._get_dsA()
        n_imgs = ds.shape[0]
        img_idx_ds = self._h5.create_dataset('image_index', data=np.arange(n_imgs))
        img_idx_ds.make_scale()
        ds.dims[0].attach_scale(img_idx_ds)
        nx_ds = self._h5.create_dataset('nx', data=np.arange(self.camera.nx))
        nx_ds.make_scale()
        ds.dims[2].attach_scale(nx_ds)
        ny_ds = self._h5.create_dataset('ny', data=np.arange(self.camera.ny))
        ny_ds.make_scale()

        if self.camera:
            camera_gr = self._h5.create_group('camera')
            for k, v in self.camera.model_dump().items():
                if v is None:
                    camera_gr.create_dataset(k, data='None', dtype='S5')
                else:
                    camera_gr.create_dataset(k, data=v, dtype='float32')

        if self.laser:
            laser_gr = self._h5.create_group('laser')
            for k, v in self.laser.model_dump().items():
                laser_gr.create_dataset(k, data=v, dtype='float32')

        self._h5.close()
        self._h5 = None
