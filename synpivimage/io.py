import json
import pathlib
import shutil
from typing import Dict, Union, Optional
from typing import Literal

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
        print('Preparing the folder')
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

    def write(self, img: np.ndarray, ab: Literal['A', 'B'], particles: Particles = None):
        print(f'writing {ab} image')
        if not self._enabled:
            raise ValueError('Imwriter is not enabled')
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


# def imwriteA(case_name: str,
#              imgs: Union[np.ndarray, List[np.ndarray]],
#              image_dir: Optional[Union[str, pathlib.Path]] = None,
#              suffix: str = '.tif',
#              overwrite: bool = False,
#              camera: Camera = None,
#              laser: Laser = None,
#              particles: Particles = None) -> List[pathlib.Path]:
#     """Write an A-image to a file. Calls cv2.imwrite.
#     The images will be written into the folder image_dir/case_name.
#     The images will be named imgA_000.tif, imgA_001.tif, etc.
#
#     Parameters
#     ----------
#     case_name : str
#         Inside the image_dir, a folder where all images and metadata will be written to.
#     imgs : Union[np.ndarray, List[np.ndarray]]
#         Image(s) to write to the file.
#     image_dir : str or pathlib.Path = None
#         Where to store the image(s). If None, the current working directory is used.
#     suffix : str= '.tif'
#         Suffix for the image file.
#     overwrite : bool
#         If True, overwrite the file if it exists.
#         Default is False.
#     camera : Camera
#         Camera object to write to the file.
#     laser : Laser
#         Laser object to write to the file.
#     particles : Particles
#         Particles object to write to the file.
#
#     Returns
#     -------
#     filenames: List[pathlib.Path]
#         Path to the image files written.
#     """
#     if image_dir is None:
#         image_dir = pathlib.Path.cwd() / case_name
#     else:
#         image_dir = pathlib.Path(image_dir) / case_name
#
#     if image_dir.exists() and not overwrite:
#         raise FileExistsError(f"Directory {image_dir} exists and overwrite is False")
#     if image_dir.exists() and overwrite:
#         shutil.rmtree(image_dir)
#
#     image_dir.mkdir(parents=True, exist_ok=True)
#     (image_dir / 'imgs').mkdir(parents=True, exist_ok=True)
#     (image_dir / 'meta').mkdir(parents=True, exist_ok=True)
#
#     if not isinstance(imgs, (list, tuple)):
#         imgs = [imgs]
#
#     n_imgs = len(imgs)
#     n_digits = len(str(n_imgs))
#     img_filenames = []
#     for i, img in enumerate(imgs):
#         img_filename = image_dir / f'imgs/imgA_{i:0{n_digits}d}{suffix}'
#         cv2.imwrite(str(img_filename), img)
#         img_filenames.append(img_filename)
#
#     if camera:
#         camera.save_jsonld(image_dir / 'meta/camera.json')
#     if laser:
#         laser.save_jsonld(image_dir / 'meta/laser.json')
#     if particles:
#         if not isinstance(particles, (list, tuple)):
#             particles = [particles]
#         if len(particles) != n_imgs:
#             raise ValueError("The number of particle objects must match the number of images")
#         for i, p in enumerate(particles):
#             p.save_jsonld(image_dir / f'meta/particlesA_{i:0{n_digits}d}.json')
#
#     return img_filenames
#
#     # plib_filename = parse_filename(filename, overwrite)
#     # cv2.imwrite(str(filename), img)
#     #
#     # if camera:
#     #     camera.save_jsonld(plib_filename.with_name(plib_filename.stem + '_camera'))
#     #
#     # piv_setup = {'camera': camera, 'laser': laser, 'particles': particles}
#     # for k, v in piv_setup.items():
#     #     if v:
#     #         v.save_jsonld(plib_filename.with_name(plib_filename.stem + f'_{k}'))
#
#     # metadata.update(kwargs)
#     # metawrite(plib_filename.with_suffix('.json'),
#     #           metadata,
#     #           format='json-ld')
#
#     return plib_filename


def imwrite16(filename,
              img,
              overwrite: bool = False,
              camera: Camera = None,
              laser: Laser = None,
              particles: Particles = None,
              **kwargs):
    """Write an image to a 16 bit file"""
    return imwrite(filename,
                   img.astype(np.uint16),
                   overwrite,
                   camera=camera,
                   laser=laser,
                   particles=particles,
                   **kwargs)


def imwrite8(filename,
             img,
             overwrite: bool = False,
             meta: Dict = None,
             **kwargs):
    """Write an image to a 8 bit file"""
    return imwrite(filename,
                   img.astype(np.uint8),
                   overwrite,
                   meta=meta,
                   **kwargs)


def imread(filename, flags=cv2.IMREAD_GRAYSCALE):
    """Read an image from a file. Calls cv2.imread!

    Parameters
    ----------
    filename : str
        Name of the file.
    flags : int
        Flags for reading the image.

    Returns
    -------
    ndarray
        Image read from the file.
    """
    if not pathlib.Path(filename).exists():
        raise FileNotFoundError(f"File {filename} not found")
    return cv2.imread(str(filename), flags)


def imread16(filename):
    """Read an image from a 16 bit file"""
    return imread(filename, -1).astype(np.uint16)


def imread8(filename, flags=cv2.IMREAD_GRAYSCALE):
    """Read an image from a 8 bit file"""
    return imread(filename, flags).astype(np.uint8)


def metaread(filename) -> Dict:
    """Read metadata from a image or json file. if image file is given, it looks for a .json file with the same name"""
    filename = pathlib.Path(filename)
    if filename.suffix == '.json':
        meta_filename = filename
    else:
        meta_filename = pathlib.Path(filename).with_suffix('.json')
    if not meta_filename.exists():
        raise FileNotFoundError(f"File {meta_filename} not found")
    with open(meta_filename, 'r') as f:
        return json.load(f)

# def metawrite(filename: Union[str, pathlib.Path], metadata: Dict):
#     """Write metadata to a json file"""
#     filename = pathlib.Path(filename)
#     meta_filename = filename.with_suffix('.json')
#     with open(meta_filename, 'w') as f:
#         json.dump(metadata, f, indent=4)
