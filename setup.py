import setuptools

from synpivimage._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synpivmage",
    version=__version__,
    author="Matthias Probst",
    author_email="matthias.probst@kit.edu",
    description="Tool to build synthetic Particle Image Velocimetry (PIV) images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatthiasProbst/synpivimage",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_require=['numpy',
                     'pandas',
                     'multiprocessing',
                     'pathlib',
                     'h5py',
                     'xarray',
                     'tifffile',
                     'yaml',
                     'tqdm'
                     ],
)
