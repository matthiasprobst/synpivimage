try:
    from importlib.metadata import version as _version

    __version__ = _version('pivparticledensity')
except ImportError:
    import warnings

    warnings.warn('Consider upgrading to python3.8!')
    __version__ = '999'  # unknown
