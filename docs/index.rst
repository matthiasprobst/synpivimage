SynPivImage: Synthetic Particle Image Generator
===============================================

With synthetic data for Particle Image Velocimetry (PIV) you can test your PIV algorithms and software and explore
the influence of different parameters on the PIV results. Unlike existing codes, this package focuses on transparent
full control of all parameters during the generation of the synthetic images.

Release status: alpha (`1.0.0a*`) with an upcoming stable `1.0.0`.
The core API that will remain stable is `Camera`, `Laser`, `Particles`, `take_image`, `Imwriter`, and `HDF5Writer`.


.. toctree::
    :maxdepth: 2

    simulation_overview
    user_controls
    model_scope
    getting_started/index
    api.rst
