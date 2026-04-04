User Controls
=============

This page summarizes what you can control in the simulation and how each group of parameters affects results.

Quick reference
---------------

.. list-table:: Stage-to-control mapping
   :header-rows: 1
   :widths: 25 40 35

   * - Simulation stage
     - Main controls
     - Primary effect
   * - Illumination
     - `Laser.width`, `Laser.shape_factor`
     - Out-of-plane intensity decay
   * - Particle image projection
     - `particle_image_diameter`, `fill_ratio_x`, `fill_ratio_y`,
       `focus_plane_z`, `defocus_strength`
     - Particle image size and sharpness
   * - Sensor model
     - `qe`, `shot_noise`, `baseline_noise`, `dark_noise`
     - Noise floor and signal variability
   * - Digitization
     - `sensitivity`, `bit_depth`
     - ADU scaling, saturation, quantization
   * - Acquisition calibration
     - `particle_peak_count`
     - Target peak-count reference level

Camera controls
---------------

`Camera(...)` controls the sensor and image formation:

- **Geometry:** `nx`, `ny` (sensor resolution in pixels)
- **Digital dynamic range:** `bit_depth`
- **Radiometric conversion:** `qe`, `sensitivity`
- **Noise model:** `shot_noise`, `baseline_noise`, `dark_noise`
- **Pixel integration:** `fill_ratio_x`, `fill_ratio_y`
- **Particle image size:** `particle_image_diameter`
- **Defocus behavior:** `focus_plane_z`, `defocus_strength`
- **Reproducibility:** `seed`

Laser controls
--------------

`Laser(...)` controls illumination in the out-of-plane direction:

- `width`: laser sheet thickness
- `shape_factor`: profile sharpness (Gaussian-like to top-hat-like)

Particle controls
-----------------

`Particles(...)` gives direct control over particle ensembles:

- `x`, `y`, `z`: positions
- `size`: particle diameter proxy used for image intensity distribution

Useful helpers:

- `Particles.generate(...)` for target particle density (`ppp`)
- `Particles.displace(dx, dy, dz)` to create frame B from frame A

Scene and acquisition controls
------------------------------

`take_image(laser, camera, particles, particle_peak_count=...)` sets acquisition-level behavior:

- `particle_peak_count`: target peak count calibration reference
- Camera noise settings determine detectability at low illumination
- Particle flags track in-FOV, illuminated, and out-of-plane states

Output controls
---------------

- `Imwriter(...)` for TIF + JSON-LD output
- `HDF5Writer(...)` for compact HDF5 datasets with image and particle channels

Recommended workflow
--------------------

1. Set camera and laser first.
2. Generate particles with realistic `ppp` and displacement ranges.
3. Tune `particle_peak_count`, `dark_noise`, and `particle_image_diameter` together.
4. Validate with your target PIV algorithm using both ideal and noisy settings.
