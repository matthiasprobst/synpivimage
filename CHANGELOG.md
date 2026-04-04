# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-04-03

### Added
- Scientific validation test suite in `tests/test_scientific_validation.py` covering:
  - stochastic noise reproducibility and sequence progression,
  - dark-noise RNG control and unclipped Gaussian behavior,
  - continuous out-of-plane illumination behavior,
  - polydisperse peak-count calibration,
  - defocus impact on particle-image peak intensity,
  - laser metadata unit consistency.
- New documentation pages:
  - `docs/simulation_overview.rst`
  - `docs/user_controls.rst`
  - `docs/model_scope.rst`
  explaining simulation workflow, user-controllable parameters, and modeling limitations.
- New documentation diagrams:
  - `docs/_static/piv_pipeline.svg` (end-to-end workflow)
  - `docs/_static/image_formation_chain.svg` (internal image-formation chain)

### Changed
- Camera noise modeling:
  - switched to per-camera persistent RNG state for physically consistent frame sequences,
  - removed mixed global/local randomness in camera capture path,
  - clipped negative electrons before quantization to avoid unsigned wrap-around artifacts.
- Noise model (`synpivimage/noise.py`):
  - added explicit RNG injection for shot and dark noise,
  - dark noise now uses the provided RNG source,
  - removed pre-quantization clipping of dark noise samples.
- Image formation (`synpivimage/camera.py`, `synpivimage/particles.py`):
  - added optional defocus model via `focus_plane_z` and `defocus_strength`,
  - enabled per-particle `sigmax/sigmay` during rendering.
- Core intensity calibration (`synpivimage/core.py`):
  - peak-count scaling now uses the strongest particle response in polydisperse sets
    instead of mean particle size,
  - replaced hard `exp(-2)` gating with a continuous detectability threshold tied to
    noise floor and minimum detectable counts.
- Metadata consistency:
  - laser sheet thickness metadata now uses consistent millimeter units in standard-name
    and unit fields.
- Documentation navigation:
  - updated `docs/index.rst` and `docs/getting_started/index.rst` to include the new
    simulation and user-control guidance.
  - updated `README.md` with direct links to the new documentation pages.
  - expanded `docs/user_controls.rst` with a stage-to-parameter quick-reference table.
