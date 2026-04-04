Model Scope and Limitations
===========================

`synpivimage` is designed for controlled synthetic PIV experiments. It is scientifically useful, but users should
understand which effects are modeled explicitly and which are simplified.

Modeled explicitly
------------------

- Particle illumination through a parametric laser sheet profile
- Pixel-integrated Gaussian particle images
- Sensor conversion and quantization (`qe`, `sensitivity`, `bit_depth`)
- Shot noise and Gaussian dark/read noise
- Optional defocus blur with `focus_plane_z` and `defocus_strength`
- Particle visibility status (in FOV, illuminated, out-of-plane, disabled)

Simplified or not fully modeled
-------------------------------

- Detailed optical aberrations and full diffraction pipeline
- Multiple scattering and non-linear light-matter interactions
- Complex camera non-idealities beyond the current noise/quantization chain
- Experimental artifacts from optics, calibration, and alignment errors

How to use the model responsibly
--------------------------------

1. Treat the package as a controllable benchmark generator, not a complete replica of every lab setup.
2. Match parameter ranges to your experimental hardware before drawing quantitative conclusions.
3. Perform sensitivity studies across `ppp`, particle image diameter, laser thickness/shape, and noise.
4. Validate end-to-end with your own PIV processing chain and report the synthetic assumptions used.
