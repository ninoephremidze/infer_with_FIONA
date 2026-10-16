"""infer_with_fiona: Fisher forecasts and Bayesian inference for gravitational-wave lensing,
with the lensing amplification factor F(w, y, l) evaluated by FIONA.

Subpackages and modules
  fiona/          the FIONA wave-optics solver (vendored, unmodified)
  fisher/         Fisher-matrix forecasts (waveforms, noise curves, derivatives of F)
  quadrature/     Gauss-Legendre node schedules n(w) and the node cache
  table           the interpolation table for F(x_c, y, w), the bGO switch and the prior mask
  amplification   HybridF, the F used inside the likelihood, with the LAL Fourier convention
                  and first-image time reference
  posteriors      loaders for the shipped posterior samples and for bilby results
  dm_conversions  lens parameters to dark-matter parameters (SIDM cross section, axion mass)
  paths           where data, caches and outputs live
"""
