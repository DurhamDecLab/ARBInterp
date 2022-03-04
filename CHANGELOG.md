# Changelog

<!--next-version-placeholder-->

## v1.8.0 (10/10/2021)

- Skipping v1.7 since I never got around to uploading it.
- I've dropped the atom trajectory stuff because it's pretty outdated and not that interesting.
- I have made some minor changes to how the code determines when it needs to calculate coefficients for a volume element or not; in some cases they may have been calculated more than once, wasting resources. There is also a new "allCoeffs" method to calculate the entire coefficients matrix, which may be useful for cases where the interpolant field doesn't change and we will re-use it (this is not a good idea for a use case where you are frequently updating the field and only sampling part of it). The next version might include an option to save these coefficients to a file for later use.

## v1.5.0 (04/10/2020) - now 4 dimensional!

- A few minor improvements to the existing tricubic class for interpolation of 3D fields - no changes to functionality.
- Now includes quadcubic class for interpolation of 4D fields - this is set up for time-dependent spatially 3D fields but could easily be adjusted to work with fields with four spatial components.

## v1.3.0 (15/02/2019)

- Minor improvements. Added optional "quiet" mode to the interpolator to suppress screen messages. Added new "Query" method to interpolate the field, which detects length of input array and handles it appropriately. Old "sQuery" and "rQuery" methods still work.

## v1.0.0 (24/01/2019)

- First release of tricubic interpolator. Designed for both scalar and vector fields, gradients are implemented. Methods for querying at single coordinates or multiple simultaneous coordinates. Created for simulating particle motion in magnetic fields but will operate with any suitably-formatted input. Future releases to incorporate numerical integrators for particle motion.