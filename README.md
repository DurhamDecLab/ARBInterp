ARBTools
=======

Introduction
------------

Python tools for interpolation of gridded data, either:

Tricubic interpolation of 3D gridded data, based on the scheme by Lekien and Marsden: https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.1296
Takes in gridded data from comma-separated input file, either a scalar field U as an N x 4 (x,y,z,U) array or a vector field B as an N x 6 (x, y, z, Bx, By, Bz) array.

or:

Quadcubic interpolation of 4D gridded data. Paper to follow about this.
Takes in gridded data from comma-separated input file, either a scalar field U as an N x 5 (x,y,z,t,U) array or a vector field B as an N x 7 (x, y, z, t, Bx, By, Bz) array.


Package Dependencies
--------------------
- NumPy: Required.

Usage
-----

Instantiate the 'tricubic' / 'quadcubic' class by passing a loaded field to it. It is not necessary to order the (x,y,z) / (x,y,z,t) coordinates - the code will do that itself.

NOTE: a regular field (i.e. with equal grid spacing in all dimensions) must be supplied. The spacing along all axes does not have to match, only be consistent along each.

The (3D) interpolant field does not have to be cubic, only cuboid.

If an Nx4 (3D) or Nx5 (4D) array is passed the tricubic / quadcubic class will operate in scalar mode and return both the interpolated field magnitude and partial derivatives. 

If an Nx6 (3D) or Nx7 (4D) array is passed the tricubic / quadcubic class will accept an optional kword argument 'mode' with one of the following switches:

'norm' - this takes the norm of the vector field as the interpolant. Magnitude and gradient returned.
'vector' - interpolates the x, y and z field components separately and returns them together.
'both' - returns the interpolated magnitude and gradients of the norm of the field, AND the vector components.

If no argument is passed it will default to 'vector'. If a 'mode' kword is passed but an N x 4 array is detected, the kword will be ignored.

The optional argument "quiet" can be passed, which suppresses screen print messages when the interpolation is activated. 

To query the interpolant field call the "Query" method. Input can be a single set of Cartesian coordinates (x,y,z) / (x,y,z,t) or a range of points as an array.
If a range, the first 3/4 columns are assumed to be the (x,y,z) / (x,y,z,t) coordinates. Further columns can be present e.g. velocity components - these are ignored.

Due to the boundary conditions the interpolatable volume is slightly smaller than the input field. A query at the edge will return a 'nan'. Further work will implement boundary conditions of the Neumann or Dirichlet types.

Included Files
--------------

ARBInterp.py - contains tricubic and quadcubic classes and methods for querying interpolant field.

ARB3DInterpExample.py - loads an example 3D field and queries it with sample coordinates.

Example3DVectorField.csv - part of a magnetic field, as vector components, (x, y, z, Bx, By, Bz).

Example3DScalarField.csv - the same field as Example3DVectorField, but the norm / magnitude, (x,y,z,U).

ARB4DInterpExample.py - loads an example time-varying 4D field and queries it with sample coordinates.

Example4DVectorField.csv - part of a magnetic field, as vector components, (x, y, z, t, Bx, By, Bz).

Example4DScalarField.csv - the same field as Example4DVectorField, but the norm / magnitude, (x,y,z,t,U).

ARBTrajec.py - contains function for creating random atom samples, and tracking their motion through a magnetic field with the interpolator.

ARBTrajecExample.py - creates test sample, saves it, iterates it through test field, saves output. Note: particles may drift out of the interpolation volume, in which case a 'nan' is returned in their place. Note 4D trajectories are not implemented yet but can be easily done.


Installation
------------

Download zip file from dist folder and extract. On Linux run "sudo python setup.py install" or similar. On Windows / Mac - you'll have a better idea than me!
