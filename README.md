# ARBInterp
Tricubic spline interpolation for vector and scalar fields

ARBTools
=======

Introduction
------------

Python tools for tricubic interpolation of 3D gridded data, based on the scheme by Lekien and Marsden: https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.1296

Takes in gridded data from comma-separated input file, either a scalar field U as an N x 4 (x,y,z,U) array or a vector field B as an N x 6 (x, y, z, Bx, By, Bz) array.

Package Dependencies
--------------------
- NumPy: Required.

Usage
-----

Instantiate the 'tricubic' class by passing a loaded field to it. It is not necessary to order the x, y, z coordinates - the code will do that itself.

NOTE: as of version 1.0, a regular field (i.e. with equal grid spacing in all dimensions) must be supplied. A future release will handle irregular grids, but performance will still be best with regular grids. The interpolant field does not have to be cubic, only cuboid.

If an N x 4 array is passed it will operate in scalar mode and return both the interpolated field magnitude and partial derivatives. 

If an N x 6 array is passed it will accept an optional kword argument 'mode' with one of the following switches:

'norm' - this takes the norm of the vector field as the interpolant. Magnitude and gradient returned.

'vector' - interpolates the x, y and z field components separately and returns them together.

'both' - returns the interpolated magnitude and gradients of the norm of the field, AND the vector components.

If no argument is passed it will default to 'vector'. If a 'mode' kword is passed but an N x 4 array is detected, the kword will be ignored.

There are two methods for querying the interpolant field:

sInterp - takes a single Cartesian coordinate.

rInterp - takes a range of Cartesian coordinates as an array. The first 3 columns are assumed to be the x,y,z coordinates. Further columns can be present e.g. velocity components - these are ignored.

Included Files
--------------

ARBInterp.py - contains tricubic class and methods for querying interpolant field.

ARBExample.py - loads an example field and queries it with sample coordinates.

ExampleVectorField.csv - part of a magnetic field, as vector components.

ExampleScalarField.csv - the same field as ExampleVectorField, but the norm / magnitude.



https://zenodo.org/badge/167368876.svg

