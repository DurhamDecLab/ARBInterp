from __future__ import division
import numpy as np
from ARBTools.ARBInterp import tricubic

######################################################################
	
if __name__ == '__main__':
	fieldname = "ExampleScalarField"

	print ("--- Loading field ---")
	field = np.genfromtxt(fieldname+'.csv', delimiter=',')
	
	Run = tricubic(field)	# mode kword arg will be ignored if a scalar (Nx4) input is detected

	coords=np.zeros((20,3))
	coords[:,0]=np.linspace(-2e-3,2e-3,20)
	coords[:,1]=np.linspace(-2e-3,2e-3,20)
	coords[:,2]=np.linspace(-2e-3,2e-3,20)
	
	output = Run.Query((coords[3]))
	print '\n'
	print 'Single point query, scalar field:'
	print output
	print '\n'
	
	Comps = Run.Query(coords)
	print '\n'
	print 'Multi point query, scalar field:'
	print (Comps)

	
	fieldname = "ExampleVectorField"

	print ("--- Loading field ---")
	field = np.genfromtxt(fieldname+'.csv', delimiter=',')
	
	Run = tricubic(field, mode='both')	# mode options are 'both', 'vector' or 'norm', defaults to 'vector'. Pass arg 'quiet' to suppress setup text

	output = Run.Query((coords[3]))
	print 'Single point query, vector field:'
	print '\n'
	print output
	print '\n'
	
	Comps = Run.Query(coords)
	print 'Multi point query, vector field:'
	print '\n'
	print (Comps)