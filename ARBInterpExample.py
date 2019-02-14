from __future__ import division
import numpy as np
from ARBTools.ARBInterp import tricubic

######################################################################
	
if __name__ == '__main__':
	fieldname = "ExampleScalarField"

	print ("--- Loading field ---")
	field = np.genfromtxt(fieldname+'.csv', delimiter=',')
	
	Run = tricubic(field)

	coords=np.zeros((20,3))
	coords[:,0]=np.linspace(-2e-3,2e-3,20)
	coords[:,1]=np.linspace(-2e-3,2e-3,20)
	coords[:,2]=np.linspace(-2e-3,2e-3,20)
	
	output = Run.sQuery((coords[3]))
	print output
	print '\n'
	
	Comps = Run.rQuery(coords)
	print (Comps)

	
	fieldname = "ExampleVectorField"

	print ("--- Loading field ---")
	field = np.genfromtxt(fieldname+'.csv', delimiter=',')
	
	Run = tricubic(field, mode='both')

	output = Run.sQuery((coords[3]))
	print output
	print '\n'
	
	Comps = Run.rQuery(coords)
	print (Comps)