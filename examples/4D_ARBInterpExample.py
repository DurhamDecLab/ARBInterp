import numpy as np
from ARBTools.ARBInterp import quadcubic
from sys import platform

######################################################################
	
if __name__ == '__main__':
	if platform == "win32":
		slash = "\\"
	elif platform == "linux" or "linux2":
		slash = "/"

	print ("--- Loading field ---")
	field=np.genfromtxt("ExampleFields"+slash+"Example4DScalarField.csv", delimiter=',')
	
	Run = quadcubic(field)	# mode kword arg will be ignored if a scalar (Nx5) input is detected

	coords=np.zeros((20,4))
	coords[:,0]=np.linspace(-2e-3,2e-3,20)
	coords[:,1]=np.linspace(-2e-3,2e-3,20)
	coords[:,2]=np.linspace(-2e-3,2e-3,20)
	coords[:,3]=np.linspace(-3e-6,3e-6,20)
	
	output = Run.Query((coords[3]))
	print ('\n')
	print ('Single point query, scalar field:')
	print (output)
	print ('\n')
	
	Comps = Run.Query(coords)
	print ('\n')
	print ('Multi point query, scalar field:')
	print (Comps)

######################################################################

	print ("--- Loading field ---")
	field=np.genfromtxt("ExampleFields"+slash+"Example4DVectorField.csv", delimiter=',')
	
	Run = quadcubic(field, mode='both')	# mode options are 'both', 'vector' or 'norm', defaults to 'vector'. Pass arg 'quiet' to suppress setup text

	output = Run.Query((coords[3]))
	print ('Single point query, vector field:')
	print ('\n')
	print (output)
	print ('\n')
	
	Comps = Run.Query(coords)
	print ('Multi point query, vector field:')
	print ('\n')
	print (Comps)
