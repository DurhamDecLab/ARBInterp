import numpy as np
import sys

#######################release the schmoo########################################
# Version 1.8, L4 code base
__version__="1.8"

class tricubic:
	__version__="1.8"
	def __init__(self, field, *args, **kwargs):
		self.eps = 10*np.finfo(float).eps # Machine precision of floating point number
		## Load field passed to class - can be x,y,z or norm of vector field, or scalar
		self.inputfield = field			
		## Analyse field, get shapes etc
		self.getFieldParams()
		## Make coefficient matrix
		if not hasattr(self, 'A'):
			self.makeAMatrix()

		### Mask to identify where coefficients exist
		self.alphamask = np.zeros((self.nc+1, 1 ))
		self.alphamask[-1] = 1
		## Determining which mode to run in
		if self.inputfield.shape[1] == 4:
			if not 'quiet' in args:
				print ('--- Scalar field, ignoring switches, interpolating for magnitude and gradient --- \n')
			self.Query = self.Query2
			self.sQuery = self.sQuery2
			self.rQuery = self.rQuery2
			self.calcCoefficients = self.calcCoefficients2
			self.alphan = np.zeros((64, self.nc+1 ))
			self.Bn = self.inputfield[:,3]
		elif self.inputfield.shape[1] == 6:
			if 'mode' in kwargs:
				if kwargs['mode'] == 'vector':
					if not 'quiet' in args:
						print ('--- Vector field, interpolating for vector components --- \n')
					self.Query = self.Query1
					self.sQuery = self.sQuery1
					self.rQuery = self.rQuery1
					self.calcCoefficients = self.calcCoefficients1
					self.alphax = np.zeros((64, self.nc+1+1 ))
					self.alphay = np.zeros((64, self.nc+1+1 ))
					self.alphaz = np.zeros((64, self.nc+1+1 ))
					self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = np.nan
					self.Bx = self.inputfield[:,3]
					self.By = self.inputfield[:,4]
					self.Bz = self.inputfield[:,5]
				elif kwargs['mode'] == 'norm':
					if not 'quiet' in args:
						print ('--- Vector field, interpolating for magnitude and gradient --- \n')
					self.Query = self.Query2
					self.sQuery = self.sQuery2
					self.rQuery = self.rQuery2
					self.calcCoefficients = self.calcCoefficients2
					self.alphan = np.zeros((64, self.nc+1 ))
					self.alphan[:,-1] = np.nan
					self.Bn = np.linalg.norm(self.inputfield[:,3:], axis=1)
				elif kwargs['mode'] == 'both':
					if not 'quiet' in args:
						print ('--- Vector field, interpolating vector components plus magnitude and gradient --- \n')
					self.Query = self.Query3
					self.sQuery = self.sQuery3
					self.rQuery = self.rQuery3
					self.calcCoefficients = self.calcCoefficients3
					self.alphax = np.zeros((64, self.nc+1 ))
					self.alphay = np.zeros((64, self.nc+1 ))
					self.alphaz = np.zeros((64, self.nc+1 ))
					self.alphan = np.zeros((64, self.nc+1 ))
					self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = self.alphan[:,-1] = np.nan
					self.Bx = self.inputfield[:,3]
					self.By = self.inputfield[:,4]
					self.Bz = self.inputfield[:,5]
					self.Bn = np.linalg.norm(self.inputfield[:,3:], axis=1)
				else:
					if not 'quiet' in args:
						print ('--- Vector field, invalid option, defaulting to interpolating for vector components --- \n')
					self.Query = self.Query1
					self.sQuery = self.sQuery1
					self.rQuery = self.rQuery1
					self.calcCoefficients = self.calcCoefficients1
					self.alphax = np.zeros((64, self.nc+1 ))
					self.alphay = np.zeros((64, self.nc+1 ))
					self.alphaz = np.zeros((64, self.nc+1 ))
					self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = np.nan
					self.Bx = self.inputfield[:,3]
					self.By = self.inputfield[:,4]
					self.Bz = self.inputfield[:,5]
			else:
				if not 'quiet' in args:
					print ('--- Vector field, no option selected, defaulting to interpolating for vector components --- \n')
				self.Query = self.Query1
				self.sQuery = self.sQuery1
				self.rQuery = self.rQuery1
				self.calcCoefficients = self.calcCoefficients1
				self.alphax = np.zeros((64, self.nc+1 ))
				self.alphay = np.zeros((64, self.nc+1 ))
				self.alphaz = np.zeros((64, self.nc+1 ))
				self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = np.nan
				self.Bx = self.inputfield[:,3]
				self.By = self.inputfield[:,4]
				self.Bz = self.inputfield[:,5]
		else:
			sys.exit('--- Input not shaped as expected - should be N x 4 or N x 6 ---')
			
			
	def makeAMatrix(self):
		### Creates tricubic interpolation matrix and finite difference matrix and combines them
		### Interpolation matrix
		corners=np.array(([[i,j,k] for k in range(2) for j in range(2) for i in range(2)])).astype(float).T
		exp = [[i,j,k] for k in range(4) for j in range(4) for i in range(4)]
		B = np.zeros((64,64), dtype=np.float64)

		for i in range(64):
			ex,ey,ez = exp[i][0], exp[i][1], exp[i][2]    # 
			for k in range(8):
				x,y,z=corners[0,k],corners[1,k],corners[2,k]
				B[0*8+k,i] = x**ex * y**ey * z**ez
				B[1*8+k,i] = ex*x**(abs(ex-1)) * y**ey * z**ez
				B[2*8+k,i] = x**ex * ey*y**(abs(ey-1)) * z**ez
				B[3*8+k,i] = x**ex * y**ey * ez*z**(abs(ez-1))
				B[4*8+k,i] = ex*x**(abs(ex-1)) * ey*y**(abs(ey-1)) * z**ez
				B[5*8+k,i] = ex*x**(abs(ex-1)) * y**ey * ez*z**(abs(ez-1))
				B[6*8+k,i] = x**ex * ey*y**(abs(ey-1)) * ez*z**(abs(ez-1))
				B[7*8+k,i] = ex*x**(abs(ex-1)) * ey*y**(abs(ey-1)) * ez*z**(abs(ez-1))

		# This makes a finite-difference matrix to return the components of the "b"-vector 
		# needed in the alpha calculation, in "cuboid" coordinates
		C=np.array((21,22,25,26,37,38,41,42))
		D=np.zeros((64,64))

		for i in range(8):
			D[i,C[i]] = 1

		for i,j in enumerate(range(8,16,1)):
			D[j,C[i]-1] = -0.5
			D[j,C[i]+1] = 0.5

		for i,j in enumerate(range(16,24,1)):
			D[j,C[i]-4] = -0.5
			D[j,C[i]+4] = 0.5

		for i,j in enumerate(range(24,32,1)):
			D[j,C[i]-16] = -0.5
			D[j,C[i]+16] = 0.5

		for i,j in enumerate(range(32,40,1)):
			D[j,C[i]+5] = 0.25
			D[j,C[i]-3] = -0.25
			D[j,C[i]+3] = -0.25
			D[j,C[i]-5] = 0.25

		for i,j in enumerate(range(40,48,1)):
			D[j,C[i]+17] = 0.25
			D[j,C[i]-15] = -0.25
			D[j,C[i]+15] = -0.25
			D[j,C[i]-17] = 0.25

		for i,j in enumerate(range(48,56,1)):
			D[j,C[i]+20] = 0.25
			D[j,C[i]-12] = -0.25
			D[j,C[i]+12] = -0.25
			D[j,C[i]-20] = 0.25

		for i,j in enumerate(range(56,64,1)):
			D[j,C[i]+21] = 0.125
			D[j,C[i]+13] = -0.125
			D[j,C[i]+19] = -0.125
			D[j,C[i]+11] = 0.125
			D[j,C[i]-11] = -0.125
			D[j,C[i]-19] = 0.125
			D[j,C[i]-13] = 0.125
			D[j,C[i]-21] = -0.125

		self.A = np.matmul(np.linalg.inv(B),D)

	def Query1(self, query):
		try:
			if query.shape[1] > 1:
				comps = self.rQuery1(query)
				return comps
			else:
				comps = self.sQuery1(query)
				return comps
		except IndexError:
			comps = self.sQuery1(query)
			return comps
		
	def Query2(self, query):
		try:
			if query.shape[1] > 1:
				norms, grads = self.rQuery2(query)
				return norms, grads
			else:
				norm, grad = self.sQuery2(query)
				return norm, grad
		except IndexError:
			norm, grad = self.sQuery2(query)
			return norm, grad
		
	def Query3(self, query):
		try:
			if query.shape[1] > 1:
				comps, norms, grads = self.rQuery3(query)
				return comps, norms, grads
			else:
				comps, norm, grad = self.sQuery3(query)
				return comps, norm, grad
		except IndexError:
			comps, norm, grad = self.sQuery3(query)
			return comps, norm, grad
			
	def sQuery1(self, query):
		### Removes particles that are outside of interpolation volume
		if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax:
			return np.nan
		else:
			### How many cuboids in is query point
			iu = (query[0]-self.xIntMin)/self.hx
			iv = (query[1]-self.yIntMin)/self.hy
			iw = (query[2]-self.zIntMin)/self.hz
			### Finds base coordinates of cuboid particle is in
			ix = np.floor(iu)
			iy = np.floor(iv)
			iz = np.floor(iw)
			### particle coordinates in unit cuboid
			cuboidx = iu-ix
			cuboidy = iv-iy
			cuboidz = iw-iz
			### Returns index of base cuboid
			self.queryInd = ix + iy*(self.nPos[0]) + iz*(self.nPos[0])*(self.nPos[1])
			self.queryInd = self.queryInd.astype(int)
			### Calculate alpha for cuboid if it doesn't exist
			if self.alphamask[self.queryInd]==0:
				self.calcCoefficients(self.queryInd)
			### 4-vectors for finding interpolated values
			xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
			yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
			zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
			### 4-vector summation components
			x = np.tile(xvec,16)
			y = np.tile(np.repeat(yvec,4),4)
			z = np.repeat(zvec,16)
			### interpolated values
			compx =  np.inner(self.alphax[:,self.queryInd], (x*y*z))
			compy =  np.inner(self.alphay[:,self.queryInd], (x*y*z))
			compz =  np.inner(self.alphaz[:,self.queryInd], (x*y*z))
			## Components
			return np.array((compx, compy, compz))
	
	def sQuery2(self, query):
		### Removes particles that are outside of interpolation volume
		if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax:
			return np.nan
		else:
			### How many cuboids in is query point
			iu = (query[0]-self.xIntMin)/self.hx
			iv = (query[1]-self.yIntMin)/self.hy
			iw = (query[2]-self.zIntMin)/self.hz
			### Finds base coordinates of cuboid particle is in
			ix = np.floor(iu)
			iy = np.floor(iv)
			iz = np.floor(iw)
			### particle coordinates in unit cuboid
			cuboidx = iu-ix
			cuboidy = iv-iy
			cuboidz = iw-iz
			### Returns index of base cuboid
			self.queryInd = ix + iy*(self.nPos[0]) + iz*(self.nPos[0])*(self.nPos[1])
			self.queryInd = self.queryInd.astype(int)
			### Calculate alpha for cuboid if it doesn't exist
			if self.alphamask[self.queryInd]==0:
				self.calcCoefficients(self.queryInd)
			### 4-vectors for finding interpolated values
			xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
			yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
			zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
			### 4-vectors for finding interpolated gradients
			xxvec = np.array([0, 1, 2*cuboidx, 3*cuboidx**2])
			yyvec = np.array([0, 1, 2*cuboidy, 3*cuboidy**2])
			zzvec = np.array([0, 1, 2*cuboidz, 3*cuboidz**2])
			### 4-vector summation components
			x = np.tile(xvec,16)
			y = np.tile(np.repeat(yvec,4),4)
			z = np.repeat(zvec,16)
			### 4-vector summation components
			xx = np.tile(xxvec,16)
			yy = np.tile(np.repeat(yyvec,4),4)
			zz = np.repeat(zzvec,16)
			###	interpolated values
			norm =  np.inner(self.alphan[:,self.queryInd], (x*y*z))
			grad = np.array([np.dot(self.alphan[:,self.queryInd], xx*y*z)/self.hx, np.dot(self.alphan[:,self.queryInd], x*yy*z)/self.hy, np.dot(self.alphan[:,self.queryInd], x*y*zz)/self.hz])
			## Magnitude, gradient
			return norm, grad

	def sQuery3(self, query):
		### Removes particles that are outside of interpolation volume
		if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax:
			return np.nan
		else:
			### How many cuboids in is query point
			iu = (query[0]-self.xIntMin)/self.hx
			iv = (query[1]-self.yIntMin)/self.hy
			iw = (query[2]-self.zIntMin)/self.hz
			### Finds base coordinates of cuboid particle is in
			ix = np.floor(iu)
			iy = np.floor(iv)
			iz = np.floor(iw)
			### particle coordinates in unit cuboid
			cuboidx = iu-ix
			cuboidy = iv-iy
			cuboidz = iw-iz
			### Returns index of base cuboid
			self.queryInd = ix + iy*(self.nPos[0]) + iz*(self.nPos[0])*(self.nPos[1])
			self.queryInd = self.queryInd.astype(int)
			### Calculate alpha for cuboid if it doesn't exist
			if self.alphamask[self.queryInd]==0:
				self.calcCoefficients(self.queryInd)
			### 4-vectors for finding interpolated values
			xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
			yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
			zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
			### 4-vectors for finding interpolated gradients
			xxvec = np.array([0, 1, 2*cuboidx, 3*cuboidx**2])
			yyvec = np.array([0, 1, 2*cuboidy, 3*cuboidy**2])
			zzvec = np.array([0, 1, 2*cuboidz, 3*cuboidz**2])
			### 4-vector summation components
			x = np.tile(xvec,16)
			y = np.tile(np.repeat(yvec,4),4)
			z = np.repeat(zvec,16)
			### 4-vector summation components
			xx = np.tile(xxvec,16)
			yy = np.tile(np.repeat(yyvec,4),4)
			zz = np.repeat(zzvec,16)
			###	interpolated values
			compx =  np.inner(self.alphax[:,self.queryInd], (x*y*z))
			compy =  np.inner(self.alphay[:,self.queryInd], (x*y*z))
			compz =  np.inner(self.alphaz[:,self.queryInd], (x*y*z))
			norm =  np.inner(self.alphan[:,self.queryInd], (x*y*z))
			grad = np.array([np.dot(self.alphan[:,self.queryInd], xx*y*z)/self.hx, np.dot(self.alphan[:,self.queryInd], x*yy*z)/self.hy, np.dot(self.alphan[:,self.queryInd], x*y*zz)/self.hz])
			## Components, magnitude, gradient
			return np.array((compx, compy, compz)), norm, grad
	
	def rQuery1(self, query):
		## Finds base cuboid indices of the points to be interpolated
		### Length of sample distribution ###
		N = len(query)

		### Removes particles that are outside of interpolation volume
		query[np.where(query[:,0] < self.xIntMin)[0]] = np.nan
		query[np.where(query[:,0] > self.xIntMax)[0]] = np.nan
		query[np.where(query[:,1] < self.yIntMin)[0]] = np.nan
		query[np.where(query[:,1] > self.yIntMax)[0]] = np.nan
		query[np.where(query[:,2] < self.zIntMin)[0]] = np.nan
		query[np.where(query[:,2] > self.zIntMax)[0]] = np.nan
		
		### Coords in cuboids
		iu = (query[:,0]-self.xIntMin)/self.hx
		iv = (query[:,1]-self.yIntMin)/self.hy
		iw = (query[:,2]-self.zIntMin)/self.hz

		### Finds base coordinates of cuboid particles are in ###
		ix = np.floor(iu)
		iy = np.floor(iv)
		iz = np.floor(iw)

		### Returns indices of base cuboids ###
		self.queryInds = ix + iy*(self.nPos[0]) + iz*(self.nPos[0])*(self.nPos[1])
		self.queryInds[np.where(np.isnan(self.queryInds))]=self.nc
		self.queryInds = self.queryInds.astype(int)

		### Coordinates of the sample in unit cuboid
		queryCoords = np.stack((iu-ix,iv-iy,iw-iz),axis=1)

		## Returns the interpolate magnitude and / or gradients at the query coordinates
		if len(self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]) > 0:
			list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]))
		
		# Calculate interpolated values
		x = np.tile(np.transpose(np.array([np.ones(N), queryCoords[:,0], queryCoords[:,0]**2, queryCoords[:,0]**3])), 16)
		y = np.tile(np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:,1], queryCoords[:,1]**2, queryCoords[:,1]**3])), 4, axis=1),4)
		z = np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:,2], queryCoords[:,2]**2, queryCoords[:,2]**3])), 16, axis=1)

		# Return coefficient matrix values, give NaN for invalid locations
		tx = self.alphax[:, self.queryInds]
		tx = np.transpose(tx)
		ty = self.alphay[:, self.queryInds]
		ty = np.transpose(ty)
		tz = self.alphaz[:, self.queryInds]
		tz = np.transpose(tz)

		# Return components
		compsx = np.reshape((tx * (x*y*z)).sum(axis=1), (N,1))
		compsy = np.reshape((ty * (x*y*z)).sum(axis=1), (N,1))
		compsz = np.reshape((tz * (x*y*z)).sum(axis=1), (N,1))
		
		return np.hstack((compsx, compsy, compsz))
	
	def rQuery2(self, query):
		## Finds base cuboid indices of the points to be interpolated
		### Length of sample distribution ###
		N = len(query)
		
		### Removes particles that are outside of interpolation volume
		query[np.where(query[:,0] < self.xIntMin)[0]] = np.nan
		query[np.where(query[:,0] > self.xIntMax)[0]] = np.nan
		query[np.where(query[:,1] < self.yIntMin)[0]] = np.nan
		query[np.where(query[:,1] > self.yIntMax)[0]] = np.nan
		query[np.where(query[:,2] < self.zIntMin)[0]] = np.nan
		query[np.where(query[:,2] > self.zIntMax)[0]] = np.nan
		
		### Coords in cuboids
		iu = (query[:,0]-self.xIntMin)/self.hx
		iv = (query[:,1]-self.yIntMin)/self.hy
		iw = (query[:,2]-self.zIntMin)/self.hz

		### Finds base coordinates of cuboid particles are in ###
		ix = np.floor(iu)
		iy = np.floor(iv)
		iz = np.floor(iw)

		### Returns indices of base cuboids ###
		self.queryInds = ix + iy*(self.nPos[0]) + iz*(self.nPos[0])*(self.nPos[1])
		self.queryInds[np.where(np.isnan(self.queryInds))]=self.nc
		self.queryInds = self.queryInds.astype(int)

		### Coordinates of the sample in unit cuboid
		queryCoords = np.stack((iu-ix,iv-iy,iw-iz),axis=1)

		## Returns the interpolate magnitude and / or gradients at the query coordinates
		if len(self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]) > 0:
			list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]))
		
		# Calculate interpolated values
		x = np.tile(np.transpose(np.array([np.ones(N), queryCoords[:,0], queryCoords[:,0]**2, queryCoords[:,0]**3])), 16)
		y = np.tile(np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:,1], queryCoords[:,1]**2, queryCoords[:,1]**3])), 4, axis=1),4)
		z = np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:,2], queryCoords[:,2]**2, queryCoords[:,2]**3])), 16, axis=1)

		# Derivatives
		xx = np.tile(np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,0], 3*queryCoords[:,0]**2])), 16)
		yy=np.tile(np.repeat(np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,1], 3*queryCoords[:,1]**2])), 4, axis=1), 4)
		zz=np.repeat(np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,2], 3*queryCoords[:,2]**2])), 16, axis=1)

		# Return coefficient matrix values, give NaN for invalid locations
		tn = self.alphan[:, self.queryInds]
		tn = np.transpose(tn)

		# Return magnitude
		norms = np.reshape((tn * (x*y*z)).sum(axis=1), (N,1))
		
		# Return gradient
		grads = np.transpose(np.array([((tn * (xx*y*z))/self.hx).sum(axis=1), ((tn * (x*yy*z))/self.hy).sum(axis=1), ((tn * (x*y*zz))/self.hz).sum(axis=1) ]))

		return norms, grads


	def rQuery3(self, query):
		## Finds base cuboid indices of the points to be interpolated
		### Length of sample distribution ###
		N = len(query)
		
		### Removes particles that are outside of interpolation volume
		query[np.where(query[:,0] < self.xIntMin)[0]] = np.nan
		query[np.where(query[:,0] > self.xIntMax)[0]] = np.nan
		query[np.where(query[:,1] < self.yIntMin)[0]] = np.nan
		query[np.where(query[:,1] > self.yIntMax)[0]] = np.nan
		query[np.where(query[:,2] < self.zIntMin)[0]] = np.nan
		query[np.where(query[:,2] > self.zIntMax)[0]] = np.nan
		
		### Coords in cuboids
		iu = (query[:,0]-self.xIntMin)/self.hx
		iv = (query[:,1]-self.yIntMin)/self.hy
		iw = (query[:,2]-self.zIntMin)/self.hz

		### Finds base coordinates of cuboid particles are in ###
		ix = np.floor(iu)
		iy = np.floor(iv)
		iz = np.floor(iw)

		### Returns indices of base cuboids ###
		self.queryInds = ix + iy*(self.nPos[0]) + iz*(self.nPos[0])*(self.nPos[1])
		self.queryInds[np.where(np.isnan(self.queryInds))]=self.nc
		self.queryInds = self.queryInds.astype(int)

		### Coordinates of the sample in unit cuboid
		queryCoords = np.stack((iu-ix,iv-iy,iw-iz),axis=1)

		## Returns the interpolate magnitude and / or gradients at the query coordinates
		if len(self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]) > 0:
			list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]))
		
		# Calculate interpolated values
		x = np.tile(np.transpose(np.array([np.ones(N), queryCoords[:,0], queryCoords[:,0]**2, queryCoords[:,0]**3])), 16)
		y = np.tile(np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:,1], queryCoords[:,1]**2, queryCoords[:,1]**3])), 4, axis=1),4)
		z = np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:,2], queryCoords[:,2]**2, queryCoords[:,2]**3])), 16, axis=1)

		# Derivatives
		xx=np.tile(np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,0], 3*queryCoords[:,0]**2])), 16)
		yy=np.tile(np.repeat(np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,1], 3*queryCoords[:,1]**2])), 4, axis=1), 4)
		zz=np.repeat(np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,2], 3*queryCoords[:,2]**2])), 16, axis=1)

		# Return coefficient matrix values, give NaN for invalid locations
		tx = self.alphax[:, self.queryInds]
		tx = np.transpose(tx)
		ty = self.alphay[:, self.queryInds]
		ty = np.transpose(ty)
		tz = self.alphaz[:, self.queryInds]
		tz = np.transpose(tz)
		tn = self.alphan[:, self.queryInds]
		tn = np.transpose(tn)

		# Return components and magnitude
		compsx = np.reshape((tx * (x*y*z)).sum(axis=1), (N,1))
		compsy = np.reshape((ty * (x*y*z)).sum(axis=1), (N,1))
		compsz = np.reshape((tz * (x*y*z)).sum(axis=1), (N,1))
		norms = np.reshape((tn * (x*y*z)).sum(axis=1), (N,1))
		
		# Return gradient
		grads = np.transpose(np.array([((tn * (xx*y*z))/self.hx).sum(axis=1), ((tn * (x*yy*z))/self.hy).sum(axis=1), ((tn * (x*y*zz))/self.hz).sum(axis=1) ]))

		return np.hstack((compsx, compsy, compsz)), norms, grads

	def allCoeffs(self):
		allinds = np.arange(self.nc)
		list(map(self.calcCoefficients, allinds))

			
	def getFieldParams(self):
		## Make sure coords are sorted correctly
		self.inputfield = self.inputfield[self.inputfield[:,0].argsort()] 
		self.inputfield = self.inputfield[self.inputfield[:,1].argsort(kind='mergesort')]
		self.inputfield = self.inputfield[self.inputfield[:,2].argsort(kind='mergesort')]
		
		## Analyse field
		xzero=self.inputfield[np.where(self.inputfield[:,0]==self.inputfield[0,0])[0]]
		yzero=self.inputfield[np.where(self.inputfield[:,1]==self.inputfield[0,1])[0]]
		xaxis=yzero[np.where(yzero[:,2]==yzero[0,2])[0]] 
		yaxis=xzero[np.where(xzero[:,2]==xzero[0,2])[0]] 
		zaxis=xzero[np.where(xzero[:,1]==xzero[0,1])[0]] 
		
		nPosx = len(xaxis) # These give the length of the interpolation volume along each axis
		nPosy = len(yaxis)
		nPosz = len(zaxis)

		self.nPos = np.array([nPosx-3, nPosy-3, nPosz-3]) # number of interpolatable cuboids per axis

		self.hx = np.abs((xaxis[0,0]-xaxis[1,0])) # grid spacing along each axis
		self.hy = np.abs((yaxis[0,1]-yaxis[1,1]))
		self.hz = np.abs((zaxis[0,2]-zaxis[1,2]))

		self.xIntMin = self.inputfield[1, 0] # Minimal value of x that can be interpolated
		self.yIntMin = self.inputfield[nPosx, 1]
		self.zIntMin = self.inputfield[nPosx*nPosy, 2]
		self.xIntMax = self.inputfield[-2, 0] # Maximal value of x that can be interpolated
		self.yIntMax = self.inputfield[-2*nPosx, 1]
		self.zIntMax = self.inputfield[-2*nPosx*nPosy, 2]

		## Find base indices of all interpolatable cuboids
		minI = nPosx*nPosy + nPosx + 1
		self.basePointInds = minI+np.arange(0, nPosx-3, 1)
		temp = np.array([self.basePointInds+i*nPosx for i in range(nPosy-3)])
		self.basePointInds= np.reshape(temp, (1, len(temp)*len(temp[0])))[0]
		temp = np.array([self.basePointInds+i*nPosx*nPosy for i in range(nPosz-3)])
		self.basePointInds = np.reshape(temp, (1, len(temp)*len(temp[0])))[0]
		self.basePointInds = np.sort(self.basePointInds)

		## Number of interpolatable cuboids
		self.nc = len(self.basePointInds)
		
	def calcCoefficients1(self, alphaindex):
		if self.alphamask[alphaindex] == 0:
			## Find interpolation coefficients for a cuboid
			realindex = self.basePointInds[alphaindex]
			## Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
			inds = self.neighbourInd(realindex)
			# Alpha coefficients
			self.alphax[:, alphaindex] = np.dot(self.A,self.Bx[inds])
			self.alphay[:, alphaindex] = np.dot(self.A,self.By[inds]) 
			self.alphaz[:, alphaindex] = np.dot(self.A,self.Bz[inds]) 
			self.alphamask[alphaindex] = 1

	def calcCoefficients2(self, alphaindex):
		if self.alphamask[alphaindex] == 0:
			## Find interpolation coefficients for a cuboid
			realindex = self.basePointInds[alphaindex]
			## Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
			inds = self.neighbourInd(realindex)
			# Alpha coefficients
			self.alphan[:, alphaindex] = np.dot(self.A,self.Bn[inds]) 
			self.alphamask[alphaindex] = 1
		
	def calcCoefficients3(self, alphaindex):
		if self.alphamask[alphaindex] == 0:
			## Find interpolation coefficients for a cuboid
			realindex = self.basePointInds[alphaindex]
			## Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
			inds = self.neighbourInd(realindex)
			# Alpha coefficients
			self.alphax[:, alphaindex] = np.dot(self.A,self.Bx[inds])
			self.alphay[:, alphaindex] = np.dot(self.A,self.By[inds]) 
			self.alphaz[:, alphaindex] = np.dot(self.A,self.Bz[inds]) 
			self.alphan[:, alphaindex] = np.dot(self.A,self.Bn[inds]) 
			self.alphamask[alphaindex] = 1

	def neighbourInd(self, ind0):
		# For base index ind0 this finds all 64 vertices of the 3x3x3 range of cuboids around it
		# It also returns the 7 neighbouring points
		newind0 = ind0 - 1 - (self.nPos[0]+3)*(self.nPos[1]+4)
		bInds = np.zeros(64)
		bInds[0] = newind0
		bInds[1] = bInds[0] + 1	
		bInds[2] = bInds[1] + 1
		bInds[3] = bInds[2] + 1
		bInds[4:8] = bInds[:4] + self.nPos[0]+3
		bInds[8:12] = bInds[4:8] + self.nPos[0]+3
		bInds[12:16] = bInds[8:12] + self.nPos[0]+3
		bInds[16:32] = bInds[:16] + (self.nPos[0]+3)*(self.nPos[1]+3)
		bInds[32:48] = bInds[16:32] + (self.nPos[0]+3)*(self.nPos[1]+3)
		bInds[48:] = bInds[32:48] + (self.nPos[0]+3)*(self.nPos[1]+3)
		bInds = bInds.astype(int)
		return bInds


#######################release the schmoo########################################
# Version 1.8, A2 code base

class quadcubic:
	__version__="1.8"
	def __init__(self, field, *args, **kwargs):
		self.eps = 10*np.finfo(float).eps # Machine precision of floating point number
		## Load field passed to class - can be x,y,z or norm of vector field, or scalar
		self.inputfield = field			
		## Analyse field, get shapes etc
		self.getFieldParams()
		## Make coefficient matrix
		if not hasattr(self, 'A'):
			self.makeAMatrix()
		### Mask to identify where coefficients exist
		self.alphamask = np.zeros((self.nc+1, 1 ))
		self.alphamask[-1] = 1
		
		## Determining which mode to run in
		if self.inputfield.shape[1] == 5:
			if not 'quiet' in args:
				print ('--- Scalar field, ignoring switches, interpolating for magnitude and gradient --- \n')
			self.Query = self.Query2
			self.sQuery = self.sQuery2
			self.rQuery = self.rQuery2
			self.calcCoefficients = self.calcCoefficients2
			self.alphan = np.zeros((256, self.nc+1 ))
			self.Bn = self.inputfield[:,4]
		elif self.inputfield.shape[1] == 7:
			if 'mode' in kwargs:
				if kwargs['mode'] == 'vector':
					if not 'quiet' in args:
						print ('--- Vector field, interpolating for vector components --- \n')
					self.Query = self.Query1
					self.sQuery = self.sQuery1
					self.rQuery = self.rQuery1
					self.calcCoefficients = self.calcCoefficients1
					self.alphax = np.zeros((256, self.nc+1+1 ))
					self.alphay = np.zeros((256, self.nc+1+1 ))
					self.alphaz = np.zeros((256, self.nc+1+1 ))
					self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = np.nan
					self.Bx = self.inputfield[:,4]
					self.By = self.inputfield[:,5]
					self.Bz = self.inputfield[:,6]
				elif kwargs['mode'] == 'norm':
					if not 'quiet' in args:
						print ('--- Vector field, interpolating for magnitude and gradient --- \n')
					self.Query = self.Query2
					self.sQuery = self.sQuery2
					self.rQuery = self.rQuery2
					self.calcCoefficients = self.calcCoefficients2
					self.alphan = np.zeros((256, self.nc+1 ))
					self.alphan[:,-1] = np.nan
					self.Bn = np.linalg.norm(self.inputfield[:,4:], axis=1)
				elif kwargs['mode'] == 'both':
					if not 'quiet' in args:
						print ('--- Vector field, interpolating vector components plus magnitude and gradient --- \n')
					self.Query = self.Query3
					self.sQuery = self.sQuery3
					self.rQuery = self.rQuery3
					self.calcCoefficients = self.calcCoefficients3
					self.alphax = np.zeros((256, self.nc+1 ))
					self.alphay = np.zeros((256, self.nc+1 ))
					self.alphaz = np.zeros((256, self.nc+1 ))
					self.alphan = np.zeros((256, self.nc+1 ))
					self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = self.alphan[:,-1] = np.nan
					self.Bx = self.inputfield[:,4]
					self.By = self.inputfield[:,5]
					self.Bz = self.inputfield[:,6]
					self.Bn = np.linalg.norm(self.inputfield[:,4:], axis=1)
				else:
					if not 'quiet' in args:
						print ('--- Vector field, invalid option, defaulting to interpolating for vector components --- \n')
					self.Query = self.Query1
					self.sQuery = self.sQuery1
					self.rQuery = self.rQuery1
					self.calcCoefficients = self.calcCoefficients1
					self.alphax = np.zeros((256, self.nc+1 ))
					self.alphay = np.zeros((256, self.nc+1 ))
					self.alphaz = np.zeros((256, self.nc+1 ))
					self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = np.nan
					self.Bx = self.inputfield[:,4]
					self.By = self.inputfield[:,5]
					self.Bz = self.inputfield[:,6]
			else:
				if not 'quiet' in args:
					print ('--- Vector field, no option selected, defaulting to interpolating for vector components --- \n')
				self.Query = self.Query1
				self.sQuery = self.sQuery1
				self.rQuery = self.rQuery1
				self.calcCoefficients = self.calcCoefficients1
				self.alphax = np.zeros((256, self.nc+1 ))
				self.alphay = np.zeros((256, self.nc+1 ))
				self.alphaz = np.zeros((256, self.nc+1 ))
				self.alphax[:,-1] = self.alphay[:,-1] = self.alphaz[:,-1] = np.nan
				self.Bx = self.inputfield[:,4]
				self.By = self.inputfield[:,5]
				self.Bz = self.inputfield[:,6]
		else:
			sys.exit('--- Input not shaped as expected - should be N x 5 or N x 7 ---')
		
			
	def makeAMatrix(self):
		### Creates quadcubic interpolation matrix and finite difference matrix and combines them
		### Interpolation matrix
		corners=np.array(([[i,j,k,l] for l in range(2) for k in range(2) for j in range(2) for i in range(2)])).astype(float).T
		exp = [[i,j,k,l] for l in range (4) for k in range(4) for j in range(4) for i in range(4)]

		B = np.zeros((256,256), dtype=np.float64)

		for i in range(256):
			ex,ey,ez,et = exp[i][0], exp[i][1], exp[i][2], exp[i][3]    # 
			for k in range(16):
				x,y,z,t=corners[0,k],corners[1,k],corners[2,k],corners[3,k]
				B[0*16+k,i] = x**ex * y**ey * z**ez * t**et	# 0:15, f

				B[1*16+k,i] = ex*x**(abs(ex-1)) * y**ey * z**ez * t**et	# 16:31, df/dx
				B[2*16+k,i] = x**ex * ey*y**(abs(ey-1)) * z**ez * t**et	# 32:47, df/dy
				B[3*16+k,i] = x**ex * y**ey * ez*z**(abs(ez-1)) * t**et	# 48:63, df/dz
				B[4*16+k,i] = x**ex * y**ey * z**ez * et*t**(abs(et-1))	# 64:79, df/dt

				B[5*16+k,i] = ex*x**(abs(ex-1)) * ey*y**(abs(ey-1)) * z**ez * t**et	# 80:95, d2f/dxdy
				B[6*16+k,i] = ex*x**(abs(ex-1)) * y**ey * ez*z**(abs(ez-1)) * t**et	# 96:111, d2f/dxdz
				B[7*16+k,i] = ex*x**(abs(ex-1)) * y**ey * z**ez * et*t**(abs(et-1))	# 112:127, d2f/dxdt
				B[8*16+k,i] = x**ex * ey*y**(abs(ey-1)) * ez*z**(abs(ez-1)) * t**et	# 128:143, d2f/dydz
				B[9*16+k,i] = x**ex * ey*y**(abs(ey-1)) * z**ez * et*t**(abs(et-1))	# 144:159, d2f/dydt
				B[10*16+k,i] = x**ex * y**ey * ez*z**(abs(ez-1)) * et*t**(abs(et-1))	# 160:175, d2f/dzdt

				B[11*16+k,i] = ex*x**(abs(ex-1)) * ey*y**(abs(ey-1)) * ez*z**(abs(ez-1)) * t**et	# 176:191, d3f/dxdydz
				B[12*16+k,i] = ex*x**(abs(ex-1)) * ey*y**(abs(ey-1)) * z**ez * et*t**(abs(et-1))	# 192:207, d3f/dxdydt
				B[13*16+k,i] = ex*x**(abs(ex-1)) * y**ey * ez*z**(abs(ez-1)) * et*t**(abs(et-1))	# 208:223, d3f/dxdzdt
				B[14*16+k,i] = x**ex * ey*y**(abs(ey-1)) * ez*z**(abs(ez-1)) * et*t**(abs(et-1))	# 224:239, d3f/dydzdt

				B[15*16+k,i] = ex*x**(abs(ex-1)) * ey*y**(abs(ey-1)) * ez*z**(abs(ez-1)) * et*t**(abs(et-1))	# 240:255, d4f/dxdydzdt
		

		# This makes a finite-difference matrix to return the components of the "b"-vector 
		# needed in the alpha calculation, in "cuboid" coordinates
		C = np.array((85,86,89,90, 101,102,105,106, 149,150,153,154, 165,166,169,170))
		D=np.zeros((256,256))

		for i in range(16):
			D[i,C[i]] = 1

		for i,j in enumerate(range(16,32,1)):
			D[j,C[i]-1] = -0.5
			D[j,C[i]+1] = 0.5

		for i,j in enumerate(range(32,48,1)):
			D[j,C[i]-4] = -0.5
			D[j,C[i]+4] = 0.5

		for i,j in enumerate(range(48,64,1)):
			D[j,C[i]-16] = -0.5
			D[j,C[i]+16] = 0.5

		for i,j in enumerate(range(64,80,1)):
			D[j,C[i]-64] = -0.5
			D[j,C[i]+64] = 0.5

		for i,j in enumerate(range(80,96,1)):
			D[j,C[i]+5] = 0.25
			D[j,C[i]-3] = -0.25
			D[j,C[i]+3] = -0.25
			D[j,C[i]-5] = 0.25

		for i,j in enumerate(range(96,112,1)):
			D[j,C[i]+17] = 0.25
			D[j,C[i]-15] = -0.25
			D[j,C[i]+15] = -0.25
			D[j,C[i]-17] = 0.25

		for i,j in enumerate(range(112,128,1)):
			D[j,C[i]+65] = 0.25
			D[j,C[i]-63] = -0.25
			D[j,C[i]+63] = -0.25
			D[j,C[i]-65] = 0.25

		for i,j in enumerate(range(128,144,1)):
			D[j,C[i]+20] = 0.25
			D[j,C[i]-12] = -0.25
			D[j,C[i]+12] = -0.25
			D[j,C[i]-20] = 0.25

		for i,j in enumerate(range(144,160,1)):
			D[j,C[i]+68] = 0.25
			D[j,C[i]-60] = -0.25
			D[j,C[i]+60] = -0.25
			D[j,C[i]-68] = 0.25

		for i,j in enumerate(range(160,176,1)):
			D[j,C[i]+80] = 0.25
			D[j,C[i]-48] = -0.25
			D[j,C[i]+48] = -0.25
			D[j,C[i]-80] = 0.25

		for i,j in enumerate(range(176,192,1)):
			D[j,C[i]+21] = 0.125
			D[j,C[i]+13] = -0.125
			D[j,C[i]+19] = -0.125
			D[j,C[i]+11] = 0.125
			D[j,C[i]-11] = -0.125
			D[j,C[i]-19] = 0.125
			D[j,C[i]-13] = 0.125
			D[j,C[i]-21] = -0.125

		for i,j in enumerate(range(192,208,1)):
			D[j,C[i]+69] = 0.125
			D[j,C[i]+61] = -0.125
			D[j,C[i]+67] = -0.125
			D[j,C[i]+59] = 0.125
			D[j,C[i]-59] = -0.125
			D[j,C[i]-67] = 0.125
			D[j,C[i]-61] = 0.125
			D[j,C[i]-69] = -0.125

		for i,j in enumerate(range(208,224,1)):
			D[j,C[i]+81] = 0.125
			D[j,C[i]+49] = -0.125
			D[j,C[i]+79] = -0.125
			D[j,C[i]+47] = 0.125
			D[j,C[i]-47] = -0.125
			D[j,C[i]-79] = 0.125
			D[j,C[i]-49] = 0.125
			D[j,C[i]-81] = -0.125

		for i,j in enumerate(range(224,240,1)):
			D[j,C[i]+84] = 0.125
			D[j,C[i]+52] = -0.125
			D[j,C[i]+76] = -0.125
			D[j,C[i]+44] = 0.125
			D[j,C[i]-44] = -0.125
			D[j,C[i]-76] = 0.125
			D[j,C[i]-52] = 0.125
			D[j,C[i]-84] = -0.125

		for i,j in enumerate(range(241,256,1)):
			D[j,C[i]+85] = 0.0625
			D[j,C[i]+77] = -0.0625
			D[j,C[i]+83] = -0.0625
			D[j,C[i]+75] = 0.0625
			D[j,C[i]+53] = -0.0625
			D[j,C[i]+45] = 0.0625
			D[j,C[i]+51] = 0.0625
			D[j,C[i]+43] = -0.0625
			D[j,C[i]-43] = -0.0625
			D[j,C[i]-51] = 0.0625
			D[j,C[i]-45] = 0.0625
			D[j,C[i]-53] = -0.0625
			D[j,C[i]-75] = 0.0625
			D[j,C[i]-83] = -0.0625
			D[j,C[i]-77] = -0.0625
			D[j,C[i]-85] = 0.0625

		self.A = np.matmul(np.linalg.inv(B),D)

	def Query1(self, query):
		try:
			if query.shape[1] > 1:
				comps = self.rQuery1(query)
				return comps
			else:
				comps = self.sQuery1(query)
				return comps
		except IndexError:
			comps = self.sQuery1(query)
			return comps
		
	def Query2(self, query):
		try:
			if query.shape[1] > 1:
				norms, grads = self.rQuery2(query)
				return norms, grads
			else:
				norm, grad = self.sQuery2(query)
				return norm, grad
		except IndexError:
			norm, grad = self.sQuery2(query)
			return norm, grad
		
	def Query3(self, query):
		try:
			if query.shape[1] > 1:
				comps, norms, grads = self.rQuery3(query)
				return comps, norms, grads
			else:
				comps, norm, grad = self.sQuery3(query)
				return comps, norm, grad
		except IndexError:
			comps, norm, grad = self.sQuery3(query)
			return comps, norm, grad
	
	def sQuery1(self, query):
		### Removes particles that are outside of interpolation volume
		if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax or query[3] < self.tIntMin or query[3] > self.tIntMax:
			return np.nan
		else:
			### How many cuboids in is query point
			ix = (query[0]-self.xIntMin)/self.hx
			iy = (query[1]-self.yIntMin)/self.hy
			iz = (query[2]-self.zIntMin)/self.hz
			it = (query[3]-self.tIntMin)/self.ht
			### Finds base coordinates of cuboid particle is in
			iix = np.floor(ix)
			iiy = np.floor(iy)
			iiz = np.floor(iz)
			iit = np.floor(it)
			### particle coordinates in unit cuboid
			cuboidx = ix-iix
			cuboidy = iy-iiy
			cuboidz = iz-iiz
			cuboidt = it-iit
			### Returns index of base cuboid
			self.queryInd = iix + iiy*(self.nPosx-3) + iiz*(self.nPosx-3)*(self.nPosy-3) + iit*(self.nPosx-3)*(self.nPosy-3)*(self.nPosz-3)
			self.queryInd = self.queryInd.astype(int)
			### Calculate alpha for cuboid if it doesn't exist
			if self.alphamask[self.queryInd]==0:
				self.calcCoefficients(self.queryInd)
			### 4-vectors for finding interpolated values
			xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
			yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
			zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
			tvec = np.array([1, cuboidt, cuboidt**2, cuboidt**3])
			### 4-vector summation components
			x = np.tile(xvec,64)
			y = np.tile(np.repeat(yvec,4),16)
			z = np.tile(np.repeat(zvec,16),4)
			t = np.repeat(tvec,64)
			### interpolated values
			compx =  np.inner(self.alphax[:,self.queryInd], (x*y*z*t))
			compy =  np.inner(self.alphay[:,self.queryInd], (x*y*z*t))
			compz =  np.inner(self.alphaz[:,self.queryInd], (x*y*z*t))
			return np.array((compx, compy, compz))
		
	def sQuery2(self, query):
		### Removes particles that are outside of interpolation volume
		if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax or query[3] < self.tIntMin or query[3] > self.tIntMax:
			return np.nan
		else:
			### How many cuboids in is query point
			ix = (query[0]-self.xIntMin)/self.hx
			iy = (query[1]-self.yIntMin)/self.hy
			iz = (query[2]-self.zIntMin)/self.hz
			it = (query[3]-self.tIntMin)/self.ht
			### Finds base coordinates of cuboid particle is in
			iix = np.floor(ix)
			iiy = np.floor(iy)
			iiz = np.floor(iz)
			iit = np.floor(it)
			### particle coordinates in unit cuboid
			cuboidx = ix-iix
			cuboidy = iy-iiy
			cuboidz = iz-iiz
			cuboidt = it-iit
			### Returns index of base cuboid
			self.queryInd = iix + iiy*(self.nPosx-3) + iiz*(self.nPosx-3)*(self.nPosy-3) + iit*(self.nPosx-3)*(self.nPosy-3)*(self.nPosz-3)
			self.queryInd = self.queryInd.astype(int)
			### Calculate interpolation coefficients, if they don't exist
			if self.alphamask[self.queryInd]==0:
				self.calcCoefficients(self.queryInd)
			### 4-vectors for finding interpolated values
			xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
			yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
			zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
			tvec = np.array([1, cuboidt, cuboidt**2, cuboidt**3])
			### 4-vectors for finding interpolated gradients
			xxvec = np.array([0, 1, 2*cuboidx, 3*cuboidx**2])
			yyvec = np.array([0, 1, 2*cuboidy, 3*cuboidy**2])
			zzvec = np.array([0, 1, 2*cuboidz, 3*cuboidz**2])
			ttvec = np.array([0, 1, 2*cuboidt, 3*cuboidt**2])
			### 4-vector summation components
			x = np.tile(xvec,64)
			y = np.tile(np.repeat(yvec,4),16)
			z = np.tile(np.repeat(zvec,16),4)
			t = np.repeat(tvec,64)
			### 4-vector summation components
			xx = np.tile(xxvec,64)
			yy = np.tile(np.repeat(yyvec,4),16)
			zz = np.tile(np.repeat(zzvec,16),4)
			tt = np.repeat(ttvec,64)
			###	interpolated values
			norm =  np.inner(self.alphan[:,self.queryInd], (x*y*z*t))
			grads = np.array([np.dot(self.alphan[:,self.queryInd], xx*y*z*t)/self.hx, np.dot(self.alphan[:,self.queryInd], x*yy*z*t)/self.hy, np.dot(self.alphan[:,self.queryInd], x*y*zz*t)/self.hz, np.dot(self.alphan[:,self.queryInd], x*y*z*tt)/self.ht])
			return norm, grads

	def sQuery3(self, query):
		### Removes particles that are outside of interpolation volume
		if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax or query[3] < self.tIntMin or query[3] > self.tIntMax:
			return np.nan
		else:
			### How many cuboids in is query point
			ix = (query[0]-self.xIntMin)/self.hx
			iy = (query[1]-self.yIntMin)/self.hy
			iz = (query[2]-self.zIntMin)/self.hz
			it = (query[3]-self.tIntMin)/self.ht
			### Finds base coordinates of cuboid particle is in
			iix = np.floor(ix)
			iiy = np.floor(iy)
			iiz = np.floor(iz)
			iit = np.floor(it)
			### particle coordinates in unit cuboid
			cuboidx = ix-iix
			cuboidy = iy-iiy
			cuboidz = iz-iiz
			cuboidt = it-iit
			### Returns index of base cuboid
			self.queryInd = iix + iiy*(self.nPosx-3) + iiz*(self.nPosx-3)*(self.nPosy-3) + iit*(self.nPosx-3)*(self.nPosy-3)*(self.nPosz-3)
			self.queryInd = self.queryInd.astype(int)
			### Calculate alpha for cuboid if it doesn't exist
			if self.alphamask[self.queryInd]==0:
				self.calcCoefficients(self.queryInd)
			### 4-vectors for finding interpolated values
			xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
			yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
			zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
			tvec = np.array([1, cuboidt, cuboidt**2, cuboidt**3])
			### 4-vectors for finding interpolated gradients
			xxvec = np.array([0, 1, 2*cuboidx, 3*cuboidx**2])
			yyvec = np.array([0, 1, 2*cuboidy, 3*cuboidy**2])
			zzvec = np.array([0, 1, 2*cuboidz, 3*cuboidz**2])
			ttvec = np.array([0, 1, 2*cuboidt, 3*cuboidt**2])
			### 4-vector summation components
			x = np.tile(xvec,64)
			y = np.tile(np.repeat(yvec,4),16)
			z = np.tile(np.repeat(zvec,16),4)
			t = np.repeat(tvec,64)
			### 4-vector summation components
			xx = np.tile(xxvec,64)
			yy = np.tile(np.repeat(yyvec,4),16)
			zz = np.tile(np.repeat(zzvec,16),4)
			tt = np.repeat(ttvec,64)
			###	interpolated values
			compx =  np.inner(self.alphax[:,self.queryInd], (x*y*z*t))
			compy =  np.inner(self.alphay[:,self.queryInd], (x*y*z*t))
			compz =  np.inner(self.alphaz[:,self.queryInd], (x*y*z*t))
			norm =  np.inner(self.alphan[:,self.queryInd], (x*y*z*t))
			grads = np.array([np.dot(self.alphan[:,self.queryInd], xx*y*z*t)/self.hx, np.dot(self.alphan[:,self.queryInd], x*yy*z*t)/self.hy, np.dot(self.alphan[:,self.queryInd], x*y*zz*t)/self.hz, np.dot(self.alphan[:,self.queryInd], x*y*z*tt)/self.ht])
			## Components, magnitude, gradient
			return np.array((compx, compy, compz)), norm, grads

	def rQuery1(self, query):
		## Finds base cuboid indices of the points to be interpolated
		### Length of sample distribution ###
		N = len(query)
		### Removes particles that are outside of interpolation volume
		query[np.where(query[:,0] < self.xIntMin)[0]] = np.nan
		query[np.where(query[:,0] > self.xIntMax)[0]] = np.nan
		query[np.where(query[:,1] < self.yIntMin)[0]] = np.nan
		query[np.where(query[:,1] > self.yIntMax)[0]] = np.nan
		query[np.where(query[:,2] < self.zIntMin)[0]] = np.nan
		query[np.where(query[:,2] > self.zIntMax)[0]] = np.nan
		query[np.where(query[:,3] < self.tIntMin)[0]] = np.nan
		query[np.where(query[:,3] > self.tIntMax)[0]] = np.nan
		### Coords in cuboids
		ix = (query[:,0]-self.xIntMin)/self.hx
		iy = (query[:,1]-self.yIntMin)/self.hy
		iz = (query[:,2]-self.zIntMin)/self.hz
		it = (query[:,3]-self.tIntMin)/self.ht
		### Finds base coordinates of cuboid particles are in ###
		iix = np.floor(ix)
		iiy = np.floor(iy)
		iiz = np.floor(iz)
		iit = np.floor(it)
		### Returns indices of base cuboids ###
		self.queryInds = iix + iiy*(self.nPosx-3) + iiz*(self.nPosx-3)*(self.nPosy-3) + iit*(self.nPosx-3)*(self.nPosy-3)*(self.nPosz-3)
		self.queryInds[np.where(np.isnan(self.queryInds))]=self.nc
		self.queryInds = self.queryInds.astype(int)
		### Coordinates of the sample in unit cuboid
		queryCoords = np.stack((ix-iix,iy-iiy,iz-iiz,it-iit),axis=1)		
		## Calculate alpha for cuboid if it doesn't exist
		if len(self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]) > 0:
			list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]))
		### 4-vectors for finding interpolated values
		xvec = np.transpose(np.array([np.ones(N), queryCoords[:,0], queryCoords[:,0]**2, queryCoords[:,0]**3]))
		yvec = np.transpose(np.array([np.ones(N), queryCoords[:,1], queryCoords[:,1]**2, queryCoords[:,1]**3]))
		zvec = np.transpose(np.array([np.ones(N), queryCoords[:,2], queryCoords[:,2]**2, queryCoords[:,2]**3]))
		tvec = np.transpose(np.array([np.ones(N), queryCoords[:,3], queryCoords[:,3]**2, queryCoords[:,3]**3]))
		### 4-vectors for finding interpolated gradients
		xxvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,0], 3*queryCoords[:,0]**2]))
		yyvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,1], 3*queryCoords[:,1]**2]))
		zzvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,2], 3*queryCoords[:,2]**2]))
		ttvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,3], 3*queryCoords[:,3]**2]))
		### 4-vector summation components
		x = np.tile(xvec,64)
		y = np.tile(np.repeat(yvec,4,axis=1),16)
		z = np.tile(np.repeat(zvec,16,axis=1),4)
		t = np.repeat(tvec,64,axis=1)
		### 4-vector summation components
		xx = np.tile(xxvec,64)
		yy = np.tile(np.repeat(yyvec,4,axis=1),16)
		zz = np.tile(np.repeat(zzvec,16,axis=1),4)
		tt = np.repeat(ttvec,64,axis=1)	
		# Return coefficient matrix values, give NaN for invalid locations
		tx = self.alphax[:, self.queryInds]
		tx = np.transpose(tx)
		ty = self.alphay[:, self.queryInds]
		ty = np.transpose(ty)
		tz = self.alphaz[:, self.queryInds]
		tz = np.transpose(tz)
		# Return components
		compsx = np.reshape((tx * (x*y*z*t)).sum(axis=1), (N,1))
		compsy = np.reshape((ty * (x*y*z*t)).sum(axis=1), (N,1))
		compsz = np.reshape((tz * (x*y*z*t)).sum(axis=1), (N,1))
		return np.hstack((compsx, compsy, compsz))
		
	def rQuery2(self, query):
		## Finds base cuboid indices of the points to be interpolated
		### Length of sample distribution ###
		N = len(query)
		### Removes particles that are outside of interpolation volume
		query[np.where(query[:,0] < self.xIntMin)[0]] = np.nan
		query[np.where(query[:,0] > self.xIntMax)[0]] = np.nan
		query[np.where(query[:,1] < self.yIntMin)[0]] = np.nan
		query[np.where(query[:,1] > self.yIntMax)[0]] = np.nan
		query[np.where(query[:,2] < self.zIntMin)[0]] = np.nan
		query[np.where(query[:,2] > self.zIntMax)[0]] = np.nan
		query[np.where(query[:,3] < self.tIntMin)[0]] = np.nan
		query[np.where(query[:,3] > self.tIntMax)[0]] = np.nan
		### Coords in cuboids
		ix = (query[:,0]-self.xIntMin)/self.hx
		iy = (query[:,1]-self.yIntMin)/self.hy
		iz = (query[:,2]-self.zIntMin)/self.hz
		it = (query[:,3]-self.tIntMin)/self.ht
		### Finds base coordinates of cuboid particles are in ###
		iix = np.floor(ix)
		iiy = np.floor(iy)
		iiz = np.floor(iz)
		iit = np.floor(it)
		### Returns indices of base cuboids ###
		self.queryInds = iix + iiy*(self.nPosx-3) + iiz*(self.nPosx-3)*(self.nPosy-3) + iit*(self.nPosx-3)*(self.nPosy-3)*(self.nPosz-3)
		self.queryInds[np.where(np.isnan(self.queryInds))]=self.nc
		self.queryInds = self.queryInds.astype(int)
		### Coordinates of the sample in unit cuboid
		queryCoords = np.stack((ix-iix,iy-iiy,iz-iiz,it-iit),axis=1)
		## Calculate alpha for cuboid if it doesn't exist
		if len(self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]) > 0:
			list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]))
		### 4-vectors for finding interpolated values
		xvec = np.transpose(np.array([np.ones(N), queryCoords[:,0], queryCoords[:,0]**2, queryCoords[:,0]**3]))
		yvec = np.transpose(np.array([np.ones(N), queryCoords[:,1], queryCoords[:,1]**2, queryCoords[:,1]**3]))
		zvec = np.transpose(np.array([np.ones(N), queryCoords[:,2], queryCoords[:,2]**2, queryCoords[:,2]**3]))
		tvec = np.transpose(np.array([np.ones(N), queryCoords[:,3], queryCoords[:,3]**2, queryCoords[:,3]**3]))
		### 4-vectors for finding interpolated gradients
		xxvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,0], 3*queryCoords[:,0]**2]))
		yyvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,1], 3*queryCoords[:,1]**2]))
		zzvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,2], 3*queryCoords[:,2]**2]))
		ttvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,3], 3*queryCoords[:,3]**2]))
		### 4-vector summation components
		x = np.tile(xvec,64)
		y = np.tile(np.repeat(yvec,4,axis=1),16)
		z = np.tile(np.repeat(zvec,16,axis=1),4)
		t = np.repeat(tvec,64,axis=1)
		### 4-vector summation components
		xx = np.tile(xxvec,64)
		yy = np.tile(np.repeat(yyvec,4,axis=1),16)
		zz = np.tile(np.repeat(zzvec,16,axis=1),4)
		tt = np.repeat(ttvec,64,axis=1)	
		# Return coefficient matrix values, give NaN for invalid locations
		tn = self.alphan[:, self.queryInds]
		tn = np.transpose(tn)
		# Return magnitude
		norms = np.reshape((tn * (x*y*z*t)).sum(axis=1), (N,1))
		# Return gradient
		grads = np.transpose(np.array([((tn * (xx*y*z*t))/self.hx).sum(axis=1), ((tn * (x*yy*z*t))/self.hy).sum(axis=1), ((tn * (x*y*zz*t))/self.hz).sum(axis=1), ((tn * (x*y*z*tt))/self.ht).sum(axis=1) ]))
		return norms, grads

	def rQuery3(self, query):
		## Finds base cuboid indices of the points to be interpolated
		### Length of sample distribution ###
		N = len(query)
		### Removes particles that are outside of interpolation volume
		query[np.where(query[:,0] < self.xIntMin)[0]] = np.nan
		query[np.where(query[:,0] > self.xIntMax)[0]] = np.nan
		query[np.where(query[:,1] < self.yIntMin)[0]] = np.nan
		query[np.where(query[:,1] > self.yIntMax)[0]] = np.nan
		query[np.where(query[:,2] < self.zIntMin)[0]] = np.nan
		query[np.where(query[:,2] > self.zIntMax)[0]] = np.nan
		query[np.where(query[:,3] < self.tIntMin)[0]] = np.nan
		query[np.where(query[:,3] > self.tIntMax)[0]] = np.nan
		### Coords in cuboids
		ix = (query[:,0]-self.xIntMin)/self.hx
		iy = (query[:,1]-self.yIntMin)/self.hy
		iz = (query[:,2]-self.zIntMin)/self.hz
		it = (query[:,3]-self.tIntMin)/self.ht
		### Finds base coordinates of cuboid particles are in ###
		iix = np.floor(ix)
		iiy = np.floor(iy)
		iiz = np.floor(iz)
		iit = np.floor(it)
		### Returns indices of base cuboids ###
		self.queryInds = iix + iiy*(self.nPosx-3) + iiz*(self.nPosx-3)*(self.nPosy-3) + iit*(self.nPosx-3)*(self.nPosy-3)*(self.nPosz-3)
		self.queryInds[np.where(np.isnan(self.queryInds))]=self.nc
		self.queryInds = self.queryInds.astype(int)
		### Coordinates of the sample in unit cuboid
		queryCoords = np.stack((ix-iix,iy-iiy,iz-iiz,it-iit),axis=1)		
		## Calculate alpha for cuboid if it doesn't exist
		if len(self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]) > 0:
			list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds]==0)[0]]))
		### 4-vectors for finding interpolated values
		xvec = np.transpose(np.array([np.ones(N), queryCoords[:,0], queryCoords[:,0]**2, queryCoords[:,0]**3]))
		yvec = np.transpose(np.array([np.ones(N), queryCoords[:,1], queryCoords[:,1]**2, queryCoords[:,1]**3]))
		zvec = np.transpose(np.array([np.ones(N), queryCoords[:,2], queryCoords[:,2]**2, queryCoords[:,2]**3]))
		tvec = np.transpose(np.array([np.ones(N), queryCoords[:,3], queryCoords[:,3]**2, queryCoords[:,3]**3]))
		### 4-vectors for finding interpolated gradients
		xxvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,0], 3*queryCoords[:,0]**2]))
		yyvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,1], 3*queryCoords[:,1]**2]))
		zzvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,2], 3*queryCoords[:,2]**2]))
		ttvec = np.transpose(np.array([np.zeros(N), np.ones(N), 2*queryCoords[:,3], 3*queryCoords[:,3]**2]))
		### 4-vector summation components
		x = np.tile(xvec,64)
		y = np.tile(np.repeat(yvec,4,axis=1),16)
		z = np.tile(np.repeat(zvec,16,axis=1),4)
		t = np.repeat(tvec,64,axis=1)
		### 4-vector summation components
		xx = np.tile(xxvec,64)
		yy = np.tile(np.repeat(yyvec,4,axis=1),16)
		zz = np.tile(np.repeat(zzvec,16,axis=1),4)
		tt = np.repeat(ttvec,64,axis=1)	
		# Return coefficient matrix values, give NaN for invalid locations
		tx = self.alphax[:, self.queryInds]
		tx = np.transpose(tx)
		ty = self.alphay[:, self.queryInds]
		ty = np.transpose(ty)
		tz = self.alphaz[:, self.queryInds]
		tz = np.transpose(tz)
		tn = self.alphan[:, self.queryInds]
		tn = np.transpose(tn)
		# Return components and magnitude
		compsx = np.reshape((tx * (x*y*z*t)).sum(axis=1), (N,1))
		compsy = np.reshape((ty * (x*y*z*t)).sum(axis=1), (N,1))
		compsz = np.reshape((tz * (x*y*z*t)).sum(axis=1), (N,1))
		norms = np.reshape((tn * (x*y*z*t)).sum(axis=1), (N,1))
		# Return gradient
		grads = np.transpose(np.array([((tn * (xx*y*z*t))/self.hx).sum(axis=1), ((tn * (x*yy*z*t))/self.hy).sum(axis=1), ((tn * (x*y*zz*t))/self.hz).sum(axis=1), ((tn * (x*y*z*tt))/self.ht).sum(axis=1) ]))
		return np.hstack((compsx, compsy, compsz)), norms, grads

	def allCoeffs(self):
		allinds = np.arange(self.nc)
		list(map(self.calcCoefficients, allinds))
	
	def getFieldParams(self):
		## Make sure coords are sorted correctly
		self.inputfield = self.inputfield[self.inputfield[:,0].argsort()] 
		self.inputfield = self.inputfield[self.inputfield[:,1].argsort(kind='mergesort')]
		self.inputfield = self.inputfield[self.inputfield[:,2].argsort(kind='mergesort')]
		self.inputfield = self.inputfield[self.inputfield[:,3].argsort(kind='mergesort')]

		## Analyse field
		xAxis=self.inputfield[np.where(self.inputfield[:,1]==self.inputfield[0,1])[0]]
		xAxis=xAxis[np.where(xAxis[:,2]==xAxis[0,2])[0]]
		xAxis=xAxis[np.where(xAxis[:,3]==xAxis[0,3])[0]]

		yAxis=self.inputfield[np.where(self.inputfield[:,0]==self.inputfield[0,0])[0]]
		yAxis=yAxis[np.where(yAxis[:,2]==yAxis[0,2])[0]]
		yAxis=yAxis[np.where(yAxis[:,3]==yAxis[0,3])[0]]

		zAxis=self.inputfield[np.where(self.inputfield[:,0]==self.inputfield[0,0])[0]]
		zAxis=zAxis[np.where(zAxis[:,1]==zAxis[0,1])[0]]
		zAxis=zAxis[np.where(zAxis[:,3]==zAxis[0,3])[0]]

		tAxis=self.inputfield[np.where(self.inputfield[:,0]==self.inputfield[0,0])[0]]
		tAxis=tAxis[np.where(tAxis[:,1]==tAxis[0,1])[0]]
		tAxis=tAxis[np.where(tAxis[:,2]==tAxis[0,2])[0]] 

		self.nPosx = len(xAxis) # These give the length of the interpolation volume along each axis
		self.nPosy = len(yAxis)
		self.nPosz = len(zAxis)
		self.nPost = len(tAxis)

		self.hx = np.abs((xAxis[0,0]-xAxis[1,0])) # grid spacing along each axis
		self.hy = np.abs((yAxis[0,1]-yAxis[1,1]))
		self.hz = np.abs((zAxis[0,2]-zAxis[1,2]))
		self.ht = np.abs((tAxis[0,3]-tAxis[1,3]))

		self.xIntMin = self.inputfield[1, 0] # Minimal value of x that can be interpolated
		self.yIntMin = self.inputfield[self.nPosx, 1]
		self.zIntMin = self.inputfield[self.nPosx*self.nPosy, 2]
		self.tIntMin = self.inputfield[self.nPosx*self.nPosy*self.nPosz,3] # right I think
		self.xIntMax = self.inputfield[-2, 0] # Maximal value of x that can be interpolated
		self.yIntMax = self.inputfield[-2*self.nPosx, 1]
		self.zIntMax = self.inputfield[-2*self.nPosx*self.nPosy, 2]
		self.tIntMax = self.inputfield[(-2*self.nPosx*self.nPosy*self.nPosz),3] # OK then!

		## Find base indices of all interpolatable cuboids
		minI = self.nPosx*(self.nPosy*(self.nPosz + 1) + 1) + 1
		self.basePointInds = minI+np.arange(0, self.nPosx-3, 1)
		temp = np.array([self.basePointInds+i*self.nPosx for i in range(self.nPosy-3)])
		self.basePointInds= np.reshape(temp, (1, len(temp)*len(temp[0])))[0]
		temp = np.array([self.basePointInds+i*self.nPosx*self.nPosy for i in range(self.nPosz-3)])
		self.basePointInds = np.reshape(temp, (1, len(temp)*len(temp[0])))[0]
		temp = np.array([self.basePointInds+i*self.nPosx*self.nPosy*self.nPosz for i in range(self.nPost-3)])
		self.basePointInds = np.reshape(temp, (1, len(temp)*len(temp[0])))[0]

		self.basePointInds = np.sort(self.basePointInds)

		## Number of interpolatable cuboids
		self.nc = len(self.basePointInds)
		
	def calcCoefficients1(self, alphaindex):
		if self.alphamask[alphaindex] == 0:
			## Find interpolation coefficients for a cuboid
			realindex = self.basePointInds[alphaindex]
			## Find all neighbours in 3x3x3x3 neighbouring array
			inds = self.neighbourInd(realindex)
			# Alpha coefficients
			self.alphax[:, alphaindex] = np.dot(self.A,self.Bx[inds])
			self.alphay[:, alphaindex] = np.dot(self.A,self.By[inds]) 
			self.alphaz[:, alphaindex] = np.dot(self.A,self.Bz[inds]) 
			self.alphamask[alphaindex] = 1

	def calcCoefficients2(self, alphaindex):
		if self.alphamask[alphaindex] == 0:
			## Find interpolation coefficients for a cuboid
			realindex = self.basePointInds[alphaindex]
			## Find all neighbours in 3x3x3x3 neighbouring array
			inds = self.neighbourInd(realindex)
			# Alpha coefficients
			self.alphan[:, alphaindex] = np.dot(self.A,self.Bn[inds]) 
			self.alphamask[alphaindex] = 1
		
	def calcCoefficients3(self, alphaindex):
		if self.alphamask[alphaindex] == 0:
			## Find interpolation coefficients for a cuboid
			realindex = self.basePointInds[alphaindex]
			## Find all neighbours in 3x3x3x3 neighbouring array
			inds = self.neighbourInd(realindex)
			# Alpha coefficients
			self.alphax[:, alphaindex] = np.dot(self.A,self.Bx[inds])
			self.alphay[:, alphaindex] = np.dot(self.A,self.By[inds]) 
			self.alphaz[:, alphaindex] = np.dot(self.A,self.Bz[inds]) 
			self.alphan[:, alphaindex] = np.dot(self.A,self.Bn[inds]) 
			self.alphamask[alphaindex] = 1

	def neighbourInd(self, ind0):
		# For base index ind0 this finds all 256 vertices of the 3x3x3x3 range of cuboids around it
		xinc = 1
		yinc = self.nPosx
		zinc = (self.nPosx)*(self.nPosy)
		tinc = (self.nPosx)*(self.nPosy)*(self.nPosz)

		newind0 = ind0 - xinc - yinc - zinc - tinc
		bInds = np.zeros(256) 
		bInds[0] = newind0
		bInds[1] = bInds[0] + xinc
		bInds[2] = bInds[1] + xinc
		bInds[3] = bInds[2] + xinc
		bInds[4:8] = bInds[:4] + yinc
		bInds[8:12] = bInds[4:8] + yinc
		bInds[12:16] = bInds[8:12] + yinc
		bInds[16:32] = bInds[:16] + zinc
		bInds[32:48] = bInds[16:32] + zinc
		bInds[48:64] = bInds[32:48] + zinc
		bInds[64:128] = bInds[0:64] + tinc
		bInds[128:192] = bInds[64:128] + tinc
		bInds[192:256] = bInds[128:192] + tinc
		bInds = bInds.astype(int)

		return bInds
