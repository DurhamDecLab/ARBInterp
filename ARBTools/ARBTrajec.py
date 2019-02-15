from __future__ import division
import numpy as np
import random as rand
from ARBTools.ARBInterp import tricubic

#######################release the schmoo########################################
# Version 1.3, A4 code base

class trajectories:
	def __init__(self, field, mass, moment):
		self.eps = 10*np.finfo(float).eps # Machine precision of floating point number
		self.amu = 1.66e-27         # Atomic mass units
		self.k = 1.38064852e-23         # Boltzmann constant
		mu = 9.274009994e-24	#Bohr magneton
		self.m = mass*self.amu         # Mass of atom
		self.muom = moment*mu/self.m	#	magnetic moment in magnetons over mass
		self.Interp = tricubic(field, mode='norm')
		
	def atoms(self, T, N):
		sd = np.sqrt(self.k*T/self.m)
		PSRange = np.zeros((int(N), 6))
		for q in range(PSRange.shape[0]):
			x = rand.gauss(0, 1e-3)
			y = rand.gauss(0, 1e-3)
			z = rand.gauss(0, 1e-3)
			vx = rand.gauss(0, sd)
			vy = rand.gauss(0, sd)
			vz = rand.gauss(0, sd)
			PSRange[q] = np.array([[x,y,z,vx,vy,vz]])	
		return PSRange

	def sRK3d(self, sample, h):           ### sample is the loaded distribution, h the RK timestep ###
		temp = sample.copy()
		
		### Runge-Kutta coefficients are stored in these arrays
		RK0 = np.zeros((6))
		RK1 = np.zeros((6))
		RK2 = np.zeros((6))
		RK3 = np.zeros((6))
		
		#Coeffs 1
		Norm, Grad = self.Interp.sQuery(temp)
		
		RK0[0] = h*(temp[3])
		RK0[3] = -h*self.muom*Grad[0]
		RK0[1] = h*(temp[4])
		RK0[4] = -h*self.muom*Grad[1]
		RK0[2] = h*(temp[5])
		RK0[5] = -h*self.muom*Grad[2]
		
		#New variables
		temp = sample + 0.5*RK0
		
		#Coeffs 2
		Norm, Grad = self.Interp.sQuery(temp)
		
		RK1[0] = h*(temp[3])
		RK1[3] = -h*self.muom*Grad[0]
		RK1[1] = h*(temp[4])
		RK1[4] = -h*self.muom*Grad[1]
		RK1[2] = h*(temp[5])
		RK1[5] = -h*self.muom*Grad[2]

		#New variables
		temp = sample + 0.5*RK1

		#Coeff 3
		Norm, Grad = self.Interp.sQuery(temp)   

		RK2[0] = h*(temp[3])
		RK2[3] = -h*self.muom*Grad[0]
		RK2[1] = h*(temp[4])
		RK2[4] = -h*self.muom*Grad[1]
		RK2[2] = h*(temp[5])
		RK2[5] = -h*self.muom*Grad[2]
		
		#New variables
		temp = sample + RK2

		#Coeff 4
		Norm, Grad = self.Interp.sQuery(temp)  

		RK3[0] = h*(temp[3])
		RK3[3] = -h*self.muom*Grad[0]
		RK3[1] = h*(temp[4])
		RK3[4] = -h*self.muom*Grad[1]
		RK3[2] = h*(temp[5])
		RK3[5] = -h*self.muom*Grad[2]
		
		#Final new variables
		temp = sample + (RK0 + 2*RK1 + 2*RK2 + RK3)/6
		np.copyto(sample, temp)

	def rRK3d(self, sample, h):           ### sample is the loaded distribution, h the RK timestep ###
		temp = sample.copy()
		N = int(temp.shape[0])
		
		### Runge-Kutta coefficients are stored in these arrays
		RK0 = np.zeros((N,6))
		RK1 = np.zeros((N,6))
		RK2 = np.zeros((N,6))
		RK3 = np.zeros((N,6))
		
		#Coeffs 1
		Norms, Gradlist = self.Interp.rQuery(temp)
		
		RK0[:, 0] = h*(temp[:, 3])
		RK0[:, 3] = -h*self.muom*Gradlist[:, 0]
		RK0[:, 1] = h*(temp[:, 4])
		RK0[:, 4] = -h*self.muom*Gradlist[:, 1]
		RK0[:, 2] = h*(temp[:, 5])
		RK0[:, 5] = -h*self.muom*Gradlist[:, 2]
		
		#New variables
		temp = sample + 0.5*RK0
		
		#Coeffs 2
		Norms, Gradlist = self.Interp.rQuery(temp)
		
		RK1[:, 0] = h*(temp[:, 3])
		RK1[:, 3] = -h*self.muom*Gradlist[:, 0]
		RK1[:, 1] = h*(temp[:, 4])
		RK1[:, 4] = -h*self.muom*Gradlist[:, 1]
		RK1[:, 2] = h*(temp[:, 5])
		RK1[:, 5] = -h*self.muom*Gradlist[:, 2]

		#New variables
		temp = sample + 0.5*RK1

		#Coeff 3
		Norms, Gradlist = self.Interp.rQuery(temp)   

		RK2[:, 0] = h*(temp[:, 3])
		RK2[:, 3] = -h*self.muom*Gradlist[:, 0]
		RK2[:, 1] = h*(temp[:, 4])
		RK2[:, 4] = -h*self.muom*Gradlist[:, 1]
		RK2[:, 2] = h*(temp[:, 5])
		RK2[:, 5] = -h*self.muom*Gradlist[:, 2]
		
		#New variables
		temp = sample + RK2

		#Coeff 4
		Norms, Gradlist = self.Interp.rQuery(temp)  

		RK3[:, 0] = h*(temp[:, 3])
		RK3[:, 3] = -h*self.muom*Gradlist[:, 0]
		RK3[:, 1] = h*(temp[:, 4])
		RK3[:, 4] = -h*self.muom*Gradlist[:, 1]
		RK3[:, 2] = h*(temp[:, 5])
		RK3[:, 5] = -h*self.muom*Gradlist[:, 2]
		
		#Final new variables
		temp = sample + (RK0 + 2*RK1 + 2*RK2 + RK3)/6
		np.copyto(sample, temp)
	
	def Iterate(self, sample, tm, h):
		try:
			if sample.shape[1] > 1:
				t = 0
				while t < tm:
					self.rRK3d(sample, h)		# Perform RK on distribution
					t += h
			else:
				t = 0
				while t < tm:
					self.sRK3d(sample, h)		# Perform RK on distribution
					t += h
		except IndexError:
			t = 0
			while t < tm:
				self.sRK3d(sample, h)		# Perform RK on distribution
				t += h

############################### Skookum ###############################
