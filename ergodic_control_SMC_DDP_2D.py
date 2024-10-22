'''
Trajectory optimization for ergodic control problem 

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: GPL-3.0-only
'''

import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper functions
# ===============================

# Residuals w and Jacobians J in spectral domain
def f_ergodic(x, param):
	[xx,yy] = numpy.mgrid[range(param.nbFct),range(param.nbFct)]

	phi1 = np.zeros((param.nbData,param.nbFct,2))
	dphi1 = np.zeros((param.nbData,param.nbFct,2))

	x1_s = x[0::2]
	x2_s = x[1::2]
	

	phi1[:,:,0] = np.cos(x1_s @ param.kk1.T) / param.L
	dphi1[:,:,0] = - np.sin(x1_s @ param.kk1.T) * np.matlib.repmat(param.kk1.T,param.nbData,1) / param.L
	
	phi1[:,:,1] = np.cos(x2_s @ param.kk1.T) / param.L
	dphi1[:,:,1] = - np.sin(x2_s @ param.kk1.T) * np.matlib.repmat(param.kk1.T,param.nbData,1) / param.L

	phi = phi1[:,xx.flatten(),0] * phi1[:,yy.flatten(),1]

	dphi = np.zeros((param.nbData*param.nbVarX,param.nbFct**2))
	dphi[0::2,:] = dphi1[:,xx.flatten(),0] * phi1[:,yy.flatten(),1]
	dphi[1::2,:] = phi1[:,xx.flatten(),0] * dphi1[:,yy.flatten(),1]

	w = (np.sum(phi,axis=0) / param.nbData).reshape((param.nbFct**2,1))
	J = dphi.T / param.nbData
	return w, J

# Constructs a Hadamard matrix of size n.
def hadamard_matrix(n: int) -> np.ndarray:
    # Base case: A Hadamard matrix of size 1 is just [[1]].
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2.
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    # Combine the four sub-matrices to form a Hadamard matrix of size n.
    h = np.empty((n, n), dtype=int)
    h[:half_size,:half_size] = h_half
    h[half_size:,:half_size] = h_half
    h[:half_size:,half_size:] = h_half
    h[half_size:,half_size:] = -h_half

    return h

## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.nbData = 200 # Number of datapoints
param.nbVarX = 2 # State space dimension
param.nbFct = 8 # Number of Fourier basis functsions
param.nbStates = 2 # Number of Gaussians to represent the spatial distribution
param.nbIter = 50 # Maximum number of iterations for iLQR
param.dt = 1e-2 # Time step length
param.r = 1e-8 # Control weight term

param.xlim = [0,1] # Domain limit
param.L = (param.xlim[1] - param.xlim[0]) * 2 # Size of [-param.xlim(2),param.xlim(2)]
param.om = 2 * np.pi / param.L # Omega
param.range = np.arange(param.nbFct)
param.kk1 = param.om * param.range.reshape((param.nbFct,1))
[xx,yy] = numpy.mgrid[range(param.nbFct),range(param.nbFct)]
sp = (param.nbVarX + 1) / 2 # Sobolev norm parameter

KX = np.zeros((param.nbVarX, param.nbFct, param.nbFct))
KX[0, :, :], KX[1, :, :] = np.meshgrid(param.range, param.range)
param.kk = KX.reshape(param.nbVarX, param.nbFct**2) * param.om
param.Lambda = np.power(xx**2++yy**2+1,-sp).flatten() # Weighting vector

# Enumerate symmetry operations for 2D signal ([-1,-1],[-1,1],[1,-1] and [1,1]), and removing redundant ones -> keeping ([-1,-1],[-1,1])
op = hadamard_matrix(2**(param.nbVarX-1))

# Desired spatial distribution represented as a mixture of Gaussians
param.Mu = np.zeros((2,2))
param.Mu[:,0] = [0.5, 0.7]
param.Mu[:,1] = [0.6, 0.3]

param.Sigma = np.zeros((2,2,2))
sigma1_tmp= np.array([[0.3],[0.1]])
param.Sigma[:,:,0] = sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.identity(param.nbVarX)*5e-3
sigma2_tmp= np.array([[0.1],[0.2]])
param.Sigma[:,:,1] = sigma2_tmp @ sigma2_tmp.T * 3e-1 + np.identity(param.nbVarX)*1e-2 

logs = lambda: None # Object to store logs
logs.x = []
logs.w = []
logs.g = []
logs.e = []

Priors = np.ones(param.nbStates) / param.nbStates # Mixing coefficients

# Transfer matrices (for linear system as single integrator)
Su = np.vstack([
	np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]), 
	np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
]) 
Sx = np.kron(np.ones(param.nbData), np.eye(param.nbVarX)).T

Q = np.diag(param.Lambda) # Precision matrix
R = np.eye((param.nbData-1) * param.nbVarX) * param.r # Control weight matrix (at trajectory level)


# Compute Fourier series coefficients w_hat of desired spatial distribution
# =========================================================================
# Explicit description of w_hat by exploiting the Fourier transform properties of Gaussians (optimized version by exploiting symmetries)

w_hat = np.zeros(param.nbFct**param.nbVarX)
for j in range(param.nbStates):
    for n in range(op.shape[1]):
        MuTmp = np.diag(op[:, n]) @ param.Mu[:, j]
        SigmaTmp = np.diag(op[:, n]) @ param.Sigma[:, :, j] @ np.diag(op[:, n]).T
        cos_term = np.cos(param.kk.T @ MuTmp)
        exp_term = np.exp(np.diag(-0.5 * param.kk.T @ SigmaTmp @ param.kk))
        # Eq.(22) where D=1
        w_hat = w_hat + Priors[j] * cos_term * exp_term
w_hat = w_hat / (param.L**param.nbVarX) / (op.shape[1])
w_hat = w_hat.reshape((-1,1))

# Fourier basis functions (only used for display as a discretized map)
nbRes = 40
xm1d = np.linspace(param.xlim[0], param.xlim[1], nbRes)  # Spatial range for 1D
xm = np.zeros((param.nbStates, nbRes, nbRes))  # Spatial range
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
phim = np.cos(KX[0,:].flatten().reshape((-1,1)) @ xm[0,:].flatten().reshape((1,-1))*param.om) * np.cos(KX[1,:].flatten().reshape((-1,1)) @ xm[1,:].flatten().reshape((1,-1))*param.om) * 2 ** param.nbVarX
hk = np.ones((param.nbFct,1)) * 2
hk[0,0] = 1
HK = hk[xx.flatten()] * hk[yy.flatten()]
phim = phim * np.matlib.repmat(HK,1,nbRes**param.nbVarX)

# Desired spatial distribution
g = w_hat.T @ phim

image_path = "reconstructed_distribution"
save_plot(xm,g,nbRes,image_path)

# Myopic ergodic control (for initialisation)
# ===============================
u_max = 4e0 # Maximum speed allowed
u_norm_reg = 1e-3 

xt = np.array([[0.1],[0.1]]) # Initial position
wt = np.zeros((param.nbFct**param.nbVarX,1))
u = np.zeros((param.nbData-1,param.nbVarX)) # Initial control command

for t in range(param.nbData-1):
	phi1 = np.cos(xt @ param.kk1.T) / param.L # In 1D
	dphi1 = - np.sin(xt @ param.kk1.T) * np.matlib.repmat(param.kk1.T, param.nbVarX,1) / param.L # in 1D

	phi = (phi1[0,xx.flatten()] * phi1[1,yy.flatten()]).reshape((-1,1)) # Fourier basis function
	dphi = np.vstack((
		dphi1[0,xx.flatten()] * phi1[1,yy.flatten()],
		phi1[0,xx.flatten()] * dphi1[1,yy.flatten()]
	)) # Gradient of Fourier basis functions

	wt = wt + phi
	w = wt / (t+1) #w are the Fourier series coefficients along trajectory 

	# Controller with constrained velocity norm
	u_t = -dphi @ np.diag(param.Lambda) @ (w-w_hat)
	u_t = u_t * u_max / (np.linalg.norm(u_t)+u_norm_reg) # Velocity command
	u[t] = u_t.T
	xt = xt + u_t * param.dt # Update of position

# iLQR
# ===============================

u = u.reshape((-1,1)) # Initial control command
#u = (u + np.random.normal(size=(len(u),1))).reshape((-1,1))

x0 = np.array([[0.1],[0.1]]) # Initial position

for i in range(param.nbIter):
	x = Su @ u + Sx @ x0 # System evolution
	w, J = f_ergodic(x, param) # Fourier series coefficients and Jacobian
	f = w - w_hat # Residual

	du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f - u * param.r) # Gauss-Newton update

	cost0 = f.T @ Q @ f + np.linalg.norm(u)**2 * param.r # Cost
	
	# Log data
	logs.x += [x] # Save trajectory in state space
	logs.w += [w] # Save Fourier coefficients along trajectory
	logs.g += [w.T @ phim] # Save reconstructed spatial distribution (for visualization)
	logs.e += [cost0.squeeze()] # Save reconstruction error

	# Estimate step size with backtracking line search method
	alpha = 1
	while True:
		utmp = u + du * alpha
		xtmp = Sx @ x0 + Su @ utmp
		wtmp, _ = f_ergodic(xtmp,param)
		ftmp = wtmp - w_hat 
		cost = ftmp.T @ Q @ ftmp + np.linalg.norm(utmp)**2 * param.r
		if cost < cost0 or alpha < 1e-3:
			print("Iteration {}, cost: {}".format(i,cost.squeeze()))
			break
		alpha /= 2
	
	u = u + du * alpha

	if np.linalg.norm(du * alpha) < 1E-2:
		break # Stop iLQR iterations when solution is reached

# Plots
# ===============================

plt.figure(figsize=(16,8))

# x
plt.subplot(2,3,1)
X = np.squeeze(xm[0, :, :])
Y = np.squeeze(xm[1, :, :])
G = np.reshape(g, [nbRes, nbRes])  # original distribution
G = np.where(G > 0, G, 0)
plt.contourf(X, Y, G, cmap="gray_r")
plt.plot(logs.x[0][0::2],logs.x[0][1::2], linestyle="-", color=[.7,.7,.7],label="Initial")
plt.plot(logs.x[-1][0::2],logs.x[-1][1::2], linestyle="-", color=[0,0,0],label="Final")
plt.axis("scaled")
plt.legend()
plt.title("Spatial distribution g(x)")
plt.xticks([])
plt.yticks([])

# w_hat
plt.subplot(2,3,2)
plt.title(r"Desired Fourier coefficients $\hat{w}$")
plt.imshow(np.reshape(w_hat, [param.nbFct, param.nbFct]).T, cmap="gray_r")
plt.xticks([])
plt.yticks([])

# w
plt.subplot(2,3,3)
plt.title(r"Reproduced Fourier coefficients $w$")
plt.imshow(np.reshape(logs.w[-1] / param.nbData, [param.nbFct, param.nbFct]).T, cmap="gray_r")
plt.xticks([])
plt.yticks([])

# error
plt.subplot(2,1,2)
plt.xlabel("n",fontsize=16)
plt.ylabel("$\epsilon$",fontsize=16)
plt.plot(logs.e,color=[0,0,0])

plt.show()