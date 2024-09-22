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
	return w.squeeze(), J

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

#Residuals f and Jacobians J for staying within bounded domain
def f_domain(x,param):
    sz = param.xlim[1]/ 2
    ftmp = x - sz  # Residuals
    f = (ftmp - np.sign(ftmp) * sz)  # Bounded domain
    
    J = np.eye(param.nbVarX * param.nbData)  
    id = np.abs(ftmp) < sz
    f[id] = 0
    id_indices = np.where(id)[0]
    J[id_indices, id_indices] = 0
    
    return f.squeeze(), J


## Parameters
# ===============================

param = lambda: None  # Lazy way to define an empty class in Python
param.nbData = 200  # Number of datapoints
param.nbVarX = 2  # State space dimension
param.nbFct = 8  # Number of Fourier basis functions
param.nbStates = 2  # Number of Gaussians to represent the spatial distribution
param.nbAgents = 9  # Number of agents
param.nbIter = 50  # Maximum number of iterations for iLQR
param.dt = 1e-2  # Time step length
param.r = 1e-8  # Control weight term
param.qd = 0e4  # Bounded domain weight term
param.Mu_ma = np.tile(np.array([[0.3], [0.9]]), (1, param.nbAgents)) # Target positions for agents

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
Qd = np.eye(param.nbData * param.nbVarX) * param.qd
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

#Ergodic controla as a trajectory planning problem
# =================================================
'''TO BE MODIFIED ADDING HAFTONING'''
# Initialize the random starting positions of the agents
x0 = np.random.rand(param.nbVarX, param.nbAgents)  # Initial positions of agents
# Shift the positions and assign them to param.Mu
param.Mu_ma = np.hstack((x0[:, 1:], x0[:, :1]))

# Initial control commands
# Initialize u to store control commands for all agents
u = np.zeros((param.nbVarX * (param.nbData - 1), param.nbAgents))

for m in range(param.nbAgents):
    # Initial control commands for each agent
    u[:, m] = np.tile((param.Mu_ma[:, m] - x0[:, m]) / ((param.nbData - 1) * param.dt), (param.nbData - 1))

# Initialize other matrices for agents
x = np.zeros((param.nbVarX * param.nbData, param.nbAgents))
fd = np.zeros((param.nbVarX * param.nbData, param.nbAgents))
w = np.zeros((param.nbFct ** param.nbVarX, param.nbAgents))
Jd = np.zeros((param.nbVarX * param.nbData, param.nbVarX * param.nbData, param.nbAgents))
J = np.zeros((param.nbFct ** param.nbVarX, param.nbVarX * param.nbData, param.nbAgents))
fdtmp = np.zeros((param.nbVarX * param.nbData, param.nbAgents))
wtmp = np.zeros((param.nbFct ** param.nbVarX, param.nbAgents))

# Multi-agent iLQR
for n in range(param.nbIter):
    
    x = Sx @ x0 + Su @ u #System evolution
    
    for m in range(param.nbAgents):
        
        fd[:,m], Jd[:,:,m] = f_domain(x[:,m][:, np.newaxis], param); #Residuals and Jacobians for staying within bounded domain
        w[:, m], J[:, :, m] = f_ergodic(x[:, m][:,np.newaxis], param)  # Fourier series coefficients and Jacobian for each agent
   
    w_avg = np.mean(w, axis=1)  # Average Fourier coefficients across agents
    f = w_avg - w_hat  # Residuals for ergodic control

    # Gauss-Newton update for each agent
    for m in range(param.nbAgents):
        a = (J[:, :, m].T @ Q @ J[:, :, m])
        a.shape()
        print(Jd[:, :, m].T @ Qd @ Jd[:, :, m]).shape()
        du = np.linalg.inv(Su.T @ (J[:, :, m].T @ Q @ J[:, :, m] + Jd[:, :, m].T @ Qd @ Jd[:, :, m]) + R) @ (-Su.T @ (J[:, :, m].T @ Q @ f + Jd[:, :, m].T @ Qd @ fd) - u[:, m] * param.r)

    cost0 = f.T @ Q @ f + np.linalg.norm(fd)**2*param.qd + np.linalg.norm(u)**2 * param.r  # Cost
           
	# Log data
    logs.x += [x]  # Save trajectory in state space
    logs.w += [w]  # Save Fourier coefficients along trajectory
    logs.g += [w.T @ phim]  # Save reconstructed spatial distribution (for visualization)
    logs.e += [cost0.squeeze()]  # Save reconstruction error

    # Estimate step size with backtracking line search method
    alpha = 1
    while True:
        utmp = u + du * alpha
        xtmp = Sx @ x0 + Su @ utmp
        
        for m in range(param.nbAgents):
        
            wtmp[:, m], _ = f_ergodic(x[:, m], param)  # Fourier series coefficients and Jacobian for each agent
            fdtmp[:, m] , _= f_domain(xtmp[:, m],param)  # Residuals and Jacobians for staying within bounded domain
        
        wtmp = np.mean(wtmp, axis=1)  # Average Fourier coefficients across agents
        ftmp = wtmp - w_hat  # Residuals for ergodic control
        
        cost = ftmp.T @ Q @ ftmp +  np.linalg.norm(fdtmp)**2 * param.qd+ np.linalg.norm(utmp)**2 * param.r
        if cost < cost0 or alpha < 1e-3:
            print("Iteration {}, cost: {}".format(i, cost.squeeze()))
            break
        alpha /= 2
    
    u = u + du * alpha

    if np.linalg.norm(du * alpha) < 1E-2:
        break  # Stop iLQR iterations when solution is reached

# Plots
# ===============================

import matplotlib.pyplot as plt
import numpy as np

# Initialize figure
plt.figure(figsize=(16, 8))

# Plot the spatial distribution g(x)
plt.subplot(1, 4, 1)
X = np.squeeze(xm[0, :, :])
Y = np.squeeze(xm[1, :, :])
G = np.reshape(g, [nbRes, nbRes])  # original distribution
G = np.where(G > 0, G, 0)

# Plot the spatial distribution as a contour plot
plt.contourf(X, Y, G, cmap="gray_r")

# Loop through each agent and plot its initial and final trajectories
for m in range(param.nbAgents):
    plt.plot(logs.x[0][0::2, m], logs.x[0][1::2, m], linestyle="-", color=[.7, .7, .7], label="Initial" if m == 0 else None)
    plt.plot(logs.x[-1][0::2, m], logs.x[-1][1::2, m], linestyle="-", color=[0, 0, 0], label="Final" if m == 0 else None)
    plt.plot(logs.x[-1][0, m], logs.x[-1][1, m], marker="o", markersize=5, color=[0, 0, 0])

# Plot the target positions
plt.plot(param.Mu[0, :], param.Mu[1, :], 'x', markersize=6, linewidth=4, color=[0, .6, 0], label="Target")
plt.axis("scaled")
plt.title("Spatial distribution g(x)")
plt.legend()
plt.xticks([])
plt.yticks([])

# Plot the desired Fourier coefficients w_hat
plt.subplot(1, 4, 2)
plt.title(r"Desired Fourier coefficients $\hat{w}$")
plt.imshow(np.reshape(w_hat, [param.nbFct, param.nbFct]).T, cmap="gray_r")
plt.xticks([])
plt.yticks([])

# Plot the reproduced Fourier coefficients w
plt.subplot(1, 4, 3)
plt.title(r"Reproduced Fourier coefficients $w$")
plt.imshow(np.reshape(np.mean(logs.w[-1], axis=1) / param.nbData, [param.nbFct, param.nbFct]).T, cmap="gray_r")
plt.xticks([])
plt.yticks([])

# Plot the cost/error over iterations
plt.subplot(1, 4, 4)
plt.plot(logs.e, color=[0, 0, 0])
plt.title("Cost over iterations")
plt.xlabel("Iteration")
plt.ylabel(r"Cost $\epsilon$")
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


