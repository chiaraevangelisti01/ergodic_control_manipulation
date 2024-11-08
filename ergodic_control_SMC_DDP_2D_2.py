'''
Trajectory optimization for ergodic control problem 

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

License: GPL-3.0-only
'''

import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from record_trajectories_function import generate_trajectories
import glob
import math
from scipy.linalg import block_diag

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

    #phi = phi1[:,xx.flatten(),0] * phi1[:,yy.flatten(),1]
    phi = phi1[:, xx, 0] * phi1[:, yy, 1]
    phi = phi.reshape(param.nbData, -1, order='F')

    dphi = np.zeros((param.nbData * param.nbVarPos, param.nbFct**2))

    # dphi[0:param.nbData * param.nbVarPos:2, :]= dphi1[:,xx.flatten(),0] * phi1[:,yy.flatten(),1]
    # dphi[1:param.nbData * param.nbVarPos:2, :]= phi1[:,xx.flatten(),0] * dphi1[:,yy.flatten(),1]
    dphi[0:param.nbData * param.nbVarPos:2, :] = (dphi1[:, xx, 0] * phi1[:, yy, 1]).reshape(param.nbData, -1, order='F')
    dphi[1:param.nbData * param.nbVarPos:2, :] = (phi1[:, xx, 0] * dphi1[:, yy, 1]).reshape(param.nbData, -1, order='F')

    w = (np.sum(phi, axis=0) / param.nbData).reshape((param.nbFct**2, 1))
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

def f_domain(x,param):
    sz = param.xlim[1]/ 2
    ftmp = x - sz  # Residuals
    f = ftmp - np.sign(ftmp) * sz
    
    J = np.eye(param.nbVarPos * param.nbData)  
    id = np.abs(ftmp) < sz
    f[id] = 0
    id_indices = np.where(id)[0]
    J[id_indices, id_indices] = 0
   
    return f, J

def f_reach(x ,param):
    
    f = x - param.Mu_reach  # Residuals
    J = np.eye(param.nbVarX * param.nbPoints)  # Jacobian
    return f, J

def f_curvature(x, param):
    
    dx = x[param.nbVarPos:param.nbVarPos * 2, :]  # Velocity (first derivative)
    ddx = x[param.nbVarPos*2:param.nbVarPos * 3, :]  # Acceleration
    
    dxn = np.sum(dx**2, axis=0)**(3/2)
    f = (dx[0, :] * ddx[1, :] - dx[1, :] * ddx[0, :]) / (dxn + 1E-8)
    
    s11 = np.zeros(param.nbVarX); s11[param.nbVarPos] = 1
    s12 = np.zeros(param.nbVarX); s12[param.nbVarPos + 1] = 1
    s21 = np.zeros(param.nbVarX); s21[2 * param.nbVarPos] = 1
    s22 = np.zeros(param.nbVarX); s22[2 * param.nbVarPos + 1] = 1

    Sa = np.outer(s11, s22) - np.outer(s12, s21)
    Sb = np.outer(s11, s11) + np.outer(s12, s12)

    
    for t in range(param.nbData):
        a = x[:, t].T @ Sa @ x[:, t]
        b = x[:, t].T @ Sb @ x[:, t] + 1E-8
        Jtmp =  b**(-3/2) * (Sa+Sa.T) @ x[:, t] - 3 * a * b**(-5/2) * Sb @ x[:, t]
        if t ==0:
            J = Jtmp
        else:
            J =block_diag(J,Jtmp.T)
        
    return f, J

def reference_curvature(x):
    # Compute first derivatives (velocities)
    dx = np.gradient(x[0, :])
    dy = np.gradient(x[1, :])

    # Compute second derivatives (accelerations)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Compute curvature using the formula
    dxn = (dx**2 + dy**2)**(3/2)
    fc_ref = (dx * ddy - dy * ddx) / (dxn + 1E-8)  # Avoid division by zero

    return fc_ref

def transfer_matrices(A, B):
    nbVarX, nbVarU, nbData = B.shape
    nbData = nbData + 1

    Sx = np.kron(np.ones((nbData, 1)), np.eye(nbVarX))
    Su = np.zeros((nbVarX *nbData , nbVarU * (nbData - 1))) #CHECK THIS: ADDITIONAL DIMENSIONS?
    
    for t in range(1, nbData):  
        id1 = np.arange((t - 1) * nbVarX, t * nbVarX)
        id2 = np.arange(t * nbVarX, (t + 1) * nbVarX)
        id3 = np.arange((t - 1) * nbVarU, t * nbVarU)
        
        Sx[id2, :] = A[:, :, t - 1] @ Sx[id1, :]
        Su[id2, :] = A[:, :, t - 1] @ Su[id1, :]
        Su[id2[:, None], id3] = B[:, :, t - 1]
    
    return Su, Sx

def save_plot(xm,g,nbRes,image_name):
    # Create a new figure
    plt.figure(figsize=(8, 6))
    plt.axis("off")

    # Plot the spatial distribution g(x)
    X = np.squeeze(xm[0, :, :])
    Y = np.squeeze(xm[1, :, :])
    G = np.reshape(g, [nbRes, nbRes])  # original distribution
    G = np.where(G > 0, G, 0)

    # Plot the spatial distribution as a contour plot
    plt.contourf(X, Y, G, cmap="gray_r")
    # Save the plot as a PNG file
    plt.savefig(image_name)


def create_trajectories(directory, num_agents, num_points, image_path):
    """create trajectories and load them after creation."""
    
    generate_trajectories(directory, num_agents, num_points, image_path)
    trajectories = []
    file_prefix = 'traj'

    for filename in sorted(glob.glob(os.path.join(directory, "{}*.npy".format(file_prefix)))):
        traj = np.load(filename)  # Load trajectory (shape: num_points x 2)
        trajectories.append(traj)
        #print(f"Loaded trajectory from {filename}, shape: {traj.shape}")

    return trajectories


## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.nbData = 200 # Number of datapoints
param.nbVarPos = 2 # Position space dimension
param.nbDeriv = 3 # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVarX = param.nbVarPos * param.nbDeriv # State space dimension
param.nbFct = 8 # Number of Fourier basis functsions
param.nbStates = 2 # Number of Gaussians to represent the spatial distribution
param.nbIter = 50 # Maximum number of iterations for iLQR
param.nbPoints = 1  # Number of viapoints to reach (here, final target point)
param.dt = 1e-2 # Time step length
param.qd = 1e0; #Bounded domain weight term
param.qr =1e2   # Reach target weight term
param.qc = 1e-6 #Curvature weight term
param.r = 1e-9 # Control weight term
param.nbAgents = 1
#param.Mu_reach = np.array([[0.3],[0.9]]) #Target to reach
param.Mu_reach = np.concatenate(([0.3, 0.9], np.zeros(param.nbVarX - param.nbVarPos))).reshape(-1, 1)

param.xlim = [0,1] # Domain limit
param.L = (param.xlim[1] - param.xlim[0]) * 2 # Size of [-param.xlim(2),param.xlim(2)]
param.om = 2 * np.pi / param.L # Omega
param.range = np.arange(param.nbFct)
param.kk1 = param.om * param.range.reshape((param.nbFct,1))
[xx,yy] = numpy.mgrid[range(param.nbFct),range(param.nbFct)]
sp = (param.nbVarPos + 1) / 2 # Sobolev norm parameter

KX = np.zeros((param.nbVarPos, param.nbFct, param.nbFct))
KX[0, :, :], KX[1, :, :] = np.meshgrid(param.range, param.range)
param.kk = KX.reshape(param.nbVarPos, param.nbFct**2) * param.om
param.Lambda = np.power(xx**2+yy**2+1,-sp).flatten() # Weighting vector

# Time occurrence of viapoints
tl = np.linspace(1, param.nbData, param.nbPoints + 1)
tl = np.round(tl[1:]).astype(int)  
idx = (tl - 1) * param.nbVarX + np.arange(1, param.nbVarX + 1).reshape(-1, 1)
idx= idx.flatten()
idp = np.arange(param.nbData)[:, None] * param.nbVarX + np.arange(1, param.nbVarPos + 1)
idp = idp.flatten()  # position indeces


# Enumerate symmetry operations for 2D signal ([-1,-1],[-1,1],[1,-1] and [1,1]), and removing redundant ones -> keeping ([-1,-1],[-1,1])
op = hadamard_matrix(2**(param.nbVarPos-1))

# Desired spatial distribution represented as a mixture of Gaussians
param.Mu = np.zeros((2,2))
param.Mu[:,0] = [0.5, 0.7]
param.Mu[:,1] = [0.6, 0.3]

param.Sigma = np.zeros((2,2,2))
sigma1_tmp= np.array([[0.3],[0.1]])
param.Sigma[:,:,0] = sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.identity(param.nbVarPos)*5e-3
sigma2_tmp= np.array([[0.1],[0.2]])
param.Sigma[:,:,1] = sigma2_tmp @ sigma2_tmp.T * 3e-1 + np.identity(param.nbVarPos)*1e-2 
Priors = np.ones(param.nbStates) / param.nbStates # Mixing coefficients

logs = lambda: None # Object to store logs
logs.x = []
logs.w = []
logs.g = []
logs.e = []


# Transfer matrices (for linear system as single integrator)
# Su = np.vstack([
# 	np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]), 
# 	np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
# ]) 
# Sx = np.kron(np.ones(param.nbData), np.eye(param.nbVarX)).T
# Sr = Su[idx-1, :]


#Dybamical system settings (discrete version)
A1d = np.zeros((param.nbDeriv, param.nbDeriv))
for i in range(param.nbDeriv):
    A1d += np.diag(np.ones(param.nbDeriv - i), k=i) * (param.dt ** i) / math.factorial(i)
B1d = np.zeros((param.nbDeriv, 1))
for i in range(1, param.nbDeriv + 1):
    B1d[param.nbDeriv - i, 0] = (param.dt ** i) / math.factorial(i)
A = np.kron(A1d, np.eye(param.nbVarPos))  
A = np.repeat(A[:, :, np.newaxis], param.nbData - 1, axis=2)  # Replicate along the third dimension
B = np.kron(B1d, np.eye(param.nbVarPos))  
B = np.repeat(B[:, :, np.newaxis], param.nbData - 1, axis=2)  # 

Su, Sx = transfer_matrices(A, B)
Sr = Su[idx-1, :]

Q = np.diag(param.Lambda) # Precision matrix
Qd = np.eye(param.nbData * param.nbVarPos) * param.qd
Qr = np.eye(param.nbPoints * param.nbVarX) * param.qr
R = np.eye((param.nbData-1) * param.nbVarPos) * param.r # Control weight matrix (at trajectory level)


# Compute Fourier series coefficients w_hat of desired spatial distribution
# =========================================================================
# Explicit description of w_hat by exploiting the Fourier transform properties of Gaussians (optimized version by exploiting symmetries)

w_hat = np.zeros(param.nbFct**param.nbVarPos)
for j in range(param.nbStates):
    for n in range(op.shape[1]):
        MuTmp = np.diag(op[:, n]) @ param.Mu[:, j]
        SigmaTmp = np.diag(op[:, n]) @ param.Sigma[:, :, j] @ np.diag(op[:, n]).T
        cos_term = np.cos(param.kk.T @ MuTmp)
        exp_term = np.exp(np.diag(-0.5 * param.kk.T @ SigmaTmp @ param.kk))
        # Eq.(22) where D=1
        w_hat = w_hat + Priors[j] * cos_term * exp_term
w_hat = w_hat / (param.L**param.nbVarPos) / (op.shape[1])
w_hat = w_hat.reshape((-1,1))

# Fourier basis functions (only used for display as a discretized map)
nbRes = 40
xm1d = np.linspace(param.xlim[0], param.xlim[1], nbRes)  # Spatial range for 1D
xm = np.zeros((param.nbStates, nbRes, nbRes))  # Spatial range
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
phim = np.cos(KX[0,:].flatten().reshape((-1,1)) @ xm[0,:].flatten().reshape((1,-1))*param.om) * np.cos(KX[1,:].flatten().reshape((-1,1)) @ xm[1,:].flatten().reshape((1,-1))*param.om) * 2 ** param.nbVarPos
hk = np.ones((param.nbFct,1)) * 2
hk[0,0] = 1
HK = hk[xx.flatten()] * hk[yy.flatten()]
phim = phim * np.matlib.repmat(HK,1,nbRes**param.nbVarPos)

# Desired spatial distribution
g = w_hat.T @ phim

image_path = "reconstructed_distribution"
save_plot(xm,g,nbRes,image_path)
directory = "mouse_trajectories"
ref_trajectories = create_trajectories(directory, param.nbAgents, param.nbData, image_path+'.png')
fc_ref = reference_curvature(ref_trajectories[0].T)


# Myopic ergodic control (for initialisation)
# ===============================
u_max = 4e0 # Maximum speed allowed
u_norm_reg = 1e-3 

xt = np.array([[0.1],[0.1]]) # Initial position
wt = np.zeros((param.nbFct**param.nbVarX,1))
#u = np.zeros((param.nbData-1,param.nbVarX)) # Initial control command
#u = np.random.uniform(-u_max*0.75, u_max*0.75, (param.nbData-1, param.nbVarX))

# for t in range(param.nbData-1):
# 	phi1 = np.cos(xt @ param.kk1.T) / param.L # In 1D
# 	dphi1 = - np.sin(xt @ param.kk1.T) * np.matlib.repmat(param.kk1.T, param.nbVarX,1) / param.L # in 1D

# 	phi = (phi1[0,xx.flatten()] * phi1[1,yy.flatten()]).reshape((-1,1)) # Fourier basis function
# 	dphi = np.vstack((
# 		dphi1[0,xx.flatten()] * phi1[1,yy.flatten()],
# 		phi1[0,xx.flatten()] * dphi1[1,yy.flatten()]
# 	)) # Gradient of Fourier basis functions

# 	wt = wt + phi
# 	w = wt / (t+1) #w are the Fourier series coefficients along trajectory 

# 	# Controller with constrained velocity norm
# 	u_t = -dphi @ np.diag(param.Lambda) @ (w-w_hat)
# 	u_t = u_t * u_max / (np.linalg.norm(u_t)+u_norm_reg) # Velocity command
# 	u[t] = u_t.T
# 	xt = xt + u_t * param.dt # Update of position

# iLQR
# ===============================

#u = u.reshape((-1,1)) # Initial control command
#u = (u + np.random.normal(size=(len(u),1))).reshape((-1,1))

#x0 = np.array([[0.1],[0.1]]) # Initial position
x0 = np.concatenate(([0.1, 0.1], np.zeros(param.nbVarX - param.nbVarPos))).reshape(-1, 1)
u = np.tile((param.Mu_reach[:2] - x0[:2]) / ((param.nbData - 1) * param.dt), (param.nbData - 1, 1))

for i in range(param.nbIter):
    x = Su @ u + Sx @ x0 # System evolution
    
    fd, Jd = f_domain(x[idp-1], param)
    w, J = f_ergodic(x[idp-1], param) # Fourier series coefficients and Jacobian
    fr, Jr = f_reach(x[idx-1], param) # Reach target
    fc, Jc = f_curvature(x.reshape(param.nbVarX,param.nbData, order = 'F'),param)
    f = w - w_hat # Residual
    fc_delta = fc.reshape(-1,1) - fc_ref.reshape(-1,1)
  
    
    du = np.linalg.inv(Su[idp-1, :].T @ (J.T @ Q @ J + Jd.T @ Qd @ Jd) @ Su[idp-1,:] + Su.T @ Jc.T @ Jc @ Su *param.qc+
     Sr.T @ Jr.T @ Qr @ Jr @ Sr + R) @ (-Su[idp-1,:].T @ (J.T @ Q @ f + Jd.T @ Qd @ fd) -Su.T @ Jc.T @ fc_delta *param.qc- Sr.T @ Jr.T @ Qr @ fr - u * param.r) # Gauss-Newton update
    
    cost0 = f.T @ Q @ f + np.linalg.norm(fd)**2 * param.qd  + np.linalg.norm(fc_delta)**2 * param.qc + np.linalg.norm(fr)**2 * param.qr+ np.linalg.norm(u)**2 * param.r # Cost
    
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
        fdtmp, _ = f_domain(xtmp[idp-1], param)  # Residuals and Jacobians for staying within bounded domain
        frtmp, _ = f_reach(xtmp[idx-1], param)  # Residuals and Jacobians for reaching target
        fctmp, _ = f_curvature(xtmp.reshape(param.nbVarX,param.nbData, order = 'F'),param)
        wtmp, _ = f_ergodic(xtmp[idp-1], param)
        
        ftmp = wtmp - w_hat 
        cost = ftmp.T @ Q @ ftmp + np.linalg.norm(fdtmp)**2 * param.qd + np.linalg.norm(fctmp-fc_ref)**2 * param.qc + np.linalg.norm(frtmp)**2 * param.qr+ np.linalg.norm(utmp)**2 * param.r
        if cost < cost0 or alpha < 1e-3:
            print("Iteration {}, cost: {}".format(i, cost.squeeze()))
            #print(cost.squeeze())
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
plt.plot(logs.x[0][0::param.nbVarX],logs.x[0][1::param.nbVarX], linestyle="-", color=[.7,.7,.7],label="Initial")
plt.plot(logs.x[-1][0::param.nbVarX],logs.x[-1][1::param.nbVarX], linestyle="-", color=[0,0,0],label="Final")
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