'''
iLQR (batch formulation) applied to a Franka Emika manipulator for a viapoints task in a 3D workspace

Copyright (c) 2024 Idiap Research Institute <https://www.idiap.ch/>
Written by Philip Abbet <philip.abbet@idiap.ch>,
Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: GPL-3.0-only
'''

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# Helper functions
# ===============================

# Cost and gradient for a viapoints reaching task (in object coordinate system)
def f_reach(x, param):
	ftmp, _ = fkin(x, param)
	f = logmap(ftmp, param.Mu)
	J = np.zeros([param.nbPoints * 6, param.nbPoints * param.nbVarX])
	for t in range(param.nbPoints):
		J[t*6:(t+1)*6, t*param.nbVarX:(t+1)*param.nbVarX] = Jkin_num(x[:,t], param)
	return f, J

# Forward kinematics for end-effector (from DH parameters)
def fkin(x, param):
	f = np.zeros((7, x.shape[1]))
	R = np.zeros((3, 3, x.shape[1]))
	for t in range(x.shape[1]):
		ftmp, Rtmp = fkin0(x[:,t], param)
		R[:,:,t] = Rtmp[:,:,-1]
		f[0:3,t] = ftmp[:,-1]
		f[3:7,t] = R2q(Rtmp[:,:,-1])
	return f, R

# Forward kinematics for all robot articulations, for articulatory joints and modified DH convention (a.k.a. Craig's formulation)
def fkin0(x, param):
	N = param.dh.q.shape[1]
	x = np.append(x, 0)
	Tf = np.eye(4)
	R = np.zeros((3,3,N))
	f = np.zeros((3,N+1))
	for n in range(N):
		ct = np.cos(x[n] + param.dh.q_offset[n])
		st = np.sin(x[n] + param.dh.q_offset[n])
		ca = np.cos(param.dh.alpha[n])
		sa = np.sin(param.dh.alpha[n])
		Tf = Tf @ [[ct,    -st,     0,   param.dh.r[n]   ],
				   [st*ca,  ct*ca, -sa, -param.dh.d[n]*sa],
				   [st*sa,  ct*sa,  ca,  param.dh.d[n]*ca],
				   [0,      0,      0,   1               ]]
		R[:,:,n] = Tf[0:3,0:3]
		f[:,n+1] = Tf[0:3,-1]
	return f, R

# Jacobian of forward kinematics function with numerical computation
def Jkin_num(x, param):
	e = 1E-6
	X = np.matlib.repmat(x, param.nbVarX, 1).T
	F1, _ = fkin(X, param)
	F2, _ = fkin(X + np.eye(param.nbVarX) * e, param)
	J = logmap(F2, F1) / e # Error by considering manifold
	return J

# Logarithmic map for R^3 x S^3 manifold (with e in tangent space)
def logmap(f, f0):
	e = np.ndarray((6, f.shape[1]))
	e[0:3, :] = f[0:3,:] - f0[0:3,:] # Error on R^3
	for t in range(f.shape[1]):
		H = dQuatToDxJac(f0[3:7,t])
		e[3:6,t] = 2 * H @ logmap_S3(f[3:7,t], f0[3:7,t]) # Error on S^3
	return e

# Logarithmic map for S^3 manifold (with e in ambient space)
def logmap_S3(x, x0):
	x0 = x0.reshape((4, 1))
	x = x.reshape((4, 1))

	th = acoslog(x0.T @ x)

	u = x - (x0.T @ x) * x0
	if np.linalg.norm(u) > 1e-9:
		u = np.multiply(th, u) / np.linalg.norm(u)

	return u[:, 0]

# Arcosine redefinition to make sure the distance between antipodal quaternions is zero
def acoslog(x):
	y = np.arccos(x)[0][0]
	if (x>=-1.0) and (x<0):
		y = y - np.pi
	return y

def dQuatToDxJac(q):
	return np.array([
		[-q[1], q[0], -q[3], q[2]],
		[-q[2], q[3], q[0], -q[1]],
		[-q[3], -q[2], q[1], q[0]],
	])

# Unit quaternion to rotation matrix conversion (for quaternions as [w,x,y,z])
def q2R(q):
	return np.array([
		[1.0 - 2.0 * q[2]**2 - 2.0 * q[3]**2, 2.0 * q[1] * q[2] - 2.0 * q[3] * q[0], 2.0 * q[1] * q[3] + 2.0 * q[2] * q[0]],
		[2.0 * q[1] * q[2] + 2.0 * q[3] * q[0], 1.0 - 2.0 * q[1]**2 - 2.0 * q[3]**2, 2.0 * q[2] * q[3] - 2.0 * q[1] * q[0]],
		[2.0 * q[1] * q[3] - 2.0 * q[2] * q[0], 2.0 * q[2] * q[3] + 2.0 * q[1] * q[0], 1.0 - 2.0 * q[1]**2 - 2.0 * q[2]**2],
	])

# Rotation matrix to unit quaternion conversion
def R2q(R):
	R = R.T
	K = np.array([
			[R[0,0]-R[1,1]-R[2,2], R[1,0]+R[0,1], R[2,0]+R[0,2], R[1,2]-R[2,1]],
			[R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], R[2,1]+R[1,2], R[2,0]-R[0,2]],
			[R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], R[0,1]-R[1,0]],
			[R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]],
	]) / 3.0
	e_val, e_vec = np.linalg.eig(K) # unsorted eigenvalues
	q = np.real([e_vec[3, np.argmax(e_val)], *e_vec[0:3, np.argmax(e_val)]]) # for quaternions as [w,x,y,z]
	return q

# Plot coordinate system
def plotCoordSys(ax, x, R, width=1):
	for t in range(x.shape[1]):
		ax.plot([x[0,t], x[0,t]+R[0,0,t]], [x[1,t], x[1,t]+R[1,0,t]], [x[2,t], x[2,t]+R[2,0,t]], linewidth=2*width, color=[1.0, 0.0, 0.0])
		ax.plot([x[0,t], x[0,t]+R[0,1,t]], [x[1,t], x[1,t]+R[1,1,t]], [x[2,t], x[2,t]+R[2,1,t]], linewidth=2*width, color=[0.0, 1.0, 0.0])
		ax.plot([x[0,t], x[0,t]+R[0,2,t]], [x[1,t], x[1,t]+R[1,2,t]], [x[2,t], x[2,t]+R[2,2,t]], linewidth=2*width, color=[0.0, 0.0, 1.0])
		ax.plot(x[0,t], x[1,t], x[2,t], 'o', markersize=10, color='black')


## Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 50 # Number of datapoints
param.nbIter = 30 # Maximum number of iterations for iLQR
param.nbPoints = 2 # Number of viapoints
param.nbVarX = 7 # State space dimension (x1,x2,x3,...)
param.nbVarU = param.nbVarX # Control space dimension (dx1,dx2,dx3,...)
param.nbVarF = 7 # Task space dimension (f1,f2,f3 for position, f4,f5,f6,f7 for unit quaternion)
param.r = 1e-6 # Control weighting term

Rtmp = q2R([np.cos(np.pi/3), np.sin(np.pi/3), 0.0, 0.0])
param.MuR = np.dstack((Rtmp, Rtmp))
param.Mu = np.ndarray((param.nbVarF, param.nbPoints))
param.Mu[0:3, 0] = [.6, 0, .2]
param.Mu[3:7, 0] = R2q(param.MuR[:,:,0])
param.Mu[0:3, 1] = [.3, .5, .1]
param.Mu[3:7, 1] = R2q(param.MuR[:,:,1])

# DH parameters of Franka Emika robot
param.dh = lambda: None # Lazy way to define an empty class in python
param.dh.convention = 'm' # modified DH, a.k.a. Craig's formulation
param.dh.type = ['r'] * (param.nbVarX+1) # Articulatory joints
param.dh.q = np.zeros((1, param.nbVarX+1)) # Angle about previous z
param.dh.q_offset = np.zeros((param.nbVarX+1,)) # Offset on articulatory joints
param.dh.alpha = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0] # Angle about common normal
param.dh.d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107] # Offset along previous z to the common normal
param.dh.r = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0] # Length of the common normal


# Main program
# ===============================

# Precision matrix
# Q = np.identity((param.nbVarF-1) * param.nbPoints) # Full precision matrix
Qr = np.diag([1.0,1.0,1.0,1.0,1.0,0.0] * param.nbPoints) # Precision matrix in relative coordinate frame (tool frame) (by removing orientation constraint on 3rd axis)

# Control weight matrix (at trajectory level)
R = np.eye((param.nbData-1) * param.nbVarU) * param.r

# Time occurrence of viapoints
tl = np.linspace(1, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([i + np.arange(0,param.nbVarX,1) for i in (tl*param.nbVarX)]).flatten()

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([
	np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]), 
	np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
]) 
Sx0 = np.kron(np.ones(param.nbData), np.eye(param.nbVarX)).T
Su = Su0[idx,:] # We remove the lines that are out of interest


# iLQR
# ===============================

u = np.zeros((param.nbVarU * (param.nbData-1), 1)) # Initial control command
x0 = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, 0]) # Initial robot pose

x0 = x0.reshape((-1, 1))

for i in range(param.nbIter):
	x = Su0 @ u + Sx0 @ x0 # System evolution
	x = x.reshape([param.nbVarX, param.nbData], order='F')

	f, J = f_reach(x[:,tl], param) # Residuals and Jacobians
	f = f.reshape((-1,1), order='F')

	Ra = np.zeros_like(Qr)

	for j in range(param.nbPoints):
		Rkp = np.zeros((param.nbVarF-1,param.nbVarF-1)) # Transformation matrix with both translation and rotation
		Rkp[:3,:3] = np.identity(3) # For translation
		Rkp[-3:,-3:] = q2R(param.Mu[-4:,j]) # Orientation matrix for target

		nbVarQ = param.nbVarF - 1
		Ra[j*nbVarQ:(j+1)*nbVarQ,j*nbVarQ:(j+1)*nbVarQ] = Rkp

	Q = Ra @ Qr @ Ra.T # Precision matrix in absolute coordinate frame (base frame)
	
	#du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f - u * param.r) # Gauss-Newton update
	du = np.linalg.lstsq(
		Su.T @ J.T @ Q @ J @ Su + R,
		-Su.T @ J.T @ Q @ f - u * param.r,
		rcond=-1
	)[0] # Gauss-Newton update

	# Estimate step size with backtracking line search method
	alpha = 1
	cost0 = f.T @ Q @ f + np.linalg.norm(u)**2 * param.r # Cost
	while True:
		utmp = u + du * alpha
		xtmp = Su0 @ utmp + Sx0 @ x0 # System evolution
		xtmp = xtmp.reshape([param.nbVarX, param.nbData], order='F')
		ftmp,_ = f_reach(xtmp[:,tl], param) # Residuals
		ftmp = ftmp.reshape((-1,1), order='F')
		cost = ftmp.T @ Q @ ftmp + np.linalg.norm(utmp)**2 * param.r # Cost
		if cost < cost0 or alpha < 1e-3:
			print("Iteration {}, cost: {}".format(i,cost[0][0]))
			break
		alpha /= 2

	u = u + du * alpha

	if np.linalg.norm(du * alpha) < 1E-2:
		break # Stop iLQR iterations when solution is reached


# Plots
# ===============================

ax = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')

# Plot the robot
ftmp, _ = fkin0(x[:,0], param)
ax.plot(ftmp[0,:], ftmp[1,:], ftmp[2,:], linewidth=4, color=[.8, .8, .8])

ftmp, _ = fkin0(x[:,tl[0]], param)
ax.plot(ftmp[0,:], ftmp[1,:], ftmp[2,:], linewidth=4, color=[.6, .6, .6])

ftmp, _ = fkin0(x[:,tl[1]], param)
ax.plot(ftmp[0,:], ftmp[1,:], ftmp[2,:], linewidth=4, color=[.4, .4, .4])

# Plot targets
plotCoordSys(ax, param.Mu, param.MuR * 0.1)

# Plot end-effector and trajectory
ftmp, Rtmp = fkin(x, param)
ax.plot(ftmp[0,:], ftmp[1,:], ftmp[2,:], linewidth=1, color='black')
plotCoordSys(ax, ftmp[:,0:1], Rtmp[:,:,0:1] * .05, width=2)
plotCoordSys(ax, ftmp[:,tl], Rtmp[:,:,tl] * .05, width=2)

plotCoordSys(ax, np.zeros((3,1)), np.eye(3).reshape((3, 3, 1)) * .1);

# Set axes limits and labels
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.8)
ax.set_zlim(0, 0.8)
ax.set_xlabel(r'$f_1$')
ax.set_ylabel(r'$f_2$')
ax.set_zlabel(r'$f_3$')

limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
ax.set_box_aspect(np.ptp(limits, axis=1))

plt.show()
