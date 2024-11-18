'''
Inverse Kinematics in a 3D workspace for a manipulator defined with DH parameters (standard or modified)

Copyright (c) 2024 Idiap Research Institute <https://www.idiap.ch>
Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://rcfs.ch>
License: GPL-3.0-only
'''
import copy
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# Helper functions
# ===============================

# Cost and gradient for a viapoints reaching task (in object coordinate system)
def f_reach(x, param, iter):
    ftmp, _ = fkin(x, param)
    f = logmap(ftmp, param.Mu[:,iter].reshape(-1,1))
    J = Jkin_num(x[:,0], param)
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
    N = param.nbVarX+1
    x = np.append(x, 0)
    Tf = np.eye(4)
    R = np.zeros((3,3,N))
    f = np.zeros((3,N+1))
    for n in range(N):
        ct = np.cos(x[n] + param.dh.q_offset[n])
        st = np.sin(x[n] + param.dh.q_offset[n])
        ca = np.cos(param.dh.alpha[n])
        sa = np.sin(param.dh.alpha[n])

        if param.dh.convention == "m":
            Tf = Tf @ [[ct,    -st,     0,   param.dh.r[n]   ],
                    [st*ca,  ct*ca, -sa, -param.dh.d[n]*sa],
                    [st*sa,  ct*sa,  ca,  param.dh.d[n]*ca],
                    [0,      0,      0,   1               ]]
        elif param.dh.convention == "s":
            Tf = Tf @ [[ct,    -st*ca,     st*sa,   param.dh.r[n]*ct   ],
                   [st,  ct*ca, -ct*sa, param.dh.r[n]*st],
                   [0,  sa,  ca,  param.dh.d[n]],
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
param.nbIter = 50 # Maximum number of iterations for iLQR
param.nbVarX = 6 # State space dimension (x1,x2,x3)
param.nbVarU = param.nbVarX # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 7 # Task space dimension (f1,f2,f3 for position, f4,f5,f6,f7 for unit quaternion)
param.nbPoints = 30 #viapoints of the trajectory

param.Mu = np.ndarray((param.nbVarF, param.nbPoints))
#Define a circular 2d trajectory  for 10 viapoints
r = 0.3
theta = np.linspace(0, 2*np.pi, param.nbPoints, endpoint=False)
x = r * np.cos(theta)
y = r * np.sin(theta)

#Define a sine wave trajectory
# x = np.linspace(-0.3, 0.3, param.nbPoints)
# y = 0.2 * np.sin(4 * np.pi * x ) + 0.1

z = 0.01 # planar trajectory
orientation = [0.0, 1.0, 0.0, 0.0] #perpendicular to the plane, opposite direction of z axis, could be also [0.0, 0.0, 1.0, 0]
Rtmp = q2R(orientation)
param.MuR = np.dstack([Rtmp] * param.nbPoints)
for i in range(param.nbPoints):
    param.Mu[0:3, i] = [x[i], y[i], z]
    param.Mu[3:7, i] = R2q(param.MuR[:,:,i])
param.alpha = 1e-6 # Regularization term


# Modified DH parameters of ULite6 Robot
# param.dh = lambda: None # Lazy way to define an empty class in python
# param.dh.convention = 'm' # modified DH, a.k.a. Craig's formulation
# param.dh.type = ['r'] * (param.nbVarX+1) # Articulatory joints
# param.dh.q_offset = np.array([0,-np.pi/2,-np.pi/2,0,0,0,0]) # Offset on articulatory joints
# param.dh.alpha =  [0,-np.pi/2,np.pi,np.pi/2,np.pi/2,-np.pi/2,0]# Angle about common normal
# param.dh.d = [0.2433, 0, 0, 0.2276, 0, 0.0615,0] # Offset along previous z to the common normal
# param.dh.r = [0, 0, 0.2, 0.087, 0, 0,0] # Length of the common normal

# Standard DH parameters of ULite6 Robot
param.dh = lambda: None # Lazy way to define an empty class in python
param.dh.convention = 's' # Standard DH
param.dh.type = ['r'] * (param.nbVarX+1) # Articulatory joints
param.dh.q_offset = np.array([0,-np.pi/2,-np.pi/2,0,0,0,0]) # Offset on articulatory joints
param.dh.alpha =  [-np.pi/2,np.pi,np.pi/2,np.pi/2,-np.pi/2,0,0]# Angle about common normal
param.dh.d = [0.2433, 0, 0, 0.2276, 0, 0.0615,0] # Offset along previous z to the common normal
param.dh.r = [0, 0.2, 0.087, 0, 0, 0, 0] # Length of the common normal

# Standard DH parameters of MyCobot280 (presumed)
# param.dh = lambda: None # Lazy way to define an empty class in python
# param.dh.convention = 's' # standard DH parameters formulation
# param.nbVarX = 6 # State space dimension (x1,x2,x3)
# param.dh.type = ['r'] * (param.nbVarX) # Articulatory joints
# param.dh.q_offset =  np.array([0,-np.pi/2,0,-np.pi/2,np.pi/2,0,0]) # Offset on articulatory joints
# param.dh.alpha = [np.pi/2,0,0,np.pi/2,-np.pi/2,0,0] # Angle about common normal
# param.dh.d = [0.13122,0,0,0.0634,0.07505,0.0456,0] # Offset along previous z to the common normal
# param.dh.r = [0,-0.1104,-0.096,0,0,0,0] # Length of the common normal



# Main program
# ===============================
intermediate_configs = []
viapoints_configs = []
x0 = np.zeros((6,1)) # Initial robot pose
x = copy.deepcopy(x0)


for i in range(param.nbPoints): #Compute IK for each point in the trajectory

    for j in range(param.nbIter):

        e, J = f_reach(x,param,i)
       
        if np.linalg.norm(e) < 1E-2:
            ftmp, Rtmp = fkin0(x.flatten(), param)
            viapoints_configs.append((ftmp, Rtmp))
            print(f'Convergence at iteration {j}')
            break

        #Define weighting matrix
        #Rkp = np.identity(param.nbVarF - 1)
        Rkp = np.zeros((param.nbVarF - 1, param.nbVarF - 1))
        Rkp[:3, :3] = np.identity(3) #Translation constraint
        Rkp[3:, 3:] = q2R(param.Mu[-4:, i]) #Orientation constraint

        #Update x
        J = Jkin_num(x.flatten(),param)
        dx = np.linalg.inv(J.T @ Rkp @ J + param.alpha * np.identity(J.shape[1])) @ (J.T @ Rkp @ e)
        x -= 0.1*dx

        if j == param.nbIter-1:
            ftmp, Rtmp = fkin0(x.flatten(), param)
            viapoints_configs.append((ftmp, Rtmp))
           
        else:
            ftmp, Rtmp = fkin0(x.flatten(), param)
            intermediate_configs.append((ftmp, Rtmp))
       


# Plots
# ===============================

ax = plt.figure(figsize=(12, 10)).add_subplot(projection='3d')

# Plot the robot
ftmp, _ = fkin0(x0.flatten(), param)
ax.plot(ftmp[0,:], ftmp[1,:], ftmp[2,:], linewidth=4, color=[1, 1, 1])

ftmp, Rtmp = fkin0(x.flatten(), param)
ax.plot(ftmp[0,:], ftmp[1,:], ftmp[2,:], linewidth=4, color=[0, 0, 0])
plotCoordSys(ax, ftmp[:,-1:], Rtmp[:,:,-1:] * .06, width=2)

# Plot each intermediate configuration
#for ftmp, Rtmp in intermediate_configs:
    #ax.plot(ftmp[0, :], ftmp[1, :], ftmp[2, :], linewidth=1, color='grey', alpha=0.5)  # Use alpha for transparency

for ftmp, Rtmp in viapoints_configs:
    ax.plot(ftmp[0, :], ftmp[1, :], ftmp[2, :], linewidth=1, color='green' ) 
    plotCoordSys(ax, ftmp[:,-1:], Rtmp[:,:,-1:] * .06, width=2)


# Plot targets
plotCoordSys(ax, param.Mu, param.MuR * 0.1)

# Set axes limits and labels
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.4, 0.4)
ax.set_zlim(0, 0.6)
ax.set_xlabel(r'$f_1$')
ax.set_ylabel(r'$f_2$')
ax.set_zlabel(r'$f_3$')

limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
ax.set_box_aspect(np.ptp(limits, axis=1))

plt.show()
