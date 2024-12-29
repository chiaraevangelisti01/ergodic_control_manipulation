
# Copyright (c) 2024 Idiap Research Institute <https://www.idiap.ch/>
#
# This file is inspired by RCFS <https://robotics-codes-from-scratch.github.io/>
# License: GPL-3.0-only

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import block_diag
import skimage.io


def f_ergodic(x, param, compute_jacobian=True):
    [xx, yy] = np.mgrid[range(param.nbFct), range(param.nbFct)]

    phi1 = np.zeros((param.nbData, param.nbFct, 2))
    x1_s = x[0::2].reshape((-1, 1))
    x2_s = x[1::2].reshape((-1, 1))

    phi1[:, :, 0] = np.cos(x1_s @ param.kk1.T) / param.L
    phi1[:, :, 1] = np.cos(x2_s @ param.kk1.T) / param.L

    phi = phi1[:, xx, 0] * phi1[:, yy, 1]
    phi = phi.reshape(param.nbData, -1, order="F")
    w = (np.sum(phi, axis=0) / param.nbData).reshape((param.nbFct**2, 1))

    if not compute_jacobian:
        return w, None

    dphi1 = np.zeros((param.nbData, param.nbFct, 2))
    dphi1[:, :, 0] = (
        -np.sin(x1_s @ param.kk1.T)
        * np.matlib.repmat(param.kk1.T, param.nbData, 1)
        / param.L
    )
    dphi1[:, :, 1] = (
        -np.sin(x2_s @ param.kk1.T)
        * np.matlib.repmat(param.kk1.T, param.nbData, 1)
        / param.L
    )
    dphi = np.zeros((param.nbData * param.nbVarPos, param.nbFct**2))
    dphi[0 : param.nbData * param.nbVarPos : 2, :] = (
        dphi1[:, xx, 0] * phi1[:, yy, 1]
    ).reshape(param.nbData, -1, order="F")
    dphi[1 : param.nbData * param.nbVarPos : 2, :] = (
        phi1[:, xx, 0] * dphi1[:, yy, 1]
    ).reshape(param.nbData, -1, order="F")

    J = dphi.T / param.nbData
    return w, J


def f_kernel_v(x, param, compute_jacobian=True):
    xT = x[0::2]
    yT = x[1::2]
    delta_x = -(xT[:, None] - xT[None, :])[:, :, 0]
    delta_y = -(yT[:, None] - yT[None, :])[:, :, 0]
    dist = np.sqrt(delta_x**2 + delta_y**2)
    fi = np.exp(-1.0 * dist / param.theta)
    f = fi.sum(axis=1) - 1.0
    dist[dist == 0.0] = 1e10
    T = 1.0 / (param.nbData**2)
    if not compute_jacobian:
        f *= T
        return f.reshape((-1, 1)), None
    for i in range(param.nbData):
        Jtmp = np.zeros((2))
        Jx = (-2.0) * fi[:, i] * np.divide(delta_x[:, i], (dist[:, i] * param.theta))
        Jy = (-2.0) * fi[:, i] * np.divide(delta_y[:, i], (dist[:, i] * param.theta))
        Jtmp[0:1] = Jx.sum()
        Jtmp[1:2] = Jy.sum()
        if i == 0:
            J = np.copy(Jtmp.T)
        else:
            J = block_diag(J, np.copy(Jtmp.T))
    f *= T
    J *= T
    return f.reshape((-1, 1)), J


param = lambda: None  # Lazy way to define an empty class in python
param.dt = 1e-2  # Time step size
param.nbData = 3600  # Number of datapoints
param.nbIter = 50  # Number of iterations for iLQR
param.nbVarPos = 2  # Dimension of position data
param.nbDeriv = 3  # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVarX = param.nbVarPos * param.nbDeriv  # State space dimension
param.theta = 4e-3
param.qk = 0.0  # 1e2
param.qs = 1.0

param.nbFct = 6
param.nbRes = 400

sp = (param.nbVarPos + 1) / 2  # Sobolev norm parameter
rg = np.arange(0, param.nbFct, dtype=float)
KX = np.zeros((param.nbVarPos, param.nbFct, param.nbFct))
KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
Lambda = np.array(KX[0, :].flatten() ** 2 + KX[1, :].flatten() ** 2 + 1).T ** (-sp)
Qs = np.diag(Lambda)
Qhalf = np.diag(Lambda**0.5)

xlim = [0, 1]
xm1d = np.linspace(xlim[0], xlim[1], param.nbRes)  # Spatial range for 1D
xm = np.zeros((param.nbVarPos, param.nbRes, param.nbRes))  # Spatial range
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
param.L = (xlim[1] - xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
om = 2 * np.pi / param.L
param.range = np.arange(param.nbFct)
param.kk1 = om * param.range.reshape((param.nbFct, 1))

arg1 = (
    KX[0, :, :].flatten().T[:, np.newaxis] @ xm[0, :, :].flatten()[:, np.newaxis].T * om
)
arg2 = (
    KX[1, :, :].flatten().T[:, np.newaxis] @ xm[1, :, :].flatten()[:, np.newaxis].T * om
)
phim = np.cos(arg1) * np.cos(arg2) * 2 ** (param.nbVarPos)  # Fourier basis functions
xx, yy = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
phim = phim * np.tile(HK, (param.nbRes**param.nbVarPos, 1)).T
phi_inv = (
    np.cos(arg1) * np.cos(arg2) / param.L**param.nbVarPos / param.nbRes**param.nbVarPos
)

# Load image and compute Fourier coefficients of its reversed intensity
img = skimage.io.imread("skull.png")
img = skimage.transform.resize(img[:, :, 0], (param.nbRes, param.nbRes))

g = 1.0 - img.reshape((param.nbRes * param.nbRes,))
g = g * param.nbRes**param.nbVarPos / np.sum(g) * 1.0

w_hat = (phi_inv @ g).reshape((-1, 1))
g = w_hat.T @ phim

# Initialize randomly particles between [0.1 and 0.9]
x = 0.8 * np.random.rand(2 * param.nbData) + 0.1
x = x.reshape((-1, 1))

logs = lambda: None
logs.e = []

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 3, height_ratios=[1.0, 1.0])
fig.subplots_adjust(hspace=0.5)

plt.subplot(gs[0, 0])
ax = plt.gca()
ax.imshow(img, cmap="gray", alpha=1.0)
plt.title("Image")

plt.subplot(gs[0, 1])
ax = plt.gca()
contour = ax.contourf(
    xm[0, :, :],
    xm[1, :, :],
    g.reshape(param.nbRes, param.nbRes),
    levels=50,
    cmap="gray",
    alpha=0.2,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_aspect("equal")
plt.title("Particles and distribution")
ax.invert_yaxis()

for i in range(param.nbIter):
    ws, Js = f_ergodic(np.copy(x), param)
    ws = ws - w_hat
    Js = Qhalf @ Js

    # Fast approximation
    dx = -np.linalg.pinv(Js) @ Qhalf @ ws * param.qs

    # Robust inversion matrix, however heavier to compute
    # dx = np.linalg.pinv(Js.T @ Js + np.eye(x.shape[0]) * 1e-8) @ (-Js.T @ Qhalf @ ws * param.qs)

    cost0 = ws.T @ Qs @ ws * param.qs

    if param.qk != 0:
        fk, Jk = f_kernel_v(np.copy(x), param)
        cost_k = np.linalg.norm(fk) ** 2 * param.qk  # Cost
        dx += -np.linalg.pinv(Jk) @ fk * param.qk
        cost0 += cost_k

    logs.e += [cost0.squeeze()]

    alpha = 0.05
    while True:
        xtmp = x + dx * alpha
        wstmp, _ = f_ergodic(np.copy(xtmp), param, compute_jacobian=False)
        wstmp = wstmp - w_hat
        cost = wstmp.T @ Qs @ wstmp * param.qs

        if param.qk != 0:
            fktmp, _ = f_kernel_v(np.copy(xtmp), param, compute_jacobian=False)
            cost += np.linalg.norm(fktmp) ** 2 * param.qk  # Cost

        if cost < cost0 or alpha < 1e-3:
            print(alpha)
            print("Iteration {}, cost: {}".format(i, cost.squeeze()))
            break
        alpha /= 2

    x = x + dx * alpha

plt.subplot(gs[0, 1])
ax = plt.gca()
ax.scatter(x[0::2], x[1::2], marker=".", s=5.0, c="b", alpha=1.0, label="particles")

plt.subplot(gs[0, 2])
ax = plt.gca()
ax.scatter(x[0::2], x[1::2], marker=".", s=5.0, c="b", alpha=1.0, label="particles")
ax.set_ylabel("y")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
plt.title("Particles")
ax.invert_yaxis()

plt.subplot(gs[1, 0])
plt.plot(logs.e, linestyle="-", label="Total")
plt.title("Cost")
ax = plt.gca()
ax.set_xlabel("iterations")
ax.set_ylabel("cost")
ax.set_yscale("log")
plt.legend()

plt.show()
