
# Copyright (c) 2024 Idiap Research Institute <https://www.idiap.ch/>
#
# This file is inspired by RCFS <https://robotics-codes-from-scratch.github.io/>
# License: GPL-3.0-only

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skimage.io


def offset(mat, i, j):
    """
    offset a 2D matrix by i, j
    """
    rows, cols = mat.shape
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]


def normalize_mat(mat):
    return mat / (np.sum(mat) + 1e-10)


nb_particule = 5
dim = 2
resolution = 200
dx = 1
repeat_diffusion = 1  # 50 # 0
diffusion = 0.001 # 10
alpha = 0.99*np.array([1, 1]) * diffusion
agent_radius = 0.0001  # 0.0003

# For the stability of implicit integration of Heat Equation, dt follow this rule:
dt = np.min((1.0, (dx * dx) / (4.0 * np.max(alpha))))
# dt = 1e-2
print("dt : ", dt)

xlim = [0, 1]
xm1d = np.linspace(xlim[0], xlim[1], resolution)  # Spatial range for 1D
xm = np.zeros((dim, resolution, resolution))  # Spatial range
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)

# Load image and compute Fourier coefficients of its reversed intensity
img = skimage.io.imread("skull.png")
img = skimage.transform.resize(img[:, :, 0], (resolution, resolution))
distribution = 1.0 - img
distribution = distribution * resolution**dim / np.sum(distribution) * 1.0
distribution = normalize_mat(distribution)


y_ids, x_ids = np.indices((resolution, resolution))
x_ids = x_ids.astype("float32") / (resolution - 1)
y_ids = y_ids.astype("float32") / (resolution - 1)

x = np.zeros((nb_particule, dim))

for i in range(nb_particule):
    # integrate diffusion on the distribution
    # Either : no diffusion. Seems to gives nice results, but no particule in white blank area
    # Either :but would requires less diffusion repetitions
    # diffusion on the distrib, place the agent but keep the original distib for cooling
    # heat = np.copy(distribution)
    # - requires more difusion amplitude and repetitions
    # Either keep the diffusion result, and cooling on it. Thus the diffusion should
    # be lighter otherwise we get a distribution completely flat after some time. But faster.
    heat = distribution
    for _ in range(repeat_diffusion):
        heat[1:-1, 1:-1] = dt * (
            (
                +alpha[0] * offset(heat, 1, 0)
                + alpha[0] * offset(heat, -1, 0)
                + alpha[1] * offset(heat, 0, 1)
                + alpha[1] * offset(heat, 0, -1)
                - 4.0 * diffusion * offset(heat, 0, 0)
            )
            / (dx * dx)
        ) + offset(heat, 0, 0)
    heat = heat.astype(np.float32)
    # plt.imshow(heat, cmap='gray');plt.show()

    # Find maximum heat and place agent at this position (in pixel indice scale)
    id = np.argmax(heat)
    yi, xi = np.unravel_index(id, np.array(heat).shape)
    x[i, 0:2] = (xi, yi)

    # Cool the heat source around the agent by computing the distance all over the
    # map (could be faster with just the neighborhood) and creating a kernel
    # that will be used to cool the distribution
    #elliptical kernel
    sigma_x = 0.005  # Standard deviation along x-axis
    sigma_y = 0.015  # Standard deviation along y-axis

    # Adjust the elliptical distance formula
    dists = (
        ((x_ids - xi / resolution) ** 2) / (2 * sigma_x**2)
        + ((y_ids - yi / resolution) ** 2) / (2 * sigma_y**2)
    )
    coverage_density = np.exp(-dists)  
    coverage = normalize_mat(coverage_density) / nb_particule
    
    # dists = np.sqrt((x_ids - xi / resolution) ** 2 + (y_ids - yi / resolution) ** 2)
    # coverage_density = np.exp(-(1 / agent_radius) * dists**2)
    # coverage = normalize_mat(coverage_density) / nb_particule
    plt.imshow(coverage, cmap='gray')
    plt.show()    
    distribution -= coverage
    distribution = np.maximum(distribution, 0)
    print("Particle {} added".format(i), xi, yi)
    

# Rescale x from pixel coordinate to limits
x /= resolution

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(1, 3, height_ratios=[1.0])
fig.subplots_adjust(hspace=0.5)

plt.subplot(gs[0, 0])
ax = plt.gca()
ax.imshow(img, cmap="gray")
plt.title("Image")

plt.subplot(gs[0, 1])
ax = plt.gca()
contour = ax.contourf(
    xm[0, :, :],
    xm[1, :, :],
    distribution,
    levels=50,
    cmap="gray",
    alpha=0.2,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.scatter(x[:, 0], x[:, 1], marker=".", s=8, c="b", alpha=1.0)
ax.invert_yaxis()
plt.title("Particle and distribution")

plt.subplot(gs[0, 2])
ax = plt.gca()
ax.scatter(x[:, 0], x[:, 1], marker=".", s=8, c="b")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.invert_yaxis()
plt.title("Particles")

plt.show()
