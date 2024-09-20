import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#Plotting functions to visualize output
def plot_gmm(Mu, Sigma, ax=None):
    if ax is None:
        ax = plt.gca()
    
    for i in range(Mu.shape[1]):
        # Extract mean and covariance for each Gaussian
        mean = Mu[:, i]
        cov = Sigma[:, :, i]

        # Create an ellipse representing the covariance
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='r', facecolor='none', lw=2)

        ax.add_patch(ell)
        ax.scatter(*mean, color='r', zorder=5)

def plot_trajectory(trajectory, ax=None):
    if ax is None:
        ax = plt.gca()
    
    ax.plot(trajectory[0, :], trajectory[1, :], 'b-', lw=2, label="Spiral Trajectory")
    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Trajectory")

def plot_control_inputs(control_inputs, ax=None):
    if ax is None:
        ax = plt.gca()

    # Plot both x and y components of the control inputs
    fig, ax = plt.subplots(figsize=(8, 4))
    timesteps = np.arange(len(control_inputs))

    # Plot the x and y components separately
    ax.plot(timesteps, control_inputs[:, 0], 'r-', lw=1, label="v_x ")
    ax.plot(timesteps, control_inputs[:, 1], 'b-', lw=1, label="v_y ")

    # Add labels and legend
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Velocity Components")
    ax.set_title("Control Inputs (Velocity Components)")
    ax.legend()

    plt.show()

# Parameters definition
param = lambda: None
param.nbData = 200  # Number of data points (
param.nbVarX = 2  # State space dimension (2D)
param.nbStates = 2  # Number of Gaussians in GMM
param.dt = 1e-2  # Time step length
param.xlim = [0, 1]  # Domain limits
param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-param.xlim(2), param.xlim(2)]
param.a = 5  # Spiral parameter for radius growth
param.b = 16  # Spiral parameter for the angle
param.u_max = 4.0  # Maximum allowed velocity
param.u_norm_reg = 1e-3  # Regularization term for control inputs


# Desired spatial distribution represented as a mixture of Gaussians
param.Mu = np.zeros((2, 2))
param.Mu[:, 0] = [0.5, 0.7]
param.Mu[:, 1] = [0.6, 0.3]
param.Sigma = np.zeros((2, 2, 2))
sigma1_tmp = np.array([[0.3], [0.1]])
param.Sigma[:, :, 0] = sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.identity(param.nbVarX) * 5e-3 
sigma2_tmp = np.array([[0.1], [0.2]])
param.Sigma[:, :, 1] = sigma2_tmp @ sigma2_tmp.T * 3e-1 + np.identity(param.nbVarX) * 1e-2 

# Generate the transformation matrices for each Gaussian (to match spiral trajectory with GMM)
U = np.zeros((2, 2, param.nbStates))
for i in range(param.nbStates):
    # Perform eigendecomposition
    D, V = np.linalg.eigh(param.Sigma[:, :, i])
       
    # Reconstruct U 
    U[:, :, i] = V @ np.diag(np.sqrt(D))

#Trajectory generation
#Generate 2D spiral trajectory
t = np.linspace(0, param.b * np.pi, param.nbData)  # Angle
r = np.linspace(0, 1, param.nbData)  # Radius

x0 = np.vstack((r * np.sin(t), r * np.cos(t)))  # x0 is the base spiral in 2D

#Transform the spirals to match the GMM using U
x = np.zeros((param.nbVarX, 1))
for i in range(param.nbStates):
    transformed_spiral = U[:, :, i] @ x0 + param.Mu[:, i].reshape(-1, 1)
    x = np.hstack((x, transformed_spiral))

#Compute control inputs to generate the trajectory (under u-max constraint)
control_inputs = np.zeros((param.nbData - 1, param.nbVarX))
for i in range(1, param.nbData):
    delta_pos = (x[:, i] - x[:, i - 1]) / param.dt
    delta_pos = delta_pos * (param.u_max / (np.linalg.norm(delta_pos) + param.u_norm_reg))  # Apply u_max constraint
    
    control_inputs[i - 1] = delta_pos

# Plot the GMM and trajectory
fig, ax = plt.subplots(figsize=(8, 8))
plot_gmm(param.Mu, param.Sigma, ax)
plot_trajectory(x, ax)
plt.show()

# Plot the control inputs (velocity magnitudes over time)
plot_control_inputs(control_inputs, ax)
