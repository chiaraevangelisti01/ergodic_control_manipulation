import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to plot the Gaussian Mixture Model (GMM) distribution
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

# Function to plot the trajectory
def plot_trajectory(trajectory, ax=None):
    if ax is None:
        ax = plt.gca()
    
    # Plot the trajectory
    ax.plot(trajectory[0, :], trajectory[1, :], 'b-', lw=2, label="Sine wave Trajectory")

    
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Trajectory")

# Function to plot the control inputs
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
param.nbData = 200  # Number of data points 
param.nbVarX = 2  # State space dimension (2D)
param.nbStates = 2  # Number of Gaussians in GMM
param.dt = 1e-2  # Time step length
param.xlim = [0, 1]  # Domain limits
param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-param.xlim(2), param.xlim(2)]
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


# Helper function to generate the trajectory
def modulated_sine_wave_with_transitions(param, Mu, Sigma):
    total_trajectory = np.zeros((param.nbVarX, 1))  # Initialize full trajectory

    segment_length = param.nbData // param.nbStates  # Divide trajectory into segments
    A_base = 1  # Base amplitude of sine wave

    # Loop over each Gaussian state to generate its corresponding sine wave
    for i in range(param.nbStates):
        t = np.linspace(0, 2 * np.pi, segment_length)  # Time vector for the concerned segment

        # Perform eigendecomposition to get the major and minor axes of the ellips
        D, V = np.linalg.eigh(Sigma[:, :, i])

        # Eigenvectors: V[:, 1] is the major axis, V[:, 0] is the minor axis
        major_axis_length = 2 * np.sqrt(D[1])  # The length of the major axis
        minor_axis_length = 2 * np.sqrt(D[0])  # The length of the minor axis

        # Modulated sine wave along the major axis and oscillation along the minor axis
        # Apply amplitude modulation: stronger in the center, weaker at the edges
        modulation = 0.7 + 0.7 * np.sin(np.linspace(0, np.pi, segment_length))

        x_segment = np.vstack((
            np.linspace(+major_axis_length / 2, -major_axis_length / 2, segment_length),  # Linear motion along the major axis
            A_base * modulation * np.sin( np.pi * t) * minor_axis_length / 2  # Oscillation along the minor axis with smoother modulation
        ))

        # Rotate the trajectory using the eigenvectors to align with the covariance ellipse
        modulated_wave = V[:, [1, 0]] @ x_segment + Mu[:, i].reshape(-1, 1)

        # Concatenate this segment to the full trajectory
        total_trajectory = np.hstack((total_trajectory, modulated_wave))

        # If there's another Gaussian, add a linear transition to the next one
        if i < param.nbStates - 1:
            next_mu = Mu[:, i + 1].reshape(-1, 1)
            D, V = np.linalg.eigh(Sigma[:, :, i+1])
            next_begin = next_mu + V[:, 1].reshape(-1, 1) * major_axis_length / 2
            transition_segment = np.linspace(modulated_wave[:, -1], next_begin.flatten(), segment_length).T
            total_trajectory = np.hstack((total_trajectory, transition_segment))

    return total_trajectory


# Generate sinusoidal trajectory modulated by the GMM covariances with transitions
x_modulated = modulated_sine_wave_with_transitions(param, param.Mu, param.Sigma)

# Control inputs calculation with u_max constraint
control_inputs = np.zeros((param.nbData - 1, param.nbVarX))
for i in range(1, param.nbData):
    delta_pos = (x_modulated[:, i] - x_modulated[:, i - 1]) / param.dt
    control_norm = np.linalg.norm(delta_pos)
    delta_pos = delta_pos * (param.u_max / (control_norm + param.u_norm_reg))  # Apply u_max constraint
    
    control_inputs[i - 1] = delta_pos

# Plot the GMM and modulated sine wave trajectory
fig, ax = plt.subplots(figsize=(8, 8))
plot_gmm(param.Mu, param.Sigma, ax)
plot_trajectory(x_modulated, ax)
plt.show()

# Plot the control inputs (velocity magnitudes over time)
plot_control_inputs(control_inputs, ax)

