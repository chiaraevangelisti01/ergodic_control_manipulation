import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


nbVarX = 2  # State space dimension
nbStates = 2  # Number of Gaussians to represent the spatial distribution
# Desired spatial distribution represented as a mixture of Gaussians
spread_factor = 1  # Adjust this factor to control the overall spread

import numpy as np

# spread_factor = 10  # Adjust this factor to control the overall spread

# # Define the means
# Mu = np.zeros((2, 2))
# Mu[:, 0] = [0.5 * 100, 0.7 * 100]
# Mu[:, 1] = [0.6 * 100, 0.3 * 100]

# # Define the original covariances
# Sigma = np.zeros((2, 2, 2))
# sigma1_tmp = np.array([[0.3], [0.1]])
# original_cov1 = sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.eye(2) * 5e-3

# sigma2_tmp = np.array([[0.1], [0.2]])
# original_cov2 = sigma2_tmp @ sigma2_tmp.T * 3e-1 + np.eye(2) * 1e-2

# # Function to scale the covariance matrix without changing its structure
# def scale_covariance(cov_matrix, scale_factor):
#     # Eigenvalue decomposition
#     eigvals, eigvecs = np.linalg.eigh(cov_matrix)
#     # Scale the eigenvalues
#     scaled_eigvals = eigvals * scale_factor
#     # Reconstruct the covariance matrix
#     scaled_cov_matrix = eigvecs @ np.diag(scaled_eigvals) @ eigvecs.T
#     return scaled_cov_matrix

# # Scale the covariances while preserving their structure
# Sigma[:, :, 0] = scale_covariance(original_cov1, spread_factor)
# Sigma[:, :, 1] = scale_covariance(original_cov2, spread_factor)
def analytical_force_field(x, Mu, Sigma, Priors):
        '''Compute the force field analytically at the given location using the Gaussian mixture model'''
        x = x[::-1]
        nbStates = Mu.shape[1]  # Number of Gaussian components
        grad_f = np.zeros(len(x))    # Initialize the gradient
        pdf_value = 0.0         # Initialize the total PDF value

        for k in range(nbStates):
            mu_k = Mu[:, k]
            sigma_k = Sigma[:, :, k]
            prior_k = Priors[k]
            
            # Compute the inverse and determinant of the covariance matrix
            sigma_inv = np.linalg.inv(sigma_k)
            # det_sigma = np.linalg.det(sigma_k)
            
            # # Compute the difference vector
            # diff = x - mu_k

            # # Compute the Gaussian PDF value for the current component
            # exponent = -0.5 * np.dot(diff.T, np.dot(sigma_inv, diff))
            # norm_const = prior_k / (2 * np.pi * np.sqrt(det_sigma))
            # f_k = norm_const * np.exp(exponent)
            f_k = multivariate_normal.pdf(x, mean=mu_k, cov=sigma_k)
           
            # Accumulate the total PDF value
            pdf_value += f_k*prior_k
        
            # Compute the gradient contribution from the current component
            grad_f += f_k * np.dot(sigma_inv, mu_k - x)
            
        # Normalize the gradient by the total PDF value to get the gradient of the GMM
        if pdf_value > 0:
            grad_f /= pdf_value
        

        return grad_f
    
def ff2(Mu, Sigma, Priors):
        grid_points = np.array(np.meshgrid(np.arange(0, 100), np.arange(0, 100)))
        forcefield = np.zeros((grid_points.shape[1], grid_points.shape[2], 2))  #  2D forces

        for ip, jp in np.ndindex(grid_points.shape[1], grid_points.shape[2]):  # Imaginary charge moving over the grid
              
            forcefield[ip, jp] = analytical_force_field(np.array([ip, jp]), Mu, Sigma, Priors) # Scale by grey value
                               
        return forcefield

def plot_force_field(forcefield):
        '''Plot the force field as a vector field -> for debugging'''

         # Generate the grid points with increased spacing to reduce arrow density
        grid_points_x, grid_points_y = np.meshgrid(np.arange(0, 100, 3),
                                                np.arange(0, 100, 3))

        # Subsample the force field to match the reduced number of grid points
        force_x = forcefield[::3, ::3, 0]
        force_y = forcefield[::3, ::3, 1]

        # Plot the force field with adjusted arrow scaling and head size
        plt.quiver(grid_points_x, grid_points_y, force_x, force_y, color='b',
                scale=5, scale_units='xy', headwidth=1.5, headlength=2)

        # Adjust plot settings
        plt.gca()#.invert_yaxis()
        plt.show()

Mu = np.zeros((2,2))
Mu[:,0] = [0.5*100, 0.7*100]
Mu[:,1] = [0.6*100, 0.3*100]
Sigma = np.zeros((2,2,2))
sigma1_tmp= np.array([[0.3],[0.1]])
Sigma[:,:,0] = sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.eye(nbVarX)*5e-3
sigma2_tmp= np.array([[0.1],[0.2]])
Sigma[:,:,1] = sigma2_tmp @ sigma2_tmp.T * 3e-1 + np.eye(nbVarX)*1e-2 

eigvals1, eigvecs1 = np.linalg.eigh(Sigma[:,:,0])
scaled_eigvals1 = eigvals1 * 10000
#print(scaled_eigvals1)
Sigma[:,:,0] = eigvecs1 @ np.diag(scaled_eigvals1) @ eigvecs1.T

eigvals2, eigvecs2 = np.linalg.eigh(Sigma[:,:,1])
scaled_eigvals2 = eigvals2 * 10000
Sigma[:,:,1] = eigvecs2 @ np.diag(scaled_eigvals2) @ eigvecs2.T
Priors = [0.5, 0.5]

# Define the grid for plotting
x_min = 0
x_max = 100
y_min = 0
y_max = 100

x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)

# Compute the PDF values for each Gaussian
pos = np.dstack((X, Y))


# Gaussian 1
rv1 = multivariate_normal(mean=Mu[:, 0], cov=Sigma[:, :, 0])
Z1 = rv1.pdf([pos])*0.5
print(Mu[:, 0])
print(Sigma[:, :, 0])


# Gaussian 2
rv2 = multivariate_normal(mean=Mu[:, 1], cov=Sigma[:, :, 1])
Z2 = rv2.pdf(pos)*0.5
print(Z1.shape)

# Combine the PDFs
Z_combined = Z1 + Z2

grad_Z_combined_y, grad_Z_combined_x = np.gradient(Z_combined, y, x)
scaling_factor = 1.4e6  # Adjust this factor as needed to make arrows visible
grad_Z_combined_x *= scaling_factor
grad_Z_combined_y *= scaling_factor
forcefield2 = np.stack((grad_Z_combined_x, grad_Z_combined_y), axis=-1)
plot_force_field(forcefield2)


# Plotting
plt.figure(figsize=(10.5, 10.5))
plt.axis("off")
contour =plt.contourf(X, Y, Z_combined, levels=8, cmap="gray_r", alpha=1)  # Increase levels for more contours
#cbar = plt.colorbar(contour)
#cbar.set_label("Density", color="black")
#plt.scatter(*Mu[:, 0], color="blue", marker="o", label="Mean 1")
#plt.scatter(*Mu[:, 1], color="red", marker="x", label="Mean 2")
plt.xlabel("X")
plt.ylabel("Y")
#plt.legend()
#plt.title("Gaussian Mixture Model in Grayscale (Increased Spread)")
plt.tight_layout(pad = 0)
plt.savefig('original_distribution.png')
plt.show()


forcefield = ff2(Mu, Sigma, Priors)*50
plot_force_field(forcefield)


