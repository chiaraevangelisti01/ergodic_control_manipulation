import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # For loading the image
from scipy.ndimage import zoom  # Import zoom for resampling
from PIL import Image
from scipy.stats import multivariate_normal

import time

class ElectrostaticHalftoning:

    def __init__(self, num_agents, image_path,xdom, ydom, Mu,Sigma,Priors, initialization = 'random', num_iterations = 250, target_resolution=(100, 100), displacement_threshold=15e-3):

        self.num_agents = num_agents # for sampling at steady-state
        self.image_path = image_path
        self.num_iterations = num_iterations
        self.displacement_threshold = displacement_threshold  # Convergence threshold
        self.target_resolution = target_resolution  # Target resolution for the image processing
        self.xdom = xdom #Application domain on x
        self.ydom = ydom #Application domain on y
        self.initialization = initialization
        self.Mu = Mu
        self.Sigma = Sigma
        self.Priors = Priors
        
        self.nbVarX = 2  # 2D state space for x, y coordinates
        
        self.xlim = None
        self.ylim = None
        self.forcefield = None
        self.particles = None
        self.required_particles = None
        self.sf_agent = None #Scaling factor for the forces to account for number of agents different from number of particles
        self.sf_attraction = self.num_agents**0.85, #Scaling factor for the attractive forces to account for the analytical force field
        

    ######################INPUT DATA PROCESSING FUNCTIONS###########################
    def process_image(self,image_path):
        """Convert an image to grayscale and resample it to the target resolution."""

        image = Image.open(image_path).convert('L')  # 'L' mode  for grayscale
        image = np.array(image)/255 # Normalize pixel values to [0, 1] as required by rhe algorithms

        zoom_factors = (self.target_resolution[0] / image.shape[0],self.target_resolution[1] / image.shape[1])
        rescaled_image = zoom(image,zoom_factors)

        return rescaled_image
    
    def scale_covariance(self, cov_matrix, scale_factor):
        '''Scale the covariance matrix to new domain'''
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        scaled_eigvals = eigvals * scale_factor
        scaled_cov_matrix = eigvecs @ np.diag(scaled_eigvals) @ eigvecs.T
        return scaled_cov_matrix
    
    def scale_distribution(self):
        '''Scale the covariances while preserving their structure: assuming square domain'''
        spread_factor = self.target_resolution[0] /self.xdom[1] 

        if len(self.Mu.shape) == 1:
            self.Mu = self.Mu.reshape(-1,1)
            self.Sigma = self.Sigma.reshape(2,2,1)

        for i in range(self.Mu.shape[1]):
            self.Sigma[:, :, i] = self.scale_covariance(self.Sigma[:, :, i], spread_factor**2)
            self.Mu[:,i] = self.Mu[:,i]*spread_factor

        return

     ######################ALGORITHM FUNCTIONS ###########################
    def number_particles(self):
        ''' Compute required number of particles to maintain the average grey value of the input image'''
        return int(np.round(np.sum(1 - self.image)))

    
    def analytical_force_field(self):
        '''Compute the force field analytically '''

        x = np.linspace(self.xlim[0], self.xlim[1], self.target_resolution[0])
        y = np.linspace(self.ylim[0], self.ylim[1], self.target_resolution[1])
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        Z = np.zeros(pos.shape[:2])

        for k in range(self.Mu.shape[1]):
           
            Z += self.Priors[k]*multivariate_normal.pdf(pos, mean=self.Mu[:,k], cov=self.Sigma[:,:,k])
    
        grad_y, grad_x = np.gradient(Z,y,x)
        grad_f =np.stack((grad_x, grad_y),axis=-1)

        return grad_f/Z[:,:,np.newaxis]
            
    
    def initialize_particles(self):
        '''Initialize particles in the image randomly, optimizing based on gray values'''

        # Pre-calculate all valid pixel positions with their corresponding gray values
        x_indices, y_indices = np.indices(self.image.shape)  # Generate all coordinates
        valid_pixels = np.column_stack((x_indices.flatten(), y_indices.flatten()))  # List of all coordinates

        if self.initialization == 'random':

            gray_values = np.flipud(self.image).flatten()  # Flatten the grayscale image to match coordinates
            max_grey = np.max(gray_values)
            
            # Apply probability filtering based on the grayscale values, ensuring that darker pixels have a higher chance
            probabilities = 1 - gray_values / max_grey  # Convert grey values to probabilities
            
            # Randomly choose the required number of particles based on these probabilities
            selected_indices = np.random.choice(len(valid_pixels), size=self.num_agents , p=probabilities/np.sum(probabilities), replace=False)
            particles = valid_pixels[selected_indices].T
        
        
        elif self.initialization == 'uniform':
            row_particles = int(np.sqrt(self.num_agents*self.image.shape[1]/self.image.shape[0]))
            col_particles = int(np.sqrt(self.num_agents*self.image.shape[0]/self.image.shape[1]))
          
            while row_particles*col_particles < self.num_agents:
                row_particles += 1
            
            x_pos = np.linspace(0,self.image.shape[0]-1,row_particles)
            y_pos = np.linspace(0,self.image.shape[1]-1,col_particles)
            X,Y = np.meshgrid(x_pos,y_pos)
            particles = np.vstack((X.flatten(),Y.flatten()))

            if particles.shape[1]> self.num_agents:
                selected_indices = np.random.choice(particles.shape[1], self.num_agents, replace=False)
                particles = particles[:,selected_indices]
            
        return particles.astype(np.float64)
    
    def bilinear_interpolation(self, forcefield, p):
        '''Bilinear interpolation for force field'''
        x, y = p[:2] 
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1

        x0 = np.clip(x0, 0, forcefield.shape[0] - 1)
        x1 = np.clip(x1, 0, forcefield.shape[0] - 1)
        y0 = np.clip(y0, 0, forcefield.shape[1] - 1)
        y1 = np.clip(y1, 0, forcefield.shape[1] - 1)

        tx = x - x0
        ty = y - y0

        f00 = forcefield[x0, y0]
        f01 = forcefield[x0, y1]
        f10 = forcefield[x1, y0]
        f11 = forcefield[x1, y1]

        fx0 = f00 * (1 - tx) + f10 * tx
        fx1 = f01 * (1 - tx) + f11 * tx

        fxy = fx0 * (1 - ty) + fx1 * ty
        return fxy

    def compute_force_all(self, particles):
        '''Compute force between all particles at once using vectorized operations in 2D space'''
        
        # Compute pairwise differences between all particles (delta = r_i - r_j)
        delta = particles[:, :, None] - particles[:, None, :]
        
        # Compute Euclidean distances between particles
        distances = np.linalg.norm(delta, axis=0)
        np.fill_diagonal(distances, np.inf)
        
        # Unit vector, coulomb law in 1d (1/distance^2), broadcasting to apply force direction
        forces = delta / (distances**2+ 1e-8)
        
        return np.sum(forces, axis=2)

    
    def evolve_particles(self):
        '''Compute particle movement in parallel using vectorized operations'''
        positions_over_time = [self.particles.copy()]  # Store the initial positions
        tau = 0.05#(self.displacement_threshold*self.num_iterations)/self.num_agents**1.05#0.006  # Step size, reduce it for more fine-grained movement
        shaking_strength = 0.001  # Shaking strength
        converged = False

        for iteration in range(self.num_iterations):
            max_displacement = 0  

            # Calculate repuslive forces for all particles at once
            forces = self.compute_force_all(self.particles)
            
            for i, p in enumerate(self.particles.T):
                p_old = p.copy()  

                # Apply bilinear interpolation for attractive forces
                attractive_force = self.bilinear_interpolation(self.forcefield, p)
                #print("Attractive force:", attractive_force)
                #print("Repulsive force: ", forces[:, i])
                
                # Combine repulsive and attractive forces
                total_force = forces[:, i] + attractive_force[::-1]*self.sf_attraction

                # Update particle position
                self.particles[:, i] = np.clip(p + tau * total_force, [self.xlim[0], self.ylim[0]], [self.xlim[1], self.xlim[1]])

                # Calculate displacement by comparing with the old position (before update)
                displacement = np.linalg.norm(self.particles[:, i] - p_old)
                max_displacement = max(max_displacement, displacement) #tracking maximum displacement

                if iteration % 10 == 0:
                    c1 = max(0, (np.log2(self.num_iterations) - 6) / 10)
                    shaking_strength = c1 * np.exp(-iteration / 1000)
                    self.particles[:, i] += np.hstack((np.random.uniform(-shaking_strength, shaking_strength, 2),np.zeros(self.nbVarX - 2)))  # Shake only in 2D

            positions_over_time.append(self.particles.copy())

            print(f"Iteration {iteration}: Max Displacement = {max_displacement}")

            # Check for convergence
            if max_displacement < self.displacement_threshold:
                print(f"Converged at iteration {iteration + 1}")
                converged = True
                break

        return self.particles, positions_over_time, converged
    
    def run(self):
        '''Run the electrostatic halftoning algorithm'''

        # Process the input image
        self.image = self.process_image(self.image_path)
        self.scale_distribution()

        # Initialize grid points 
        self.xlim = (0, self.image.shape[0])  # Number of rows in the image defines the x-limits
        self.ylim = (0, self.image.shape[1])  # Number of columns in the image defines the y-limits

        # Compute the required number of particles to maintain the average grey value
        self.required_particles = self.number_particles()
        self.sf_agents = self.num_agents / self.required_particles
        
        #Compute the force field (plot for debugging)
        self.forcefield = self.analytical_force_field()
        self.plot_force_field()
        
        # Initialize particles in the  domain
        self.particles = self.initialize_particles()
        
        # Evolve the particles
        final_particles, positions_over_time, converged = self.evolve_particles()
        
        # Scale their positions to the application domain
        self.plot_positions(positions_over_time)

        agents = self.scale_positions(final_particles, self.xdom[0], self.xdom[1], self.ydom[0], self.ydom[1])
        return agents

    ######################OUTPUR DATA PROCESSING FUNCTIONS###########################
    def scale_positions(self,positions,xlow, xup, ylow, yup):
        '''Scale agent positions to a new domain [xlow, xup] * [ylow, yup]'''
    
        # Scale the x coordinates
        positions[0,:] = (positions[0, :] - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (yup - ylow) + ylow
        
        # Scale the y coordinates
        positions[1,:] = (positions[1, :] - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (xup - xlow) + xlow
        
        # Return the scaled positions as a 2xN array
        return positions

    ######################PLOTTING FUNCTIONS###########################

    def plot_positions(self, positions_over_time):
        '''Plot particle evolution, initial, and final positions in a single figure with subplots'''
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 6))  # Create 3 subplots side by side

        # Extract initial and final positions
        initial_positions = positions_over_time[0]
        final_positions = positions_over_time[-1]
        #Underlying image (left plot)
        ax = axes[0]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        ax.set_title("Underlying Image")

        
        # Evolution over time (middle plot)
        ax = axes[1]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        num_iterations = len(positions_over_time)
        color = plt.cm.Blues  # Using a colormap for the evolution
        
        for i, positions in enumerate(positions_over_time):
            ax.scatter(positions[1, :], positions[0, :], label=f"Iteration {i+1}",
                    s=3,  # Marker size
                    color=color(i / num_iterations), 
                    alpha=0.2 + (0.8 * (i / num_iterations)))  # Alpha increases with iterations
        
        ax.set_title("Evolution of Particles")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        # Initial positions (left plot)
        ax = axes[2]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        ax.scatter(initial_positions[1, :], initial_positions[0, :], color='red', s=3, alpha=0.8)
        ax.set_title("Initial Particle Positions")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        # Final positions (right plot)
        ax = axes[3]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        ax.scatter(final_positions[1, :], final_positions[0, :], color='blue', s=3, alpha=0.9)
        ax.set_title("Final Particle Positions")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        # Final particles (no image)
        ax = axes[4]
        ax.set_aspect('equal')
        ax.scatter(final_positions[1, :], final_positions[0, :], color='black', s=3, alpha=0.9)
        ax.set_title("Final Particle Positions no image")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    
    def plot_force_field(self):
        '''Plot the force field as a vector field -> for debugging'''

         # Generate the grid points with increased spacing to reduce arrow density
        grid_points_x, grid_points_y = np.meshgrid(np.arange(self.xlim[0], self.xlim[1], 3),
                                                np.arange(self.ylim[0], self.ylim[1], 3))

        # Subsample the force field to match the reduced number of grid points
        force_x = self.forcefield[::3, ::3, 0]
        force_y = self.forcefield[::3, ::3, 1]

        # Plot the force field with adjusted arrow scaling and head size
        plt.quiver(grid_points_x, grid_points_y, force_x, force_y, color='b',
                scale=0.1, scale_units='xy', headwidth=1.5, headlength=2)

        # Adjust plot settings
        plt.gca()#.invert_yaxis()
        plt.show()

    ######################PAST FUNCTIONS###########################
    
    def sample_agents(self,final_particles):
        '''Sample a given number of agenys from the particles array at random'''
        
        # Randomly choose indices for the particles to sample (without replacement)
        sampled_indices = np.random.choice(self.required_particles, self.num_agents, replace=False)
        
        # Select the columns corresponding to the sampled indices
        sampled_particles = final_particles[:, sampled_indices]
    
        return sampled_particles
    
    
    def compute_force_field(self):
        '''Compute the force field using vectorized operations.'''

        x_grid, y_grid = np.meshgrid(np.arange(self.xlim[0], self.xlim[1]), np.arange(self.ylim[0], self.ylim[1]), indexing='ij')

        # Stack the grid coordinates into shape (grid_size, 2) where each row is a point [x, y]
        grid_points = np.stack([x_grid, y_grid], axis=-1)  # Shape: (n_x, n_y, 2)

        # Reshape the grid points to (N, 2) where N = n_x * n_y for easy broadcasting
        grid_points_flat = grid_points.reshape(-1, 2)  # Shape: (N, 2)

        # Compute pairwise differences (each point with every other point)
        deltas = grid_points_flat[:, np.newaxis, :] - grid_points_flat[np.newaxis, :, :]  # Shape: (N, N, 2)

        distances = np.linalg.norm(deltas, axis=-1)
        np.fill_diagonal(distances, np.inf) # Avoid division by zero by setting diagonal (self-interaction) to infinit

        unit_vectors = deltas / distances[:, :, np.newaxis]  # Shape: (N, N, 2)
        forces = unit_vectors /distances[:, :, np.newaxis]  # Shape: (N, N, 2)

        flipped_image = np.flipud(self.image)  # Flip image y-axis to match the grid points

        # Reshape the flipped_image to a 1D array, add 2 dimensions (2d forcefield, broadcast for multiplication
        grey_values_flat = (1 - flipped_image).reshape(-1)  # Shape: (N,)
        grey_values_stacked = np.stack([grey_values_flat] * 2, axis=-1)  # Shape: (N, 2)
        grey_values_broadcasted = grey_values_stacked[:, np.newaxis, :]  # Shape: (N, N, 2)

        # Element-wise multiply the grayscale values with the forcefield contributions
        forcefield_contribution = forces * grey_values_broadcasted*self.scaling_factor

        total_forces = forcefield_contribution.sum(axis=0)  # get the total force acting on each point
        total_forces = total_forces.reshape(grid_points.shape)  # Reshape total forces back to grid shape (n_x, n_y, 2)

        return total_forces


    
# # #Example usages
num_agents = 200
num_iterations =500

nbVarX = 2  # State space dimension
nbStates = 2  # Number of Gaussians to represent the spatial distribution
# Desired spatial distribution represented as a mixture of Gaussians
Mu = np.zeros((2,2))
Mu[:,0] = [0.5, 0.7]
Mu[:,1] = [0.6, 0.3]
Sigma = np.zeros((2,2,2))
sigma1_tmp= np.array([[0.3],[0.1]])
Sigma[:,:,0] = sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.eye(nbVarX)*5e-3
sigma2_tmp= np.array([[0.1],[0.2]])
Sigma[:,:,1] = sigma2_tmp @ sigma2_tmp.T * 3e-1 + np.eye(nbVarX)*1e-2 
# Mu = np.array([0.5, 0.7])
# sigma1_tmp= np.array([[0.3],[0.1]])
# Sigma= sigma1_tmp @ sigma1_tmp.T * 5e-1 + np.eye(nbVarX)*5e-3

Priors = np.ones(nbStates) / nbStates # Mixing coeffic

# # #image_path = "black_circle.png"
# #image_path = "dog_grey.jpg"
image_path = "original_distribution.png"

halftoning = ElectrostaticHalftoning(num_agents, image_path, [0,1], [0,1], Mu, Sigma, Priors,"random", num_iterations)
agents = halftoning.run()


