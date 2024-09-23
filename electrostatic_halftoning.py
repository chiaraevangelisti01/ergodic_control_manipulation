import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # For loading the image
from scipy.ndimage import zoom  # Import zoom for resampling
from PIL import Image

class ElectrostaticHalftoning:
    def __init__(self, num_agents, num_iterations, image_path, target_resolution=(20, 20), displacement_threshold=1e-3):
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.displacement_threshold = displacement_threshold  # Convergence threshold
        self.nbVarX = 2  # 2D state space for x, y coordinates

        self.image = self.to_grayscale(image_path)
        # Resample the image to the target resolution
        self.image = self.resample_image(self.image, target_resolution)

        # Extract xlim and ylim from the resampled image shape
        self.xlim = (0, self.image.shape[0])  # Number of rows in the image defines the x-limits
        self.ylim = (0, self.image.shape[1])  # Number of columns in the image defines the y-limits

        self.forcefield = None
        self.particles = None
    
    from PIL import Image

    def to_grayscale(self,image_path):
        """Convert an image to grayscale using Pillow."""
        image = Image.open(image_path).convert('L')  # 'L' mode is for grayscale
        return np.array(image)

    def resample_image(self, image, target_resolution):
        """Resample the input image to the target resolution."""
        zoom_factors = (
            target_resolution[0] / image.shape[0],
            target_resolution[1] / image.shape[1]
        )
        return zoom(image, zoom_factors)  # Use zoom to resample the image

    def compute_force(self, p, g):
        # Calculate force between two points (Coulomb's law-like)-> force based on distance, may be modified
        distance = np.linalg.norm(np.array(p) - np.array(g))
        if distance == 0:
            return np.array([0, 0])  # No force on same point
        return (np.array(p) - np.array(g)) / distance**2
    
    def compute_force_field(self, grid_points):
        forcefield = np.zeros((grid_points.shape[0], grid_points.shape[1], 2))  # Shape for 2D forces

        for ip, jp in np.ndindex(grid_points.shape[:2]):  # Loop over the grid for point p
            p = np.array([ip, jp])  # Get the position of the current grid point p

            for ig, jg in np.ndindex(grid_points.shape[:2]):  # Loop over the grid for point g
                if (ip, jp) != (ig, jg):  # Avoid self-interaction
                    g = np.array([ig, jg])  # Get the position of the other grid point g
                    forcefield[ip, jp] += self.compute_force(p, g)  # Add force contribution from g to p
        return forcefield

    def bilinear_interpolation(self, forcefield, p):
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

    def select_random_pixel(self):
        x = np.random.randint(0, self.image.shape[0])
        y = np.random.randint(0, self.image.shape[1])
        return x, y

    def initialize_particles(self):
        particles = np.zeros((self.nbVarX, self.num_agents))  # Initialize particle positions array with shape (nbVarX, num_agents)
        max_grey = np.min(self.image)
        
        for i in range(self.num_agents):
            attempts = 0
            while True:
                x, y = self.select_random_pixel()
                if (x, y) not in particles.T and np.random.uniform(0, max_grey) < self.image[x, y]:
                    particles[:, i] = [x, y]  # Fill particle's position
                    break
                attempts += 1
                if attempts > 1000:  # Avoid infinite loop
                    particles[:, i] = [x, y]
                    break
        
        return particles

    def evolve_particles(self, particles, forcefield):
        positions_over_time = []
        tau = 0.01  # Step size, reduce it for more fine-grained movement
        shaking_strength = 0.01  # Shaking strength
        converged = False

        for iteration in range(self.num_iterations):
            max_displacement = 0  # Track maximum displacement for convergence

            for i, p in enumerate(particles.T):
                p_old = p.copy()  # Copy the particle position to avoid reference issues
                
                forcep = self.bilinear_interpolation(forcefield, p[:2])

                for j, q in enumerate(particles.T):
                    if i != j:
                        forcep += self.compute_force(p[:2], q[:2])

                # Update particle position
                particles[:, i] = p + tau * np.hstack((forcep, np.zeros(self.nbVarX - 2)))

                # Calculate displacement by comparing with the old position (before update)
                displacement = np.linalg.norm(particles[:, i] - p_old)
                max_displacement = max(max_displacement, displacement)

                # Print debugging information for forces and displacement
                print(f"Iteration {iteration}, Particle {i} Force applied: {forcep}, Displacement: {displacement}")

                if iteration % 10 == 0:
                    shaking_factor = (1 - iteration / self.num_iterations)
                    particles[:, i] += np.hstack((
                        np.random.uniform(-shaking_strength, shaking_strength, 2) * shaking_factor,
                        np.zeros(self.nbVarX - 2)))  # Shake only in 2D

            positions_over_time.append(particles.copy())

            # Print maximum displacement for this iteration
            print(f"Iteration {iteration}: Max Displacement = {max_displacement}")

            # Check for convergence
            if max_displacement < self.displacement_threshold:
                print(f"Converged at iteration {iteration + 1}")
                converged = True
                break

        return particles, positions_over_time, converged


    def plot_positions(self, positions_over_time):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])  # Align the image correctly

        # Overlay particles on the grid
        for i, positions in enumerate(positions_over_time):
            ax.scatter(positions[1, :], positions[0, :], label=f"Iteration {i+1}", color='blue')

        plt.xlim([0, self.ylim[1]])
        plt.ylim([0, self.xlim[1]])
        plt.show()

    def run(self):
        grid_points = np.array(np.meshgrid(np.arange(self.xlim[0], self.xlim[1]), np.arange(self.ylim[0], self.ylim[1])))
        self.forcefield = self.compute_force_field(grid_points)
        self.particles = self.initialize_particles()

        final_particles, positions_over_time, converged = self.evolve_particles(self.particles, self.forcefield)

        self.plot_positions(positions_over_time)

        return final_particles, converged

# Example usages
num_agents = 10
num_iterations = 100
#image = np.random.rand(15, 15)  # Synthetic grid of random values for grayscale image


image_path ="C:/Users/Chiara/Documents/CHIARA/Scuola/UNIVERSITA/MAGISTRALE/Semester III/Semestral project/ergodic_control_manipulation/dog_grey.jpg"

halftoning = ElectrostaticHalftoning(num_agents, num_iterations, image_path)
final_positions, has_converged = halftoning.run()

if has_converged:
    print("The system has converged.")
