import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # For loading the image
from scipy.ndimage import zoom  # Import zoom for resampling
from PIL import Image

class ElectrostaticHalftoning:
    def __init__(self, num_agents, num_iterations, image_path, target_resolution=(20, 20), displacement_threshold=1e-4):
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
    
    def compute_force(self, p, g, normalize=True):
        # Calculate force between two points (Coulomb's law-like)
        distance_vector = np.array(p) - np.array(g)
        distance = np.linalg.norm(distance_vector)
        
        if distance == 0:
            return np.array([0, 0])  # No force if distance is 0 (same point)
        
        # Compute the force direction
        force = distance_vector / (distance**2)  # Coulomb force with inverse square law
        
        if normalize:
            # Normalize the force by its magnitude
            force_norm = np.linalg.norm(force)
            if force_norm > 0:
                force = force / force_norm
        
        return force

    
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
        max_grey = np.max(self.image)
    
        
        for i in range(self.num_agents):
            attempts = 0
            while True:
                x, y = self.select_random_pixel()
                #print("Local grey",self.image[x, y])
                if (x, y) not in particles.T and np.random.uniform(0, max_grey) > self.image[x, y]:
                    particles[:, i] = [x, y]  # Fill particle's position
                    break
                attempts += 1
                if attempts > 1000:  # Avoid infinite loop
                    particles[:, i] = [x, y]
                    break
        
        return particles

    def evolve_particles(self, particles, forcefield):
        positions_over_time = []
        tau = 0.1  # Step size, reduce it for more fine-grained movement
        shaking_strength = 0.001  # Shaking strength
        converged = False

        N = self.image.shape[0] * self.image.shape[1]  # Total number of pixels
        M = self.num_agents  # Number of particles
        scal_fac = 10

        for iteration in range(self.num_iterations):
            max_displacement = 0  # Track maximum displacement for convergence

            for i, p in enumerate(particles.T):
                p_old = p.copy()  # Copy the particle position to avoid reference issues
                
                forcep = self.bilinear_interpolation(forcefield, p[:2])/N
                if iteration % 10 == 0:
                    print( "Attraction",forcep)
                
                tot = np.array([0.0, 0.0])
                for j, q in enumerate(particles.T):
                    if i != j:
                        forcep += self.compute_force(p[:2], q[:2])/(M*scal_fac)
                        tot += self.compute_force(p[:2], q[:2])/(M*scal_fac)

                if iteration % 10 == 0:
                        print("Repulsion",tot) 
                        print( "Total",forcep)
                      

                # Update particle position
                particles[:, i] = p + tau * np.hstack((forcep, np.zeros(self.nbVarX - 2)))
                particles[0, :] = np.clip(particles[0, :], 0, self.xlim[1] )  # X-coordinates (rows)
                particles[1, :] = np.clip(particles[1, :], 0, self.ylim[1] )  # Y-coordinates (columns)

                # Calculate displacement by comparing with the old position (before update)
                displacement = np.linalg.norm(particles[:, i] - p_old)
                max_displacement = max(max_displacement, displacement)

                # if iteration % 10 == 0:
                #     shaking_factor = (1 - iteration / self.num_iterations)**2
                #     particles[:, i] += np.hstack((
                #         np.random.uniform(-shaking_strength, shaking_strength, 2) * shaking_factor,
                #         np.zeros(self.nbVarX - 2)))  # Shake only in 2D

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

        # Display the background image in grayscale
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])

        # Display the pixel values at each grid point (without flipping the y-axis)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel_value = int(self.image[i, j])
                # Ensure the text is centered on the pixels
                ax.text(j + 0.5, self.image.shape[0] - i - 0.5,  # Invert y-axis for text
                    str(pixel_value), fontsize=8, ha='center', va='center', color='red')

        # Set up a colormap for the fading color effect
        num_iterations = len(positions_over_time)
        color = plt.cm.Blues  # Using a blue colormap

        # Overlay particles on the grid, ensuring correct placement
        for i, positions in enumerate(positions_over_time):
            ax.scatter(positions[1, :], positions[0, :], label=f"Iteration {i+1}",
                    s=10,  # Adjusted size for smaller markers
                    color=color(i / num_iterations),  # Color fades with iterations
                    alpha=0.2 + (0.8 * (i / num_iterations)))  # Alpha starts low and increases with iterations

        # Set the limits of the plot
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
num_agents = 50
num_iterations =100 
#image = np.random.rand(15, 15)  # Synthetic grid of random values for grayscale image

image_path ="C:/Users/Chiara/Documents/CHIARA/Scuola/UNIVERSITA/MAGISTRALE/Semester III/Semestral project/ergodic_control_manipulation/dog_grey.jpg"

halftoning = ElectrostaticHalftoning(num_agents, num_iterations, image_path)
final_positions, has_converged = halftoning.run()

if has_converged:
    print("The system has converged.")
