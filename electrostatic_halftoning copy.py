import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # For loading the image
from scipy.ndimage import zoom  # Import zoom for resampling
from PIL import Image

class ElectrostaticHalftoning:

    def __init__(self, num_agents, image_path, num_iterations = 250, target_resolution=(13, 13), displacement_threshold=7e-3):

        self.num_agents = num_agents # for sampling at steady-state
        self.image_path = image_path
        self.num_iterations = num_iterations
        self.displacement_threshold = displacement_threshold  # Convergence threshold
        self.target_resolution = target_resolution  # Target resolution for the image processing
        
        self.nbVarX = 2  # 2D state space for x, y coordinates
        
        self.xlim = None
        self.ylim = None
        self.forcefield = None
        self.particles = None
        self.required_particles = None


    def process_image(self,image_path):
        """Convert an image to grayscale and resample it to the target resolution."""

        image = Image.open(image_path).convert('L')  # 'L' mode  for grayscale
        image = np.array(image)/255 # Normalize pixel values to [0, 1] as required by rhe algorithms

        zoom_factors = (self.target_resolution[0] / image.shape[0],self.target_resolution[1] / image.shape[1])

        return zoom(image,zoom_factors)


    def number_particles(self):
        ''' Compute required number of particles to maintain the average grey value of the input image'''
        required_particles = 0
        for i in range(self.image.shape[0]):  # Iterate over rows
            for j in range(self.image.shape[1]):  # Iterate over columns
                u_x = self.image[i, j]
                required_particles += (1-u_x )
        
        return int(np.round(required_particles))


    def compute_force(self, p, g):
        '''Compute the force between two points p and g'''
        # Calculate force between two points (Coulomb's law-like)
        direction_vector = np.array(g) - np.array(p) # Vector from p to g (when forcefield p is the particle and g is the source of attraction)
        magnitude = np.linalg.norm(direction_vector) # Distance between the two points
        
        if magnitude == 0:
            return np.array([0, 0])  # No force if distance is 0 (same point)
        else:
            unit_vector = direction_vector / magnitude  # Unit vector in the direction of the force
        
       #Force magnitude 1/r , charges set to 1 to obtain neutrality
        force = unit_vector / (magnitude)  # Coulomb force with inverse law --> NOT SQUARE BECAUSE 2D
        
        return force


    def compute_force_field(self):
        '''Store force field attractive force for each grid point'''
        grid_points = np.array(np.meshgrid(np.arange(self.xlim[0], self.xlim[1]), np.arange(self.ylim[0], self.ylim[1])))
        forcefield = np.zeros((grid_points.shape[1], grid_points.shape[2], 2))  #  2D forces

        for ip, jp in np.ndindex(grid_points.shape[1], grid_points.shape[2]):  # Imaginary charge moving over the grid
            p = np.array([ip, jp])  
            
            for ig, jg in np.ndindex(grid_points.shape[1], grid_points.shape[2]): 
                
                if (ip, jp) != (ig, jg):  # Avoid self-interaction
                    g = np.array([ig, jg])  
                    
                    # Include image influence: darker areas exert stronger attractive force
                    flipped_ig = self.image.shape[0] - ig - 1 #image has inverted y-axis compared to grid_points
                    grey_value = self.image[flipped_ig, jg] # Grey value at position g

                    forcefield[ip, jp] += self.compute_force(p, g)* ((1 - grey_value)*1) # Scale by grey value
                               
        return forcefield
    
   
    def initialize_particles(self):
        '''Initialize particles in the image randomly'''

        particles = np.zeros((self.nbVarX, self.required_particles)) 
        max_grey = np.max(self.image)
        i = 0
    
        for i in range(self.required_particles):
            attempts = 0
            while True:
                x = np.random.randint(0, self.image.shape[0])
                y = np.random.randint(0, self.image.shape[1])
                if (x, y) not in particles.T and np.random.uniform(0, max_grey) > self.image[self.image.shape[0]-x-1, y]:
                    particles[:, i] = [x, y]  
                    break
                attempts += 1
                if attempts > 1000:  # Avoid infinite loop
                    particles[:, i] = [x, y]
                    break
    
        return particles


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

    
    def evolve_particles(self):
        '''Compute particle movement'''
        positions_over_time = []
        tau = 0.006  # Step size, reduce it for more fine-grained movement
        shaking_strength = 0.001  # Shaking strength
        converged = False

        for iteration in range(self.num_iterations):
            max_displacement = 0  

            for i, p in enumerate(self.particles.T):
                p_old = p.copy()  
                
                #Attractive force from the grid --> it was already computed influence from the entire grid
                forcep = self.bilinear_interpolation(self.forcefield, p)
                if iteration % 30 == 0:
                    print( "Attraction",forcep)
                
                tot = np.array([0.0, 0.0])
                for j, q in enumerate(self.particles.T):
                    if i != j:
                        forcep -= self.compute_force(p, q) #Repulsive forces are negative
                        tot -= self.compute_force(p, q)

                if iteration % 30 == 0:
                        print("Repulsion",tot) 
                        print( "Total",forcep)
                      

                # Update particle position
                self.particles[:, i] = p + tau * forcep

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


    def plot_positions(self, positions_over_time):
        '''Plot particle evolution, initial, and final positions in a single figure with subplots'''
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create 3 subplots side by side

        # Extract initial and final positions
        initial_positions = positions_over_time[0]
        final_positions = positions_over_time[-1]

        # Evolution over time (middle plot)
        ax = axes[0]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        num_iterations = len(positions_over_time)
        color = plt.cm.Blues  # Using a colormap for the evolution
        
        for i, positions in enumerate(positions_over_time):
            ax.scatter(positions[1, :], positions[0, :], label=f"Iteration {i+1}",
                    s=15,  # Marker size
                    color=color(i / num_iterations), 
                    alpha=0.2 + (0.8 * (i / num_iterations)))  # Alpha increases with iterations
        
        ax.set_title("Evolution of Particles")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        # Initial positions (left plot)
        ax = axes[1]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        ax.scatter(initial_positions[1, :], initial_positions[0, :], color='red', s=30, alpha=0.8)
        ax.set_title("Initial Particle Positions")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        # Final positions (right plot)
        ax = axes[2]
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])
        ax.scatter(final_positions[1, :], final_positions[0, :], color='blue', s=30, alpha=0.9)
        ax.set_title("Final Particle Positions")
        ax.set_xlim([0, self.ylim[1]])
        ax.set_ylim([0, self.xlim[1]])
        ax.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    
    def plot_force_field(self):

        # Generate the grid points
        grid_points_x, grid_points_y = np.meshgrid(np.arange(self.xlim[0], self.xlim[1]),np.arange(self.ylim[0], self.ylim[1]))

        # Plot the force field, scaling the arrows by the force magnitude
        plt.quiver(grid_points_x, grid_points_y, self.forcefield[:, :, 0], self.forcefield[:, :, 1], color='b',scale=10, scale_units='xy')  

        # Adjust plot settings
        plt.gca().invert_yaxis()
        plt.show()


    def run(self):
        '''Run the electrostatic halftoning algorithm'''

        # Process the input image
        self.image = self.process_image(self.image_path)

        # Initialize grid points 
        self.xlim = (0, self.image.shape[0])  # Number of rows in the image defines the x-limits
        self.ylim = (0, self.image.shape[1])  # Number of columns in the image defines the y-limits

        # Compute the required number of particles to maintain the average grey value
        self.required_particles = self.number_particles()
        
        #Compute the force field (plot for debugging)
        self.forcefield = self.compute_force_field()
        self.plot_force_field()
        
        # Initialize particles in the  domain
        self.particles = self.initialize_particles()

        # Evolve the particles
        final_particles, positions_over_time, converged = self.evolve_particles()

        self.plot_positions(positions_over_time)

        return final_particles, converged

# Example usages
num_agents = 50
num_iterations =500
#image = np.random.rand(15, 15)  # Synthetic grid of random values for grayscale image

image_path ="C:/Users/Chiara/Documents/CHIARA/Scuola/UNIVERSITA/MAGISTRALE/Semester_III/Semestral project/ergodic_control_manipulation/dog_grey.jpg"

halftoning = ElectrostaticHalftoning(num_agents, image_path,num_iterations)
final_positions, has_converged = halftoning.run()

if has_converged:
    print("The system has converged.")
