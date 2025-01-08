import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import io, transform


class DiffusionBasedPlacement:
    def __init__(self, xdom, ydom, image_path, resolution = 100, diffusion_strength = 0.001, num_particles = 1000, kernel_type='circular', repeat_diffusion=1, agent_radius=0.0001, agent_sigmax=0.005, agent_sigmay=0.015):
        self.resolution = resolution
        self.diffusion_strength = diffusion_strength
        self.num_particles = num_particles
        self.xdom = xdom
        self.ydom = ydom
        self.image_path = image_path
        self.kernel_type = kernel_type
        self.agent_radius = agent_radius
        self.agent_sigmax = agent_sigmax
        self.agent_sigmay = agent_sigmay
        self.repeat_diffusion = repeat_diffusion
        
        self.dx = 1
        self.alpha = 0.99*np.array([1, 1]) * self.diffusion_strength
        self.reference_radius = 0.0004  
        # self.alpha = (
        #     np.array([1, 1])
        #     * self.diffusion_strength
        #     * (self.resolution**0.75 / ((self.num_particles)**0.7))
        #     * ((self.agent_radius / self.reference_radius)**0.7)
        #     )


        self.dt = np.min((1.0, (self.dx * self.dx) / (4.0 * np.max(self.alpha))))
        
        self.particles = None
        self.img = None

    def normalize_mat(self, mat):
        return mat / (np.sum(mat) + 1e-10)

    def offset(self, mat, i, j):
        rows, cols = mat.shape
        rows = rows - 2
        cols = cols - 2
        return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]

    def process_image(self):
        self.img = io.imread(self.image_path)
        self.img = transform.resize(self.img[:, :, 0], (self.resolution, self.resolution))
        distribution = 1.0 - self.img
        distribution = distribution * self.resolution**2 / np.sum(distribution) * 1.0
        distribution = self.normalize_mat(distribution)
        return distribution

    def cool_heat_source(self, xi, yi):
        if self.kernel_type == 'elliptical':
            dists = (
                ((self.x_ids - xi / self.resolution) ** 2) / (2 * self.agent_sigmax ** 2)
                + ((self.y_ids - yi / self.resolution) ** 2) / (2 * self.agent_sigmay ** 2)
            )
            coverage_density = np.exp(-dists)
        elif self.kernel_type == 'circular':
            dists = np.sqrt((self.x_ids - xi / self.resolution) ** 2 + (self.y_ids - yi / self.resolution) ** 2)
            coverage_density = np.exp(-(1 / self.agent_radius) * dists**2)
        
        return self.normalize_mat(coverage_density) / self.num_particles

    def place_particles(self):
        y_ids, x_ids = np.indices((self.resolution, self.resolution))
        self.x_ids = x_ids.astype("float32") / (self.resolution - 1)
        self.y_ids = y_ids.astype("float32") / (self.resolution - 1)

        particles = np.zeros((self.num_particles, 2))
        distribution = self.process_image()

        for i in range(self.num_particles):
            heat = distribution
            for _ in range(self.repeat_diffusion):
                heat[1:-1, 1:-1] = self.dt * (
                    (
                        + self.alpha[0] * self.offset(heat, 1, 0)
                        + self.alpha[0] * self.offset(heat, -1, 0)
                        + self.alpha[1] * self.offset(heat, 0, 1)
                        + self.alpha[1] * self.offset(heat, 0, -1)
                        - 4.0 * self.diffusion_strength * self.offset(heat, 0, 0)
                    )
                    / (self.dx * self.dx)
                ) + self.offset(heat, 0, 0)
            heat = heat.astype(np.float32)
            id = np.argmax(heat)
            yi, xi = np.unravel_index(id, np.array(heat).shape)
            particles[i, 0:2] = (xi, yi)

            coverage = self.cool_heat_source(xi, yi)
            # plt.imshow(coverage, cmap='gray')
            # plt.show()      
            distribution -= coverage
            distribution = np.maximum(distribution, 0)
            print("Particle {} added".format(i), xi, yi)
            

        # Rescale particles to the original domain
        particles[:, 0] = particles[:, 0] / self.resolution * (self.xdom[1] - self.xdom[0]) + self.xdom[0]
        #particles[:, 1] = particles[:, 1] / self.resolution * (self.ydom[1] - self.ydom[0]) + self.ydom[0]
        particles[:, 1] = (self.resolution - particles[:, 1]) / self.resolution * (self.ydom[1] - self.ydom[0]) + self.ydom[0]

        self.particles = particles


    def run(self):
        self.process_image()
        self.place_particles()
        self.plot_particles()
        
        return self.particles

    def plot_particles(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the original image on the first subplot
        ax[0].imshow(self.img, cmap="gray")
        ax[0].set_title("Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Plot the particle positions on the second subplot
        ax[1].scatter(self.particles[:, 0], self.particles[:, 1], c='b', s=8)
        ax[1].set_xlim(self.xdom)
        ax[1].set_ylim(self.ydom)
        ax[1].set_title("Particle Positions")
        ax[1].set_xlabel("X-axis")
        ax[1].set_ylabel("Y-axis")

        plt.tight_layout()
        plt.show()



# if __name__ == "__main__":
#     resolution = 100
#     diffusion_strength = 0.001
#     num_particles = 1000
#     xdom = [0, 1]
#     ydom = [0, 1]
#     image_path = "spatial_distribution.png"
#     kernel_type = "circular"
#     agent_radius = 0.0005

#     placement = DiffusionBasedPlacement(xdom, ydom, image_path, resolution, diffusion_strength, num_particles, kernel_type, agent_radius=agent_radius)
#     particles = placement.run()
#     placement.plot_particles()
