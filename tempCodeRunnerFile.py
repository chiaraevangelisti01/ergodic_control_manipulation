    fig, ax = plt.subplots(figsize=(10, 10))

        # Display the background image in grayscale
        ax.imshow(self.image, cmap='gray', origin='upper', extent=[0, self.ylim[1], 0, self.xlim[1]])  # Align the image correctly

        # Display the pixel values at each grid point
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel_value = int(self.image[i, j])
                ax.text(j + 0.5, i + 0.5, str(pixel_value), fontsize=8, ha='center', va='center', color='red')

        # Set up a colormap for the fading color effect
        num_iterations = len(positions_over_time)
        color = plt.cm.Blues  # Using a blue colormap

        # Overlay particles on the grid, with size and color fading
        for i, positions in enumerate(positions_over_time):
            ax.scatter(positions[1, :], positions[0, :], label=f"Iteration {i+1}",
                    s=10,  # Adjusted size for smaller markers
                    color=color(i / num_iterations),  # Color fades with iterations
                    alpha=0.2 + (0.8 * (i / num_iterations)))  # Alpha starts low and increases with iterations

        # Set the limits of the plot
        plt.xlim([0, self.ylim[1]])
        plt.ylim([0, self.xlim[1]])

        plt.show()  