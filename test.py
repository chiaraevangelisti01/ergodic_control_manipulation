# main_script.py

from record_trajectories_function import generate_trajectories

def main():
    directory = './mouse_trajectories'
    image_path = 'reconstructed_distribution.png'
    num_agents = 5
    num_points = 100

    generate_trajectories(directory, num_agents, num_points, image_path)

if __name__ == "__main__":
    main()
