import os
import time
import glob
import argparse
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from scipy.ndimage import zoom
from scipy.interpolate import interp1d  

def generate_trajectories(directory, num_agents, num_points, image_path):
    """Main function to generate trajectories and save them, with an image background."""
    global display_trajectories, mouse_pose, x, y, capture_on, delta_t, t_prev
    def clear():
        mouse_pose.clear()
        x.clear()
        y.clear()
    
    def resample_trajectory(x, y, num_points):
        """Resample the trajectory to have exactly num_points."""
        if len(x) < 2 or len(y) < 2:
            return np.array(x), np.array(y)  # Not enough points to resample

        # Create an array of evenly spaced points between 0 and 1 for the original and new trajectory
        t_original = np.linspace(0, 1, len(x))  # Time scale of the original data
        t_resampled = np.linspace(0, 1, num_points)  # Time scale for resampling

        # Interpolate x and y positions
        x_resampled = interp1d(t_original, x, kind='linear')(t_resampled)
        y_resampled = interp1d(t_original, y, kind='linear')(t_resampled)

        return x_resampled, y_resampled

    def on_click(event, directory, file_prefix, num_points,num_agents):
        """Toggle capture_on state on left mouse button click"""
        global capture_on
        # Always allow toggling off to stop and save the current trajectory
        if capture_on or (event.button == 1 and len(display_trajectories) < num_agents):  
            if not capture_on:
                # Start capture and create a plot for the new trajectory
                clear()
                (tr,) = plt.plot([], [])
                display_trajectories.append(tr)
            else:
                # Stop capture and resample trajectory to have exactly num_points
                x_resampled, y_resampled = resample_trajectory(x, y, num_points)
                filename = os.path.join(
                    directory, "{}{}.npy".format(file_prefix, len(display_trajectories) - 1)  # Use -1 because it was appended earlier
                )
                np.save(filename, np.asarray([x_resampled, y_resampled]).T)

            capture_on = not capture_on

    def on_capture_mouse_motion(event):
        global t_prev, capture_on
        if not capture_on:
            return
        t = time.time()
        if (t - t_prev) >= delta_t:
            mouse_pose.append((event.xdata, event.ydata))
            t_prev = t
    
    def display(i):
        global capture_on
        if not capture_on or len(list(mouse_pose)) == 0:
            return
        p = mouse_pose.popleft()

        x.append(p[0])
        y.append(p[1])

        display_trajectories[-1].set_data(x, y)
        return display_trajectories
    
    def on_key(event):
        # if event.key == 'a':
        # Do something
        pass

    display_trajectories = []
    mouse_pose = deque(maxlen=10)
    x = deque(maxlen=1000)
    y = deque(maxlen=1000)
    capture_on = False

    # Time between two recorded mouse poses
    delta_t = 0.02
    t_prev = 0

    file_prefix = "traj"
    if not os.path.exists(directory):
        os.makedirs(directory)

    xmin, xmax = 0, 1.0
    ymin, ymax = 0, 1.0
    
    fig = plt.figure(figsize=(10, 6))
    
    # Replace by an image or distributions

    ax = plt.gca()
    img = np.array(Image.open(image_path).convert('L'))
    ax.imshow(img,cmap='gray', origin='upper', extent=[xmin-0.16, xmax+0.15, ymin-0.15, ymax+0.16],aspect = 'equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect(
        "button_press_event", lambda event: on_click(event, directory, file_prefix, num_points,num_agents)
    )
    fig.canvas.mpl_connect("motion_notify_event", on_capture_mouse_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    

    anim = animation.FuncAnimation(
        fig, display, 1000, interval=20, blit=False, repeat=True
    )
    plt.show()
    

    # # Read and plot trajectories saved in .npy files
    # fig = plt.figure(figsize=(10, 6))
    # ax = plt.gca()
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # filepath = os.path.join(directory, "{}*.npy".format(file_prefix))
    # for filename in sorted(glob.glob(filepath)):
    #     tr = np.load(filename)
    #     plt.plot(tr[:, 0], tr[:, 1], label=filename)
    # plt.legend()
    # plt.show()


   
