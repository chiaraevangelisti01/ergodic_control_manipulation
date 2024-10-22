#!/usr/bin/env python3
"""
Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
"""

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


# Create a directory if it does not exist to save trajectories
# Tracking the mouse motion in the image

display_trajectories = []
mouse_pose = deque(maxlen=10)
x = deque(maxlen=1000)
y = deque(maxlen=1000)
capture_on = False

# Time between two recorded mouse poses
delta_t = 0.02
t_prev = 0

num_agents = 0 # to track how many trajectories need to be recorded


def clear():
    mouse_pose.clear()
    x.clear()
    y.clear()


def on_click(event, directory, file_prefix):
    """Toggle capture_on state on left mouse button click"""
    global capture_on, num_agents
    # Always allow toggling off to stop and save the current trajectory
    if capture_on or (event.button == 1 and len(display_trajectories) < num_agents):  
        if not capture_on:
            # Start capture and create a plot for the new trajectory
            clear()
            (tr,) = plt.plot([], [])
            display_trajectories.append(tr)
        else:
            # Stop capture and save trajectory
            filename = os.path.join(
                directory, "{}{}.npy".format(file_prefix, len(display_trajectories) - 1)  # Use -1 because it was appended earlier
            )
            np.save(filename, np.asarray([x, y]).T)

        capture_on = not capture_on

def on_key(event):
    # if event.key == 'a':
    # Do something
    pass


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


def main():
    global num_agents
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Relative path to directory to create and save trajectories",
    )
    parser.add_argument(
    "-a",
    "--agents",
    type=int,
    required=True,
    help="Number of agents (trajectories) to create",
    )

    args = parser.parse_args()
    # directory = 'tmp'
    directory = args.directory
    num_agents = args.agents
    file_prefix = "traj"
    if not os.path.exists(directory):
        os.makedirs(directory)

    xmin, xmax = 0, 1.0
    ymin, ymax = 0, 1.0

    fig = plt.figure(figsize=(10, 6))
    # Replace by an image or distributions

    ax = plt.gca()
    img = np.array(Image.open('reconstructed_distribution.png').convert('L'))
    ax.imshow(img,cmap='gray', origin='upper', extent=[xmin-0.16, xmax+0.15, ymin-0.15, ymax+0.16],aspect = 'equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect(
        "button_press_event", lambda event: on_click(event, directory, file_prefix)
    )
    fig.canvas.mpl_connect("motion_notify_event", on_capture_mouse_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = animation.FuncAnimation(
        fig, display, 1000, interval=20, blit=False, repeat=True
    )
    plt.show()

    # Read and plot trajectories saved in .npy files
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    filepath = os.path.join(directory, "{}*.npy".format(file_prefix))
    for filename in sorted(glob.glob(filepath)):
        tr = np.load(filename)
        plt.plot(tr[:, 0], tr[:, 1], label=filename)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
