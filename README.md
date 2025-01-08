# Repository Overview

This repository is organized into three main branches for modularity and clarity:

---

## 1. **Main Branch**
- Contains:
  - The **report**, which provides a comprehensive explanation of the approaches and results.
  - The `requirements.txt` file, listing all dependencies required for the project.

---

## 2. **Initialization Branch**
This branch contains three folders, each with a specific purpose:

### **a. Preparatory Files**
- Includes scripts used to investigate the techniques independently of the ergodic control problem.
- Purpose:
  - Understand the individual approaches.
  - Refine and improve the techniques.

### **b. Single Agent**
- Focuses on **ergodic control trajectory optimization**:
  - **Sine pattern** optimization.
  - **Cosine pattern** optimization.
  - **Mouse trajectories** optimization:
    - The `mouse_trajectories` file is directly called within the trajectory optimization script.
    - Stores trajectories in the `mouse_trajectories` folder.
    - Overwrites previous trajectories whenever new ones are generated.

### **c. Multi-Agent**
- Contains classes for two main methods:
  - **Electrostatic halftoning**.
  - **Diffusion-based approach**.
- Usage:
  - Can be called independently to test each method.
  - Alternatively, integrated into the **ergodic control trajectory optimization problem**.
- Each folder includes images required to run the scripts.

---

## 3. **IK_Tracking Branch**
- Contains a single file for **weighted inverse kinematics**:
  - Takes a given trajectory as input to perform the calculations.

---

## Notes
- Each folder includes any required images for running the scripts.
- Ensure you follow the dependencies listed in `requirements.txt` before running any file.
