import sys
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

def update_gauss_markov_process(previous_noise, theta, sigma, delta_t=1):
    """ Update the Gauss-Markov process to generate correlated noise.
        `delta_t` is the time step and is typically 1 for frame-by-frame correlation. """
    return theta * previous_noise + np.random.normal(0, np.sqrt((1 - theta**2) * sigma**2), size=3)

def add_noise_to_matrix(matrix, rot_noise, trans_noise):
    """ Add noise to the rotation and translation components of a transformation matrix. """
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    # Convert rotation matrix to Euler angles, add noise, and convert back
    euler_angles = matrix_to_euler_angles(torch.tensor(rotation_matrix.T), convention="XYZ")
    noisy_euler_angles = euler_angles + (rot_noise * np.pi / 180)  # noise is already in degrees
    noisy_rotation_matrix = euler_angles_to_matrix(noisy_euler_angles, convention="XYZ").numpy().T

    # Add noise to translation vector
    noisy_translation_vector = translation_vector + trans_noise
    
    # Construct new noisy matrix
    noisy_matrix = np.eye(4)
    noisy_matrix[:3, :3] = noisy_rotation_matrix
    noisy_matrix[:3, 3] = noisy_translation_vector
    
    return noisy_matrix

def process_folder(source_folder, dest_folder, theta, sigma_deg, sigma_m):
    os.makedirs(dest_folder)
    
    # Initialize Gauss-Markov processes for rotation and translation
    rotation_noise = np.random.normal(0, sigma_deg, size=3)
    translation_noise = np.random.normal(0, sigma_m, size=3)
    
    # rot_noises = [rotation_noise]
    # tra_noises = [translation_noise]

    for filename in tqdm(sorted(os.listdir(source_folder))):
        if filename.endswith(".txt"):
            file_path = os.path.join(source_folder, filename)
            matrix = np.loadtxt(file_path)
            
            # Update noises
            rotation_noise = update_gauss_markov_process(rotation_noise, theta, sigma_deg)
            translation_noise = update_gauss_markov_process(translation_noise, theta, sigma_m)
            
            # rot_noises.append(rotation_noise)
            # tra_noises.append(translation_noise)

            # Add correlated noise to the matrix
            noisy_matrix = add_noise_to_matrix(matrix, rotation_noise, translation_noise)
            noisy_file_path = os.path.join(dest_folder, filename)
            np.savetxt(noisy_file_path, noisy_matrix)

    # plt.plot(list(range(len(rot_noises))), [r[0] for r in rot_noises])
    # plt.show()
    # plt.plot(list(range(len(tra_noises))), [r[0] for r in tra_noises])
    # plt.show()


theta = 0.95  # Correlation coefficient (close to 1 for high correlation)

for dataset in ["Company_speedx100", "Restaurant_speedx100", "ScienceLab_speedx100", "Subway_speedx100"]:
    sigma_deg = 0.3  # Long-term standard deviation for rotation noise in degrees
    sigma_cm = 2  # Long-term standard deviation for translation noise in mete
    
    root_dir = f"/CT/EventSLAM/static00/data/synthetic/{dataset}/train"
    source_folder = f'{root_dir}/pose'
    dest_folder = f'{root_dir}/pose_deg={sigma_deg}_cm={sigma_cm}'

    process_folder(source_folder, dest_folder, theta, sigma_deg, sigma_cm / 100)

    sigma_deg = 3.  # Long-term standard deviation for rotation noise in degrees
    sigma_cm = 2  # Long-term standard deviation for translation noise in meters

    root_dir = f"/CT/EventSLAM/static00/data/synthetic/{dataset}/train"
    source_folder = f'{root_dir}/pose'
    dest_folder = f'{root_dir}/pose_deg={sigma_deg}_cm={sigma_cm}'

    process_folder(source_folder, dest_folder, theta, sigma_deg, sigma_cm / 100)
