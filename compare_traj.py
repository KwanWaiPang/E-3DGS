#!/usr/bin/env python3
import numpy as np
import json
import os
import sys
from os import path
from tqdm import trange, tqdm
import torch
from pytorch3d.transforms import matrix_to_euler_angles
import matplotlib.pyplot as plt


def get_errors(pose_folder1, pose_folder2):

    angle_errors = []
    position_errors = []

    mean_position_error = None

    def getAngle(P, Q):
        R = np.dot(P, Q.T)
        cos_theta = (np.trace(R)-1)/2
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta) * (180/np.pi)

    for pose_filename in tqdm(sorted(os.listdir(os.path.join(pose_folder1)))):
        if "start" in pose_filename:
            continue
        # if not imgfn.endswith('png'):
        #     continue

        # if not os.path.exists(os.path.join(base_path, pose_folder2, pose_filename.replace(".txt", "_train.txt"))):
            # continue

        c2w_1 = np.loadtxt(path.join(os.path.join(pose_folder1, pose_filename)))
        try:
            c2w_2 = np.loadtxt(path.join(os.path.join(pose_folder2, pose_filename.replace(".txt", ".txt"))))
        except:
            c2w_2 = np.loadtxt(path.join(os.path.join(pose_folder2, pose_filename.replace(".txt", "_train.txt"))))

        w2c_1 = np.linalg.inv(c2w_1)
        w2c_2 = np.linalg.inv(c2w_2)

        angles = matrix_to_euler_angles(torch.tensor((w2c_1[:3, :3] @ w2c_2[:3, :3].T).T), convention="XYZ").numpy() * 180 / np.pi

        angle_errors.append(np.linalg.norm(angles))
        diff = w2c_1[:3, 3] - w2c_2[:3, 3]
        position_errors.append(np.linalg.norm(diff))

        if mean_position_error is None:
            mean_position_error = diff
        else:
            mean_position_error += diff

    angle_errors = np.array(angle_errors)[2:]
    position_errors = np.array(position_errors)[2:]

    return angle_errors, position_errors


base_path = ""
pose_folder1 = "/CT/EventSLAM/static00/data/capture_main/shot_009/train/pose"
pose_folder2 = "/CT/EventSLAM/work/gaussian-splatting/trainings/post_siggraph/capture_main/shot_009_pose/optimized_poses"

angle_errors1, position_errors1 = get_errors(pose_folder1, pose_folder2)

pose_folder1 = "/CT/EventSLAM/static00/data/synthetic/Company_speedx100/train/pose"
pose_folder2 = "/CT/EventSLAM/static00/data/synthetic/Company_speedx100/train/pose_deg=0.3_cm=2"
angle_errors2, position_errors2 = get_errors(pose_folder1, pose_folder2)

min_len = min(angle_errors1.shape[0], angle_errors2.shape[0]) // 3

angle_errors1 = angle_errors1[min_len:2*min_len]
angle_errors2 = angle_errors2[min_len:2*min_len]

position_errors1 = position_errors1[min_len:2*min_len]
position_errors2 = position_errors2[min_len:2*min_len]

plt.rcParams['figure.figsize'] = [9, 3]

plt.plot(np.arange(0, angle_errors1.shape[0]/50, 1/50), angle_errors1[:], color="red", label="E-3DGS-Real")
plt.plot(np.arange(0, angle_errors2.shape[0]/50, 1/50), angle_errors2[:], color="blue", label="E-3DGS-Synthetic-Hard")
# plt.plot(range(angle_errors.shape[0]), angle_errors[:, 1], label="y")
# plt.plot(range(angle_errors.shape[0]), angle_errors[:, 2], label="z")
plt.legend()
plt.ylabel("Rotation error (degree)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig(f"{os.path.dirname(pose_folder2)}/angles.png", dpi=500)

plt.clf()

plt.plot(np.arange(0, angle_errors1.shape[0]/50, 1/50), position_errors1[:] * 100, color="red", label="E-3DGS-Real")
plt.plot(np.arange(0, angle_errors2.shape[0]/50, 1/50), position_errors2[:] * 100, color="blue", label="E-3DGS-Synthetic-Hard")
# plt.plot(range(angle_errors.shape[0]), position_errors[:, 1], label="y")
# plt.plot(range(angle_errors.shape[0]), position_errors[:, 2], label="z")
plt.legend()
plt.ylabel("Translation error (cm)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig(f"{os.path.dirname(pose_folder2)}/pos.png", dpi=500)

with open(f"{os.path.dirname(pose_folder2)}/errors.txt", "w") as f:
    ang_avg = np.average(angle_errors, axis=0)
    ang_std = np.std(angle_errors, axis=0)
    pos_avg = np.average(position_errors, axis=0)
    pos_std = np.std(position_errors, axis=0)

    f.write("ANGLE\n")
    f.write("RMSE:\n")
    f.write(str(np.sqrt(((angle_errors) ** 2).mean(axis=0))) + "\n")
    f.write("STD:\n")
    f.write(str(ang_std) + "\n")
    f.write("AVG:\n")
    f.write(str(ang_avg) + "\n")

    f.write("\n\nPOSITION\n")
    f.write("RMSE:\n")
    f.write(str(np.sqrt(((position_errors) ** 2).mean(axis=0))) + "\n")
    f.write("STD:\n")
    f.write(str(pos_std) + "\n")
    f.write("AVG:\n")
    f.write(str(pos_avg) + "\n")

print(np.mean(angle_errors))
print(np.mean(position_errors))
print(mean_position_error)
