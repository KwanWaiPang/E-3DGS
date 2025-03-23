#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple, List
from scene.cameras import ZFAR, CameraManager
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix
import numpy as np
import glob
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Rotation
import numba
import torch
from torch import nn
from utils.general_utils import PILtoTorch


class EventManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_frames = dict()
        self.distortion_params = None
        self.K = None

    def add_camera(self, name, event_frame):
        self.event_frames[name] = event_frame

    def set_distortion_params(self, distortion_params, K, W, H):
        self.distortion_params = distortion_params
        self.K = K

        # Pre-compute the maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion_params, np.eye(3), K, (W, H), cv2.CV_16SC2)


    def get_frame(self, name_T1, name_T0):
        ev_frame = self.event_frames[name_T1] - self.event_frames[name_T0]

        if (np.abs(self.distortion_params) > 1e-3).any():
            # ev_frame = cv2.fisheye.undistortImage(ev_frame, self.K, self.distortion_params, Knew=self.K)
            ev_frame = cv2.remap(ev_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        ev_frame = Image.fromarray(np.clip(ev_frame, -40, 40))
        ev_frame = PILtoTorch(ev_frame, None)
        return ev_frame

    def contains(self, name):
        # return name in self.se3_store
        return name in self.event_frames.keys()
    


class CameraInfo(NamedTuple):
    uid: int
    Rs: List[np.array]
    Ts: List[np.array]
    FovY: np.array
    FovX: np.array
    image: Image.Image
    image_paths: List[str]
    image_names: List[str]
    event_mgr: EventManager
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    train_cam_mgr: CameraManager
    test_cam_mgr: CameraManager
    nerf_normalization: dict
    ply_path: str
    color_mask: np.array

def rotate_around_axis(matrix, axis=[0, np.cos(np.pi/6), np.sin(np.pi/6)], angle_step=np.pi/45):
    R3 = Rotation.from_rotvec(np.array(axis) * angle_step).as_matrix().astype(matrix.dtype)
    R4 = np.eye(4, dtype=R3.dtype)
    R4[:3, :3] = R3

    w2c = np.linalg.inv(matrix)
    c2w = np.linalg.inv(w2c.dot(R4.T))

    return c2w

@numba.jit()
def accumulate_events(xs, ys, ts, ps, out, resolution_level, polarity_offset):
    for i in range(len(xs)):
        x, y, t, p = xs[i], ys[i], ts[i], ps[i]
        if p > 0:    
            p *= polarity_offset
        out[y // resolution_level, x // resolution_level] += p

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.Rs[0], cam.Ts[0])
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def find_files(dir, exts, exclude=None):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        if exclude is not None:
            files_grabbed = list(filter(lambda x: all([token not in os.path.basename(x) for token in exclude]), files_grabbed))
        return files_grabbed
    else:
        return []

def readCamerasFromTransforms(path, cam_mgr, event_mgr, extension=".png", eval=True, max_events=1_000_000, pose_folder="pose", adaptive_window=True):
    cam_infos = []

    # WARNING: Detection of tum-vie dataset done on the basis of path
    is_tum_vie = 'tum-vie' in path

    Rs = []
    Ts = []
    file_paths = []
    if is_tum_vie:
        if eval:
            W, H = 1024, 1024
        else:
            W, H = 1280, 720
    else:
        W, H = 346, 260

    # camera parameters files
    intrinsics_files = find_files(os.path.join(path, "intrinsics"), exts=['*.txt'], exclude=["start"])
    pose_files = find_files(os.path.join(path, pose_folder), exts=['*.txt'], exclude=["usable", "start"])
    print('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    print('raw pose_files: {}'.format(len(pose_files)))

    pose_files.sort()

    xs, ys, ts, ps = [np.array([])] * 4
    event_data = []

    if not eval:
        event_file = find_files('{}/events'.format(path), exts=['*.npz'])
        print(event_file)
        if len(event_file) == 1:
            event_data = np.load(event_file[0])
            xs, ys, ts, ps = event_data['x'], event_data['y'], event_data['t'], event_data['p']
        elif len(event_file) > 1:
            # WARNING: In case of multiple event files, the one with "undistort" is preferred.
            #  Reason: The undistorted and distorted events were placed in same folder.
            selected = [f for f in event_file if "undistort" in os.path.basename(f)]
            if len(selected) > 0:
                f = selected[0]
            else:
                f = event_file[0]
            print("\n\n\tREADING: ", f)
            event_data = np.load(f)
            xs, ys, ts, ps = event_data['x'], event_data['y'], event_data['t'], event_data['p']

    ps = ps.astype(np.int8)
    ps[ps != 1] = -1


    # if os.path.exists(os.path.join(path, "events", "start.txt")):
    pose_timestamps = list(map(lambda x: int(os.path.basename(x).split('_')[-1].rsplit('.', 1)[0]), pose_files))

    # WARNING: Detection of EventNeRF (real life) datasets on the basis of paths.
    if "vrudnev" in path and "/nerf/" not in path and len(ts) > 0:
        assert ts[0] >=0
        assert ts[-1] <=1
        ts = ts * 1000

    if os.path.exists(os.path.join(path, "intrinsics", "color_mask_undistorted.png")):
        color_mask = cv2.imread(os.path.join(path, "intrinsics", "color_mask_undistorted.png"))
        # color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        color_mask = (color_mask != 0).astype(np.uint8)
    else:
        color_mask = None

    if color_mask is None:
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        color_mask[0::2, 0::2, 0] = 1  # r
        color_mask[0::2, 1::2, 1] = 1  # g
        color_mask[1::2, 0::2, 1] = 1  # g
        color_mask[1::2, 1::2, 2] = 1  # b
        
    if is_tum_vie:
        color_mask[:] = 1

    cam_cnt = len(pose_files)
    
    for idx in range(cam_cnt):
        file_name = Path(pose_files[idx]).stem  # Removes parents and extension.
        file_name = file_name + "_" + os.path.basename(path) + "_" + os.path.basename(os.path.dirname(path))
        rgb_path = os.path.join(path, "rgb", file_name + extension)

        # COLMAP format (Y down, X right, Z forward)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.loadtxt(pose_files[idx])
        # if 'test' in os.path.basename(path):
        #     c2w = rotate_around_axis(c2w, axis=[1,0,0], angle_step=10* np.pi /180)
        #     c2w = rotate_around_axis(c2w, axis=[0,1,0], angle_step=-50* np.pi /180)
        # change from OpenGL/Blender camera axes (Y up, Z back, X right) to COLMAP (Y down, X right, Z forward)
        # c2w[:3, 1:3] *= -1


        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w).astype(np.float32)
        R = w2c[:3,:3]
        T = w2c[:3, 3]

        Rs.append(R)
        Ts.append(T)
        file_paths.append(rgb_path)
        cam_mgr.add_camera(file_name, w2c.T)

    MIN_EVENTS=0 if eval else 100
    MAX_EVENTS=0 if eval else max_events
    # MAX_EVENTS = min(MAX_EVENTS, int(0.04 * len(ts)))
    event_counts = []

    # indices =  np.arange(0, cam_cnt)
    # end_times = (indices)/(cam_cnt-1)
    ends = np.searchsorted(ts, np.array(pose_timestamps) + 1e-8)

    for i in tqdm(range(cam_cnt)):
        if i == 0:
            event_frame = np.zeros((H, W), dtype=np.float32)
            i1, i2 = 0, ends[0] 
            event_count = i2
        else:
            i1, i2 = ends[i-1], ends[i]
            event_count = i2 - i1 + event_counts[-1] # Technically equal to just i2.

        assert(event_count == i2)
        if i1 < i2:
            accumulate_events(xs[i1:i2], ys[i1:i2], ts[i1:i2], ps[i1:i2], event_frame, 1, (0.3645 / 0.25) if is_tum_vie else 1.)
            
        if event_mgr is not None:
            event_mgr.add_camera(Path(file_paths[i]).stem, event_frame.copy())
        event_counts.append(event_count)

    del xs, ys, ts, ps, event_data

    if len(intrinsics_files) == 2:
        K = np.loadtxt(f'{path}/intrinsics/intrinsics.txt')

        # WARNING: For all datasets events are already undistorted. 
        # For tum-vie it is done during training. Hence read distortion params only for if it is tum-vie dataset
        if is_tum_vie:
            distortion_params = np.loadtxt(f'{path}/intrinsics/radial_distortion.txt')
        else:
            distortion_params = np.array([0,0,0,0]).astype(np.float32)
        # assert (np.abs(dist) < 0.1).all()

        fovx = 2 * np.math.atan(W/(2 * K[0, 0]))
    else:
        K = np.loadtxt(intrinsics_files[0])
        
        fovx = 2 * np.math.atan(W/(2 * K[0, 0]))
        distortion_params = np.array([0,0,0,0]).astype(np.float32)
        # fovx = 81.202583 * math.pi / 180

    if event_mgr is not None:
        event_mgr.set_distortion_params(distortion_params, K[:3, :3], W, H)

    # histogram = None

    MAX_EVENTS_LIST = [MAX_EVENTS, MAX_EVENTS // 30]

    event_windows = []

    if adaptive_window is False:
        ratio = MAX_EVENTS / event_counts[-1]
        MAX_WINLEN = pose_timestamps[-1] * ratio
        if eval:
            MAX_WINLEN = 0
        MAX_WINLEN_LIST = [MAX_WINLEN, MAX_WINLEN // 30]

    for i in tqdm(range(cam_cnt)):
        if adaptive_window:
            if event_counts[i].sum() - event_counts[0].sum() < MAX_EVENTS:
                continue

            target_event_counts = [
                np.random.random() * 9 * max_e // 10 + max_e / 10
                for max_e in MAX_EVENTS_LIST
                ]

            starts = [None] * len(MAX_EVENTS_LIST)
            for x in range(i, -1, -1):
                for s in range(len(target_event_counts)):
                    if event_counts[i] - event_counts[x] >= target_event_counts[s] and starts[s] is None:
                        starts[s] = x
                if all([s is not None for s in starts]):
                    break
            else:
                continue
        else:
            if pose_timestamps[i] - pose_timestamps[0] < MAX_WINLEN:
                continue

            target_event_windows = [
                np.random.random() * max_e + 1
                for max_e in MAX_WINLEN_LIST
                ]

            starts = [None] * len(MAX_WINLEN_LIST)
            for x in range(i, -1, -1):
                for s in range(len(target_event_windows)):
                    if pose_timestamps[i] - pose_timestamps[x] >= target_event_windows[s] and starts[s] is None:
                        starts[s] = x
                if all([s is not None for s in starts]):
                    break
            else:
                continue

        starts = [i] + starts

        image_paths = [file_paths[s] for s in starts]
        image_names = [Path(image_path).stem for image_path in image_paths]

        image = None
        dirname = os.path.dirname(file_paths[i])
        filename = os.path.basename(file_paths[i])
        f_idx = int(filename.split('_')[0])
        img_name = os.path.join(dirname, f"{f_idx:06d}.png")
        
        if eval and os.path.exists(img_name):
            image = Image.open(img_name).convert("RGB")

        curr_Rs = [Rs[s] for s in starts]
        curr_Ts = [Ts[s] for s in starts]

        event_windows.append([i -s for s in starts])

        fovy = focal2fov(fov2focal(fovx, W), H)
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(
            uid=idx,
            Rs=curr_Rs,
            Ts=curr_Ts,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_paths=image_paths,
            image_names=image_names,
            event_mgr=event_mgr,
            width=W,
            height=H,
            ))
            
    return cam_infos, color_mask

def readNerfSyntheticInfo(path, eval, extension=".png", max_events=1_000_000, pose_folder="pose", adaptive_window=True):
    train_cam_mgr = CameraManager()
    test_cam_mgr = CameraManager()
    event_mgr = EventManager()
    # path = os.path.dirname(path)
    print("Reading Training Transforms")
    train_cam_infos1, color_mask = readCamerasFromTransforms(os.path.join(path, "train"), train_cam_mgr, event_mgr, extension, eval, max_events, pose_folder=pose_folder, adaptive_window=adaptive_window)
    # train_cam_infos2, color_mask = readCamerasFromTransforms(os.path.join(path, "../scene2/train"), train_cam_mgr, extension, eval)
    print("Reading Test Transforms")
    test_cam_infos1, _ = readCamerasFromTransforms(os.path.join(path, "validation"), test_cam_mgr, None, extension, True, 0)
    # test_cam_infos2, color_mask = readCamerasFromTransforms(os.path.join(path, "../scene2/validation"), test_cam_mgr, extension, eval)
    
    train_cam_infos = train_cam_infos1# + train_cam_infos2
    test_cam_infos = test_cam_infos1# + test_cam_infos2

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if True or not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 50_000
        print(f"Generating random point cloud ({num_pts})...")

        projection_matrix = getProjectionMatrix(znear=0.1, zfar=ZFAR, fovX=train_cam_infos[0].FovX, fovY=train_cam_infos[0].FovY).transpose(0,1).cuda()

        image_names = list(train_cam_mgr.R6T_store.keys())
        r6ts = [train_cam_mgr.R6T_store[name].detach() for name in image_names]
        r6ts = torch.stack(r6ts, dim=0)

        points = train_cam_mgr.sample_points_in_frustum(num_pts // len(image_names), train_cam_mgr.RT_to_w2c(r6ts), projection_matrix).reshape(-1, 4)[:, :3] # [c, n, 4] -> [c*n, 3]
        
        num_pts = points.shape[0]

        # # We create random points inside the bounds of the synthetic Blender scenes
        xyz = points.cpu().numpy()
        # xyz = (np.random.random((num_pts, 3)) * 2. - 1.) * 10 #* nerf_normalization2["radius"]# + nerf_normalization2["translate"]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           train_cam_mgr=train_cam_mgr,
                           test_cam_mgr=test_cam_mgr,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           color_mask=color_mask)
    return scene_info
