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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import pytorch3d.transforms as T3D


ZFAR = 1000.0
ZNEAR = 0.01


class CameraManager(nn.Module):
    def __init__(self, learnable=False, precondition=False):
        super().__init__()
        self.learnable = learnable
        self.precondition = precondition
        self.se3_store = nn.ParameterDict()
        self.w2c_store = nn.ParameterDict()
        self.R6T_store = nn.ParameterDict()
        self.R6T_error_store = nn.ParameterDict()
        self.p_inv_store = nn.ParameterDict()
        self.calibration_error = nn.Parameter(torch.tensor(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        ).cuda(), requires_grad=False)

    def add_camera(self, name, w2c):
        # se3 = T3D.se3_log_map(torch.tensor(w2c).unsqueeze(0)).squeeze(0)
        # self.se3_store[name] = nn.Parameter(se3, requires_grad=True).cuda()
        # self.w2c_store[name] = nn.Parameter(torch.tensor(w2c), requires_grad=True).cuda()
        assert abs(np.linalg.norm(w2c[0, :3]) - 1) < 1e-3
        assert abs(np.linalg.norm(w2c[1, :3]) - 1) < 1e-3
        assert abs(np.linalg.norm(w2c[2, :3]) - 1) < 1e-3
        assert (np.abs(np.cross(w2c[0, :3], w2c[1, :3]) - w2c[2, :3]) < 1e-3).all()
        assert (np.abs(w2c[:3, 3]) < 1e-3).all()

        self.R6T_store[name] = nn.Parameter(torch.tensor(np.concatenate((w2c[:2, :3].flatten(), w2c[3, :3]))).cuda(), requires_grad=False)
        self.R6T_error_store[name] = nn.Parameter(torch.tensor(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)).cuda(), requires_grad=True)

    def contains(self, name):
        # return name in self.se3_store
        return name in self.w2c_store
    
    def RT_to_w2c(self, R6T):
        R = R6T[..., :6].reshape((*R6T.shape[:-1], 2, 3))
        T = R6T[..., 6:]

        R1 = R[..., 0, :] / torch.norm(R[..., 0, :], dim=-1, keepdim=True)

        R2 = R[..., 1, :]
        R2 = R2 - torch.bmm(R1.view(-1, 1, 3), R2.view(-1, 3, 1)).view(*R6T.shape[:-1], 1) * R1
        R2 = R2 / torch.norm(R2, dim=-1, keepdim=True)

        R3 = torch.cross(R1, R2, dim=-1)
        # R2 = torch.cross(R3, R1, dim=-1)

        w2c = torch.zeros((*R6T.shape[:-1], 4, 4), dtype=R.dtype).to(R.device)
        w2c[..., 0, :3] = R1
        w2c[..., 1, :3] = R2
        w2c[..., 2, :3] = R3
        w2c[..., 3, :3] = T
        w2c[..., 3, 3] = 1
        return w2c

    def get_pose(self, name, bypass=False):
        # se3 = self.se3_store[name]
        # if not bypass and self.precondition:
        #     p_inv = self.p_inv_store[name]
        #     se3 = p_inv @ se3

        # if not self.learnable:
        #     se3 = se3.detach()
        # w2c = T3D.se3_exp_map(se3.unsqueeze(0)).squeeze(0)
        # return w2c
    
        # res = self.w2c_store[name]
        # if not bypass and self.precondition:
        #     p_inv = self.p_inv_store[name]
        #     res = (p_inv @ res.flatten()).reshape((4,4))
        # return res

        R6T = self.R6T_store[name]
        R6T_error = self.R6T_error_store[name]
        if not bypass and self.precondition:
            p_inv = self.p_inv_store[name]
            R6T = (p_inv @ R6T)
        if not self.learnable:
            R6T = R6T.detach()
            R6T_error = R6T_error.detach()

        return self.RT_to_w2c(R6T) @ self.RT_to_w2c(R6T_error) # self.RT_to_w2c(self.calibration_error)
    
    def compute_preconditioner(self, projection_matrix):
        image_names = list(self.R6T_store.keys())
        r6ts = [self.R6T_store[name].detach() for name in image_names]

        r6ts = torch.stack(r6ts, dim=0)

        points = self.sample_points_in_frustum(200, self.RT_to_w2c(r6ts), projection_matrix) # [c, 200, 4]

        def project(p, r6ts):
            w2cs = self.RT_to_w2c(r6ts)
            full_proj = torch.bmm(w2cs, projection_matrix.unsqueeze(0).repeat(w2cs.shape[0], 1, 1))
            projected_hom = torch.bmm(p, full_proj)
            projected = projected_hom / projected_hom[:, :, 3:4]
            return projected[:, :, :2]

        BATCH_SIZE = 4

        J = torch.zeros((*points.shape[:2], 2, r6ts.shape[-1]), dtype=torch.float32).cuda()

        print(points.shape[0])
        for i in range(0, points.shape[0], BATCH_SIZE):
            print(i)
            batched_j = torch.autograd.functional.jacobian(
                project, 
                (points[i: i + BATCH_SIZE], r6ts[i: i + BATCH_SIZE]), 
                vectorize=True, 
                strategy='forward-mode'
            )[1]

            assert batched_j.shape[0] == BATCH_SIZE or i + BATCH_SIZE > points.shape[0]
            assert batched_j.shape[3] == BATCH_SIZE or i + BATCH_SIZE > points.shape[0]

            J[i: i + BATCH_SIZE] = torch.diagonal(batched_j, dim1=0, dim2=3).permute(3, 0, 1, 2)

        J = J.view(J.shape[0], -1, J.shape[-1])
        assert J.shape[-1] == r6ts.shape[-1]
        
        # Step 1: Compute the matrix product
        M = torch.bmm(J.transpose(-2, -1), J)

        M  = (
            M +  
            1e-1 * torch.diag_embed(torch.diagonal(M, dim1=-1, dim2=-2)) + 
            1e-8 * torch.eye(J.shape[-1], dtype=M.dtype, device=M.device)
            )
        
        # Step 2: Compute the eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        
        # Ensure all eigenvalues are non-negative
        eigenvalues = torch.clamp(eigenvalues, min=0)
        
        # Step 3: Compute the matrix square root of the inverse
        Lambda_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(eigenvalues))
        P_inv = torch.bmm(torch.bmm(eigenvectors, Lambda_inv_sqrt), eigenvectors.transpose(-1, -2))

        for i, name in enumerate(image_names):
            self.p_inv_store[name] = nn.Parameter(P_inv[i]).cuda()
            self.R6T_store[name] = nn.Parameter(torch.inverse(P_inv[i]) @ r6ts[i], requires_grad=True).cuda()   

            if not torch.allclose(r6ts[i], self.p_inv_store[name] @ self.R6T_store[name]):
                print(r6ts[i], self.p_inv_store[name] @ self.R6T_store[name])
    
    def sample_points_in_frustum(self, num_points, w2cs, projection_matrix):
        # Inverse the projection matrix
        inv_proj_matrix = torch.linalg.inv(projection_matrix)

        # Sample depth in camera space and transform to clip space to ensure uniform distribution.
        points_z_cam_space = torch.rand((w2cs.shape[0], num_points, 1), device="cuda") * (ZFAR /25 - (ZNEAR + 1e-6)) + ZNEAR + 1e-6
        points_z_cam_space = torch.cat(
            (
                torch.zeros_like(points_z_cam_space),
                torch.zeros_like(points_z_cam_space),
                points_z_cam_space,
                torch.ones_like(points_z_cam_space),
            ),
            axis=-1
        )
        
        points_clip_space = torch.matmul(points_z_cam_space, projection_matrix.unsqueeze(0))
        points_clip_space = points_clip_space / points_clip_space[..., 3:4]
        # Modify xy from 0 to [-1, 1]
        points_clip_space[:, :, :2] = torch.rand((w2cs.shape[0], num_points, 2), device="cuda") * 2 - 1

        # Transform points to the camera space using the inverse projection matrix
        random_points_cam_space = torch.matmul(points_clip_space, inv_proj_matrix.unsqueeze(0))
        random_points_cam_space = random_points_cam_space / random_points_cam_space[:, :, 3:4] # Convert from homogeneous to Cartesian coordinates

        # Filter points that are outside the near and far planes
        assert (random_points_cam_space[:, :, 2] < ZFAR).all()
        assert (random_points_cam_space[:, :, 2] > ZNEAR).all()

        # random_points_cam_space = random_points_cam_space.unsqueeze(0).repeat(w2cs.shape[0], 1, 1)

        random_points_world_space = torch.bmm(random_points_cam_space, torch.linalg.inv(w2cs))
        random_points_world_space = random_points_world_space / random_points_world_space[:, :, 3:4] # Convert from homogeneous to Cartesian coordinates

        return random_points_world_space
    
# class CameraManager(nn.Module):
#     def __init__(self, learnable=False):
#         super().__init__()
#         self.learnable = learnable
#         self.w2c_store = nn.ParameterDict()

#     def add_camera(self, name, w2c):
#         self.w2c_store[name] = nn.Parameter(torch.tensor(w2c), requires_grad=True).cuda()

#     def contains(self, name):
#         return name in self.w2c_store

#     def get_pose(self, name):
#         res = self.w2c_store[name]
#         if not self.learnable:
#             res = res.detach()
#         return res

class Camera(nn.Module):
    def __init__(self, colmap_id, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ref_image_name=None, # Name of the image with which the event frame is to be computed
                 cam_mgr=None,
                 event_mgr=None,
                 W=-1,
                 H=-1
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        # self.R = R
        # self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.ref_image_name = ref_image_name
        self.cam_mgr : CameraManager = cam_mgr
        self.event_mgr = event_mgr

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.image_width = W
        self.image_height = H

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            self.original_image = None
        
        self.zfar = ZFAR
        self.znear = ZNEAR

        self.trans = trans
        self.scale = scale

        self._event_frame = None

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        # if image_name == "r_01005":
        #     print(image_name, np.round(self.world_view_transform.detach().cpu().numpy(),2))
        #     print(image_name, np.round(self.projection_matrix.detach().cpu().numpy(),2))
        #     print(image_name, np.round(self.full_proj_transform.detach().cpu().numpy(),2))
        #     exit(0)
        # import pdb; pdb.set_trace()
        # self.cam_mgr.compute_preconditioner(self.image_name, self)
        # self.org_transform = self.world_view_transform.detach().cpu()

    @property
    def world_view_transform(self):
        return self.cam_mgr.get_pose(self.image_name)
    
    @property
    def full_proj_transform(self):
        return self.world_view_transform @ self.projection_matrix
    
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]
    
    @property
    def event_frame(self):
        if self.event_mgr is None:
            return None
        return self.event_mgr.get_frame(name_T1=self.ref_image_name, name_T0=self.image_name)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

