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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from collections import defaultdict

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * 2.5) ** 2
    C2 = (0.03 * 2.5) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def TV_loss(gaussians, visibility_filter, cells_per_unit=100):
    _xyz = gaussians._xyz[visibility_filter]
    _features_dc = gaussians._features_dc[visibility_filter]
    _features_rest = gaussians._features_rest[visibility_filter]
    _scaling = gaussians._scaling[visibility_filter]
    _rotation = gaussians._rotation[visibility_filter]
    _opacity = gaussians._opacity[visibility_filter]

    hashes = torch.floor(cells_per_unit * _xyz).int()
    
    min_hash = torch.min(hashes, dim=0).values
    max_hash = torch.max(hashes, dim=0).values

    # Create a defaultdict to store indices for each cell
    hash_table = defaultdict(list)

    # Assuming _xyz contains the coordinates of your Gaussians
    for i in range(len(_xyz)):
        x, y, z = hashes[i].tolist()  # Get the cell coordinates

        # Append the index to the corresponding cell
        hash_table[(x, y, z)].append(i)

    # Initialize lists to store the losses for each property
    opacity_losses = []
    rotation_losses = []
    scaling_losses = []
    features_dc_losses = []
    features_rest_losses = []

    # Now you can compute the loss for each cell
    # import pdb; pdb.set_trace()
    # for x in range(int(min_hash[0].item()), int(max_hash[0].item()) + 1):
    #     for y in range(int(min_hash[1].item()), int(max_hash[1].item()) + 1):
    #         for z in range(int(min_hash[2].item()), int(max_hash[2].item()) + 1):
    for i, (x, y, z) in enumerate(list(hash_table.keys())):
                current_cell_indices = hash_table[(x, y, z)]
                if len(current_cell_indices) == 0:
                    continue
                # next_cell_indices = hash_table.get((x, y, z + 1), [])

                # Extract the relevant subsets from the original tensors
                current_cell_opacity = _opacity[current_cell_indices]
                # next_cell_opacity = _opacity[next_cell_indices]

                current_cell_rotation = _rotation[current_cell_indices]
                # next_cell_rotation = _rotation[next_cell_indices]

                current_cell_scaling = _scaling[current_cell_indices]
                # next_cell_scaling = _scaling[next_cell_indices]

                current_cell_features_dc = _features_dc[current_cell_indices]
                # current_cell_features_dc = _features_dc[current_cell_indices]
                
                current_cell_features_rest = _features_rest[current_cell_indices]
                # current_cell_features_rest = _features_rest[visibility_filter]

                # Compute the loss for this pair of cells as the difference between averages
                loss_opacity       = torch.abs(torch.mean(current_cell_opacity, dim=0, keepdim=True) - current_cell_opacity)
                loss_rotation      = torch.abs(torch.mean(current_cell_rotation, dim=0, keepdim=True) - current_cell_rotation)
                loss_scaling       = torch.abs(torch.mean(current_cell_scaling, dim=0, keepdim=True) - current_cell_scaling)
                loss_features_dc   = torch.abs(torch.mean(current_cell_features_dc, dim=0, keepdim=True) - current_cell_features_dc)
                loss_features_rest = torch.abs(torch.mean(current_cell_features_rest, dim=0, keepdim=True) - current_cell_features_rest)

                # Compute the loss for each property as the difference between averages
                opacity_losses.append(loss_opacity)
                rotation_losses.append(loss_rotation)
                scaling_losses.append(loss_scaling)
                features_dc_losses.append(loss_features_dc)
                features_rest_losses.append(loss_features_rest)

    return \
        0.05 + torch.mean(torch.concat(opacity_losses)) + \
        0.1 + torch.mean(torch.concat(rotation_losses)) + \
        0.05 + torch.mean(torch.concat(scaling_losses)) + \
        0.05 + torch.mean(torch.concat(features_dc_losses)) + \
        0.05 + torch.mean(torch.concat(features_rest_losses))