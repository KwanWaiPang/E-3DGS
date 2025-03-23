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
import torch
import torchvision
import random
import numpy as np
from random import randint
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def training(dataset, opt, pipe):
    eps = dataset.tonemap_eps

    opt.pose_lr = 0.001
    dataset.pose_learnable = True
    
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.eval = True
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False, test_optim=True)
    print(f"\nNumber of Test Views: {len(scene.getTestCameras())}\n")

    bg_color = [dataset.bg_color] * 3
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0

    render_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "renders_optim")
    os.makedirs(render_path, exist_ok=True)

    for cam in tqdm(scene.getTestCameras()):
        optimizer = torch.optim.Adam(gaussians.camera_manager.parameters(), lr=max(1e-8, opt.pose_lr))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        cam = cam[0]

        # progress_bar = tqdm(range(5000))
        min_loss = 1000
        min_loss_i = 0
        min_loss_img = None
        for i in range(5000):
            optimizer.zero_grad(set_to_none = True)

            render_pkg = render(cam, gaussians, pipe, background)
            render_img = render_pkg['render']
            
            l_render_img = torch.log(render_img**2.2+eps) - torch.log(torch.tensor(0.2**2.2+eps, device=render_img.device)) + torch.log(torch.tensor(eps, device=render_img.device))
            l_org_img = torch.log(cam.original_image**2.2+eps)

            l_org_mean = l_org_img.view((3, -1)).mean(dim=-1)
            l_render_mean = l_render_img.view((3, -1)).mean(dim=-1)

            l_render_img = l_render_img + l_org_mean.reshape((3,1,1)) - l_render_mean.reshape((3,1,1))

            error = l_render_img - l_org_img
            error = torch.abs(error)

            loss = torch.mean(error)

            loss.backward()
            optimizer.step()
            scheduler.step(loss)            

            # if i == 0:
            #     l_img = torch.log(torch.clamp(render_img, 0.2)**2.2+eps) - torch.log(torch.tensor(0.2**2.2+eps, device=render_img.device)) + torch.log(torch.tensor(eps, device=render_img.device))
            #     rendering = (torch.exp(l_img) - eps) ** (1 / 2.2)
            #     torchvision.utils.save_image(rendering, os.path.join(render_path, cam.image_name + "_pre_optim.png"))

            if loss.item() < min_loss:
                min_loss = loss.item()
                min_loss_i = i
                min_loss_img = render_img.detach()

            if i - min_loss_i > 100:
                break
        
        # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            # progress_bar.update(1)
        print(i)
        
        rendering = min_loss_img
        l_img = torch.log(torch.clamp(rendering, 0.2)**2.2+eps) - torch.log(torch.tensor(0.2**2.2+eps, device=rendering.device)) + torch.log(torch.tensor(eps, device=rendering.device))
        rendering = (torch.exp(l_img) - eps) ** (1 / 2.2)

        ndarr = rendering.mul(255).permute(1, 2, 0).to("cpu", torch.float32).numpy()
        np.save(os.path.join(render_path, cam.image_name + ".npy"), ndarr)

        torchvision.utils.save_image(rendering, os.path.join(render_path, cam.image_name + ".png"))
        # progress_bar.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    training(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
