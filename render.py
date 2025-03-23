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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import numpy as np

eps = None

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view_tuple in enumerate(tqdm(views, desc="Rendering progress")):
        view = view_tuple[0]
        # view.image_width *= 2
        # view.image_height *= 2
        rendering = render(view, gaussians, pipeline, background)["render"]

        # For correction the 0.2 difference
        l_img = torch.log(torch.clamp(rendering, 0.2)**2.2+eps) - torch.log(torch.tensor(0.2**2.2+eps, device=rendering.device)) + torch.log(torch.tensor(eps, device=rendering.device))

        # For correcting contrast threshold in generic setting
        # l_img = torch.log(rendering**2.2+eps) * 2
        # l_img += - l_img.mean() + l_img.mean() / 2

        if "vrudnev" in model_path:
            # For correcting contrast threshold in EventNeRF. 
            l_img = torch.log(rendering**2.2+eps) * 2
            l_img += l_img[0,0,0] / 2 - l_img[0,0,0]
        
        # l_img = torch.log(rendering**2.2+eps) * 0.7 * 1.7
        # l_img -= 0.7

        # Reverting back, in all cases
        rendering = (torch.exp(l_img) - eps) ** (1 / 2.2)

        if view.original_image is not None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
        
        ndarr = rendering.mul(255).permute(1, 2, 0).to("cpu", torch.float32).numpy()
        # im = Image.fromarray(ndarr)
        # im.save(fp, format=format)
        np.save(os.path.join(render_path, view.image_name + ".npy"), ndarr)

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    global eps
    eps = dataset.tonemap_eps

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        dataset.eval = True
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, test_optim=True)

        bg_color = [dataset.bg_color] * 3
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras()[::max(1, len(scene.getTrainCameras())//100)], gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras()[::max(1, len(scene.getTestCameras())//100)], gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)