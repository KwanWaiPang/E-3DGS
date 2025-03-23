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
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

color_mask = None
THR = None
eps = None

def get_loss(pkg1, pkg2, cam):
    global color_mask
    diff = (torch.log(pkg1['render']**2.2+eps)-torch.log(pkg2['render']**2.2+eps)) * color_mask
    gt = cam.event_frame.cuda()*THR * color_mask
    error = diff - gt
    # error = render_pkg1['render'] - viewpoint_cams[0].original_image
    error = torch.abs(error)
    error = torch.where(error > THR, error, error * 0.2)

    min_b, max_b = 0.2, 1.2

    reg1 = torch.where(pkg1['render'] > max_b, pkg1['render'] - max_b, 0)
    reg2 = torch.where(pkg1['render'] < min_b, min_b - pkg1['render'], 0)
    reg3 = torch.where(pkg2['render'] > max_b, pkg2['render'] - max_b, 0)
    reg4 = torch.where(pkg2['render'] < min_b, min_b - pkg2['render'], 0)

    reg = reg1 + reg2 + reg3 + reg4

    # scaling_reg = torch.mean(torch.min(scene.gaussians.get_scaling, dim=-1).values)
    event_mask = torch.abs(gt) > 1e-2
    n_px_events = torch.sum(event_mask)
    n_px_no_events = torch.sum(color_mask) - n_px_events

    alpha = 0.3
    error[~event_mask] = alpha * (error[~event_mask] / n_px_no_events)
    error[event_mask] = (1 - alpha) * (error[event_mask] / n_px_events)

    return error / 3, torch.mean(reg * 0.1)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from):
    dataset.adaptive_event_window = bool(dataset.adaptive_event_window)
    
    first_iter = 0
    if opt.pose_lr > 0:
        dataset.pose_learnable = True
    else:
        dataset.pose_learnable = False

    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    print(f"\nNumber of Training Views: {len(scene.getTrainCameras())}")
    print(f"Number of Test Views: {len(scene.getTestCameras())}\n")

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [dataset.bg_color] * 3
    # bg_color = [159. /255] * 3
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    global color_mask
    color_mask = torch.from_numpy(scene.color_mask).to(dataset.data_device).permute(2, 0, 1)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    avg_pt_grad_norms = []
    avg_pose_grad_norms = []

    ema_pose_grad = None
    ema_pose_grad_squared = None

    grad_threshold_scheduler = get_expon_lr_func(lr_init=opt.densify_grad_threshold,
                                                lr_final=opt.densify_grad_threshold_final,
                                                max_steps=opt.opacity_reset_until)


    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cams = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        event_training = True

        # prev_idx = int((len(viewpoint_cams) - 2) * min(1, iteration / opt.event_schedule_end)) + 1
        prev_idx1 = 1
        prev_idx2 = -1
        render_pkg1 = render(viewpoint_cams[0], gaussians, pipe, background)
        if event_training:
            render_pkg2 = render(viewpoint_cams[prev_idx1], gaussians, pipe, background)
            render_pkg3 = render(viewpoint_cams[prev_idx2], gaussians, pipe, background)

        image1, viewspace_point_tensor1, visibility_filter1, radii1 = render_pkg1["render"], render_pkg1["viewspace_points"], render_pkg1["visibility_filter"], render_pkg1["radii"]
        if event_training:
            image2, viewspace_point_tensor2, visibility_filter2, radii2 = render_pkg2["render"], render_pkg2["viewspace_points"], render_pkg2["visibility_filter"], render_pkg2["radii"]
            image3, viewspace_point_tensor3, visibility_filter3, radii3 = render_pkg3["render"], render_pkg3["viewspace_points"], render_pkg3["visibility_filter"], render_pkg3["radii"]
        # Loss

        global THR, eps
        THR = dataset.event_threshold
        eps = dataset.tonemap_eps

        if event_training:
            isotropic_filter = (visibility_filter1 | visibility_filter2 | visibility_filter3).unsqueeze(-1)
            isotropic_loss = torch.abs(scene.gaussians.get_scaling - torch.mean(scene.gaussians.get_scaling, dim=-1, keepdim=True))
            isotropic_loss = torch.sum(isotropic_filter * isotropic_loss) / torch.sum(isotropic_filter)
            isotropic_loss = isotropic_loss * (10. if iteration < 20000 else 1.) * dataset.lambda_isotropic_reg

            if (iteration % opt.opacity_reset_interval < 300):
                isotropic_loss *= 0

            calibration_reg1 = gaussians.camera_manager.RT_to_w2c(gaussians.camera_manager.R6T_error_store[viewpoint_cams[0].image_name]) - torch.tensor(np.eye(4), device="cuda")
            calibration_reg1 = torch.norm(calibration_reg1)
            
            calibration_reg2 = gaussians.camera_manager.RT_to_w2c(gaussians.camera_manager.R6T_error_store[viewpoint_cams[prev_idx1].image_name]) - torch.tensor(np.eye(4), device="cuda")
            calibration_reg2 = torch.norm(calibration_reg2)
            
            calibration_reg3 = gaussians.camera_manager.RT_to_w2c(gaussians.camera_manager.R6T_error_store[viewpoint_cams[prev_idx2].image_name]) - torch.tensor(np.eye(4), device="cuda")
            calibration_reg3 = torch.norm(calibration_reg3)

            error1, reg1 = get_loss(render_pkg1, render_pkg2, viewpoint_cams[prev_idx1])
            error2, reg2 = get_loss(render_pkg1, render_pkg3, viewpoint_cams[prev_idx2])

            L_recon1 = (torch.sum(error1) + dataset.lambda_color_range * reg1) * 0.65
            L_recon2 = (torch.sum(error2) + dataset.lambda_color_range * reg2) * 0.65
            L_pose = dataset.lambda_pose_reg * (calibration_reg1 + calibration_reg2 + calibration_reg3)

            if dataset.n_event_losses == 1:
                L_recon1 = 2 * L_recon1
                L_recon2 = 0 * L_recon2
            elif dataset.n_event_losses != 2:
                raise ValueError(f"n_event_losses must be a value from {1, 2}. Got: {dataset.n_event_losses}")

            loss = L_recon1 + L_recon2 + isotropic_loss + L_pose
        else:
            error = render_pkg1['render'] - viewpoint_cams[0].original_image
            error = torch.abs(error)

            loss = torch.mean(error)

        loss = loss / 5

        # if iteration > 1000:
            # loss = loss + 1 * scaling_reg
        # loss = (1.0 - opt.lambda_dssim) * loss + opt.lambda_dssim * (1.0 - ssim(diff, gt))

        # visibility_filter = visibility_filter1 | visibility_filter2

        # if iteration % 100 == 0:
        #     tvl = TV_loss(gaussians, visibility_filter, cells_per_unit=5 if iteration < 10000 else 25)

        #     loss = loss + tvl * 10

        loss.backward()

        # image = render_pkg1['render']
        # gt_image = viewpoint_cams[0].original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss.backward()

        iter_end.record()
        
        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter1] = torch.max(gaussians.max_radii2D[visibility_filter1], radii1[visibility_filter1])
                gaussians.max_radii2D[visibility_filter2] = torch.max(gaussians.max_radii2D[visibility_filter2], radii2[visibility_filter2])
                gaussians.max_radii2D[visibility_filter3] = torch.max(gaussians.max_radii2D[visibility_filter3], radii3[visibility_filter3])
                
                # avg_pt_grad_norms.append((viewspace_point_tensor1.grad[visibility_filter1, :2].norm(dim=-1) > (opt.densify_grad_threshold /100)).sum().cpu().item())
                # avg_pt_grad_norms.append((viewspace_point_tensor2.grad[visibility_filter2, :2].norm(dim=-1) > (opt.densify_grad_threshold /100)).sum().cpu().item())

                # grad1 = viewpoint_cams[0].cam_mgr.R6T_store[viewpoint_cams[0].image_name].grad.abs().cpu()
                # grad2 = viewpoint_cams[1].cam_mgr.R6T_store[viewpoint_cams[1].image_name].grad.abs().cpu()

                # # avg_pose_grad_norms.append(grad1)
                # # avg_pose_grad_norms.append(grad2)

                # if ema_pose_grad is None:
                #     ema_pose_grad = 0.5 * grad1 + 0.5 * grad2
                #     ema_pose_grad_squared = 0.5 * (grad1 ** 2) + 0.5 * (grad2 ** 2)
                # else:
                #     alpha = 0.01
                #     ema_pose_grad = (1 - 2 * alpha) * ema_pose_grad + alpha * grad1 + alpha * grad2
                #     ema_pose_grad_squared = (1 - 2 * alpha) * ema_pose_grad_squared + alpha * (grad1 ** 2) + alpha * (grad2 ** 2)

                # sd = torch.sqrt(ema_pose_grad_squared - (ema_pose_grad ** 2))

                # diff1 = grad1 - ema_pose_grad
                # diff2 = grad2 - ema_pose_grad

                # n_sd1 = (((diff1 / sd) > 2).sum() + 1)
                # n_sd2 = (((diff2 / sd) > 2).sum() + 1)

                # print(n_sd1, n_sd2)

                gaussians.add_densification_stats(viewspace_point_tensor1, visibility_filter1, None)
                gaussians.add_densification_stats(viewspace_point_tensor2, visibility_filter2, None)
                gaussians.add_densification_stats(viewspace_point_tensor3, visibility_filter3, None)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # coef = np.corrcoef(np.array([avg_pose_grad_norms, avg_pt_grad_norms]))
                    # print(coef)

                    # plt.clf()
                    # plt.ylim(min(avg_pose_grad_norms), max(avg_pose_grad_norms))
                    # plt.scatter(avg_pt_grad_norms, avg_pose_grad_norms, alpha=0.2)
                    # plt.savefig("plt.png")

                    size_threshold = viewpoint_cams[0].image_width // 16 if iteration > opt.opacity_reset_interval else None
                    # tmp = iteration / opt.densify_until_iter 
                    grad_threshold = grad_threshold_scheduler(iteration)

                    gaussians.densify_and_prune(grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if (iteration > 10 and iteration % opt.opacity_reset_interval == 1 and iteration < opt.opacity_reset_until):
                    gaussians.reset_opacity()

        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()
        
        # Log and save
        training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
        if (iteration % saving_iterations == 0):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        # Optimizer step
        if iteration < opt.iterations:
            gaussians.step_optimizer(iteration, skip_pose_optimizer=(iteration % opt.opacity_reset_interval < 300))

        if (iteration % saving_iterations == 0):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            # torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if gaussians.camera_manager.learnable:
                pose_out = os.path.join(scene.model_path, "optimized_poses")
                os.makedirs(pose_out, exist_ok=True)
                for key in gaussians.camera_manager.R6T_store.keys():
                    pose_file = key.rsplit('_', 2)[0] + '.txt'

                    w2c = gaussians.camera_manager.get_pose(key).detach().cpu().numpy()
                    c2w = np.linalg.inv(w2c).T
                    np.savetxt(os.path.join(pose_out, pose_file), c2w)
                tmp = gaussians.camera_manager.RT_to_w2c(gaussians.camera_manager.calibration_error)
                tmp = tmp.detach().cpu().numpy()
                print(tmp)
                np.savetxt(os.path.join(pose_out, "calib_error.txt"), tmp)

def prepare_output_and_logger(args, args2, args3):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args), **vars(args2), **vars(args3))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration == 1 or iteration % testing_iterations == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(0, len(scene.getTestCameras()), max(1, len(scene.getTestCameras()) // 10))]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0, len(scene.getTrainCameras()), max(1, len(scene.getTrainCameras()) // 10))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                event_loss = 0.0
                
                for idx, viewpoints in enumerate(config['cameras']):
                    viewpoint = viewpoints[0]
                    render_pkg1 = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    render_pkg2 = renderFunc(viewpoints[1], scene.gaussians, *renderArgs)
                    render_raw = render_pkg1["render"]
                    flattened = render_raw.view(3, -1)
                    # min_rgb = flattened.min(dim=-1).values[:, None, None]
                    # max_rgb = flattened.max(dim=-1).values[:, None, None]

                    if config['name'] == 'train':
                        loss, reg = get_loss(render_pkg1, render_pkg2, viewpoints[1])
                    # render_raw = (render_raw - min_rgb) / (max_rgb - min_rgb)
                    image = torch.clamp(render_raw, 0.0, 1.0)
                    if viewpoint.original_image is not None:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if config['name'] == 'train':
                        event_frame = viewpoints[1].event_frame.to("cuda") * color_mask
                        event_frame = (event_frame - event_frame.min()) / (event_frame.max() - event_frame.min())
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if config['name'] == 'train':
                            tb_writer.add_images(config['name'] + "_view_{}/event_loss".format(viewpoint.image_name), torch.clamp(loss[None] / 1e-5, 0., 1.), global_step=iteration)
                        if iteration == testing_iterations:
                            if viewpoint.original_image is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if config['name'] == 'train':
                                tb_writer.add_images(config['name'] + "_view_{}/event_frame".format(viewpoint.image_name), event_frame[None], global_step=iteration)
                    
                    event_loss += torch.sum(loss).cpu().item()
                    # l1_test += l1_loss(image, gt_image).mean().double()
                    # psnr_test += psnr(image, gt_image).mean().double()
                event_loss /= len(config['cameras'])
                # psnr_test /= len(config['cameras'])
                # l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: Event Loss {}".format(iteration, config['name'], event_loss))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - event_loss', event_loss, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
            print(f"[Image Stats ({config['name']})] min: {render_raw.min()}, max: {render_raw.max()}, avg {render_raw.mean()}")

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tmp = scene.gaussians.get_opacity[None]
            color = torch.concat((tmp, tmp*0., 1-tmp), dim=-1)

            tb_writer.add_mesh(
                "scene/gaussians", 
                vertices=scene.gaussians.get_xyz[None], 
                colors=color*255,
                global_step=iteration
                )
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=2000)
    parser.add_argument("--save_iterations", type=int, default=60000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(is_random=True)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
