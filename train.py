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

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import json
import time
import uuid
import imageio
import datetime
import numpy as np
from tqdm import tqdm
from random import randint

import torch

from src.config import cfg, update_argparser, update_config

from src.utils.system_utils import seed_everything
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.bounding_utils import decide_main_bounding
from src.utils import mono_utils
from src.utils import loss_utils

from src.dataloader.data_pack import DataPack, compute_iter_idx
from src.sparse_voxel_model import SparseVoxelModel

import svraster_cuda


def training(args):
    # Init and load data pack
    data_pack = DataPack(cfg.data, cfg.model.white_background)

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    tr_cam_indices = compute_iter_idx(len(tr_cams), cfg.procedure.n_iter)

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init()

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            force_rerun=False)

    if cfg.regularizer.lambda_mast3r_metric_depth:
        mono_utils.prepare_mast3r_metric_depth(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            mast3r_repo_path=cfg.regularizer.mast3r_repo_path)

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        cfg_bounding=cfg.bounding,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,  # Not used
        suggested_bounding=data_pack.suggested_bounding,  # Can be None
    )

    # Init voxel model
    voxel_model = SparseVoxelModel(cfg.model)

    if args.load_iteration:
        loaded_iter = voxel_model.load_iteration(args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init(
            bounding=bounding,
            cfg_init=cfg.init,
            cameras=tr_cams)

    first_iter = loaded_iter if loaded_iter else 1
    print(f"Start optmization from iters={first_iter}.")

    # Init optimizer
    voxel_model.optimizer_init(cfg.optimizer)
    if loaded_iter and args.load_optimizer:
        voxel_model.optimizer_load_iteration(loaded_iter)

    # Init lr warmup scheduler
    if first_iter <= cfg.optimizer.n_warmup:
        rate = max(first_iter - 1, 0) / cfg.optimizer.n_warmup
        for param_group in voxel_model.optimizer.param_groups:
            param_group["base_lr"] = param_group["lr"]
            param_group["lr"] = rate * param_group["base_lr"]

    # Init subdiv
    remain_subdiv_times = sum(
        (i >= first_iter)
        for i in range(
            cfg.procedure.subdivide_from, cfg.procedure.subdivide_until+1,
            cfg.procedure.subdivide_every
        )
    )
    subdivide_scale = cfg.procedure.subdivide_target_scale ** (1 / remain_subdiv_times)
    subdivide_prop = max(0, (subdivide_scale - 1) / 7)
    print(f"Subdiv: times={remain_subdiv_times:2d} scale-each-time={subdivide_scale*100:.1f}% prop={subdivide_prop*100:.1f}%")

    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0

    tr_render_opt = {
        'track_max_w': False,
        'lambda_R_concen': cfg.regularizer.lambda_R_concen,
        'output_T': False,
        'output_depth': False,
        'ss': 1.0,  # disable supersampling at first
        'rand_bg': cfg.regularizer.rand_bg,
        'use_auto_exposure': cfg.auto_exposure.enable,
    }

    sparse_depth_loss = loss_utils.SparseDepthLoss(
        iter_end=cfg.regularizer.sparse_depth_until)
    depthanythingv2_loss = loss_utils.DepthAnythingv2Loss(
        iter_from=cfg.regularizer.depthanythingv2_from,
        iter_end=cfg.regularizer.depthanythingv2_end,
        end_mult=cfg.regularizer.depthanythingv2_end_mult)
    mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
        iter_from=cfg.regularizer.mast3r_metric_depth_from,
        iter_end=cfg.regularizer.mast3r_metric_depth_end,
        end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training")
    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth

        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

        if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
            tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.procedure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        # Pick a Camera
        cam = tr_cams[tr_cam_indices[iteration-1]]

        # Get gt image
        gt_image = cam.image.cuda()
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image

        # Render
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color']

        # Loss
        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss = cfg.regularizer.lambda_photo * photo_loss

        if need_sparse_depth:
            loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

        if cfg.regularizer.lambda_mask:
            gt_T = 1 - cam.mask.cuda()
            loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

        if need_depthanythingv2:
            loss += cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)

        if need_mast3r_metric_depth:
            loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim:
            loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
        if cfg.regularizer.lambda_T_concen:
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
        if cfg.regularizer.lambda_T_inside:
            loss += cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
        if need_nd_loss:
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss:
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)

        # Backward to get gradient of current iteration
        voxel_model.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Grid-level regularization
        grid_reg_interval = iteration >= cfg.regularizer.tv_from and iteration <= cfg.regularizer.tv_until
        if cfg.regularizer.lambda_tv_density and grid_reg_interval:
            lambda_tv_mult = cfg.regularizer.tv_decay_mult ** (iteration // cfg.regularizer.tv_decay_every)
            svraster_cuda.grid_loss_bw.total_variation(
                grid_pts=voxel_model._geo_grid_pts,
                vox_key=voxel_model.vox_key,
                weight=cfg.regularizer.lambda_tv_density * lambda_tv_mult,
                vox_size_inv=voxel_model.vox_size_inv,
                no_tv_s=True,
                tv_sparse=cfg.regularizer.tv_sparse,
                grid_pts_grad=voxel_model._geo_grid_pts.grad)

        # Optimizer step
        voxel_model.optimizer.step()

        # Learning rate warmup scheduler step
        if iteration <= cfg.optimizer.n_warmup:
            rate = iteration / cfg.optimizer.n_warmup
            for param_group in voxel_model.optimizer.param_groups:
                param_group["lr"] = rate * param_group["base_lr"]

        if iteration in cfg.optimizer.lr_decay_ckpt:
            for param_group in voxel_model.optimizer.param_groups:
                ori_lr = param_group["lr"]
                param_group["lr"] *= cfg.optimizer.lr_decay_mult
                print(f'LR decay of {param_group["name"]}: {ori_lr} => {param_group["lr"]}')

        ######################################################
        # Gradient statistic should happen before adaptive op
        ######################################################

        need_stat = (
            iteration >= 500 and \
            iteration <= cfg.procedure.subdivide_until)
        if need_stat:
            voxel_model.subdiv_meta += voxel_model._subdiv_p.grad

        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        need_pruning = (
            iteration % cfg.procedure.prune_every == 0 and \
            iteration >= cfg.procedure.prune_from and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = (
            iteration % cfg.procedure.subdivide_every == 0 and \
            iteration >= cfg.procedure.subdivide_from and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        # Do nothing in last 500 iteration
        need_pruning &= (iteration <= cfg.procedure.n_iter-500)
        need_subdividing &= (iteration <= cfg.procedure.n_iter-500)

        if need_pruning or need_subdividing:
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            torch.cuda.empty_cache()

        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_all_iter = max(1, cfg.procedure.prune_until - cfg.procedure.prune_every)
            prune_now_iter = max(0, iteration - cfg.procedure.prune_every)
            prune_iter_rate = max(0, min(1, prune_now_iter / prune_all_iter))
            thres_inc = max(0, cfg.procedure.prune_thres_final - cfg.procedure.prune_thres_init)
            prune_thres = cfg.procedure.prune_thres_init + thres_inc * prune_iter_rate

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)

            voxel_model.pruning(prune_mask)

            # Prune statistic (for the following subdivision)
            kept_idx = (~prune_mask).argwhere().squeeze(1)
            for k, v in stat_pkg.items():
                stat_pkg[k] = v[kept_idx]

            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')
            torch.cuda.empty_cache()

        if need_subdividing:
            # Exclude some voxels
            size_thres = stat_pkg['min_samp_interval'] * cfg.procedure.subdivide_samp_thres
            large_enough_mask = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest_mask = voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            valid_mask = large_enough_mask & non_finest_mask

            # Get some statistic for subdivision priority
            priority = voxel_model.subdiv_meta.squeeze(1) * valid_mask

            # Compute priority rank (larger value has higher priority)
            rank = torch.zeros_like(priority)
            rank[priority.argsort()] = torch.arange(len(priority), dtype=torch.float32, device="cuda")

            # Determine the number of voxels to subdivided
            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = rank.quantile(1 - subdivide_prop)

            # Compute subdivision mask
            subdivide_mask = (rank > thres) & valid_mask

            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (rank > rank[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            ori_n = voxel_model.num_voxels
            if subdivide_mask.sum() > 0:
                voxel_model.subdividing(subdivide_mask, cfg.procedure.subdivide_save_gpu)
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            voxel_model.subdiv_meta.zero_()  # reset subdiv meta
            remain_subdiv_times -= 1
            torch.cuda.empty_cache()

        ######################################################
        # End of adaptive voxels procedure
        ######################################################

        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log += ema_p * (loss - ema_loss_for_log)
            ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)
            if iteration % 10 == 0:
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save
            # I run OOM when this runs, so I'm commenting it out for now
            # training_report(
            #     data_pack=data_pack,
            #     voxel_model=voxel_model,
            #     iteration=iteration,
            #     loss=loss,
            #     psnr=psnr,
            #     elapsed=elapsed,
            #     ema_psnr=ema_psnr_for_log,
            #     pg_view_every=args.pg_view_every,
            #     test_iterations=args.test_iterations)

            if iteration in args.checkpoint_iterations or iteration == cfg.procedure.n_iter:
                voxel_model.save_iteration(iteration, quantize=args.save_quantized)
                if args.save_optimizer:
                    voxel_model.optimizer_save_iteration(iteration)
                print(f"[SAVE] path={voxel_model.latest_save_path}")
                voxel_model.save_ply(os.path.join(voxel_model.model_path, "voxel_model.ply"))


def training_report(data_pack, voxel_model, iteration, loss, psnr, elapsed, ema_psnr, pg_view_every, test_iterations):

    voxel_model.freeze_vox_geo()

    # Progress view
    if pg_view_every > 0 and (iteration % pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = 0
        view = test_cameras[pg_idx]
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]

        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_alpha)[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(voxel_model.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(voxel_model.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")

    # Report test and samples of training set
    if iteration in test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(voxel_model.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        outdir = os.path.join(voxel_model.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            json.dump(stat, f, indent=4)

    voxel_model.unfreeze_vox_geo()



if __name__ == "__main__":

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster optimization."
        "You can specify a list of config files to overwrite the default setups."
        "All config fields can also be overwritten by command line.")
    parser.add_argument('--cfg_files', default=[], nargs='*')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[-1])
    parser.add_argument("--pg_view_every", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--load_optimizer", action='store_true')
    parser.add_argument("--save_optimizer", action='store_true')
    parser.add_argument("--save_quantized", action='store_true')
    args, cmd_lst = parser.parse_known_args()

    # Update config from files and command line
    update_config(args.cfg_files, cmd_lst)

    # Global init
    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Setup output folder and dump config
    if not cfg.model.model_path:
        datetime_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
        unique_str = str(uuid.uuid4())[:6]
        folder_name = f"{datetime_str}-{unique_str}"
        cfg.model.model_path = os.path.join(f"./output", folder_name)

    os.makedirs(cfg.model.model_path, exist_ok=True)
    with open(os.path.join(cfg.model.model_path, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    print(f"Output folder: {cfg.model.model_path}")

    # Apply scheduler scaling
    if cfg.procedure.sche_mult != 1:
        sche_mult = cfg.procedure.sche_mult

        for key in ['geo_lr', 'sh0_lr', 'shs_lr']:
            cfg.optimizer[key] /= sche_mult
        cfg.optimizer.n_warmup = round(cfg.optimizer.n_warmup * sche_mult)
        cfg.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.optimizer.lr_decay_ckpt]

        for key in [
                'dist_from', 'tv_from', 'tv_until',
                'n_dmean_from', 'n_dmean_end',
                'n_dmed_from', 'n_dmed_end',
                'depthanythingv2_from', 'depthanythingv2_end',
                'mast3r_metric_depth_from', 'mast3r_metric_depth_end']:
            cfg.regularizer[key] = round(cfg.regularizer[key] * sche_mult)

        for key in [
                'n_iter',
                'prune_from', 'prune_every', 'prune_until',
                'subdivide_from', 'subdivide_every', 'subdivide_until']:
            cfg.procedure[key] = round(cfg.procedure[key] * sche_mult)
        cfg.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.procedure.reset_sh_ckpt]

    # Update negative iterations
    for i in range(len(args.test_iterations)):
        if args.test_iterations[i] < 0:
            args.test_iterations[i] += cfg.procedure.n_iter + 1
    for i in range(len(args.checkpoint_iterations)):
        if args.checkpoint_iterations[i] < 0:
            args.checkpoint_iterations[i] += cfg.procedure.n_iter + 1

    # Launch training loop
    training(args)
    print("Everything done.")
