# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import torch
from svraster_cuda.meta import MAX_NUM_LEVELS

from src.utils import octree_utils
from src.utils.activation_utils import exp_linear_11

class SVInOut:

    def save(self, path, quantize=False):
        '''
        Save the necessary attributes and parameters for reproducing rendering.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'active_sh_degree': self.active_sh_degree,
            'ss': self.ss,
            'scene_center': self.scene_center.data.contiguous(),
            'inside_extent': self.inside_extent.data.contiguous(),
            'scene_extent': self.scene_extent.data.contiguous(),
            'octpath': self.octpath.data.contiguous(),
            'octlevel': self.octlevel.data.contiguous(),
            '_geo_grid_pts': self._geo_grid_pts.data.contiguous(),
            '_sh0': self._sh0.data.contiguous(),
            '_shs': self._shs.data.contiguous(),
        }

        if quantize:
            quantize_state_dict(state_dict)
            state_dict['quantized'] = True
        else:
            state_dict['quantized'] = False

        for k, v in state_dict.items():
            if torch.is_tensor(v):
                state_dict[k] = v.cpu()
        torch.save(state_dict, path)
        self.latest_save_path = path
        
    def save_ply(self, path):
        '''
        Save the voxel data as a PLY file with spherical harmonics, opacity, octree information, 
        and grid points for each voxel.
        '''
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except ImportError:
            print("Error: plyfile package is required. Install with pip install plyfile")
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get basic position and octree data
        positions = self.vox_center.detach().cpu().numpy()  # Shape: (N, 3)
        octpath = self.octpath.detach().cpu().numpy().squeeze()  # Shape: (N)
        octlevel = self.octlevel.detach().cpu().numpy().squeeze()  # Shape: (N)
        
        # Get DC components (sh0)
        sh0 = self._sh0.detach().cpu().numpy()  # Shape: (N, 3)
        
        # Get higher-order SH coefficients and reshape
        shs = self._shs.detach().reshape(self._shs.shape[0], -1).cpu().numpy()  # Flatten from (N, SH_DEGREE_COEFFS, 3) to (N, 3*SH_DEGREE_COEFFS)
        
        # Get grid point data
        vox_key = self.vox_key.detach().cpu().numpy()  # Shape: (N, 8)
        
        raw_grid_values = self._geo_grid_pts.detach().cpu()
        grid_values = raw_grid_values.numpy().flatten()  # Shape: (M)
        
        # Define attribute list
        attributes = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),          # positions
            ('octpath', 'u4'), ('octlevel', 'u1'),    # octree data
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')  # DC components
        ]
        
        # Add rest of SH coefficients based on the actual data shape
        for i in range(shs.shape[1]):
            attributes.append((f'f_rest_{i}', 'f4'))
        
        # Add grid point values for each corner (no need to store positions as they're derivable)
        for i in range(8):
            attributes.append((f'grid{i}_value', 'f4'))
        
        # Create elements array
        elements = np.empty(len(positions), dtype=attributes)
        
        # Fill position and octree data
        elements['x'] = positions[:, 0]
        elements['y'] = positions[:, 1]
        elements['z'] = positions[:, 2]
        elements['octpath'] = octpath
        elements['octlevel'] = octlevel
        
        # Fill DC components
        elements['f_dc_0'] = sh0[:, 0] 
        elements['f_dc_1'] = sh0[:, 1]
        elements['f_dc_2'] = sh0[:, 2]
        
        # Fill higher-order SH coefficients (using actual shape)
        for i in range(shs.shape[1]):
            elements[f'f_rest_{i}'] = shs[:, i]
        
        # Fill grid point values for each voxel (skip storing positions)
        for i in range(8):
            for vox_idx in range(len(positions)):
                grid_idx = vox_key[vox_idx, i]
                elements[f'grid{i}_value'][vox_idx] = grid_values[grid_idx]
        
        # Add comments to the PLY file
        header_comments = []

        # Add scene center (which is a 3D vector)
        header_comments.append(f"scene_center {self.scene_center[0].item()} {self.scene_center[1].item()} {self.scene_center[2].item()}")

        # Add scene extent (which is a scalar)
        header_comments.append(f"scene_extent {self.scene_extent.item()}")

        # Add active_sh_degree
        header_comments.append(f"active_sh_degree {self.active_sh_degree}")

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], comments=header_comments).write(path)
        
        print(f"PLY file saved to: {path} with {len(positions)} points")

    def load(self, path):
        '''
        Load the saved models.
        '''
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        if state_dict.get('quantized', False):
            dequantize_state_dict(state_dict)

        self.active_sh_degree = state_dict['active_sh_degree']
        self.ss = state_dict['ss']

        self.scene_center = state_dict['scene_center'].cuda()
        self.inside_extent = state_dict['inside_extent'].cuda()
        self.scene_extent = state_dict['scene_extent'].cuda()

        self.octpath = state_dict['octpath'].cuda()
        self.octlevel = state_dict['octlevel'].cuda().to(torch.int8)
        self.vox_center, self.vox_size = octree_utils.octpath_decoding(
            self.octpath, self.octlevel, self.scene_center, self.scene_extent)
        self.grid_pts_key, self.vox_key = octree_utils.build_grid_pts_link(self.octpath, self.octlevel)

        self._geo_grid_pts = state_dict['_geo_grid_pts'].cuda().requires_grad_()
        self._sh0 = state_dict['_sh0'].cuda().requires_grad_()
        self._shs = state_dict['_shs'].cuda().requires_grad_()

        N = len(self.octpath)
        self._subdiv_p = torch.full([N, 1], 1.0, dtype=torch.float32, device="cuda").requires_grad_()
        self.subdiv_meta = torch.zeros([N, 1], dtype=torch.float32, device="cuda")

        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32, device="cuda")

        self.loaded_path = path

    def save_iteration(self, iteration, quantize=False):
        path = os.path.join(self.model_path, "checkpoints", f"iter{iteration:06d}_model.pt")
        self.save(path, quantize=quantize)
        self.latest_save_iter = iteration

    def load_iteration(self, iteration=-1):
        if iteration == -1:
            # Find the maximum iteration if it is -1.
            fnames = os.listdir(os.path.join(self.model_path, "checkpoints"))
            loaded_iter = max(int(re.sub("[^0-9]", "", fname)) for fname in fnames)
        else:
            loaded_iter = iteration

        path = os.path.join(self.model_path, "checkpoints", f"iter{loaded_iter:06d}_model.pt")
        self.load(path)

        self.loaded_iter = iteration

        return loaded_iter


# Quantization utilities to reduce size when saving model.
# It can reduce ~70% model size with minor PSNR drop.
def quantize_state_dict(state_dict):
    state_dict['_geo_grid_pts'] = quantization(state_dict['_geo_grid_pts'])
    state_dict['_sh0'] = [quantization(v) for v in state_dict['_sh0'].split(1, dim=1)]
    state_dict['_shs'] = [quantization(v) for v in state_dict['_shs'].split(1, dim=1)]

def dequantize_state_dict(state_dict):
    state_dict['_geo_grid_pts'] = dequantization(state_dict['_geo_grid_pts'])
    state_dict['_sh0'] = torch.cat(
        [dequantization(v) for v in state_dict['_sh0']], dim=1)
    state_dict['_shs'] = torch.cat(
        [dequantization(v) for v in state_dict['_shs']], dim=1)

def quantization(src_tensor, max_iter=10):
    src_shape = src_tensor.shape
    src_vals = src_tensor.flatten().contiguous()
    order = src_vals.argsort()
    quantile_ind = (torch.linspace(0,1,257) * (len(order) - 1)).long().clamp_(0, len(order)-1)
    codebook = src_vals[order[quantile_ind]].contiguous()
    codebook[0] = -torch.inf
    ind = torch.searchsorted(codebook, src_vals)

    codebook = codebook[1:]
    ind = (ind - 1).clamp_(0, 255)

    diff_l = (src_vals - codebook[ind-1]).abs()
    diff_m = (src_vals - codebook[ind]).abs()
    ind = ind - 1 + (diff_m < diff_l)
    ind.clamp_(0, 255)

    for _ in range(max_iter):
        codebook = torch.zeros_like(codebook).index_reduce_(
            dim=0,
            index=ind,
            source=src_vals,
            reduce='mean',
            include_self=False)
        diff_l = (src_vals - codebook[ind-1]).abs()
        diff_r = (src_vals - codebook[(ind+1).clamp_max_(255)]).abs()
        diff_m = (src_vals - codebook[ind]).abs()
        upd_mask = torch.minimum(diff_l, diff_r) < diff_m
        if upd_mask.sum() == 0:
            break
        shift = (diff_r < diff_l) * 2 - 1
        ind[upd_mask] += shift[upd_mask]
        ind.clamp_(0, 255)

    codebook = torch.zeros_like(codebook).index_reduce_(
        dim=0,
        index=ind,
        source=src_vals,
        reduce='mean',
        include_self=False)

    return dict(
        index=ind.reshape(src_shape).to(torch.uint8),
        codebook=codebook,
    )

def dequantization(quant_dict):
    return quant_dict['codebook'][quant_dict['index'].long()]
