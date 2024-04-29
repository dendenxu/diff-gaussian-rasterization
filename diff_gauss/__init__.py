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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

BLOCK_X = 16
BLOCK_Y = 16


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    tile_mask,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        tile_mask,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        tile_mask,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            tile_mask,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, tile_mask, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, depth, alpha, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_alpha, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, tile_mask, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                grad_out_alpha,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None, tile_mask=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if tile_mask is None:
            tile_mask = torch.Tensor([]).bool()
        # TODO: in sampler `typed` will change the type of the tensor
        if tile_mask.dtype != torch.bool:
            tile_mask = tile_mask.bool()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            tile_mask,
            raster_settings,
        )


def mark_visible(positions: torch.Tensor, viewmatrix: torch.Tensor, projmatrix: torch.Tensor):
    # Mark visible points (based on frustum culling for camera) with a boolean
    with torch.no_grad():
        visible = _C.mark_visible(
            positions,
            viewmatrix,
            projmatrix)

    return visible


def compute_cov_4d(scaling_xyzt: torch.Tensor, rotation_l: torch.Tensor, rotation_r: torch.Tensor):
    # Mark visible points (based on frustum culling for camera) with a boolean
    return _ComputeCov4D.apply(
        scaling_xyzt,
        rotation_l,
        rotation_r)


class _ComputeCov4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scaling_xyzt,
        rotation_l,
        rotation_r
    ):
        cov, ms, cov_t = _C.compute_cov_4d(scaling_xyzt, rotation_l, rotation_r)
        ctx.save_for_backward(scaling_xyzt, rotation_l, rotation_r)
        return cov, ms, cov_t

    @staticmethod
    def backward(ctx, grad_out_cov, grad_out_ms, grad_out_cov_t):

        # Restore necessary values from context
        scaling_xyzt, rotation_l, rotation_r = ctx.saved_tensors

        # Restructure args as C++ method expects them
        grad_scaling_xyzt, grad_rotation_l, grad_rotation_r = _C.compute_cov_4d_backward(
            scaling_xyzt,
            rotation_l,
            rotation_r,
            grad_out_cov,
            grad_out_ms,
            grad_out_cov_t,
        )

        grads = (
            grad_scaling_xyzt,
            grad_rotation_l,
            grad_rotation_r,
        )

        return grads


def compute_sh_4d(deg: int, deg_t: int, sh: torch.Tensor, dirs: torch.Tensor = None, dirs_t: torch.Tensor = None, l: float = None):
    # Mark visible points (based on frustum culling for camera) with a boolean
    return _ComputeSH4D.apply(
        deg,
        deg_t,
        sh,
        dirs,
        dirs_t,
        l)


class _ComputeSH4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        deg,
        deg_t,
        sh,
        dirs,
        dirs_t,
        l
    ):
        if dirs is None:
            dirs = torch.Tensor([])
        if dirs_t is None:
            dirs_t = torch.Tensor([])
        if l is None:
            l = 0.0
        rgb = _C.compute_sh_4d(deg, deg_t, sh, dirs, dirs_t, l)
        ctx.deg = deg
        ctx.deg_t = deg_t
        ctx.l = l
        ctx.save_for_backward(sh, dirs, dirs_t)
        return rgb

    @staticmethod
    def backward(ctx, grad_out_rgb):

        # Restore necessary values from context
        deg = ctx.deg
        deg_t = ctx.deg_t
        l = ctx.l
        sh, dirs, dirs_t = ctx.saved_tensors

        # Restructure args as C++ method expects them
        grad_sh, grad_dirs, grad_dirs_t = _C.compute_sh_4d_backward(
            deg, deg_t, sh, dirs, dirs_t, l,
            grad_out_rgb,
        )

        grads = (
            None,
            None,
            grad_sh,
            grad_dirs,
            grad_dirs_t,
            None,
        )

        return grads
