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


def fused_preprocess_4d(xyz3, cov6, ms3, cov_t1, occ1, t1, feat, t, world_view_transform, full_proj_transform, cam_pos, deg, deg_t, duration):
    # Mark visible points (based on frustum culling for camera) with a boolean
    with torch.no_grad():
        mask, occ1, xyz3, rgb3 = _C.fused_preprocess_4d(xyz3, cov6, ms3, cov_t1, occ1, t1, feat, t, world_view_transform, full_proj_transform, cam_pos, deg, deg_t, duration)

    return mask, occ1, xyz3, rgb3  # mask and output


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
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, tile_mask, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, means2D)
        return color, depth, alpha, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_alpha, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, tile_mask, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha, means2D = ctx.saved_tensors

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
                absgrad_means2D, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            absgrad_means2D, grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

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

        means2D.absgrad = absgrad_means2D  # let the user select their grad

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


def compute_sh_4d(deg: int, deg_t: int, sh: torch.Tensor, dir: torch.Tensor = None, dir_t: torch.Tensor = None, l: float = None):
    if dir is None:
        dir = torch.Tensor([])
    if dir_t is None:
        dir_t = torch.Tensor([])
    if l is None:
        l = 0.0
    return _ComputeSH4D.apply(
        deg,
        deg_t,
        sh,
        dir,
        dir_t,
        l)


class _ComputeSH4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        deg,
        deg_t,
        sh,
        dir,
        dir_t,
        l
    ):
        rgb = _C.compute_sh_4d(deg, deg_t, sh, dir, dir_t, l)
        ctx.deg = deg
        ctx.deg_t = deg_t
        ctx.l = l
        ctx.save_for_backward(sh, dir, dir_t)
        return rgb

    @staticmethod
    def backward(ctx, grad_out_rgb):

        # Restore necessary values from context
        deg = ctx.deg
        deg_t = ctx.deg_t
        l = ctx.l
        sh, dir, dir_t = ctx.saved_tensors

        # Restructure args as C++ method expects them
        grad_sh, grad_dir, grad_dir_t = _C.compute_sh_4d_backward(
            deg, deg_t, sh, dir, dir_t, l,
            grad_out_rgb,
        )

        grads = (
            None,
            None,
            grad_sh,
            grad_dir,
            grad_dir_t,
            None,
        )

        return grads


def align_with(p: int, a: int = 128):
    p = (p + a - 1) // a * a
    return p


def interpret_geomBuffer(geomBuffer: torch.Tensor, N: int):
    # N: Number of points rendered
    ptr = geomBuffer.data_ptr()
    p = align_with(ptr, 128) - ptr

    off = 4 * N
    depths = geomBuffer[p:p + off].view(torch.float)
    p = align_with(p + off, 128)

    off = 3 * N
    clamped = geomBuffer[p:p + off].view(torch.bool).view(N, 3)
    p = align_with(p + off, 128)

    off = 4 * N
    internal_radii = geomBuffer[p:p + off].view(torch.int)
    p = align_with(p + off, 128)

    off = 2 * 4 * N
    means2D = geomBuffer[p:p + off].view(torch.float).view(N, 2)
    p = align_with(p + off, 128)

    off = 6 * 4 * N
    cov3D = geomBuffer[p:p + off].view(torch.float).view(N, 6)
    p = align_with(p + off, 128)

    off = 4 * 4 * N
    conic_opacity = geomBuffer[p:p + off].view(torch.float).view(N, 4)
    p = align_with(p + off, 128)

    off = 3 * 4 * N
    rgb = geomBuffer[p:p + off].view(torch.float).view(N, 3)
    p = align_with(p + off, 128)

    off = 4 * N
    tiles_touched = geomBuffer[p:p + off].view(torch.int)
    p = align_with(p + off, 128)

    off = 4 * N
    point_offsets = geomBuffer[p:p + off].view(torch.int)

    return dict(
        depths=depths,
        clamped=clamped,
        internal_radii=internal_radii,
        means2D=means2D,
        cov3D=cov3D,
        conic_opacity=conic_opacity,
        rgb=rgb,
        tiles_touched=tiles_touched,
        point_offsets=point_offsets
    )


def interpret_binningBuffer(binningBuffer: torch.Tensor, N: int):
    # N: Number of tile-gaussian pairs
    ptr = binningBuffer.data_ptr()
    p = align_with(ptr, 128) - ptr

    off = 4 * N
    point_list = binningBuffer[p:p + off].view(torch.int)
    p = align_with(p + off, 128)

    off = 4 * N
    point_list_unsorted = binningBuffer[p:p + off].view(torch.int)
    p = align_with(p + off, 128)

    off = 8 * N
    point_list_keys = binningBuffer[p:p + off].view(torch.long)
    p = align_with(p + off, 128)

    off = 8 * N
    point_list_keys_unsorted = binningBuffer[p:p + off].view(torch.long)
    p = align_with(p + off, 128)

    return dict(
        point_list=point_list,
        point_list_unsorted=point_list_unsorted,
        point_list_keys=point_list_keys,
        point_list_keys_unsorted=point_list_keys_unsorted,

        # Little Endian
        depths=point_list_keys.view(torch.float).view(N, 2)[:, 0],
        tile_ids=point_list_keys.view(torch.int).view(N, 2)[:, 1],
    )


def interpret_imgBuffer(imgBuffer: torch.Tensor, N: int, M: int):
    # N: Number of pixels
    # M: Number of tiles
    ptr = imgBuffer.data_ptr()
    p = align_with(ptr, 128) - ptr

    off = 4 * N
    n_contrib = imgBuffer[p:p + off].view(torch.int)
    p = align_with(p + off, 128)

    off = 2 * 4 * M
    ranges = imgBuffer[p:p + off].view(torch.int).view(M, 2)
    p = align_with(p + off, 128)

    return dict(
        n_contrib=n_contrib,
        ranges=ranges
    )
