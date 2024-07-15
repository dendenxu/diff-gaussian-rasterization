/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& tile_mask,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& out_alpha,
	const bool debug);
		
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fusedPreprocess4D(
		const torch::Tensor& means3D,
		const torch::Tensor& cov,
		const torch::Tensor& ms,
		const torch::Tensor& cov_t,
		const torch::Tensor& opacities,
		const torch::Tensor& t1,
		const torch::Tensor& sh,
		const torch::Tensor& t,
		const torch::Tensor& viewmatrix,
		const torch::Tensor& projmatrix,
		const torch::Tensor& cam_pos,
		const int deg,
		const int deg_t,
		const float duration
		);

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeCov4D(
		torch::Tensor& scaling_xyzt,
		torch::Tensor& rotation_l,
		torch::Tensor& rotation_r);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeCov4DBackward(
		torch::Tensor& scaling_xyzt,
		torch::Tensor& rotation_l,
		torch::Tensor& rotation_r,
		torch::Tensor& dL_dcov,
		torch::Tensor& dL_dms,
		torch::Tensor& dL_dcov_t);

torch::Tensor computeSH4D(
	const int deg,
	const int deg_t,
	torch::Tensor& sh,
	torch::Tensor& dir,
	torch::Tensor& dir_t,
	const float duration
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeSH4DBackward(
	const int deg,
	const int deg_t,
	torch::Tensor& sh,
	torch::Tensor& dir,
	torch::Tensor& dir_t,
	const float duration,
	torch::Tensor& dL_drgb
);