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

#ifndef CUDA_RASTERIZER_FORWARD_HALF_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_HALF_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "auxiliary.h"
#include "auxiliary_half.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess_half(int P, int D, int M,
		const __half* orig_points,
		const glm_half3* scales,
		const __half scale_modifier,
		const glm_half4* rotations,
		const __half* opacities,
		const __half* shs,
		bool* clamped,
		const __half* cov3D_precomp,
		const __half* colors_precomp,
		const __half* viewmatrix,
		const __half* projmatrix,
		const glm_half3* cam_pos,
		const int W, int H,
		const __half focal_x, __half focal_y,
		const __half tan_fovx, __half tan_fovy,
		int* radii,
		__half2* points_xy_image,
		__half* depths,
		__half* cov3Ds,
		__half* colors,
		__half4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render_half(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const __half2* points_xy_image,
		const __half* features,
		const __half* depths,
		const __half4* conic_opacity,
		__half* out_alpha,
		uint32_t* n_contrib,
		const __half* bg_color,
		__half* out_color,
		__half* out_depth);
}


#endif