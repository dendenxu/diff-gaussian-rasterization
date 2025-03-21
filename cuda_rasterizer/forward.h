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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const bool* tile_mask,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		// float* comp,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* depths,
		const float4* conic_opacity,
		float* out_alpha,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_depth);

	void fusedPreprocess4D(int P,
		const int deg,
		const int deg_t,
		const int M,
		const glm::vec3* means3D,
		const float* cov,
		const glm::vec3* ms,
		const float* cov_t,
		const float* opacities,
		const float* t1,
		const float* shs,
		const float* t,
		const float* viewmatrix,
		const float* projmatrix,
		const float* cam_pos,
		const float duration,
		bool* mask,
		float* occ1,
		glm::vec3* xyz3,
		glm::vec3* rgb3);

	void fusedPreprocess4DSparse(int P,
		const int deg,
		const int deg_t,
		const int M,
		const glm::vec3* means3D,
		const float* cov,
		const glm::vec3* ms,
		const float* cov_t,
		const float* opacities,
		const float* t1,
		const glm::vec3* bases,
		const float* shs,
		const float* t,
		const int* inverse,
		const float* viewmatrix,
		const float* projmatrix,
		const float* cam_pos,
		const float duration,
		bool* mask,
		float* occ1,
		glm::vec3* xyz3,
		glm::vec3* rgb3);

	void computeCov3D(int P,
		const glm::vec3* scaling_xyz,
		const glm::vec4* rotation_l,
		float* cov);

	void computeCov4D(int P,
		const glm::vec4* scaling_xyzt,
		const glm::vec4* rotation_l,
		const glm::vec4* rotation_r,
		float* cov,
		glm::vec3* ms,
		float* cov_t);

	void computeSH4D(
		int P,
		int deg, int deg_t, int max_coeffs, 
		const float* sh, 
		const glm::vec3* dir, 
		const float* dir_t, 
		const float time_duration,
		glm::vec3* rgb);
}


#endif