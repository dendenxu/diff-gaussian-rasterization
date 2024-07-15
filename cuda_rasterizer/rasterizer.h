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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "cuda_fp16.h"

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void fusedPreprocess4D(
			const int P,
			const int deg,
			const int deg_t,
			const int M,
			const float* means3D,
			const float* cov,
			const float* ms,
			const float* cov_t,
			const float* opacities,
			const float* t1,
			const float* sh,
			const float* t,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float duration,
			bool* mask,
			float* occ1,
			float* xyz3,
			float* rgb3);

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static void computeCov4D(
			int P,
			const float* scaling_xyzt,
			const float* rotation_l,
			const float* rotation_r,
			float* cov,
			float* ms,
			float* cov_t);

		static void computeSH4D(
			int P,
			int deg, int deg_t, int max_coeffs, 
			const float* shs, 
			const float* dir, 
			const float* dir_t, 
			const float time_duration,
			float* rgb);

		static void computeSH4DBackward(
			int P,
			int deg, int deg_t, int max_coeffs, 
			const float* shs, 
			const float* dir, 
			const float* dir_t, 
			const float time_duration,
			const float* dL_drgb,
			float* dL_dsh,
			float* dL_ddir,
			float* dL_ddir_t);

		static void computeCov4DBackward(
			int P,
			const float* scaling_xyzt,
			const float* rotation_l,
			const float* rotation_r,
			const float* dL_dcov,
			const float* dL_dms,
			const float* dL_dcov_t,
			float* dL_dscaling_xyzt,
			float* dL_drotation_l,
			float* dL_drotation_r);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const bool* tile_mask,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, const float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_depth,
			float* out_alpha,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, const float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* accum_alphas,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			const float* dL_dpix_dalpha,
			float* dL_dmean2D,
			float* dL_dabsmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif