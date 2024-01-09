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

#include "forward_half.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm_half3 computeColorFromSHHalf(int idx, int deg, int max_coeffs, const glm_half3* means, const glm_half3 campos, const __half* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm_half3 pos = means[idx];
	glm_half3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm_half3* sh = ((glm_half3*)shs) + idx * max_coeffs;
	glm_half3 result = SH_C0_HALF * sh[0];

	if (deg > 0)
	{
		__half x = dir.x;
		__half y = dir.y;
		__half z = dir.z;
		result = result - SH_C1_HALF * y * sh[1] + SH_C1_HALF * z * sh[2] - SH_C1_HALF * x * sh[3];

		if (deg > 1)
		{
			__half xx = x * x, yy = y * y, zz = z * z;
			__half xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2_HALF[0] * xy * sh[4] +
				SH_C2_HALF[1] * yz * sh[5] +
				SH_C2_HALF[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2_HALF[3] * xz * sh[7] +
				SH_C2_HALF[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3_HALF[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3_HALF[1] * xy * z * sh[10] +
					SH_C3_HALF[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3_HALF[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3_HALF[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3_HALF[5] * z * (xx - yy) * sh[14] +
					SH_C3_HALF[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, Half(0.0f));
}

// Forward version of 2D covariance matrix computation
__device__ __half3 computeCov2DHalf(const __half3& mean, __half focal_x, __half focal_y, __half tan_fovx, __half tan_fovy, const __half* cov3D, const __half* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	__half3 t = transformPoint4x3Half(mean, viewmatrix);

	const __half limx = 1.3f * tan_fovx;
	const __half limy = 1.3f * tan_fovy;
	const __half txtz = t.x / t.z;
	const __half tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm_half33 J = glm_half33(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm_half33 W = glm_half33(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm_half33 T = W * J;

	glm_half33 Vrk = glm_half33(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm_half33 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { __half(cov[0][0]), __half(cov[0][1]), __half(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3DHalf(const glm_half3 scale, const __half mod, const glm_half4 rot, __half* cov3D)
{
	// Create scaling matrix
	glm_half33 S = glm_half33(Half(1.0f));
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm_half4 q = rot;// / glm::length(rot);
	__half r = q.x;
	__half x = q.y;
	__half y = q.z;
	__half z = q.w;

	// Compute rotation matrix from quaternion
	glm_half33 R = glm_half33(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm_half33 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm_half33 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessHalfCUDA(int P, int D, int M,
	const __half* orig_points,
	const glm_half3* scales,
	const float scale_modifier,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	__half2* points_xy_image,
	__half* depths,
	__half* cov3Ds,
	__half* rgb,
	__half4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	__half3 p_view;
	if (!in_frustum_half(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	__half3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	__half4 p_hom = transformPoint4x4Half(p_orig, projmatrix);
	__half p_w = 1.0f / (p_hom.w + 0.0000001f);
	__half3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const __half* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3DHalf(scales[idx], Half(scale_modifier), rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	__half3 cov = computeCov2DHalf(p_orig, Half(focal_x), Half(focal_y), Half(tan_fovx), Half(tan_fovy), cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	__half det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	__half det_inv = 1.f / det;
	__half3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	__half mid = 0.5f * (cov.x + cov.z);
	__half lambda1 = mid + sqrt(max(Half(0.1f), mid * mid - det));
	__half lambda2 = mid - sqrt(max(Half(0.1f), mid * mid - det));
	__half my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	__half2 point_image = { ndc2PixHalf(p_proj.x, W), ndc2PixHalf(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRectHalf(point_image, int(Half(my_radius)), rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm_half3 result = computeColorFromSHHalf(idx, D, M, (glm_half3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = int(Half(my_radius));
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one __half4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderHalfCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const __half2* __restrict__ points_xy_image,
	const __half* __restrict__ features,
	const __half* __restrict__ depths,
	const __half4* __restrict__ conic_opacity,
	__half* __restrict__ out_alpha,
	uint32_t* __restrict__ n_contrib,
	const __half* __restrict__ bg_color,
	__half* __restrict__ out_color,
	__half* __restrict__ out_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	__half2 pixf = { (Half)pix.x, (Half)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ __half2 collected_xy[BLOCK_SIZE];
	__shared__ __half4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	__half T = Half(1.0f);
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	__half C[CHANNELS] = { Half(0) };
	__half D = Half(0);

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			__half2 xy = collected_xy[j];
			__half2 d = { xy.x - pixf.x, xy.y - pixf.y };
			__half4 con_o = collected_conic_opacity[j];
			__half power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			__half alpha = min(Half(0.9999f), con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			__half test_T = T * (1 - alpha);

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] = C[ch] + features[collected_id[j] * CHANNELS + ch] * alpha * T;
			D = D + depths[collected_id[j]] * alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// Early stopping
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		out_alpha[pix_id] = 1 - T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
	}
}

void FORWARD::render_half(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const __half2* means2D,
	const __half* colors,
	const __half* depths,
	const __half4* conic_opacity,
	__half* out_alpha,
	uint32_t* n_contrib,
	const __half* bg_color,
	__half* out_color,
	__half* out_depth)
{
	renderHalfCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		out_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_depth);
}

void FORWARD::preprocess_half(int P, int D, int M,
	const __half* means3D,
	const glm_half3* scales,
	const float scale_modifier,
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
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	__half2* means2D,
	__half* depths,
	__half* cov3Ds,
	__half* rgb,
	__half4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessHalfCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}