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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float4* conic_opacity,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	const bool* tile_mask)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				if (tile_mask != nullptr && !tile_mask[key])
				{
					continue;
				}
				else 
				{

					// constexpr float alpha_threshold = 1.0f / 255.0f;
					// const float opacity_power_threshold = log(conic_opacity[idx].w / alpha_threshold);
					// glm::vec2 max_pos;
					// const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
					// const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};
					// float max_opac_factor = max_contrib_power_rect_gaussian_float<BLOCK_X-1, BLOCK_Y-1>(conic_opacity[idx], points_xy[idx], tile_min, tile_max, max_pos);

					// if (max_opac_factor > opacity_power_threshold) {
					// 	continue;
					// }

					key <<= 32;
					key |= *((uint32_t*)&depths[idx]);
					gaussian_keys_unsorted[off] = key;
					gaussian_values_unsorted[off] = idx;
					off++;
				}
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

void CudaRasterizer::Rasterizer::computeCov3D(int P,
	const float* scaling_xyz,
	const float* rotation_l,
	float* cov)
{
	FORWARD::computeCov3D(
		P,
		(glm::vec3*)scaling_xyz,
		(glm::vec4*)rotation_l,
		cov);
}

void CudaRasterizer::Rasterizer::computeCov3DBackward(int P,
	const float* scaling_xyz,
	const float* rotation_l,
	const float* dL_dcov,
	float* dL_dscaling_xyz,
	float* dL_drotation_l)
{
	BACKWARD::computeCov3DBackward(
		P,
		(glm::vec3*)scaling_xyz,
		(glm::vec4*)rotation_l,
		dL_dcov,
		(glm::vec3*)dL_dscaling_xyz,
		(glm::vec4*)dL_drotation_l);
}

void CudaRasterizer::Rasterizer::computeCov4D(int P,
	const float* scaling_xyzt,
	const float* rotation_l,
	const float* rotation_r,
	float* cov,
	float* ms,
	float* cov_t)
{
	FORWARD::computeCov4D(
		P,
		(glm::vec4*)scaling_xyzt,
		(glm::vec4*)rotation_l,
		(glm::vec4*)rotation_r,
		cov,
		(glm::vec3*)ms,
		cov_t);
}

void CudaRasterizer::Rasterizer::computeCov4DBackward(int P,
	const float* scaling_xyzt,
	const float* rotation_l,
	const float* rotation_r,
	const float* dL_dcov,
	const float* dL_dms,
	const float* dL_dcov_t,
	float* dL_dscaling_xyzt,
	float* dL_drotation_l,
	float* dL_drotation_r)
{
	BACKWARD::computeCov4DBackward(
		P,
		(glm::vec4*)scaling_xyzt,
		(glm::vec4*)rotation_l,
		(glm::vec4*)rotation_r,
		dL_dcov,
		(glm::vec3*)dL_dms,
		dL_dcov_t,
		(glm::vec4*)dL_dscaling_xyzt,
		(glm::vec4*)dL_drotation_l,
		(glm::vec4*)dL_drotation_r);
}


void CudaRasterizer::Rasterizer::computeSH4D(int P,
	int deg, int deg_t, int max_coeffs, 
	const float* shs, 
	const float* dir, 
	const float* dir_t, 
	const float time_duration,
	float* rgb)
{
	FORWARD::computeSH4D(
		P,
		deg,
		deg_t,
		max_coeffs,
		shs,
		(glm::vec3*)dir,
		dir_t,
		time_duration,
		(glm::vec3*)rgb
	);
}

void CudaRasterizer::Rasterizer::computeSH4DBackward(
	int P,
	int deg, int deg_t, int max_coeffs, 
	const float* shs, 
	const float* dir, 
	const float* dir_t, 
	const float time_duration,
	const float* dL_drgb,
	float* dL_dsh,
	float* dL_ddir,
	float* dL_ddir_t
)
{
	BACKWARD::computeSH4DBackward(
		P,
		deg,
		deg_t,
		max_coeffs,
		shs,
		(glm::vec3*)dir,
		dir_t,
		time_duration,
		(glm::vec3*)dL_drgb,
		dL_dsh,
		(glm::vec3*)dL_ddir,
		dL_ddir_t
	);
}

// Marginalization & color computation
void CudaRasterizer::Rasterizer::fusedPreprocess4DSparse(
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
	const float* base,
	const float* sh,
	const float* t,
	const int* inverse,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float duration,
	bool* mask,
	float* occ1,
	float* xyz3,
	float* rgb3)
{
	FORWARD::fusedPreprocess4DSparse(
		P,
		deg,
		deg_t,
		M,
		(glm::vec3*)means3D,
		cov,
		(glm::vec3*)ms,
		cov_t,
		opacities,
		t1,
		(glm::vec3*)base,
		sh,
		t,
		inverse,
		viewmatrix,
		projmatrix,
		cam_pos,
		duration,
		mask,
		occ1,
		(glm::vec3*)xyz3,
		(glm::vec3*)rgb3);
}

// Marginalization & color computation
void CudaRasterizer::Rasterizer::fusedPreprocess4D(
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
	float* rgb3)
{
	FORWARD::fusedPreprocess4D(
		P,
		deg,
		deg_t,
		M,
		(glm::vec3*)means3D,
		cov,
		(glm::vec3*)ms,
		cov_t,
		opacities,
		t1,
		sh,
		t,
		viewmatrix,
		projmatrix,
		cam_pos,
		duration,
		mask,
		occ1,
		(glm::vec3*)xyz3,
		(glm::vec3*)rgb3);
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N, size_t M)
{
	ImageState img;
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, M, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
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
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size); // memory allocation
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height, tile_grid.x * tile_grid.y);
	char* img_chunkptr = imageBuffer(img_chunk_size); // memory allocation
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height, tile_grid.x * tile_grid.y);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		tile_mask,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size); // memory allocation
	BinningState binState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		geomState.point_offsets,
		binState.point_list_keys_unsorted,
		binState.point_list_unsorted,
		radii,
		tile_grid,
		tile_mask)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binState.list_sorting_space,
		binState.sorting_size,
		binState.point_list_keys_unsorted, binState.point_list_keys,
		binState.point_list_unsorted, binState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, 
		block,
		imgState.ranges,
		binState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		out_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
	const float* accum_alphas,
	const float* dL_dpix,
	const float* dL_dpix_depth,
	const float* dL_dpix_alpha,
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
	bool debug)
{

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height, tile_grid.x * tile_grid.y);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		accum_alphas,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		dL_dpix_alpha,
		(float3*)dL_dmean2D,
		(float3*)dL_dabsmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}