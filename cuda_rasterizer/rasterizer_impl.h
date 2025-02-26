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

#include <cstdint>
#include <iostream>
#include <vector>
#include "rasterizer.h"
// #include "auxiliary_half.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		float* depths;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* tiles_touched;
		uint32_t* point_offsets;
		char* scanning_space;
		size_t scan_size;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint32_t* n_contrib;
		uint2* ranges;

		static ImageState fromChunk(char*& chunk, size_t N, size_t M);
	};

	struct BinningState
	{
		uint32_t* point_list;
		uint32_t* point_list_unsorted;
		uint64_t* point_list_keys;
		uint64_t* point_list_keys_unsorted;
		char* list_sorting_space;
		size_t sorting_size;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	template<typename T> 
	size_t required(size_t P, size_t N)
	{
		char* size = nullptr;
		T::fromChunk(size, P, N);
		return ((size_t)size) + 128;
	}
};
