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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		opacity.contiguous().data_ptr<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(), 
		tile_mask.contiguous().data_ptr<bool>(),
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		out_depth.contiguous().data_ptr<float>(),
		out_alpha.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, out_alpha, radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dabsmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  // just for storing intermediate results
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  out_alpha.contiguous().data_ptr<float>(),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_depth.contiguous().data_ptr<float>(),
	  dL_dout_alpha.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dabsmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),  
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_ddepths.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  debug);
  }

  return std::make_tuple(dL_dabsmeans2D, dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::empty({P}, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}

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
	)
{ 
  const int P = means3D.size(0);
  int M = 0;
  if(sh.size(0) != 0) M = sh.size(1);

  torch::Tensor mask = torch::empty({P, 1}, means3D.options().dtype(at::kBool));
  torch::Tensor occ1 = torch::empty({P, 1}, means3D.options());
  torch::Tensor xyz3 = torch::empty({P, 3}, means3D.options());
  torch::Tensor rgb3 = torch::empty({P, 3}, means3D.options());
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::fusedPreprocess4D(P, deg, deg_t, M,
		means3D.contiguous().data_ptr<float>(),
		cov.contiguous().data_ptr<float>(),
		ms.contiguous().data_ptr<float>(),
		cov_t.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		t1.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		t.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		cam_pos.contiguous().data_ptr<float>(),
		duration,
		mask.contiguous().data_ptr<bool>(),
		occ1.contiguous().data_ptr<float>(),
		xyz3.contiguous().data_ptr<float>(),
		rgb3.contiguous().data_ptr<float>());
	}
	return std::make_tuple(mask, occ1, xyz3, rgb3);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fusedPreprocess4DSparse(
	const torch::Tensor& means3D,
	const torch::Tensor& cov,
	const torch::Tensor& ms,
	const torch::Tensor& cov_t,
	const torch::Tensor& opacities,
	const torch::Tensor& t1,
	const torch::Tensor& base,
	const torch::Tensor& sh,
	const torch::Tensor& t,
	const torch::Tensor& inverse,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& cam_pos,
	const int deg,
	const int deg_t,
	const float duration
	)
{ 
  const int P = means3D.size(0);
  int M = 0;
  if(sh.size(0) != 0) M = sh.size(1);

  torch::Tensor mask = torch::empty({P, 1}, means3D.options().dtype(at::kBool));
  torch::Tensor occ1 = torch::empty({P, 1}, means3D.options());
  torch::Tensor xyz3 = torch::empty({P, 3}, means3D.options());
  torch::Tensor rgb3 = torch::empty({P, 3}, means3D.options());
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::fusedPreprocess4DSparse(P, deg, deg_t, M,
		means3D.contiguous().data_ptr<float>(),
		cov.contiguous().data_ptr<float>(),
		ms.contiguous().data_ptr<float>(),
		cov_t.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		t1.contiguous().data_ptr<float>(),
		base.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		t.contiguous().data_ptr<float>(),
		inverse.contiguous().data_ptr<int>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		cam_pos.contiguous().data_ptr<float>(),
		duration,
		mask.contiguous().data_ptr<bool>(),
		occ1.contiguous().data_ptr<float>(),
		xyz3.contiguous().data_ptr<float>(),
		rgb3.contiguous().data_ptr<float>());
	}
	return std::make_tuple(mask, occ1, xyz3, rgb3);
}

torch::Tensor computeCov3D(
		torch::Tensor& scaling_xyz,
		torch::Tensor& rotation_l)
{
	const int P = scaling_xyz.size(0);
	torch::Tensor cov = torch::empty({P, 6}, scaling_xyz.options());

	if(P != 0)
	{
		CudaRasterizer::Rasterizer::computeCov3D(P,
			scaling_xyz.contiguous().data_ptr<float>(),
			rotation_l.contiguous().data_ptr<float>(),
			cov.contiguous().data_ptr<float>());
	}

	return cov;
}

std::tuple<torch::Tensor, torch::Tensor> computeCov3DBackward(
		torch::Tensor& scaling_xyz,
		torch::Tensor& rotation_l,
		torch::Tensor& dL_dcov)
{
	const int P = scaling_xyz.size(0);
	torch::Tensor dL_dscaling_xyz = torch::zeros({P, 3}, scaling_xyz.options());
	torch::Tensor dL_drotation_l = torch::zeros({P, 4}, scaling_xyz.options());

	if(P != 0)
	{
		CudaRasterizer::Rasterizer::computeCov3DBackward(P,
			scaling_xyz.contiguous().data_ptr<float>(),
			rotation_l.contiguous().data_ptr<float>(),
			dL_dcov.contiguous().data_ptr<float>(),
			dL_dscaling_xyz.contiguous().data_ptr<float>(),
			dL_drotation_l.contiguous().data_ptr<float>());
	}

	return std::make_tuple(dL_dscaling_xyz, dL_drotation_l);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeCov4D(
		torch::Tensor& scaling_xyzt,
		torch::Tensor& rotation_l,
		torch::Tensor& rotation_r)
{ 
  const int P = scaling_xyzt.size(0);
  
  torch::Tensor cov = torch::empty({P, 6}, scaling_xyzt.options());
  torch::Tensor ms = torch::empty({P, 3}, scaling_xyzt.options());
  torch::Tensor cov_t = torch::empty({P, 1}, scaling_xyzt.options());
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::computeCov4D(P,
		scaling_xyzt.contiguous().data_ptr<float>(),
		rotation_l.contiguous().data_ptr<float>(),
		rotation_r.contiguous().data_ptr<float>(),
		cov.contiguous().data_ptr<float>(),
		ms.contiguous().data_ptr<float>(),
		cov_t.contiguous().data_ptr<float>());
  }
  
  return std::make_tuple(cov, ms, cov_t);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeCov4DBackward(
		torch::Tensor& scaling_xyzt,
		torch::Tensor& rotation_l,
		torch::Tensor& rotation_r,
		torch::Tensor& dL_dcov,
		torch::Tensor& dL_dms,
		torch::Tensor& dL_dcov_t)
{ 
  const int P = scaling_xyzt.size(0);
  
  torch::Tensor dL_dscaling_xyzt = torch::zeros({P, 4}, scaling_xyzt.options());
  torch::Tensor dL_drotation_l = torch::zeros({P, 4}, scaling_xyzt.options());
  torch::Tensor dL_drotation_r = torch::zeros({P, 4}, scaling_xyzt.options());
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::computeCov4DBackward(P,
		scaling_xyzt.contiguous().data_ptr<float>(),
		rotation_l.contiguous().data_ptr<float>(),
		rotation_r.contiguous().data_ptr<float>(),
		dL_dcov.contiguous().data_ptr<float>(),
		dL_dms.contiguous().data_ptr<float>(),
		dL_dcov_t.contiguous().data_ptr<float>(),
		dL_dscaling_xyzt.contiguous().data_ptr<float>(),
		dL_drotation_l.contiguous().data_ptr<float>(),
		dL_drotation_r.contiguous().data_ptr<float>());
  }
  
  return std::make_tuple(dL_dscaling_xyzt, dL_drotation_l, dL_drotation_r);
}


torch::Tensor computeSH4D(
	const int deg,
	const int deg_t,
	torch::Tensor& sh,
	torch::Tensor& dir,
	torch::Tensor& dir_t,
	const float duration
)
{ 
	const int P = sh.size(0);
	int M = 0;
	if(sh.size(0) != 0) M = sh.size(1);

	torch::Tensor rgb = torch::zeros({P, 3}, sh.options());

	if(P != 0)
	{
		CudaRasterizer::Rasterizer::computeSH4D(P,
			deg, deg_t, M,
			sh.contiguous().data_ptr<float>(),
			dir.contiguous().data_ptr<float>(),
			dir_t.contiguous().data_ptr<float>(),
			duration,
			rgb.contiguous().data_ptr<float>()
		);
	}

	return rgb;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeSH4DBackward(
	const int deg,
	const int deg_t,
	torch::Tensor& sh,
	torch::Tensor& dir,
	torch::Tensor& dir_t,
	const float duration,
	torch::Tensor& dL_drgb
)
{ 
	const int P = sh.size(0);
	int M = 0;
	if(sh.size(0) != 0) M = sh.size(1);

	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, sh.options());
	torch::Tensor dL_ddir = torch::zeros({P, 3}, sh.options());
	torch::Tensor dL_ddir_t = torch::zeros({P, 1}, sh.options());

	if(P != 0)
	{
		CudaRasterizer::Rasterizer::computeSH4DBackward(P,
			deg, deg_t, M,
			sh.contiguous().data_ptr<float>(),
			dir.contiguous().data_ptr<float>(),
			dir_t.contiguous().data_ptr<float>(),
			duration,
			dL_drgb.contiguous().data_ptr<float>(),
			dL_dsh.contiguous().data_ptr<float>(),
			dL_ddir.contiguous().data_ptr<float>(),
			dL_ddir_t.contiguous().data_ptr<float>()
		);
	}

	return std::make_tuple(dL_dsh, dL_ddir, dL_ddir_t);
}