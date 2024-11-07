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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("fused_preprocess_4d", &fusedPreprocess4D);
  m.def("fused_preprocess_4d_sparse", &fusedPreprocess4DSparse);
  m.def("compute_cov_4d", &computeCov4D);
  m.def("compute_cov_4d_backward", &computeCov4DBackward);
  m.def("compute_cov_3d", &computeCov3D);
  m.def("compute_cov_3d_backward", &computeCov3DBackward);
  m.def("compute_sh_4d", &computeSH4D);
  m.def("compute_sh_4d_backward", &computeSH4DBackward);
}