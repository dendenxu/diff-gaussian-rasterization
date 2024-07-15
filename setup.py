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

from setuptools import setup
from os.path import dirname, join, abspath
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

dirname(abspath(__file__))

setup(
    name="diff_gauss",
    packages=['diff_gauss'],
    ext_modules=[
        CUDAExtension(
            name="diff_gauss._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                # "cuda_rasterizer/forward_half.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"],
            extra_compile_args={"nvcc": [
                "-O3", 
                "-Xcompiler", 
                "-fno-gnu-unique", 
                # "-G",
                "-I" + join(dirname(abspath(__file__)), "third_party/glm/")]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
