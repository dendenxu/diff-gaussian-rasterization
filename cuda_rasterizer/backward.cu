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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH_4D(int idx, int deg, int deg_t, int max_coeffs,
	const float* shs, const glm::vec3* dirs, const float* dirs_t, const float time_duration,
	const glm::vec3* dL_drgb, float* dL_dshs, glm::vec3* dL_ddir, float* dL_ddir_t)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 dir = dirs[idx];
	const float dir_t = dirs_t[idx];

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_drgb[idx];

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	glm::vec3 dRGBdt(0, 0, 0);

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = ((glm::vec3*)dL_dshs) + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float l0m0 = SH_C0;

	float dRGBdsh0 = l0m0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;

	if (deg > 0){
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		float l1m1 = -1 * SH_C1 * y;
		float l1m0 = SH_C1 * z;
		float l1p1 = -1 * SH_C1 * x;

		float dl1m1_dy = -1 * SH_C1;
		float dl1m0_dz = SH_C1;
		float dl1p1_dx = -1 * SH_C1;

		dL_dsh[1] = l1m1 * dL_dRGB;
		dL_dsh[2] = l1m0 * dL_dRGB;
		dL_dsh[3] = l1p1 * dL_dRGB;

		dRGBdx = dl1p1_dx * sh[3];
		dRGBdy = dl1m1_dy * sh[1];
		dRGBdz = dl1m0_dz * sh[2];

		if (deg > 1){
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float l2m2 = SH_C2[0] * xy;
			float l2m1 = SH_C2[1] * yz;
			float l2m0 = SH_C2[2] * (2.0 * zz - xx - yy);
			float l2p1 = SH_C2[3] * xz;
			float l2p2 = SH_C2[4] * (xx - yy);

			float dl2m2_dx = SH_C2[0] * y;
			float dl2m2_dy = SH_C2[0] * x;
			float dl2m1_dy = SH_C2[1] * z;
			float dl2m1_dz = SH_C2[1] * y;
			float dl2m0_dx = -2 * SH_C2[2] * x;
			float dl2m0_dy = -2 * SH_C2[2] * y;
			float dl2m0_dz = 4 * SH_C2[2] * z;
			float dl2p1_dx = SH_C2[3] * z;
			float dl2p1_dz = SH_C2[3] * x;
			float dl2p2_dx = 2 * SH_C2[4] * x;
			float dl2p2_dy = -2 * SH_C2[4] * y;

			dL_dsh[4] = l2m2 * dL_dRGB;
			dL_dsh[5] = l2m1 * dL_dRGB;
			dL_dsh[6] = l2m0 * dL_dRGB;
			dL_dsh[7] = l2p1 * dL_dRGB;
			dL_dsh[8] = l2p2 * dL_dRGB;

			dRGBdx += (
				dl2m2_dx * sh[4] + dl2m0_dx * sh[6] + dl2p1_dx * sh[7] + dl2p2_dx * sh[8]
			);
			dRGBdy += (
				dl2m2_dy * sh[4] + dl2m1_dy * sh[5] + dl2m0_dy * sh[6] + dl2p2_dy * sh[8]
			);
			dRGBdz += (
				dl2m1_dz * sh[5] + dl2m0_dz * sh[6] + dl2p1_dz * sh[7]
			);

			if (deg > 2){
				float l3m3 = SH_C3[0] * y * (3 * xx - yy);
				float l3m2 = SH_C3[1] * xy * z;
				float l3m1 = SH_C3[2] * y * (4 * zz - xx - yy);
				float l3m0 = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
				float l3p1 = SH_C3[4] * x * (4 * zz - xx - yy);
				float l3p2 = SH_C3[5] * z * (xx - yy);
				float l3p3 = SH_C3[6] * x * (xx - 3 * yy);

				float dl3m3_dx = SH_C3[0] * y * 6 * x;
				float dl3m3_dy = SH_C3[0] * (3 * xx - 3 * yy);
				float dl3m2_dx = SH_C3[1] * yz;
				float dl3m2_dy = SH_C3[1] * xz;
				float dl3m2_dz = SH_C3[1] * xy;
				float dl3m1_dx = -SH_C3[2] * y * 2 * x;
				float dl3m1_dy = SH_C3[2] * (4 * zz - xx - 3 * yy);
				float dl3m1_dz = SH_C3[2] * y * 8 * z;
				float dl3m0_dx = -SH_C3[3] * z * 6 * x;
				float dl3m0_dy = -SH_C3[3] * z * 6 * y;
				float dl3m0_dz = SH_C3[3] * (6 * zz - 3 * xx - 3 * yy);
				float dl3p1_dx = SH_C3[4] * (4 * zz - 3 * xx - yy);
				float dl3p1_dy = -SH_C3[4] * x * 2 * y;
				float dl3p1_dz = SH_C3[4] * x * 8 * z;
				float dl3p2_dx = SH_C3[5] * z * 2 * x;
				float dl3p2_dy = -SH_C3[5] * z * 2 * y;
				float dl3p2_dz = SH_C3[5] * (xx - yy);
				float dl3p3_dx = SH_C3[6] * (3 * xx - 3 * yy);
				float dl3p3_dy = -SH_C3[6] * x * 6 * y;

				dL_dsh[9] = l3m3 * dL_dRGB;
				dL_dsh[10] = l3m2 * dL_dRGB;
				dL_dsh[11] = l3m1 * dL_dRGB;
				dL_dsh[12] = l3m0 * dL_dRGB;
				dL_dsh[13] = l3p1 * dL_dRGB;
				dL_dsh[14] = l3p2 * dL_dRGB;
				dL_dsh[15] = l3p3 * dL_dRGB;

				dRGBdx += (
					dl3m3_dx * sh[9] +
					dl3m2_dx * sh[10] +
					dl3m1_dx * sh[11] +
					dl3m0_dx * sh[12] +
					dl3p1_dx * sh[13] +
					dl3p2_dx * sh[14] +
					dl3p3_dx * sh[15]
				);

				dRGBdy += (
					dl3m3_dy * sh[9] +
					dl3m2_dy * sh[10] +
					dl3m1_dy * sh[11] +
					dl3m0_dy * sh[12] +
					dl3p1_dy * sh[13] +
					dl3p2_dy * sh[14] +
					dl3p3_dy * sh[15]
				);

				dRGBdz += (
					dl3m2_dz * sh[10] +
					dl3m1_dz * sh[11] +
					dl3m0_dz * sh[12] +
					dl3p1_dz * sh[13] +
					dl3p2_dz * sh[14]
				);

				if (deg_t > 0){
					float t1 = cos(2 * MY_PI * dir_t / time_duration);
					float dt1_dt = sin(2 * MY_PI * dir_t / time_duration) * 2 * MY_PI / time_duration;

					dL_dsh[16] = t1 * l0m0 * dL_dRGB;
					dL_dsh[17] = t1 * l1m1 * dL_dRGB;
					dL_dsh[18] = t1 * l1m0 * dL_dRGB;
					dL_dsh[19] = t1 * l1p1 * dL_dRGB;
					dL_dsh[20] = t1 * l2m2 * dL_dRGB;
					dL_dsh[21] = t1 * l2m1 * dL_dRGB;
					dL_dsh[22] = t1 * l2m0 * dL_dRGB;
					dL_dsh[23] = t1 * l2p1 * dL_dRGB;
					dL_dsh[24] = t1 * l2p2 * dL_dRGB;
					dL_dsh[25] = t1 * l3m3 * dL_dRGB;
					dL_dsh[26] = t1 * l3m2 * dL_dRGB;
					dL_dsh[27] = t1 * l3m1 * dL_dRGB;
					dL_dsh[28] = t1 * l3m0 * dL_dRGB;
					dL_dsh[29] = t1 * l3p1 * dL_dRGB;
					dL_dsh[30] = t1 * l3p2 * dL_dRGB;
					dL_dsh[31] = t1 * l3p3 * dL_dRGB;
	

					dRGBdt += dt1_dt * (
						l0m0 * sh[16] +
						l1m1 * sh[17] +
						l1m0 * sh[18] +
						l1p1 * sh[19] + 
						l2m2 * sh[20] +
						l2m1 * sh[21] +
						l2m0 * sh[22] +
						l2p1 * sh[23] +
						l2p2 * sh[24] + 
						l3m3 * sh[25] +
						l3m2 * sh[26] +
						l3m1 * sh[27] +
						l3m0 * sh[28] +
						l3p1 * sh[29] +
						l3p2 * sh[30] +
						l3p3 * sh[31]);

					dRGBdx += t1 * (
						dl1p1_dx * sh[19] + 
						dl2m2_dx * sh[20] + 
						dl2m0_dx * sh[22] + 
						dl2p1_dx * sh[23] + 
						dl2p2_dx * sh[24] + 
						dl3m3_dx * sh[25] +
						dl3m2_dx * sh[26] +
						dl3m1_dx * sh[27] +
						dl3m0_dx * sh[28] +
						dl3p1_dx * sh[29] +
						dl3p2_dx * sh[30] +
						dl3p3_dx * sh[31]
					);

					dRGBdy += t1 * (
						dl1m1_dy * sh[17] +
						dl2m2_dy * sh[20] + 
						dl2m1_dy * sh[21] + 
						dl2m0_dy * sh[22] + 
						dl2p2_dy * sh[24] + 
						dl3m3_dy * sh[25] +
						dl3m2_dy * sh[26] +
						dl3m1_dy * sh[27] +
						dl3m0_dy * sh[28] +
						dl3p1_dy * sh[29] +
						dl3p2_dy * sh[30] +
						dl3p3_dy * sh[31]
					);

					dRGBdz += t1 * (
						dl1m0_dz * sh[18] +
						dl2m1_dz * sh[21] + 
						dl2m0_dz * sh[22] + 
						dl2p1_dz * sh[23] +
						dl3m2_dz * sh[26] +
						dl3m1_dz * sh[27] +
						dl3m0_dz * sh[28] +
						dl3p1_dz * sh[29] +
						dl3p2_dz * sh[30]
					);

					if (deg_t > 1){
						float t2 = cos(2 * MY_PI * dir_t * 2 / time_duration);
						float dt2_dt = sin(2 * MY_PI * dir_t * 2 / time_duration) * 2 * MY_PI * 2 / time_duration;


						dL_dsh[32] = t2 * l0m0 * dL_dRGB;
						dL_dsh[33] = t2 * l1m1 * dL_dRGB;
						dL_dsh[34] = t2 * l1m0 * dL_dRGB;
						dL_dsh[35] = t2 * l1p1 * dL_dRGB;
						dL_dsh[36] = t2 * l2m2 * dL_dRGB;
						dL_dsh[37] = t2 * l2m1 * dL_dRGB;
						dL_dsh[38] = t2 * l2m0 * dL_dRGB;
						dL_dsh[39] = t2 * l2p1 * dL_dRGB;
						dL_dsh[40] = t2 * l2p2 * dL_dRGB;
						dL_dsh[41] = t2 * l3m3 * dL_dRGB;
						dL_dsh[42] = t2 * l3m2 * dL_dRGB;
						dL_dsh[43] = t2 * l3m1 * dL_dRGB;
						dL_dsh[44] = t2 * l3m0 * dL_dRGB;
						dL_dsh[45] = t2 * l3p1 * dL_dRGB;
						dL_dsh[46] = t2 * l3p2 * dL_dRGB;
						dL_dsh[47] = t2 * l3p3 * dL_dRGB;

						dRGBdt += dt2_dt * (
							l0m0 * sh[32] +
							l1m1 * sh[33] +
							l1m0 * sh[34] +
							l1p1 * sh[35] + 
							l2m2 * sh[36] +
							l2m1 * sh[37] +
							l2m0 * sh[38] +
							l2p1 * sh[39] +
							l2p2 * sh[40] + 
							l3m3 * sh[41] +
							l3m2 * sh[42] +
							l3m1 * sh[43] +
							l3m0 * sh[44] +
							l3p1 * sh[45] +
							l3p2 * sh[46] +
							l3p3 * sh[47]);

						dRGBdx += t2 * (
							dl1p1_dx * sh[35] + 
							dl2m2_dx * sh[36] + 
							dl2m0_dx * sh[38] + 
							dl2p1_dx * sh[39] + 
							dl2p2_dx * sh[40] + 
							dl3m3_dx * sh[41] +
							dl3m2_dx * sh[42] +
							dl3m1_dx * sh[43] +
							dl3m0_dx * sh[44] +
							dl3p1_dx * sh[45] +
							dl3p2_dx * sh[46] +
							dl3p3_dx * sh[47]
						);

						dRGBdy += t2 * (
							dl1m1_dy * sh[33] +
							dl2m2_dy * sh[36] + 
							dl2m1_dy * sh[37] + 
							dl2m0_dy * sh[38] + 
							dl2p2_dy * sh[40] + 
							dl3m3_dy * sh[41] +
							dl3m2_dy * sh[42] +
							dl3m1_dy * sh[43] +
							dl3m0_dy * sh[44] +
							dl3p1_dy * sh[45] +
							dl3p2_dy * sh[46] +
							dl3p3_dy * sh[47]
						);

						dRGBdz += t2 * (
							dl1m0_dz * sh[34] +
							dl2m1_dz * sh[37] + 
							dl2m0_dz * sh[38] + 
							dl2p1_dz * sh[39] +
							dl3m2_dz * sh[42] +
							dl3m1_dz * sh[43] +
							dl3m0_dz * sh[44] +
							dl3p1_dz * sh[45] +
							dl3p2_dz * sh[46]
						);
					}
				}
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	dL_ddir[idx].x = glm::dot(dRGBdx, dL_dRGB);
	dL_ddir[idx].y = glm::dot(dRGBdy, dL_dRGB);
	dL_ddir[idx].z = glm::dot(dRGBdz, dL_dRGB);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_ddir_t[idx] = -glm::dot(dRGBdt, dL_dRGB);
}

__global__ void computeSH4DBackwardCUDA(int P,
								int deg, int deg_t, int max_coeffs, 
								const float* sh, const glm::vec3* dir, const float* dir_t, const float time_duration,
								const glm::vec3* dL_drgb, float* dL_dsh, glm::vec3* dL_ddir, float* dL_ddir_t)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	computeColorFromSH_4D(
		idx,
		deg,
		deg_t,
		max_coeffs,
		sh,
		dir,
		dir_t,
		time_duration,
		dL_drgb,
		dL_dsh,
		dL_ddir,
		dL_ddir_t
	);
}


void BACKWARD::computeSH4DBackward(
	int P,
	int deg, int deg_t, int max_coeffs, 
	const float* sh, 
	const glm::vec3* dir, 
	const float* dir_t, 
	const float time_duration,
	const glm::vec3* dL_drgb,
	float* dL_dsh,
	glm::vec3* dL_ddir,
	float* dL_ddir_t
) 
{
	computeSH4DBackwardCUDA << <(P + 255) / 256, 256 >> > (
		P,
		deg,
		deg_t,
		max_coeffs,
		sh,
		dir,
		dir_t,
		time_duration,
		dL_drgb,
		dL_dsh,
		dL_ddir,
		dL_ddir_t
	);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


__global__ void computeCov3DBackwardCUDA(int P,
	const glm::vec3* scaling_xyz,
	const glm::vec4* rotation_l,
	const float* dL_dcov,
	glm::vec3* dL_dscaling_xyz,
	glm::vec4* dL_drotation_l)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	computeCov3D(
		idx,
		scaling_xyz[idx],
		1.0f,
		rotation_l[idx],
		// dL_dcov + idx * 6,
		// dL_dscaling_xyz + idx,
		// dL_drotation_l + idx);
		dL_dcov,
		dL_dscaling_xyz,
		dL_drotation_l);
}

void BACKWARD::computeCov3DBackward(
	int P,
	const glm::vec3* scaling_xyz,
	const glm::vec4* rotation_l,
	const float* dL_dcov,
	glm::vec3* dL_dscaling_xyz,
	glm::vec4* dL_drotation_l)
{
	computeCov3DBackwardCUDA << <(P + 255) / 256, 256 >> > (
		P,
		scaling_xyz,
		rotation_l,
		dL_dcov,
		dL_dscaling_xyz,
		dL_drotation_l);
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov4DBackward(
	const glm::vec4 scaling_xyzt,
	const glm::vec4 rotation_l, 
	const glm::vec4 rotation_r,
	const float* dL_dcov, 
	const glm::vec3 dL_dms, 
	const float dL_dcov_t,
	glm::vec4 &dL_dscaling_xyzt,
	glm::vec4 &dL_drotation_l, 
	glm::vec4 &dL_drotation_r)
{
	glm::mat4 S = glm::mat4(1.0f);
	S[0][0] = scaling_xyzt.x;
	S[1][1] = scaling_xyzt.y;
	S[2][2] = scaling_xyzt.z;
	S[3][3] = scaling_xyzt.w;

	const float l_l = glm::length(rotation_l);
	const float a = rotation_l.x / l_l;
	const float b = rotation_l.y / l_l;
	const float c = rotation_l.z / l_l;
	const float d = rotation_l.w / l_l;

	const float l_r = glm::length(rotation_r);
	const float p = rotation_r.x / l_r;
	const float q = rotation_r.y / l_r;
	const float r = rotation_r.z / l_r;
	const float s = rotation_r.w / l_r;

	glm::mat4 M_l = glm::mat4(
		a, -b, -c, -d,
		b, a,-d, c,
		c, d, a,-b,
		d,-c, b, a
	);

	glm::mat4 M_r = glm::mat4(
		p, q, r, s,
		-q, p,-s, r,
		-r, s, p,-q,
		-s,-r, q, p
	);
	// glm stores in column major
	glm::mat4 R = M_r * M_l;
	glm::mat4 M = S * R;
	glm::mat4 Sigma = glm::transpose(M) * M;
	float cov_t = Sigma[3][3];

	glm::mat3 cov11 = glm::mat3(Sigma);
	glm::vec3 cov12 = glm::vec3(Sigma[0][3], Sigma[1][3], Sigma[2][3]);

	glm::vec3 dL_dcov12 = -glm::vec3(
		dL_dcov[0] * cov12[0] + dL_dcov[1] * cov12[1] * 0.5 + dL_dcov[2] * cov12[2] * 0.5,
		dL_dcov[1] * cov12[0] * 0.5 + dL_dcov[3] * cov12[1] + dL_dcov[4] * cov12[2] * 0.5,
		dL_dcov[2] * cov12[0] * 0.5 + dL_dcov[4] * cov12[1] * 0.5 + dL_dcov[5] * cov12[2]
	) * 2.0f / cov_t;

	dL_dcov12 += dL_dms / cov_t;

	float dL_dcov_t_w_ms_cov = dL_dcov_t;
	float dL_dms_dot_cov12 = glm::dot(dL_dms, cov12);
	dL_dcov_t_w_ms_cov += -dL_dms_dot_cov12 / (cov_t * cov_t);
	dL_dcov_t_w_ms_cov += (
		cov12[0] * cov12[0] * dL_dcov[0] + cov12[0] * cov12[1] * dL_dcov[1] +
		cov12[0] * cov12[2] * dL_dcov[2] + cov12[1] * cov12[1] * dL_dcov[3] +
		cov12[1] * cov12[2] * dL_dcov[4] + cov12[2] * cov12[2] * dL_dcov[5]
    ) / (cov_t * cov_t);

	glm::mat4 dL_dSigma = glm::mat4(
		dL_dcov[0], 0.5f * dL_dcov[1], 0.5f * dL_dcov[2], 0.5f * dL_dcov12[0],
		0.5f * dL_dcov[1], dL_dcov[3], 0.5f * dL_dcov[4], 0.5f * dL_dcov12[1],
		0.5f * dL_dcov[2], 0.5f * dL_dcov[4], dL_dcov[5], 0.5f * dL_dcov12[2],
		0.5f * dL_dcov12[0], 0.5f * dL_dcov12[1], 0.5f * dL_dcov12[2], dL_dcov_t_w_ms_cov
	);
	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat4 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat4 Rt = glm::transpose(R);
	glm::mat4 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	dL_dscaling_xyzt.x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscaling_xyzt.y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscaling_xyzt.z = glm::dot(Rt[2], dL_dMt[2]);
	dL_dscaling_xyzt.w = glm::dot(Rt[3], dL_dMt[3]);

	dL_dMt[0] *= scaling_xyzt.x;
	dL_dMt[1] *= scaling_xyzt.y;
	dL_dMt[2] *= scaling_xyzt.z;
	dL_dMt[3] *= scaling_xyzt.w;

	glm::mat4 dL_dml_t = dL_dMt * M_r;
	glm::vec4 dL_drot_l;
	dL_drot_l.x = dL_dml_t[0][0] + dL_dml_t[1][1] + dL_dml_t[2][2] + dL_dml_t[3][3];
	dL_drot_l.y = dL_dml_t[0][1] - dL_dml_t[1][0] + dL_dml_t[2][3] - dL_dml_t[3][2];
	dL_drot_l.z = dL_dml_t[0][2] - dL_dml_t[1][3] - dL_dml_t[2][0] + dL_dml_t[3][1];
	dL_drot_l.w = dL_dml_t[0][3] + dL_dml_t[1][2] - dL_dml_t[2][1] - dL_dml_t[3][0];

	glm::mat4 dL_dmr_t = M_l * dL_dMt;
	glm::vec4 dL_drot_r;
	dL_drot_r.x = dL_dmr_t[0][0] + dL_dmr_t[1][1] + dL_dmr_t[2][2] + dL_dmr_t[3][3];
	dL_drot_r.y = -dL_dmr_t[0][1] + dL_dmr_t[1][0] + dL_dmr_t[2][3] - dL_dmr_t[3][2];
	dL_drot_r.z = -dL_dmr_t[0][2] - dL_dmr_t[1][3] + dL_dmr_t[2][0] + dL_dmr_t[3][1];
	dL_drot_r.w = -dL_dmr_t[0][3] + dL_dmr_t[1][2] - dL_dmr_t[2][1] + dL_dmr_t[3][0];

	float4 dL_drotation_l_f = dnormvdv(float4{rotation_l.x, rotation_l.y, rotation_l.z, rotation_l.w}, float4{dL_drot_l.x, dL_drot_l.y, dL_drot_l.z, dL_drot_l.w});
	float4 dL_drotation_r_f = dnormvdv(float4{rotation_r.x, rotation_r.y, rotation_r.z, rotation_r.w}, float4{dL_drot_r.x, dL_drot_r.y, dL_drot_r.z, dL_drot_r.w});
	dL_drotation_l.x = dL_drotation_l_f.x;
	dL_drotation_l.y = dL_drotation_l_f.y;
	dL_drotation_l.z = dL_drotation_l_f.z;
	dL_drotation_l.w = dL_drotation_l_f.w;
	dL_drotation_r.x = dL_drotation_r_f.x;
	dL_drotation_r.y = dL_drotation_r_f.y;
	dL_drotation_r.z = dL_drotation_r_f.z;
	dL_drotation_r.w = dL_drotation_r_f.w;
}

__global__ void computeCov4DBackwardCUDA(int P,
	const glm::vec4* scaling_xyzt,
	const glm::vec4* rotation_l,
	const glm::vec4* rotation_r,
	const float* dL_dcov, 
	const glm::vec3* dL_dms, 
	const float* dL_dcov_t,
	glm::vec4* dL_dscaling_xyzt,
	glm::vec4* dL_drotation_l, 
	glm::vec4* dL_drotation_r)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	computeCov4DBackward(
		scaling_xyzt[idx],
		rotation_l[idx],
		rotation_r[idx],
		dL_dcov + idx * 6,
		dL_dms[idx],
		dL_dcov_t[idx],
		dL_dscaling_xyzt[idx],
		dL_drotation_l[idx],
		dL_drotation_r[idx]);
}


void BACKWARD::computeCov4DBackward(
	int P,
	const glm::vec4* scaling_xyzt,
	const glm::vec4* rotation_l,
	const glm::vec4* rotation_r,
	const float* dL_dcov, 
	const glm::vec3* dL_dms, 
	const float* dL_dcov_t,
	glm::vec4* dL_dscaling_xyzt,
	glm::vec4* dL_drotation_l, 
	glm::vec4* dL_drotation_r)
{
	computeCov4DBackwardCUDA << <(P + 255) / 256, 256 >> > (
		P,
		scaling_xyzt,
		rotation_l,
		rotation_r,
		dL_dcov,
		dL_dms,
		dL_dcov_t,
		dL_dscaling_xyzt,
		dL_drotation_l,
		dL_drotation_r);
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);
	
	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	glm::vec3 dL_dmean2;
	float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
	dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
	dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];
	
	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_dmean2;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ accum_alphas,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dpixel_alphas,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dabsmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - accum_alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_red = 0;
	float accum_rea = 0;
	float dL_dpixel[C];
	float dL_dpixel_depth;
	float dL_dpixel_alpha;
	if (inside) 
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dpixel_alpha = dL_dpixel_alphas[pix_id];
	}
	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = __expf(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			const float dep = collected_depths[j];
			accum_red = last_alpha * last_depth + (1.f - last_alpha) * accum_red;
			last_depth = dep;
			dL_dalpha += (dep-accum_red) * dL_dpixel_depth;
			atomicAdd(&(dL_ddepths[global_id]), dpixel_depth_ddepth * dL_dpixel_depth);
			
			accum_rea = last_alpha + (1.f - last_alpha) * accum_rea;
			dL_dalpha += (1 - accum_rea) * dL_dpixel_alpha;


			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Set background depth value == 0, thus no contribution for
			// dL_dalpha

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			const float dL_dmean2D_x = dL_dG * dG_ddelx * ddelx_dx;
			const float dL_dmean2D_y = dL_dG * dG_ddely * ddely_dy;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_x);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_y);
			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dabsmean2D[global_id].x, abs(dL_dmean2D_x));
			atomicAdd(&dL_dabsmean2D[global_id].y, abs(dL_dmean2D_y));

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAShared(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ accum_alphas,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dpixel_alphas,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dabsmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// Heuristic, gaussians are likely to be updated by the same block (same tile)
	// Thus it should be faster to first aggregate the gradients inside this block and update them to the global memory in just one go
	__shared__ float3 s_dL_dmean2D[BLOCK_SIZE];
	__shared__ float3 s_dL_dabsmean2D[BLOCK_SIZE];
	__shared__ float4 s_dL_dconic2D[BLOCK_SIZE];
	__shared__ float s_dL_dopacity[BLOCK_SIZE];
	__shared__ float s_dL_dcolors[C * BLOCK_SIZE];
	__shared__ float s_dL_ddepths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - accum_alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_red = 0;
	float accum_rea = 0;
	float dL_dpixel[C];
	float dL_dpixel_depth;
	float dL_dpixel_alpha;
	if (inside) 
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dpixel_alpha = dL_dpixel_alphas[pix_id];
	}
	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];

			// Shared gradient accumulation in this block
			s_dL_dmean2D[block.thread_rank()].x = 0.0f;
			s_dL_dmean2D[block.thread_rank()].y = 0.0f;
			s_dL_dabsmean2D[block.thread_rank()].x = 0.0f;
			s_dL_dabsmean2D[block.thread_rank()].y = 0.0f;
			s_dL_dconic2D[block.thread_rank()].x = 0.0f;
			s_dL_dconic2D[block.thread_rank()].y = 0.0f;
			s_dL_dconic2D[block.thread_rank()].w = 0.0f;
			for (int i = 0; i < C; i++)
				s_dL_dcolors[i * BLOCK_SIZE + block.thread_rank()] = 0.0f;
			s_dL_dopacity[block.thread_rank()] = 0.0f;
			s_dL_ddepths[block.thread_rank()] = 0.0f;
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = __expf(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(s_dL_dcolors[ch * BLOCK_SIZE + j]), dchannel_dcolor * dL_dchannel);
			}
			const float dep = collected_depths[j];
			accum_red = last_alpha * last_depth + (1.f - last_alpha) * accum_red;
			last_depth = dep;
			dL_dalpha += (dep-accum_red) * dL_dpixel_depth;
			atomicAdd(&(s_dL_ddepths[j]), dpixel_depth_ddepth * dL_dpixel_depth);
			
			accum_rea = last_alpha + (1.f - last_alpha) * accum_rea;
			dL_dalpha += (1 - accum_rea) * dL_dpixel_alpha;


			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Set background depth value == 0, thus no contribution for
			// dL_dalpha

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			const float dL_dmean2D_x = dL_dG * dG_ddelx * ddelx_dx;
			const float dL_dmean2D_y = dL_dG * dG_ddely * ddely_dy;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&s_dL_dmean2D[j].x, dL_dmean2D_x);
			atomicAdd(&s_dL_dmean2D[j].y, dL_dmean2D_y);
			atomicAdd(&s_dL_dabsmean2D[j].x, abs(dL_dmean2D_x));
			atomicAdd(&s_dL_dabsmean2D[j].y, abs(dL_dmean2D_y));

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&s_dL_dconic2D[j].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&s_dL_dconic2D[j].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&s_dL_dconic2D[j].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(s_dL_dopacity[j]), G * dL_dalpha);
		}
		block.sync();

		if (range.x + progress < range.y && s_dL_dmean2D[block.thread_rank()].x != 0.0) // not exactly zero
		{
			const int global_id = collected_id[block.thread_rank()];

			// Shared gradient accumulation in this block
			atomicAdd(&dL_dmean2D[global_id].x, s_dL_dmean2D[block.thread_rank()].x);
			atomicAdd(&dL_dmean2D[global_id].y, s_dL_dmean2D[block.thread_rank()].y);
			atomicAdd(&dL_dabsmean2D[global_id].x, s_dL_dabsmean2D[block.thread_rank()].x);
			atomicAdd(&dL_dabsmean2D[global_id].y, s_dL_dabsmean2D[block.thread_rank()].y);
			atomicAdd(&dL_dconic2D[global_id].x, s_dL_dconic2D[block.thread_rank()].x);
			atomicAdd(&dL_dconic2D[global_id].y, s_dL_dconic2D[block.thread_rank()].y);
			atomicAdd(&dL_dconic2D[global_id].w, s_dL_dconic2D[block.thread_rank()].w);
			for (int i = 0; i < C; i++)
				atomicAdd(&(dL_dcolors[global_id * C + i]), s_dL_dcolors[i * BLOCK_SIZE + block.thread_rank()]);
			atomicAdd(&(dL_dopacity[global_id]), s_dL_dopacity[block.thread_rank()]);
			atomicAdd(&(dL_ddepths[global_id]), s_dL_ddepths[block.thread_rank()]);
		}

	}
}


__device__ float warpReduceSum(float value) {
    auto warp = cg::coalesced_threads();
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        value += warp.shfl_down(value, offset);
    }
    return value;
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAWarp(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ accum_alphas,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dpixel_alphas,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dabsmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	auto warp = cg::coalesced_threads();

	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// By default, the thread rank should be computed as block.thread_index().x + BLOCK_X * block.thread_index().y
	// And warps should be arranged in a 2 * 16 (row * col) fashion
	// We want to make this more localized, so thread_ranks are remapped so that a warp is responsible for 4 * 8 smaller patch
	// const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	// Compute warp ID within the block
	// Here we divide the 16x16 block into 8 warps, each warp covers a 4x8 area.
	// Warp IDs are assigned row-wise.
	const int warpId = (ty / 4) * 2 + (tx / 8);
	const int laneId = (ty % 4) * 8 + (tx % 8);

	// Compute thread's position within its warp
	// Threads are linearly indexed within each warp from 0 to 31, using row-major ordering within the 4x8 block
	const int local_warp_x = (warpId % 2) * 8;
	const int local_warp_y = (warpId / 2) * 4;

	const uint2 pix = {
		pix_min.x + local_warp_x + (laneId % 8),
		pix_min.y + local_warp_y + (laneId / 8)
	};

	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// // Heuristic, gaussians are likely to be updated by the same block (same tile)
	// // Thus it should be faster to first aggregate the gradients inside this block and update them to the global memory in just one go
	// __shared__ float3 s_dL_dmean2D[BLOCK_SIZE];
	// __shared__ float3 s_dL_dabsmean2D[BLOCK_SIZE];
	// __shared__ float4 s_dL_dconic2D[BLOCK_SIZE];
	// __shared__ float s_dL_dopacity[BLOCK_SIZE];
	// __shared__ float s_dL_dcolors[C * BLOCK_SIZE];
	// __shared__ float s_dL_ddepths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - accum_alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_red = 0;
	float accum_rea = 0;
	float dL_dpixel[C];
	float dL_dpixel_depth;
	float dL_dpixel_alpha;
	if (inside) 
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dpixel_alpha = dL_dpixel_alphas[pix_id];
	}
	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	auto local_rank = block.thread_rank();

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + local_rank;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[local_rank] = coll_id;
			collected_xy[local_rank] = points_xy_image[coll_id];
			collected_conic_opacity[local_rank] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + local_rank] = colors[coll_id * C + i];
			collected_depths[local_rank] = depths[coll_id];

			// // Shared gradient accumulation in this block
			// s_dL_dmean2D[local_rank].x = 0.0f;
			// s_dL_dmean2D[local_rank].y = 0.0f;
			// s_dL_dabsmean2D[local_rank].x = 0.0f;
			// s_dL_dabsmean2D[local_rank].y = 0.0f;
			// s_dL_dconic2D[local_rank].x = 0.0f;
			// s_dL_dconic2D[local_rank].y = 0.0f;
			// s_dL_dconic2D[local_rank].w = 0.0f;
			// for (int i = 0; i < C; i++)
			// 	s_dL_dcolors[i * BLOCK_SIZE + local_rank] = 0.0f;
			// s_dL_dopacity[local_rank] = 0.0f;
			// s_dL_ddepths[local_rank] = 0.0f;
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			float2 xy = { 0.0f };
			float2 d = { 0.0f };
			float4 con_o = { 0.0f };
			float power = 0.0f;
			float G = 0.0f;
			float alpha = 0.0f;
			float dchannel_dcolor = 0.0f;
			float dpixel_depth_ddepth = 0.0f;
			float dL_dalpha = 0.0f;
			float dep = 0.0f;
			float bg_dot_dpixel = 0.0f;
			float dL_dG = 0.0f;
			float gdx = 0.0f;
			float gdy = 0.0f;
			float dG_ddelx = 0.0f;
			float dG_ddely = 0.0f;

			float w_dL_dcolors[C] = { 0.0f };
			float w_dL_ddepths = 0.0f;
			float2 w_dL_dmean2D = { 0.0f };
			float2 w_dL_dabsmean2D = { 0.0f };
			float4 w_dL_dconic2D = { 0.0f };
			float w_dL_dopacity = 0.0f;

			int global_id;
			bool early_stop = false;

			// if (done) {
			// 	early_stop = true;
			// 	goto reduce;
			// }

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor) {
				early_stop = true;
				goto reduce;
			}
				// continue;

			// Compute blending values, as before.
			xy = collected_xy[j];
			d = { xy.x - pixf.x, xy.y - pixf.y };
			con_o = collected_conic_opacity[j];
			power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) {
				early_stop = true;
				goto reduce; // early stopping
			}
				// continue;

			G = __expf(power);
			alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) {
				early_stop = true;
				goto reduce; // early stopping
			}
				// continue;

			T = T / (1.f - alpha);
			dchannel_dcolor = alpha * T;
			dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				// atomicAdd(&(s_dL_dcolors[ch * BLOCK_SIZE + j]), dchannel_dcolor * dL_dchannel);
				w_dL_dcolors[ch] = dchannel_dcolor * dL_dchannel;
			}
			dep = collected_depths[j];
			accum_red = last_alpha * last_depth + (1.f - last_alpha) * accum_red;
			last_depth = dep;
			dL_dalpha += (dep-accum_red) * dL_dpixel_depth;
			// atomicAdd(&(s_dL_ddepths[j]), dpixel_depth_ddepth * dL_dpixel_depth);
			w_dL_ddepths = dpixel_depth_ddepth * dL_dpixel_depth;

			accum_rea = last_alpha + (1.f - last_alpha) * accum_rea;
			dL_dalpha += (1 - accum_rea) * dL_dpixel_alpha;

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Set background depth value == 0, thus no contribution for
			// dL_dalpha

			// Helpful reusable temporary variables
			dL_dG = con_o.w * dL_dalpha;
			gdx = G * d.x;
			gdy = G * d.y;
			dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			// atomicAdd(&s_dL_dmean2D[j].x, dL_dG * dG_ddelx * ddelx_dx);
			// atomicAdd(&s_dL_dmean2D[j].y, dL_dG * dG_ddely * ddely_dy);
			w_dL_dmean2D.x = dL_dG * dG_ddelx * ddelx_dx;
			w_dL_dmean2D.y = dL_dG * dG_ddely * ddely_dy;
			w_dL_dabsmean2D.x = abs(w_dL_dmean2D.x);
			w_dL_dabsmean2D.y = abs(w_dL_dmean2D.y);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// atomicAdd(&s_dL_dconic2D[j].x, -0.5f * gdx * d.x * dL_dG);
			// atomicAdd(&s_dL_dconic2D[j].y, -0.5f * gdx * d.y * dL_dG);
			// atomicAdd(&s_dL_dconic2D[j].w, -0.5f * gdy * d.y * dL_dG);
			w_dL_dconic2D.x = -0.5f * gdx * d.x * dL_dG;
			w_dL_dconic2D.y = -0.5f * gdx * d.y * dL_dG;
			w_dL_dconic2D.w = -0.5f * gdy * d.y * dL_dG;

			// Update gradients w.r.t. opacity of the Gaussian
			// atomicAdd(&(s_dL_dopacity[j]), G * dL_dalpha);
			w_dL_dopacity = G * dL_dalpha;

		reduce:
			early_stop = warpReduceSum(early_stop);

			// If the whole warp votes for early stop, no need to do any further reduction or computation
			if (!early_stop) {
					
				// Call reduce sum and append results to __shared__ memory
				for (int ch = 0; ch < C; ch++) {
					w_dL_dcolors[ch] = warpReduceSum(w_dL_dcolors[ch]);
				}
				w_dL_ddepths = warpReduceSum(w_dL_ddepths);
				w_dL_dmean2D.x = warpReduceSum(w_dL_dmean2D.x);
				w_dL_dmean2D.y = warpReduceSum(w_dL_dmean2D.y);
				w_dL_dabsmean2D.x = warpReduceSum(w_dL_dabsmean2D.x);
				w_dL_dabsmean2D.y = warpReduceSum(w_dL_dabsmean2D.y);
				w_dL_dconic2D.x = warpReduceSum(w_dL_dconic2D.x);
				w_dL_dconic2D.y = warpReduceSum(w_dL_dconic2D.y);
				w_dL_dconic2D.w = warpReduceSum(w_dL_dconic2D.w);
				w_dL_dopacity = warpReduceSum(w_dL_dopacity);

				// Use a single thread from each warp to perform block level reduction
				if (local_rank % warp.size() == 0) {
					// for (int ch = 0; ch < C; ch++) {
					// 	atomicAdd(&(s_dL_dcolors[ch * BLOCK_SIZE + j]), w_dL_dcolors[ch]);
					// }
					// atomicAdd(&(s_dL_ddepths[j]), w_dL_ddepths);
					// atomicAdd(&s_dL_dmean2D[j].x, w_dL_dmean2D.x);
					// atomicAdd(&s_dL_dmean2D[j].y, w_dL_dmean2D.y);
					// atomicAdd(&s_dL_dabsmean2D[j].x, w_dL_dabsmean2D.x);
					// atomicAdd(&s_dL_dabsmean2D[j].y, w_dL_dabsmean2D.y);
					// atomicAdd(&s_dL_dconic2D[j].x, w_dL_dconic2D.x);
					// atomicAdd(&s_dL_dconic2D[j].y, w_dL_dconic2D.y);
					// atomicAdd(&s_dL_dconic2D[j].w, w_dL_dconic2D.w);
					// atomicAdd(&(s_dL_dopacity[j]), w_dL_dopacity);
					global_id = collected_id[j];

					// Shared gradient accumulation in this block
					for (int i = 0; i < C; i++)
						atomicAdd(&(dL_dcolors[global_id * C + i]), w_dL_dcolors[i]);
					atomicAdd(&dL_dmean2D[global_id].x, w_dL_dmean2D.x);
					atomicAdd(&dL_dmean2D[global_id].y, w_dL_dmean2D.y);
					atomicAdd(&dL_dabsmean2D[global_id].x, w_dL_dabsmean2D.x);
					atomicAdd(&dL_dabsmean2D[global_id].y, w_dL_dabsmean2D.y);
					atomicAdd(&dL_dconic2D[global_id].x, w_dL_dconic2D.x);
					atomicAdd(&dL_dconic2D[global_id].y, w_dL_dconic2D.y);
					atomicAdd(&dL_dconic2D[global_id].w, w_dL_dconic2D.w);
					atomicAdd(&(dL_dopacity[global_id]), w_dL_dopacity);
					atomicAdd(&(dL_ddepths[global_id]), w_dL_ddepths);
				}
			}
		}
		// block.sync();

		// if (range.x + progress < range.y)
		// {
		// 	const int global_id = point_list[range.y - progress - 1];

		// 	// Shared gradient accumulation in this block
		// 	atomicAdd(&dL_dmean2D[global_id].x, s_dL_dmean2D[local_rank].x);
		// 	atomicAdd(&dL_dmean2D[global_id].y, s_dL_dmean2D[local_rank].y);
		// 	atomicAdd(&dL_dabsmean2D[global_id].x, s_dL_dabsmean2D[local_rank].x);
		// 	atomicAdd(&dL_dabsmean2D[global_id].y, s_dL_dabsmean2D[local_rank].y);
		// 	atomicAdd(&dL_dconic2D[global_id].x, s_dL_dconic2D[local_rank].x);
		// 	atomicAdd(&dL_dconic2D[global_id].y, s_dL_dconic2D[local_rank].y);
		// 	atomicAdd(&dL_dconic2D[global_id].w, s_dL_dconic2D[local_rank].w);
		// 	for (int i = 0; i < C; i++)
		// 		atomicAdd(&(dL_dcolors[global_id * C + i]), s_dL_dcolors[i * BLOCK_SIZE + local_rank]);
		// 	atomicAdd(&(dL_dopacity[global_id]), s_dL_dopacity[local_rank]);
		// 	atomicAdd(&(dL_ddepths[global_id]), s_dL_ddepths[local_rank]);
		// }

	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* accum_alphas,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixel_depths,
	const float* dL_dpixel_alphas,
	float3* dL_dmean2D,
	float3* dL_dabsmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		accum_alphas,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dpixel_alphas,
		dL_dmean2D,
		dL_dabsmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths);
}