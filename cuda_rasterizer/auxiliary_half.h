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

#ifndef CUDA_RASTERIZER_AUXILIARY_HALF_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_HALF_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include "cuda_fp16.h"
#include <glm/glm.hpp>
#include <limits>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Add the other vector half types
struct __half1 { __half x; };
struct __align__(4) __half3 { __half x, y, z; };
struct __align__(4) __half4 { __half x, y, z, w; };

// __host__ __device__ __forceinline__ __half operator __half(const int &i) { return __int2half_rn(i); }
// __host__ __device__ __forceinline__ __half operator __half(const float &f) { return __float2half(f); }
// __host__ __device__ __forceinline__ __half operator __half(const double &f) { return __double2half(f); }

__host__ __device__ __forceinline__ __half 	operator+ ( const __half& h ) { return h; }
__host__ __device__ __forceinline__ __half 	operator- ( const __half& h ) { return __hneg(h); }
__host__ __device__ __forceinline__ __half 	operator* ( const __half& lh, const __half& rh ) { return __hmul(lh, rh); }
__host__ __device__ __forceinline__ __half 	operator+ ( const __half& lh, const __half& rh ) { return __hadd(lh, rh); }
__host__ __device__ __forceinline__ __half 	operator- ( const __half& lh, const __half& rh ) { return __hsub(lh, rh); }
__host__ __device__ __forceinline__ __half 	operator/ ( const __half& lh, const __half& rh ) { return __hdiv(lh, rh); }

__host__ __device__ __forceinline__ __half 	operator* ( const float& lh, const __half& rh ) { return __hmul(__float2half(lh), rh); }
__host__ __device__ __forceinline__ __half 	operator+ ( const float& lh, const __half& rh ) { return __hadd(__float2half(lh), rh); }
__host__ __device__ __forceinline__ __half 	operator- ( const float& lh, const __half& rh ) { return __hsub(__float2half(lh), rh); }
__host__ __device__ __forceinline__ __half 	operator/ ( const float& lh, const __half& rh ) { return __hdiv(__float2half(lh), rh); }

__host__ __device__ __forceinline__ __half 	operator* ( const __half& lh, const float& rh ) { return __hmul(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ __half 	operator+ ( const __half& lh, const float& rh ) { return __hadd(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ __half 	operator- ( const __half& lh, const float& rh ) { return __hsub(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ __half 	operator/ ( const __half& lh, const float& rh ) { return __hdiv(lh, __float2half(rh)); }

__host__ __device__ __forceinline__ bool 	operator!= ( const __half& lh, const __half& rh ) { return __hne(lh, rh); }
__host__ __device__ __forceinline__ bool 	operator<  ( const __half& lh, const __half& rh ) { return __hlt(lh, rh); }
__host__ __device__ __forceinline__ bool 	operator<= ( const __half& lh, const __half& rh ) { return __hle(lh, rh); }
__host__ __device__ __forceinline__ bool 	operator== ( const __half& lh, const __half& rh ) { return __heq(lh, rh); }
__host__ __device__ __forceinline__ bool 	operator>  ( const __half& lh, const __half& rh ) { return __hgt(lh, rh); }
__host__ __device__ __forceinline__ bool 	operator>= ( const __half& lh, const __half& rh ) { return __hge(lh, rh); }

__host__ __device__ __forceinline__ bool 	operator!= ( const float& lh, const __half& rh ) { return __hne(__float2half(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator<  ( const float& lh, const __half& rh ) { return __hlt(__float2half(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator<= ( const float& lh, const __half& rh ) { return __hle(__float2half(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator== ( const float& lh, const __half& rh ) { return __heq(__float2half(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator>  ( const float& lh, const __half& rh ) { return __hgt(__float2half(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator>= ( const float& lh, const __half& rh ) { return __hge(__float2half(lh), rh); }

__host__ __device__ __forceinline__ bool 	operator!= ( const __half& lh, const float& rh ) { return __hne(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ bool 	operator<  ( const __half& lh, const float& rh ) { return __hlt(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ bool 	operator<= ( const __half& lh, const float& rh ) { return __hle(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ bool 	operator== ( const __half& lh, const float& rh ) { return __heq(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ bool 	operator>  ( const __half& lh, const float& rh ) { return __hgt(lh, __float2half(rh)); }
__host__ __device__ __forceinline__ bool 	operator>= ( const __half& lh, const float& rh ) { return __hge(lh, __float2half(rh)); }

__host__ __device__ __forceinline__ bool 	operator!= ( const int& lh, const __half& rh ) { return __hne(__int2half_rn(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator<  ( const int& lh, const __half& rh ) { return __hlt(__int2half_rn(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator<= ( const int& lh, const __half& rh ) { return __hle(__int2half_rn(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator== ( const int& lh, const __half& rh ) { return __heq(__int2half_rn(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator>  ( const int& lh, const __half& rh ) { return __hgt(__int2half_rn(lh), rh); }
__host__ __device__ __forceinline__ bool 	operator>= ( const int& lh, const __half& rh ) { return __hge(__int2half_rn(lh), rh); }

__host__ __device__ __forceinline__ bool 	operator!= ( const __half& lh, const int& rh ) { return __hne(lh, __int2half_rn(rh)); }
__host__ __device__ __forceinline__ bool 	operator<  ( const __half& lh, const int& rh ) { return __hlt(lh, __int2half_rn(rh)); }
__host__ __device__ __forceinline__ bool 	operator<= ( const __half& lh, const int& rh ) { return __hle(lh, __int2half_rn(rh)); }
__host__ __device__ __forceinline__ bool 	operator== ( const __half& lh, const int& rh ) { return __heq(lh, __int2half_rn(rh)); }
__host__ __device__ __forceinline__ bool 	operator>  ( const __half& lh, const int& rh ) { return __hgt(lh, __int2half_rn(rh)); }
__host__ __device__ __forceinline__ bool 	operator>= ( const __half& lh, const int& rh ) { return __hge(lh, __int2half_rn(rh)); }

__host__ __device__ __forceinline__ __half 	abs ( const __half& h ) { return __habs(h); }
__host__ __device__ __forceinline__ __half 	exp ( const __half& h ) { return hexp(h); }
__host__ __device__ __forceinline__ __half 	log ( const __half& h ) { return hlog(h); }
__host__ __device__ __forceinline__ __half 	sqrt ( const __half& h ) { return hsqrt(h); }
__host__ __device__ __forceinline__ __half 	ceil ( const __half& h ) { return hceil(h); }
__host__ __device__ __forceinline__ __half 	min ( const __half& lh, const __half& rh ) { return __hmin(lh, rh); }
__host__ __device__ __forceinline__ __half 	max ( const __half& lh, const __half& rh ) { return __hmax(lh, rh); }


class Half {
public:
    __half value;

    // Constructor
    __device__ __host__ __forceinline__ Half(const __half& val = __float2half(0.0f)) : value(val) {}
    explicit __device__ __host__ __forceinline__ Half(const Half& val) : value(val.value) {}
    explicit __device__ __host__ __forceinline__ Half(const int& val) : value(__int2half_rn(val)) {}
    explicit __device__ __host__ __forceinline__ Half(const float& val) : value(__float2half(val)) {}
    explicit __device__ __host__ __forceinline__ Half(const double& val) : value(__double2half(val)) {}
    explicit __device__ __host__ __forceinline__ Half(const unsigned int& val) : value(__uint2half_rn(val)) {}

    // Conversion
    __device__ __host__ __forceinline__ operator __half() const { return value; }
    explicit __device__ __host__ __forceinline__ operator int() const { return __half2int_rn(value); }
    explicit __device__ __host__ __forceinline__ operator float() const { return __half2float(value); }
	__device__ __host__ __forceinline__ static Half bitcast(uint16_t x) { return Half(__ushort_as_half(x));	}

    // __device__ __host__ __forceinline__ Half& operator = (const Half& h) { value = h.value; return *this; }
    __device__ __host__ __forceinline__ Half& operator +=(const Half& h) { value = value + h.value; return *this; }
    __device__ __host__ __forceinline__ Half& operator -=(const Half& h) { value = value - h.value; return *this; }
    __device__ __host__ __forceinline__ Half& operator *=(const Half& h) { value = value * h.value; return *this; }
    __device__ __host__ __forceinline__ Half& operator /=(const Half& h) { value = value / h.value; return *this; }

    __device__ __host__ __forceinline__ Half& operator +=(const float& h) { value = value + h; return *this; }
    __device__ __host__ __forceinline__ Half& operator -=(const float& h) { value = value - h; return *this; }
    __device__ __host__ __forceinline__ Half& operator *=(const float& h) { value = value * h; return *this; }
    __device__ __host__ __forceinline__ Half& operator /=(const float& h) { value = value / h; return *this; }

};

__host__ __device__ __forceinline__ Half 	operator+ ( const Half& h ) { return h; }
__host__ __device__ __forceinline__ Half 	operator- ( const Half& h ) { return Half(__hneg(h.value)); }
__host__ __device__ __forceinline__ Half 	operator* ( const Half& lh, const Half& rh ) { return Half(__hmul(lh.value, rh.value)); }
__host__ __device__ __forceinline__ Half 	operator+ ( const Half& lh, const Half& rh ) { return Half(__hadd(lh.value, rh.value)); }
__host__ __device__ __forceinline__ Half 	operator- ( const Half& lh, const Half& rh ) { return Half(__hsub(lh.value, rh.value)); }
__host__ __device__ __forceinline__ Half 	operator/ ( const Half& lh, const Half& rh ) { return Half(__hdiv(lh.value, rh.value)); }

__host__ __device__ __forceinline__ bool 	operator!= ( const Half& lh, const Half& rh ) { return __hne(lh.value, rh.value); }
__host__ __device__ __forceinline__ bool 	operator<  ( const Half& lh, const Half& rh ) { return __hlt(lh.value, rh.value); }
__host__ __device__ __forceinline__ bool 	operator<= ( const Half& lh, const Half& rh ) { return __hle(lh.value, rh.value); }
__host__ __device__ __forceinline__ bool 	operator== ( const Half& lh, const Half& rh ) { return __heq(lh.value, rh.value); }
__host__ __device__ __forceinline__ bool 	operator>  ( const Half& lh, const Half& rh ) { return __hgt(lh.value, rh.value); }
__host__ __device__ __forceinline__ bool 	operator>= ( const Half& lh, const Half& rh ) { return __hge(lh.value, rh.value); }

__host__ __device__ __forceinline__ Half 	operator* ( const Half& lh, const __half& rh ) { return Half(__hmul(lh.value, rh)); }
__host__ __device__ __forceinline__ Half 	operator+ ( const Half& lh, const __half& rh ) { return Half(__hadd(lh.value, rh)); }
__host__ __device__ __forceinline__ Half 	operator- ( const Half& lh, const __half& rh ) { return Half(__hsub(lh.value, rh)); }
__host__ __device__ __forceinline__ Half 	operator/ ( const Half& lh, const __half& rh ) { return Half(__hdiv(lh.value, rh)); }

__host__ __device__ __forceinline__ bool 	operator!= ( const Half& lh, const __half& rh ) { return __hne(lh.value, rh); }
__host__ __device__ __forceinline__ bool 	operator<  ( const Half& lh, const __half& rh ) { return __hlt(lh.value, rh); }
__host__ __device__ __forceinline__ bool 	operator<= ( const Half& lh, const __half& rh ) { return __hle(lh.value, rh); }
__host__ __device__ __forceinline__ bool 	operator== ( const Half& lh, const __half& rh ) { return __heq(lh.value, rh); }
__host__ __device__ __forceinline__ bool 	operator>  ( const Half& lh, const __half& rh ) { return __hgt(lh.value, rh); }
__host__ __device__ __forceinline__ bool 	operator>= ( const Half& lh, const __half& rh ) { return __hge(lh.value, rh); }

__host__ __device__ __forceinline__ __half 	operator* ( const __half& lh, const Half& rh ) { return __hmul(lh, rh.value); }
__host__ __device__ __forceinline__ __half 	operator+ ( const __half& lh, const Half& rh ) { return __hadd(lh, rh.value); }
__host__ __device__ __forceinline__ __half 	operator- ( const __half& lh, const Half& rh ) { return __hsub(lh, rh.value); }
__host__ __device__ __forceinline__ __half 	operator/ ( const __half& lh, const Half& rh ) { return __hdiv(lh, rh.value); }

__host__ __device__ __forceinline__ bool 	operator!= ( const __half& lh, const Half& rh ) { return __hne(lh, rh.value); }
__host__ __device__ __forceinline__ bool 	operator<  ( const __half& lh, const Half& rh ) { return __hlt(lh, rh.value); }
__host__ __device__ __forceinline__ bool 	operator<= ( const __half& lh, const Half& rh ) { return __hle(lh, rh.value); }
__host__ __device__ __forceinline__ bool 	operator== ( const __half& lh, const Half& rh ) { return __heq(lh, rh.value); }
__host__ __device__ __forceinline__ bool 	operator>  ( const __half& lh, const Half& rh ) { return __hgt(lh, rh.value); }
__host__ __device__ __forceinline__ bool 	operator>= ( const __half& lh, const Half& rh ) { return __hge(lh, rh.value); }

__host__ __device__ __forceinline__ Half 	operator* ( const Half& lh, const float& rh ) { return Half(lh.value * rh); }
__host__ __device__ __forceinline__ Half 	operator+ ( const Half& lh, const float& rh ) { return Half(lh.value + rh); }
__host__ __device__ __forceinline__ Half 	operator- ( const Half& lh, const float& rh ) { return Half(lh.value - rh); }
__host__ __device__ __forceinline__ Half 	operator/ ( const Half& lh, const float& rh ) { return Half(lh.value / rh); }

__host__ __device__ __forceinline__ bool 	operator!= ( const Half& lh, const float& rh ) { return lh.value != rh; }
__host__ __device__ __forceinline__ bool 	operator<  ( const Half& lh, const float& rh ) { return lh.value <  rh; }
__host__ __device__ __forceinline__ bool 	operator<= ( const Half& lh, const float& rh ) { return lh.value <= rh; }
__host__ __device__ __forceinline__ bool 	operator== ( const Half& lh, const float& rh ) { return lh.value == rh; }
__host__ __device__ __forceinline__ bool 	operator>  ( const Half& lh, const float& rh ) { return lh.value >  rh; }
__host__ __device__ __forceinline__ bool 	operator>= ( const Half& lh, const float& rh ) { return lh.value >= rh; }

__host__ __device__ __forceinline__ float 	operator* ( const float& lh, const Half& rh ) { return __half2float(lh * rh.value); }
__host__ __device__ __forceinline__ float 	operator+ ( const float& lh, const Half& rh ) { return __half2float(lh + rh.value); }
__host__ __device__ __forceinline__ float 	operator- ( const float& lh, const Half& rh ) { return __half2float(lh - rh.value); }
__host__ __device__ __forceinline__ float 	operator/ ( const float& lh, const Half& rh ) { return __half2float(lh / rh.value); }

__host__ __device__ __forceinline__ bool 	operator!= ( const float& lh, const Half& rh ) { return lh != rh.value; }
__host__ __device__ __forceinline__ bool 	operator<  ( const float& lh, const Half& rh ) { return lh <  rh.value; }
__host__ __device__ __forceinline__ bool 	operator<= ( const float& lh, const Half& rh ) { return lh <= rh.value; }
__host__ __device__ __forceinline__ bool 	operator== ( const float& lh, const Half& rh ) { return lh == rh.value; }
__host__ __device__ __forceinline__ bool 	operator>  ( const float& lh, const Half& rh ) { return lh >  rh.value; }
__host__ __device__ __forceinline__ bool 	operator>= ( const float& lh, const Half& rh ) { return lh >= rh.value; }

__host__ __device__ __forceinline__ Half 	operator* ( const Half& lh, const int& rh ) { return Half(lh.value * rh); }
__host__ __device__ __forceinline__ Half 	operator+ ( const Half& lh, const int& rh ) { return Half(lh.value + rh); }
__host__ __device__ __forceinline__ Half 	operator- ( const Half& lh, const int& rh ) { return Half(lh.value - rh); }
__host__ __device__ __forceinline__ Half 	operator/ ( const Half& lh, const int& rh ) { return Half(lh.value / rh); }

__host__ __device__ __forceinline__ bool 	operator!= ( const Half& lh, const int& rh ) { return lh.value != rh; }
__host__ __device__ __forceinline__ bool 	operator<  ( const Half& lh, const int& rh ) { return lh.value <  rh; }
__host__ __device__ __forceinline__ bool 	operator<= ( const Half& lh, const int& rh ) { return lh.value <= rh; }
__host__ __device__ __forceinline__ bool 	operator== ( const Half& lh, const int& rh ) { return lh.value == rh; }
__host__ __device__ __forceinline__ bool 	operator>  ( const Half& lh, const int& rh ) { return lh.value >  rh; }
__host__ __device__ __forceinline__ bool 	operator>= ( const Half& lh, const int& rh ) { return lh.value >= rh; }

__host__ __device__ __forceinline__ int 	operator* ( const int& lh, const Half& rh ) { return __half2int_rn(lh * rh.value); }
__host__ __device__ __forceinline__ int 	operator+ ( const int& lh, const Half& rh ) { return __half2int_rn(lh + rh.value); }
__host__ __device__ __forceinline__ int 	operator- ( const int& lh, const Half& rh ) { return __half2int_rn(lh - rh.value); }
__host__ __device__ __forceinline__ int 	operator/ ( const int& lh, const Half& rh ) { return __half2int_rn(lh / rh.value); }

__host__ __device__ __forceinline__ bool 	operator!= ( const int& lh, const Half& rh ) { return lh != rh.value; }
__host__ __device__ __forceinline__ bool 	operator<  ( const int& lh, const Half& rh ) { return lh <  rh.value; }
__host__ __device__ __forceinline__ bool 	operator<= ( const int& lh, const Half& rh ) { return lh <= rh.value; }
__host__ __device__ __forceinline__ bool 	operator== ( const int& lh, const Half& rh ) { return lh == rh.value; }
__host__ __device__ __forceinline__ bool 	operator>  ( const int& lh, const Half& rh ) { return lh >  rh.value; }
__host__ __device__ __forceinline__ bool 	operator>= ( const int& lh, const Half& rh ) { return lh >= rh.value; }

__host__ __device__ __forceinline__ __half 	min ( const __half& lh, const float& rh ) { return __hmin(lh, Half(rh)); }
__host__ __device__ __forceinline__ __half 	max ( const __half& lh, const float& rh ) { return __hmax(lh, Half(rh)); }
__host__ __device__ __forceinline__ float 	min ( const float& lh, const __half& rh ) { return min(lh, Half(rh)); }
__host__ __device__ __forceinline__ float 	max ( const float& lh, const __half& rh ) { return max(lh, Half(rh)); }


template <>
struct std::numeric_limits<Half> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = true;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 10;

  static Half min() { return Half::bitcast(0x0001); }

  static Half lowest() { return Half::bitcast(0xfbff); }

  static Half max() { return Half::bitcast(0x7bff); }

  static Half epsilon() { return Half::bitcast(0x1800); }

  static Half round_error() { return Half(0.5f); }

  static Half infinity() { return Half::bitcast(0x7c00); }

  static Half quiet_NaN() { return Half::bitcast(0x7fff); }

  static Half signaling_NaN() { return Half::bitcast(0x7fff); }

  static Half denorm_min() { return Half::bitcast(0x0001); }
};

typedef glm::vec<3, Half, glm::lowp> glm_half3;
typedef glm::vec<4, Half, glm::lowp> glm_half4;
typedef glm::mat<3, 3, Half, glm::lowp> glm_half33;
typedef glm::mat<4, 4, Half, glm::lowp> glm_half44;

__host__ __device__ __forceinline__ glm_half3	operator* ( const glm_half3& lh, const __half& rh ) { return glm_half3(lh.x * rh, lh.y * rh, lh.z * rh); }
__host__ __device__ __forceinline__ glm_half3	operator+ ( const glm_half3& lh, const __half& rh ) { return glm_half3(lh.x + rh, lh.y * rh, lh.z + rh); }
__host__ __device__ __forceinline__ glm_half3	operator- ( const glm_half3& lh, const __half& rh ) { return glm_half3(lh.x - rh, lh.y * rh, lh.z - rh); }
__host__ __device__ __forceinline__ glm_half3	operator/ ( const glm_half3& lh, const __half& rh ) { return glm_half3(lh.x / rh, lh.y * rh, lh.z / rh); }

__host__ __device__ __forceinline__ glm_half3	operator* ( const __half& lh, const glm_half3& rh ) { return glm_half3(lh * rh.x, lh * rh.y, lh * rh.z); }
__host__ __device__ __forceinline__ glm_half3	operator+ ( const __half& lh, const glm_half3& rh ) { return glm_half3(lh + rh.x, lh * rh.y, lh + rh.z); }
__host__ __device__ __forceinline__ glm_half3	operator- ( const __half& lh, const glm_half3& rh ) { return glm_half3(lh - rh.x, lh * rh.y, lh - rh.z); }
__host__ __device__ __forceinline__ glm_half3	operator/ ( const __half& lh, const glm_half3& rh ) { return glm_half3(lh / rh.x, lh * rh.y, lh / rh.z); }

// Spherical harmonics coefficients
__device__ const __half SH_C0_HALF = Half(0.28209479177387814f);
__device__ const __half SH_C1_HALF = Half(0.4886025119029199f);
__device__ const __half SH_C2_HALF[] = {
	Half(1.0925484305920792f),
	Half(-1.0925484305920792f),
	Half(0.31539156525252005f),
	Half(-1.0925484305920792f),
	Half(0.5462742152960396f)
};
__device__ const __half SH_C3_HALF[] = {
	Half(-0.5900435899266435f),
	Half(2.890611442640554f),
	Half(-0.4570457994644658f),
	Half(0.3731763325901154f),
	Half(-0.4570457994644658f),
	Half(1.445305721320277f),
	Half(-0.5900435899266435f)
};

__forceinline__ __device__ __half ndc2PixHalf(__half v, int S)
{
	return __float2half(((__half2float(v) + 1.0f) * S - 1.0f) * 0.5f);
}

__forceinline__ __device__ void getRectHalf(const __half2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((__half2float(p.x) - max_radius - 0.5) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((__half2float(p.y) - max_radius - 0.5) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((__half2float(p.x) + max_radius + BLOCK_X - 1 + 0.5) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((__half2float(p.y) + max_radius + BLOCK_Y - 1 + 0.5) / BLOCK_Y)))
	};
}

__forceinline__ __device__ __half3 transformPoint4x3Half(const __half3& p, const __half* matrix)
{
	__half3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ __half4 transformPoint4x4Half(const __half3& p, const __half* matrix)
{
	__half4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ __half3 transformVec4x3Half(const __half3& p, const __half* matrix)
{
	__half3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ __half3 transformVec4x3TransposeHalf(const __half3& p, const __half* matrix)
{
	__half3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ __half sigmoid_half(__half x)
{
	return __float2half(1.0f / (1.0f + expf(__half2float(-x))));
}

__forceinline__ __device__ __half dist2_half(__half2 d)
{
	return d.x * d.x + d.y * d.y;
}

__forceinline__ __device__ bool in_frustum_half(int idx,
	const __half* orig_points,
	const __half* viewmatrix,
	const __half* projmatrix,
	bool prefiltered,
	__half3& p_view,
	const __half padding = __float2half(0.01f) // padding in ndc space
	)
{
	__half3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	__half4 p_hom = transformPoint4x4Half(p_orig, projmatrix);
	__half p_w = __float2half(1.0f / (__half2float(p_hom.w) + 0.0000001f));
	__half3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3Half(p_orig, viewmatrix); // write this outside

	return (p_proj.z > __float2half(-1.f) - padding) &&
		   (p_proj.z < __float2half(1.f) + padding) &&
		   (p_proj.x > __float2half(-1.f) - padding) &&
		   (p_proj.x < __float2half(1.f) + padding) &&
		   (p_proj.y > __float2half(-1.f) - padding) &&
		   (p_proj.y < __float2half(1.f) + padding);
}
#endif
