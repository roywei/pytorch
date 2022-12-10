/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_X86SIMD_VEC_S32X4_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_X86SIMD_VEC_S32X4_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_s32x4 {
public:
    union {
        __m128i v;
        int32_t raw[4];
    } __attribute__((aligned(16)));

    INLINE vec_s32x4() = default;
    INLINE vec_s32x4(int32_t f) { v = _mm_set1_epi32(f); }
    INLINE vec_s32x4(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
        v = _mm_setr_epi32(i0, i1, i2, i3);
    }
    INLINE vec_s32x4(__m128i const &x) { v = x; }

    static INLINE vec_s32x4 load(const int32_t *p) {
        return _mm_loadu_si128((const __m128i *)p);
    }
    static INLINE vec_s32x4 load_aligned(const int32_t *p) {
        return _mm_load_si128((const __m128i *)p);
    }
    static INLINE void store(vec_s32x4 v, int32_t *p) {
        _mm_storeu_si128((__m128i *)p, v.v);
    }
    static INLINE void store_aligned(vec_s32x4 v, int32_t *p) {
        _mm_store_si128((__m128i *)p, v.v);
    }
};

INLINE vec_s32x4 operator+(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_add_epi32(a.v, b.v);
}

INLINE vec_s32x4 operator-(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_sub_epi32(a.v, b.v);
}
INLINE vec_s32x4 operator-(vec_s32x4 const &a) {
    return _mm_sub_epi32(_mm_setzero_si128(), a.v);
}

INLINE vec_s32x4 operator*(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_mullo_epi32(a.v, b.v);
}

// INLINE vec_s32x4 operator/(vec_s32x4 const &a, vec_s32x4 const &b) {
//     return _mm_div_epi32(a.v, b.v);
// }

INLINE vec_s32x4 operator~(vec_s32x4 const &a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
}
INLINE vec_s32x4 operator&(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_and_si128(a.v, b.v);
}
INLINE vec_s32x4 operator|(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_or_si128(a.v, b.v);
}
INLINE vec_s32x4 operator^(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_xor_si128(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask8 operator!(vec_s32x4 const &a) {
    return _mm_cmp_epi32_mask(a.v, _mm_setzero_si128(), _MM_CMPINT_EQ);
}
INLINE __mmask8 operator==(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask8 operator!=(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask8 operator>(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask8 operator<(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask8 operator>=(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask8 operator<=(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_s32x4 sc_select(
        __mmask8 mask, vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_mask_blend_epi32(mask, b.v, a.v);
}
#endif

INLINE vec_s32x4 operator<<(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_sllv_epi32(a.v, b.v);
}
INLINE vec_s32x4 operator>>(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_srlv_epi32(a.v, b.v);
}

// operator /

INLINE vec_s32x4 sc_max(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_max_epi32(a.v, b.v);
}
INLINE vec_s32x4 sc_min(vec_s32x4 const &a, vec_s32x4 const &b) {
    return _mm_min_epi32(a.v, b.v);
}

INLINE vec_s32x4 sc_abs(vec_s32x4 const &a) {
    return _mm_abs_epi32(a.v);
}
#endif
