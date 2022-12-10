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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_X86SIMD_VEC_U16X32_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_X86SIMD_VEC_U16X32_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
#ifdef __AVX512F__
class vec_u16x32 {
public:
    union {
        __m512i v;
        uint16_t raw[32];
    } __attribute__((aligned(64)));

    INLINE vec_u16x32() = default;
    INLINE vec_u16x32(uint16_t f) { v = _mm512_set1_epi16(f); }
    INLINE vec_u16x32(__m512i const &x) { v = x; }

    static INLINE vec_u16x32 load(const uint16_t *p) {
        return _mm512_loadu_si512((const __m512i *)p);
    }
    static INLINE vec_u16x32 load_aligned(const uint16_t *p) {
        return _mm512_load_si512((const __m512i *)p);
    }
    static INLINE void store(vec_u16x32 v, uint16_t *p) {
        _mm512_storeu_si512((__m512i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u16x32 v, uint16_t *p) {
        _mm512_store_si512((__m512i *)p, v.v);
    }
};

INLINE vec_u16x32 operator+(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_adds_epu16(a.v, b.v);
}

INLINE vec_u16x32 operator-(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_subs_epu16(a.v, b.v);
}
INLINE vec_u16x32 operator-(vec_u16x32 const &a) {
    return _mm512_subs_epu16(_mm512_setzero_si512(), a.v);
}

// _mm_mulhi_epu16 was supported, but the high 16 bits of result was return.
// INLINE vec_u16x32 operator*(vec_u16x32 const &a, vec_u16x32 const &b) {
//     return _mm512_mulhi_epu16(a.v, b.v);
// }

// INLINE vec_u16x32 operator/(vec_u16x32 const &a, vec_u16x32 const &b) {
//     return _mm512_div_epu16(a.v, b.v);
// }

INLINE vec_u16x32 operator~(vec_u16x32 const &a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi16(0xFFFF));
}
INLINE vec_u16x32 operator&(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_and_si512(a.v, b.v);
}
INLINE vec_u16x32 operator|(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_or_si512(a.v, b.v);
}
INLINE vec_u16x32 operator^(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_xor_si512(a.v, b.v);
}

INLINE __mmask32 operator!(vec_u16x32 const &a) {
    return _mm512_cmp_epu16_mask(a.v, _mm512_setzero_si512(), _MM_CMPINT_EQ);
}
INLINE __mmask32 operator==(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask32 operator!=(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask32 operator>(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask32 operator<(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask32 operator>=(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask32 operator<=(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u16x32 sc_select(
        __mmask32 mask, vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_mask_blend_epi16(mask, b.v, a.v);
}

INLINE vec_u16x32 operator<<(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_sllv_epi16(a.v, b.v);
}
INLINE vec_u16x32 operator>>(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_srlv_epi16(a.v, b.v);
}

INLINE vec_u16x32 sc_max(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_max_epu16(a.v, b.v);
}
INLINE vec_u16x32 sc_min(vec_u16x32 const &a, vec_u16x32 const &b) {
    return _mm512_min_epu16(a.v, b.v);
}
#endif
#endif
