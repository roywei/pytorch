/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_DNNL_UTILS_HPP
#define BACKEND_DNNL_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdlib.h>
#include <utility>
#include <vector>

#include "backend/dnnl/common.hpp"

#include "utils/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace utils {
template <typename F, typename T,
        typename U = decltype(std::declval<F>()(std::declval<T>()))>
std::vector<U> fmap(const std::vector<T> &vec, const F &f) {
    std::vector<U> result;
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
    return result;
}

/** sorts an array of values using @p comparator. While sorting the array
 * of value, the function permutes an array of @p keys accordingly.
 *
 * @note The arrays of @p keys can be omitted. In this case the function
 *       sorts the array of @vals only.
 */
template <typename T, typename U, typename F>
inline void simultaneous_sort(T *vals, U *keys, size_t size, F comparator) {
    if (size == 0) return;

    for (size_t i = 0; i < size - 1; ++i) {
        bool swapped = false;
        for (size_t j = 0; j < size - i - 1; j++) {
            if (comparator(vals[j], vals[j + 1]) > 0) {
                std::swap(vals[j], vals[j + 1]);
                if (keys) std::swap(keys[j], keys[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false) break;
    }
}

/* Sorts an array of @p vals using @p comparator. Uses @p vals_2nd_level as a
 * second level comparing criteria in case comparator returns 0 (equal values)
 * for @p vals elements.
 * While sorting the array of @p vals, the function permutes an array of
 * @p vals_2nd_level and @p keys accordingly.
 */
template <typename T, typename U, typename F>
inline void simultaneous_sort(
        T *vals, T *vals_2nd_level, U *keys, size_t size, F comparator) {
    if (size == 0) return;

    for (size_t i = 0; i < size - 1; ++i) {
        bool swapped = false;

        for (size_t j = 0; j < size - i - 1; j++) {
            auto res = comparator(vals[j], vals[j + 1]);
            if (res == 0)
                res = comparator(vals_2nd_level[j], vals_2nd_level[j + 1]);

            if (res > 0) {
                std::swap(vals[j], vals[j + 1]);
                std::swap(vals_2nd_level[j], vals_2nd_level[j + 1]);
                std::swap(keys[j], keys[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false) break;
    }
}

template <typename T>
inline T rnd_up(const T a, const T b) {
    return (a + b - 1) / b * b;
}

inline uintptr_t mod_ptr(void *ptr, size_t bytes) {
    return reinterpret_cast<uintptr_t>(ptr) & (bytes - 1);
}

inline bool is_aligned_ptr(void *ptr, size_t bytes) {
    return mod_ptr(ptr, bytes) == 0;
}

inline std::pair<std::vector<float>, std::vector<float>> compute_scales(
        float src_scale, float dst_scale, std::vector<float> weight_scales) {
    auto scale_size = weight_scales.size();
    std::vector<float> bias_scales(scale_size), op_scales(scale_size);

    for (size_t i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scale * weight_scales[i];
        op_scales[i] = dst_scale / bias_scales[i];
    }
    return std::make_pair(std::move(bias_scales), std::move(op_scales));
}

inline std::pair<bool, int64_t> try_reverse_axis(
        const int64_t axis, const int32_t rank) {
    // oneDNN can not operate on the negative axis
    const auto new_axis = (axis < 0) ? rank + axis : axis;
    if (new_axis < 0 || new_axis >= static_cast<int64_t>(rank))
        return std::make_pair(false, axis);
    return std::make_pair(true, new_axis);
}

inline int op_scale_mask(dim scale_size) {
    return scale_size > 1 ? 2 : 0;
}

inline int tensor_scale_mask(dim scale_size, bool grouped) {
    return scale_size > 1 ? grouped ? 3 : 1 : 0;
}

inline int tensor_zp_mask(dim zp_size) {
    return zp_size > 1 ? 1 : 0;
}

inline bool compare_float(
        float ref, float given, float rtol = 1e-5f, float atol = 1e-6f) {
    const float diff = std::abs(given - ref);
    const float bigger
            = std::abs(ref) > std::abs(given) ? std::abs(ref) : std::abs(given);
    return diff <= rtol * bigger + atol;
}

} // namespace utils
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
