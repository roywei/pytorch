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

#ifndef TEST_API_COMMON_HPP
#define TEST_API_COMMON_HPP

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "src/interface/partition_cache.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

using dim_t = int64_t;
using dims_t = std::vector<dim_t>;

#define SKIP_IF(cond, msg) \
    do { \
        if (cond) { \
            std::cout << "[  SKIPPED ] " << (msg) << std::endl; \
            return; \
        } \
    } while (0)

struct dnnl_graph_test_conv_attr_name_t {
    dnnl_graph_test_conv_attr_name_t()
        : strides("strides")
        , pads_begin("pads_begin")
        , pads_end("pads_end")
        , dilations("dilations")
        , groups("groups") {}
    dnnl_graph_test_conv_attr_name_t(std::string strides,
            std::string pads_begin, std::string pads_end, std::string dilations,
            std::string groups)
        : strides(std::move(strides))
        , pads_begin(std::move(pads_begin))
        , pads_end(std::move(pads_end))
        , dilations(std::move(dilations))
        , groups(std::move(groups)) {}
    std::string strides;
    std::string pads_begin;
    std::string pads_end;
    std::string dilations;
    std::string groups;
};

struct dnnl_graph_test_conv_attr_value_t {
    dnnl_graph_test_conv_attr_value_t(dims_t strides, dims_t pads_begin,
            dims_t pads_end, dims_t dilations, dims_t groups)
        : strides(std::move(strides))
        , pads_begin(std::move(pads_begin))
        , pads_end(std::move(pads_end))
        , dilations(std::move(dilations))
        , groups(std::move(groups)) {}
    dims_t strides;
    dims_t pads_begin;
    dims_t pads_end;
    dims_t dilations;
    dims_t groups;
};

struct dnnl_graph_test_conv_shapes_t {
    dnnl_graph_test_conv_shapes_t(
            dims_t input_dims, dims_t weight_dims, dims_t output_dims)
        : input_ndim(static_cast<dim_t>(input_dims.size()))
        , weight_ndim(static_cast<dim_t>(weight_dims.size()))
        , output_ndim(static_cast<dim_t>(output_dims.size()))
        , input_dims(std::move(input_dims))
        , weight_dims(std::move(weight_dims))
        , output_dims(std::move(output_dims)) {}
    dim_t input_ndim;
    dim_t weight_ndim;
    dim_t output_ndim;
    dims_t input_dims;
    dims_t weight_dims;
    dims_t output_dims;
};

struct dnnl_graph_test_conv_layout_t {
    dnnl_graph_layout_type_t input_layout;
    dnnl_graph_layout_type_t weight_layout;
    dnnl_graph_layout_type_t output_layout;
};

/*
    conv2d attribute:
    strides, pad_begin, pad_end, dilations, groups

    logic tensors:
    input, weight, output
*/
struct dnnl_graph_test_conv_params {
    dnnl_graph_engine_kind_t engine;
    dnnl_graph_op_kind_t op_kind;
    dnnl_graph_partition_policy_t policy;
    dnnl_graph_data_type_t data_type;
    dnnl_graph_test_conv_attr_name_t attr_name;
    dnnl_graph_test_conv_attr_value_t attr_value;
    dnnl_graph_test_conv_layout_t tensor_layout;
    dnnl_graph_test_conv_shapes_t tensor_dims;
};

extern dnnl_graph_engine_kind_t api_test_engine_kind;

#ifdef DNNL_GRAPH_WITH_SYCL
struct allocator_handle_t {
    dnnl_graph_allocator_t *allocator = nullptr;
    ~allocator_handle_t() { dnnl_graph_allocator_destroy(allocator); }
    explicit operator bool() const noexcept {
        return static_cast<bool>(allocator);
    }
};
static allocator_handle_t allocator_handle;
#endif // DNNL_GRAPH_WITH_SYCL

struct engine_handle_t {
    dnnl_graph_engine_t *engine = nullptr;
    ~engine_handle_t() { dnnl_graph_engine_destroy(engine); }
    explicit operator bool() const noexcept {
        return static_cast<bool>(engine);
    }
};
static engine_handle_t engine_handle;

void api_test_dnnl_graph_engine_create(
        dnnl_graph_engine_t **engine, dnnl_graph_engine_kind_t engine_kind);

void api_test_dnnl_graph_graph_create(
        dnnl_graph_graph_t **graph, dnnl_graph_engine_kind_t engine_kind);

dnnl::graph::engine &cpp_api_test_dnnl_graph_engine_create(
        dnnl::graph::engine::kind engine_kind);

inline int get_compiled_partition_cache_size() {
    int result = 0;
    auto status = dnnl::graph::impl::get_compiled_partition_cache_size(&result);
    if (status != dnnl::graph::impl::status::success) return -1;
    return result;
}

inline dnnl_graph_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty()
            ? 0
            : std::accumulate(dims.begin(), dims.end(), (dnnl_graph_dim_t)1,
                    std::multiplies<dnnl_graph_dim_t>());
}

#endif
