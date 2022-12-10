/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#ifndef BACKEND_DNNL_PASSES_UTILS_HPP
#define BACKEND_DNNL_PASSES_UTILS_HPP

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op.hpp"
#include "interface/value.hpp"
#include "utils/utils.hpp"

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct op_executable_t;
using pd_cache_t = std::unordered_map<op_t *, dnnl::primitive_desc>;

class primitive_attr_mgr_t {
public:
    primitive_attr_mgr_t() = default;

    // Disable assignment and copy
    primitive_attr_mgr_t(const primitive_attr_mgr_t &) = delete;
    primitive_attr_mgr_t(primitive_attr_mgr_t &&) = delete;
    primitive_attr_mgr_t &operator=(const primitive_attr_mgr_t &) = delete;
    primitive_attr_mgr_t &operator=(primitive_attr_mgr_t &&) = delete;

    int64_t init_attr() {
        auto ret = data_.insert({counter++, dnnl::primitive_attr()});
        return ret.first->first;
    }

    dnnl::primitive_attr &get_attr(int64_t key) {
        assertm(key != -1, "invalid key");
        return data_[key];
    }

private:
    std::unordered_map<int64_t, dnnl::primitive_attr> data_;
    int64_t counter {0};
};

// The subgraph_t class is a subclass of graph_t, which is used as the only
// parameter of transformation passes. Each transformation pass will process the
// subgraph_t object, and after that, the content of subgraph_t object will be
// changed.
class subgraph_t : public impl::graph_t {
public:
    subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
            bool reset_layout = true);

    subgraph_t(const std::vector<op_ptr> &ops, bool reset_layout = true);

    std::vector<op_ptr> &get_mutable_ops() {
        return const_cast<std::vector<op_ptr> &>(get_ops());
    }

    // The inputs and outputs logical tensors given by users at compilation
    // stage
    std::vector<impl::logical_tensor_t> ins_;
    std::vector<impl::logical_tensor_t> outs_;

    // The engine that the subgraph is compiled for
    const dnnl::engine *p_engine_;

    // The primitive attr manager that is used to hold each op's fusion
    // information
    primitive_attr_mgr_t prm_attr_mgr_;

    // The custom cache to store the created primitive desc
    pd_cache_t pd_cache_;

    // The vector to tell which op in the subgraph is constant and will only run
    // once
    std::vector<bool> is_constant_;

    // The executable for each op in subgraph
    std::vector<std::shared_ptr<op_executable_t>> execs_;
};

class subgraph_visualizer_t {
public:
    subgraph_visualizer_t() = default;

    subgraph_visualizer_t(size_t partition_id,
            const std::function<std::string(const value_t *)> &mem_info_func
            = {})
        : enabled_(false)
        , mem_info_func_(mem_info_func)
#ifdef DNNL_GRAPH_ENABLE_DUMP
        , partition_id_(partition_id)
        , index_(0)
#endif
    {
        MAYBE_UNUSED(partition_id);
        // Set DNNL_GRAPH_DUMP=2 to enable dump subgraph
        enabled_ = impl::utils::getenv_int_user("DUMP", 0) > 1;
    }

    status_t run(const std::shared_ptr<subgraph_t> &sg,
            const std::string &name_suffix, bool is_layout_sensitive,
            bool is_memory_sensitive = false);

private:
    bool enabled_;
    std::function<std::string(const value_t *)> mem_info_func_;
#ifdef DNNL_GRAPH_ENABLE_DUMP
    size_t partition_id_;
    size_t index_;
#endif
};

class subgraph_validator_t {
public:
    subgraph_validator_t() = default;
    status_t run(const std::shared_ptr<subgraph_t> &sg);
};

// The pass_pipeline_t class is used to manage all transformation passes to run
// on a subgraph. Users should add passes need to run to the pipeline with a
// user defined order. And then call the run() method to run those added passes.
// After running each pass, the pipeline will choose to visualize the processed
// subgraph by using the visualizer.
class pass_pipeline_t {
public:
    using pass_signature
            = std::function<impl::status_t(std::shared_ptr<subgraph_t> &)>;

    pass_pipeline_t(const subgraph_visualizer_t &vis)
        : visualizer_(vis)
        , is_layout_sensitive_(false)
        , is_memory_sensitive_(false) {};

    // Reset the visualize arguments
    void reset_visualize_arg(
            bool is_layout_sensitive, bool is_memory_sensitive) {
        is_layout_sensitive_ = is_layout_sensitive;
        is_memory_sensitive_ = is_memory_sensitive;
    }

    // Add a pass to the pipeline. The current visualize arguments will be
    // recorded for the added pass and be used when visualize the subgraph
    // processed by this pass.
    void add_pass(const pass_signature &apass, const std::string &name) {
        passes_.emplace_back(apass);
        names_.emplace_back(name);
        is_layout_sensitives_.push_back(is_layout_sensitive_);
        is_memory_sensitives_.push_back(is_memory_sensitive_);
    }

    // Run all added passes
    impl::status_t run(std::shared_ptr<subgraph_t> &sg) {
        impl::status_t ret;
        for (size_t i = 0; i < passes_.size(); i++) {
            ret = passes_[i](sg);
            if (ret != impl::status::success) { return ret; }

            // Dump the subgraph to dot file
            visualizer_.run(sg, names_[i], is_layout_sensitives_[i],
                    is_memory_sensitives_[i]);

            // Validate the subgraph after each pass
            ret = validator_.run(sg);
            if (ret != impl::status::success) { return ret; }
        }
        return impl::status::success;
    }

private:
    // The added passes and their names
    std::vector<pass_signature> passes_;
    std::vector<std::string> names_;

    // The recorded visualize arguments for each pass
    std::vector<bool> is_layout_sensitives_;
    std::vector<bool> is_memory_sensitives_;

    subgraph_visualizer_t visualizer_;
    subgraph_validator_t validator_;

    // The current visualize arguments
    bool is_layout_sensitive_;
    bool is_memory_sensitive_;
};

#define BACKEND_DNNL_ADD_PASS(pipeline, pass) pipeline.add_pass(pass, #pass)

void insert_op_before(std::shared_ptr<impl::op_t> &inserted_op,
        std::shared_ptr<impl::op_t> &base_op, size_t offset);

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t offset);

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t base_offset,
        size_t inserted_offset);

void insert_op_after(std::shared_ptr<impl::op_t> &inserted_op,
        std::shared_ptr<impl::op_t> &base_op, size_t offset);

void insert_op_after(op_t *inserted_op, op_t *base_op, size_t offset);

void insert_op_after(op_t *inserted_op, op_t *base_op, size_t output_offset,
        size_t input_offset);

void fuse_op_to_successor(
        op_t *op, std::vector<std::shared_ptr<op_t>> &subgraph);

void fuse_op_to_predecessor(op_t *op,
        std::vector<std::shared_ptr<op_t>> &subgraph, size_t in_offset = 0);

status_t set_given_inputs_outputs(std::shared_ptr<subgraph_t> &sg,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs);

status_t set_given_inputs_outputs(std::vector<std::shared_ptr<op_t>> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs);

void set_all_layout_to_any(std::vector<std::shared_ptr<op_t>> &subgraph);

void set_weight_bias_constant(std::vector<std::shared_ptr<op_t>> &subgraph);

inline bool is_preprocess_op(impl::op_t &op) {
    static const std::set<impl::op_kind_t> preprocess_ops = {op_kind::permute,
            op_kind::to_group, op_kind::from_group, op_kind::expand,
            op_kind::squeeze, impl::op_kind::StaticReshape,
            impl::op_kind::StaticTranspose};
    return preprocess_ops.count(op.get_kind()) != 0;
}

void replace_op(std::shared_ptr<op_t> &org_op, std::shared_ptr<op_t> &new_op);

void merge_common_eltwise_attrs(
        std::shared_ptr<op_t> &org_op, std::shared_ptr<op_t> &new_op);

inline const std::map<op_kind_t, dnnl::algorithm> &get_eltwise_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &eltwise_alg_map = {
            {impl::op_kind::Abs, dnnl::algorithm::eltwise_abs},
            {impl::op_kind::Clamp, dnnl::algorithm::eltwise_clip_v2},
            {impl::op_kind::Elu, dnnl::algorithm::eltwise_elu},
            {impl::op_kind::Exp, dnnl::algorithm::eltwise_exp},
            {impl::op_kind::GELU, dnnl::algorithm::eltwise_gelu_erf},
            {impl::op_kind::HardSwish, dnnl::algorithm::eltwise_hardswish},
            {impl::op_kind::HardTanh, dnnl::algorithm::eltwise_clip_v2},
            {impl::op_kind::Log, dnnl::algorithm::eltwise_log},
            {impl::op_kind::ReLU, dnnl::algorithm::eltwise_relu},
            {impl::op_kind::Round, dnnl::algorithm::eltwise_round},
            {impl::op_kind::Sigmoid, dnnl::algorithm::eltwise_logistic},
            {impl::op_kind::Sqrt, dnnl::algorithm::eltwise_sqrt},
            {impl::op_kind::Square, dnnl::algorithm::eltwise_square},
            {impl::op_kind::Tanh, dnnl::algorithm::eltwise_tanh},
            {impl::op_kind::Pow, dnnl::algorithm::eltwise_pow}};
    return eltwise_alg_map;
}

inline dnnl::algorithm get_eltwise_bwd_alg(op_kind_t kind, bool use_dst) {
    using algo = dnnl::algorithm;
    switch (kind) {
        case impl::op_kind::EluBackprop:
            if (use_dst) return algo::eltwise_elu_use_dst_for_bwd;
            return algo::eltwise_elu;
        case impl::op_kind::GELUBackprop: return algo::eltwise_gelu_erf;
        case impl::op_kind::HardTanhBackprop:
            if (use_dst) return algo::eltwise_clip_v2_use_dst_for_bwd;
            return algo::eltwise_clip_v2;
        case impl::op_kind::HardSwishBackprop: return algo::eltwise_hardswish;
        case impl::op_kind::ReLUBackprop:
            if (use_dst) return algo::eltwise_relu_use_dst_for_bwd;
            return algo::eltwise_relu;
        case impl::op_kind::SigmoidBackprop:
            if (use_dst) return algo::eltwise_logistic_use_dst_for_bwd;
            return algo::eltwise_logistic;
        case impl::op_kind::SqrtBackprop:
            if (use_dst) return algo::eltwise_sqrt_use_dst_for_bwd;
            return algo::eltwise_sqrt;
        case impl::op_kind::TanhBackprop:
            if (use_dst) return algo::eltwise_tanh_use_dst_for_bwd;
            return algo::eltwise_tanh;
        default: return algo::undef;
    }
}

inline const std::map<op_kind_t, dnnl::algorithm> &get_reduction_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &reduction_alg_map = {
            {impl::op_kind::ReduceL1,
                    dnnl::algorithm::reduction_norm_lp_power_p_sum},
            {impl::op_kind::ReduceL2, dnnl::algorithm::reduction_norm_lp_sum},
            {impl::op_kind::ReduceMax, dnnl::algorithm::reduction_max},
            {impl::op_kind::ReduceMean, dnnl::algorithm::reduction_mean},
            {impl::op_kind::ReduceMin, dnnl::algorithm::reduction_min},
            {impl::op_kind::ReduceProd, dnnl::algorithm::reduction_mul},
            {impl::op_kind::ReduceSum, dnnl::algorithm::reduction_sum}};
    return reduction_alg_map;
}

inline bool is_eltwise_kind(op_kind_t kind) {
    const std::set<op_kind_t> eltwise_kinds {impl::op_kind::Abs,
            impl::op_kind::Clamp, impl::op_kind::Elu, impl::op_kind::Exp,
            impl::op_kind::GELU, impl::op_kind::HardTanh,
            impl::op_kind::HardSwish, impl::op_kind::Log, impl::op_kind::ReLU,
            impl::op_kind::Round, impl::op_kind::Sigmoid,
            impl::op_kind::SoftPlus, impl::op_kind::Sqrt, impl::op_kind::Square,
            impl::op_kind::Tanh, impl::op_kind::Pow};
    return eltwise_kinds.find(kind) != eltwise_kinds.end();
}

inline bool is_eltwise_bwd_kind(op_kind_t kind) {
    const std::set<op_kind_t> eltwise_bwd_kinds {impl::op_kind::EluBackprop,
            impl::op_kind::GELUBackprop, impl::op_kind::HardTanhBackprop,
            impl::op_kind::HardSwishBackprop, impl::op_kind::ReLUBackprop,
            impl::op_kind::SigmoidBackprop, impl::op_kind::SqrtBackprop,
            impl::op_kind::TanhBackprop};
    return eltwise_bwd_kinds.find(kind) != eltwise_bwd_kinds.end();
}

inline bool is_binary_kind(op_kind_t kind) {
    const static std::set<impl::op_kind_t> binary_kinds
            = {impl::op_kind::Add, impl::op_kind::Subtract,
                    impl::op_kind::Multiply, impl::op_kind::Divide,
                    impl::op_kind::Minimum, impl::op_kind::Maximum};
    return binary_kinds.find(kind) != binary_kinds.end();
}

inline bool is_reduction_kind(op_kind_t kind) {
    const static std::set<impl::op_kind_t> reduction_kinds
            = {impl::op_kind::ReduceL1, impl::op_kind::ReduceL2,
                    impl::op_kind::ReduceMax, impl::op_kind::ReduceMean,
                    impl::op_kind::ReduceMin, impl::op_kind::ReduceProd,
                    impl::op_kind::ReduceSum};
    return reduction_kinds.find(kind) != reduction_kinds.end();
}

std::vector<value_t *> get_constant_block_output_values(
        const std::vector<std::shared_ptr<op_t>> &subgraph);

impl::status_t infer_shape(std::shared_ptr<subgraph_t> &sg);

const std::map<op_kind_t, dnnl::algorithm> &get_binary_alg_map();

// (3, 4) * (3, 4) is doable
// (1, 4) * (3, 4) is doable
// (3, 4, 5) * (4, 5) is doable
// (3, 4, 5) * (1, 5) is doable
// (3, 4, 5) * (2, 4, 5) is NOT doable
bool binary_doable(
        const std::vector<dim_t> &shape_0, const std::vector<dim_t> &shape_1);

bool prelu_doable(const std::vector<dim_t> &src_dims,
        const std::vector<dim_t> &wei_dims, const std::string &data_format,
        const bool per_channel_broadcast);

// Checks whether chain of Reshape, Transpose, Reshape is fusible
// to dnnl_shuffle. Returns following pair:
// (is_fusible, (axis, groups))
// axis and groups store relevant information only when 'is_fusible = true'.
std::pair<bool, std::pair<size_t, int64_t>> shuffle_fusible(
        const impl::op_t *reshape0, impl::op_t *reshape1,
        impl::op_t *transpose);

// For some shapes, post binary will run into oneDNN's ref path and has poor
// performance. So, we check the shape in this function and only make
// per_tensor, per_channel, per_mb_w(MatMul) and full tensor broadcast
// binary able to be fused.
bool post_binary_fusible(const impl::op_t *base_op, const impl::op_t *bin_op);

// oneDNN support post depthwise conv fusion. This function is used to check if
// two conv ops can be fused as a conv + depthwise pattern.
bool post_depthwise_conv_fusible(
        const impl::op_t *base_conv_op, const impl::op_t *post_conv_op);

// Get the map between base op kind and fusible post ops kinds. The map is
// determined by oneDNN's fusion capability and may change. For example, a
// dnnl_eltwise op can't fuse dnnl_eltwise op, but dnnl_convolution can.
const std::unordered_map<impl::op_kind_t, std::unordered_set<impl::op_kind_t>> &
get_post_ops_fusible_map();

const std::string &kind2str(impl::op_kind_t kind);

std::shared_ptr<impl::value_t> insert_empty_scratchpad(
        std::shared_ptr<op_t> &op);

// This function is used to check if a dnnl_reorder op is converted from or act
// as a TypeCast op. This function will only return true for a dnnl_reorder op
// which only has different input/output data type.
bool is_typecast(const impl::op_t *op);

// get the dense strides of a given shape
// eg. (3, 4, 5) -> (20, 5, 1)
inline dims get_dense_strides(const dims &shape) {
    dims strides(shape.size());
    for (auto it = shape.begin(); it < shape.end(); ++it) {
        const auto val = std::accumulate(
                std::next(it), shape.end(), 1, std::multiplies<dim_t>());
        const auto dist = std::distance(shape.begin(), it);
        strides[static_cast<size_t>(dist)] = val;
    }
    return strides;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
