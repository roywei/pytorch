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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_PATTERN_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_PATTERN_UTILS_HPP

#include <algorithm>
#include <memory>
#include <vector>
#include <unordered_set>

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "backend/graph_compiler/compiler_backend.hpp"
#include "backend/graph_compiler/compiler_partition_impl.hpp"

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

class pattern_utils_t {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops);
    inline void set_partitions(dnnl::graph::impl::graph_t &backend_graph,
            std::vector<std::vector<op_t *>> &fusion_ops);

    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;
};

inline void pattern_utils_t::match(dnnl::graph::impl::graph_t &backend_graph,
        std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!impl::utils::pm::match_pattern(cur_op, pgraph, candidate_fusion)) {
            return status::success;
        }
        fusion_ops.emplace_back(candidate_fusion);
        return status::success;
    });
}

static bool check_logical_tensor_validity(const impl::logical_tensor_t &lt) {
    if (lt.layout_type != layout_type::strided) { return false; }
    if (lt.ndims <= 0) { return false; }
    std::vector<int64_t> size {lt.dims, lt.dims + lt.ndims};
    std::vector<int64_t> strides {
            lt.layout.strides, lt.layout.strides + lt.ndims};
    std::vector<int> indices(lt.ndims);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j) -> bool {
        if (strides[i] == strides[j]) { return size[i] < size[j]; }
        return strides[i] < strides[j];
    });
    if (strides[indices[0]] != 1) { return false; }
    for (int i = 1; i < lt.ndims; ++i) {
        if (strides[indices[i]]
                != strides[indices[i - 1]] * size[indices[i - 1]]) {
            return false;
        }
    }
    return true;
}

static bool check_inputs_outputs_validity(
        const std::vector<impl::logical_tensor_t> &inputs,
        const std::vector<impl::logical_tensor_t> &outputs) {
    for (auto &lt : inputs) {
        if (!check_logical_tensor_validity(lt)) { return false; }
    }
    for (auto &lt : outputs) {
        if (!check_logical_tensor_validity(lt)) { return false; }
    }
    return true;
}

inline void pattern_utils_t::set_partitions(
        dnnl::graph::impl::graph_t &backend_graph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    std::vector<op_t *> fusion_ops_set;
    std::unordered_set<op_t *> visit;

    for (auto &pairs : fusion_ops) {
        fusion_ops_set.clear();
        visit.clear();
        auto pimpl = std::make_shared<compiler_partition_impl_t>(
                backend_graph.get_engine_kind());

        for (size_t i = 0; i < pairs.size(); ++i) {
            visit.insert(pairs[i]);
            fusion_ops_set.push_back(pairs[i]);
        }

        for (auto &cur_op : fusion_ops_set) {
            for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
                auto in_value = cur_op->get_input_value(j);
                if (!in_value->has_producer()
                        || !visit.count(&in_value->get_producer())) {
                    pimpl->add_input_tensor(in_value);
                }
            }

            for (size_t j = 0; j < cur_op->num_outputs(); ++j) {
                auto out_value = cur_op->get_output_value(j);
                // if out_value has no consumer
                // OR any of its consumers are not inside the pattern
                // it is output tensor
                bool is_output = out_value->get_consumers().empty();
                for (auto &consumer : out_value->get_consumers()) {
                    if (!visit.count(&consumer.get_op())) {
                        is_output = true;
                        break;
                    }
                }
                if (is_output) { pimpl->add_output_tensor(out_value); }
            }
        }

        if (!check_inputs_outputs_validity(
                    pimpl->get_inputs(), pimpl->get_outputs())) {
            continue;
        }

        // transfer the matched op's ownership from graph to partition
        for (size_t i = 0; i < pairs.size(); ++i) {
            pimpl->add_op(pairs[i]->shared_from_this());
            // claim the op belong to the partition
            pairs[i]->set_partition(pimpl.get());
        }
        backend_graph.add_partition(pimpl);
    }
}

} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
