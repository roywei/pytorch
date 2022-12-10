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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_DYNAMIC_TRANSPOSE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_DYNAMIC_TRANSPOSE_HPP

#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/traits.hpp>

namespace sc {
namespace ops {

// not truly dynamic, just used to align with current llga's transpose's schema
class dynamic_transpose_op : public sc_op,
                             public op_traits::auto_copyable_t,
                             public op_traits::constant_optimizable_t {
public:
    dynamic_transpose_op(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    ir_module_ptr get_func(context_ptr ctx) override;
    sc_op_ptr constant_optimize(sc_graph_t &graph) override;

private:
    std::vector<int> real_axes_;
};
} // namespace ops
} // namespace sc

#endif
