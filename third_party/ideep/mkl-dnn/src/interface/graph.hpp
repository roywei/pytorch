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

#ifndef INTERFACE_GRAPH_HPP
#define INTERFACE_GRAPH_HPP

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.h"

#include "interface/c_types_map.hpp"
#include "interface/engine.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"
#include "interface/op_schema.hpp"
#include "interface/partition.hpp"
#include "interface/partition_impl.hpp"
#include "interface/value.hpp"

#include "utils/compatible.hpp"
#include "utils/id.hpp"
#include "utils/utils.hpp"

namespace impl = dnnl::graph::impl;

namespace dnnl {
namespace graph {
namespace impl {
void rewrite(impl::graph_t &agraph,
        const std::vector<std::vector<impl::op_t *>> &fusion_ops);
}
} // namespace graph
} // namespace dnnl

struct dnnl_graph_graph : public impl::utils::id_t {
    using op_t = impl::op_t;
    using value_t = impl::value_t;
    using op_ptr = std::shared_ptr<op_t>;
    using value_ptr = std::shared_ptr<value_t>;
    using logical_tensor_t = impl::logical_tensor_t;
    using logical_tensor_wrapper_t = impl::logical_tensor_wrapper_t;
    using op_schema_t = impl::op_schema_t;
    using op_schema_registry_t = impl::op_schema_registry_t;
    using id_t = impl::utils::id_t;

private:
    /*! \brief added ops*/
    std::vector<op_ptr> ops_ {};

    /*! \brief The engine kind on which the operator will be evaluated */
    impl::engine_kind_t engine_kind_ {};

    std::vector<std::shared_ptr<impl::partition_impl_t>> partition_impls_;

    bool is_built_ {false};

public:
    dnnl_graph_graph(impl::engine_kind_t kind = impl::engine_kind::cpu)
        : engine_kind_(kind) {};

    // deep copy (except that the partition_impls_ is shallow copy)
    dnnl_graph_graph(const dnnl_graph_graph &other)
        : id_t(other)
        , ops_(deep_copy(other.ops_))
        , engine_kind_(other.engine_kind_)
        , partition_impls_(other.partition_impls_) {};

    dnnl_graph_graph(const std::vector<op_ptr> &ops) : ops_(ops) {};

    dnnl_graph_graph &operator=(const dnnl_graph_graph &other) = delete;

    ~dnnl_graph_graph() = default;

    impl::engine_kind_t get_engine_kind() const { return engine_kind_; }

    /*!
     * \brief Check whether an operator can be added
     * \param l_n An operator in frameworks' graph.
     * \return Whether the operator is supported
     */
    impl::status_t add_op(const op_t *l_n) {
        if (!l_n) return impl::status::invalid_op;

        if (std::none_of(ops_.begin(), ops_.end(),
                    [&l_n](const std::vector<op_ptr>::value_type &op) {
                        return op->get_id() == l_n->get_id();
                    })) {
            const impl::op_schema_t *opm
                    = impl::op_schema_registry_t::get_op_schema(
                            l_n->get_kind());
            op_t tmp_ln = *l_n;
            if (opm != nullptr) {
                opm->set_default_attribute(&tmp_ln);
                if (!opm->verify(&tmp_ln)) { return impl::status::invalid_op; }
            }
            ops_.push_back(std::make_shared<op_t>(tmp_ln));
            auto back_op = ops_.back().get();
            for (size_t i = 0; i < back_op->num_outputs(); i++)
                back_op->get_output_value(i)->set_producer(*back_op);
        }
        return impl::status::success;
    }

    op_t *create_op(dnnl_graph_op_kind_t kind, std::string name = "") {
        ops_.push_back(std::make_shared<op_t>(kind, std::move(name)));
        return ops_.back().get();
    }

    void delete_op(op_t *op) {
        if (!op) return;

        auto pos = std::find_if(ops_.begin(), ops_.end(),
                [op](const op_ptr &n) -> bool { return *n == *op; });
        if (pos != ops_.end()) ops_.erase(pos);
    }

    /*!
     * \brief Get all the ops of this graph, inlcuding original ops and fused.
     * \return vector of ops pointers
     */
    const std::vector<op_ptr> &get_ops() const { return ops_; }

    /*! \brief how many ops in the graph */
    size_t num_ops() const { return ops_.size(); }

    /*!
     * \brief Get the output ops of this graph.
     * \return vector of output op pointers
     */
    std::vector<op_t *> get_output_ops() {
        std::vector<op_t *> outputs;
        for (const op_ptr &n : ops_) {
            size_t num_consumers = 0;
            for (size_t i = 0; i < n->num_outputs(); i++) {
                num_consumers += n->num_output_consumers(i);
            }

            if (num_consumers == 0) { outputs.push_back(n.get()); }
        }
        return outputs;
    }

    /*!
     * \brief Get the input values (values whose producer are not in the graph)
     * of this graph.
     * \return vector of input values pointers
     */
    std::vector<value_t *> get_input_values() {
        std::vector<value_t *> in_vals;
        for (const op_ptr &n : ops_) {
            for (const value_ptr &in_val : n->get_input_values()) {
                if (!in_val->has_producer()) {
                    in_vals.emplace_back(in_val.get());
                    continue;
                }

                op_t &producer = in_val->get_producer();
                if (std::none_of(ops_.begin(), ops_.end(),
                            [&producer](const op_ptr &op) {
                                return op.get() == &producer;
                            })) {
                    in_vals.emplace_back(in_val.get());
                }
            }
        }

        return in_vals;
    }

    /*!
     * \brief Get the output values (values whose comsumers are not all in the
     * graph) of this graph.
     * \return vector of output values pointers
     */
    std::vector<value_t *> get_output_values() {
        std::vector<value_t *> out_vals;
        for (const op_ptr &n : ops_) {
            for (const value_ptr &out_val : n->get_output_values()) {
                std::vector<value_t::consumer_t> consumers
                        = out_val->get_consumers();

                bool has_outer_consumer = false;
                for (const value_t::consumer_t &csm : consumers) {
                    op_t &csm_op = csm.get_op();
                    if (std::none_of(ops_.begin(), ops_.end(),
                                [&csm_op](const op_ptr &op) {
                                    return op.get() == &csm_op;
                                })) {
                        has_outer_consumer = true;
                        break;
                    }
                }

                if (consumers.empty() || has_outer_consumer)
                    out_vals.emplace_back(out_val.get());
            }
        }

        return out_vals;
    }

    void add_partition(const std::shared_ptr<impl::partition_impl_t> &pimpl) {
        partition_impls_.push_back(pimpl);
    }

    std::vector<std::shared_ptr<impl::partition_impl_t>> &get_partitions() {
        return partition_impls_;
    }

    /*!
     * \brief Get partition numbers
     * \return partition numbers
     */
    size_t get_num_partitions() const { return partition_impls_.size(); }

    /*!
     * \brief get list of partitions
     * \param list of partitions
     */
    void get_ordered_partitions(std::vector<impl::partition_t *> &partitions);

    /*!
     * \brief Build backend graph after add op is done
     */
    impl::status_t build_graph();

    // This function is used to infer shape for all the ops in a graph.
    // Before calling this function, the inputs value of the graph should
    // have valid shape
    impl::status_t infer_shape() {
        using value_ptr = std::shared_ptr<value_t>;

        // Check inputs shape
        for (value_t *in : get_input_values()) {
            logical_tensor_t lt = in->get_logical_tensor();
            if (logical_tensor_wrapper_t(lt).is_shape_unknown())
                return impl::status::invalid_shape;
        }

        // call each op's infer shape function in topological order
        return impl::topo_order_visit(get_output_ops(), [](impl::op_t *op) {
            std::vector<logical_tensor_t> tmp_inputs, tmp_outputs;
            std::vector<logical_tensor_t *> tmp_inputs_ptr, tmp_outputs_ptr;

            // avoid re-allocating
            tmp_inputs.reserve(op->num_inputs());
            tmp_outputs.reserve(op->num_outputs());
            tmp_inputs_ptr.reserve(op->num_inputs());
            tmp_outputs_ptr.reserve(op->num_outputs());

            for (const value_ptr &in : op->get_input_values()) {
                tmp_inputs.emplace_back(in->get_logical_tensor());
                tmp_inputs_ptr.emplace_back(&tmp_inputs.back());
            }
            for (const value_ptr &out : op->get_output_values()) {
                tmp_outputs.emplace_back(out->get_logical_tensor());
                tmp_outputs_ptr.emplace_back(&tmp_outputs.back());
            }

            const op_schema_t *opm
                    = op_schema_registry_t::get_op_schema(op->get_kind());
            // can't infer shape for cur op: no schema
            if (!opm) return impl::status::invalid_op;

            impl::status_t ret
                    = opm->shape_infer(op, tmp_inputs_ptr, tmp_outputs_ptr);

            if (ret != impl::status::success)
                return impl::status::invalid_shape;

            for (size_t i = 0; i < op->num_outputs(); i++) {
                op->get_output_value(i)->set_logical_tensor(tmp_outputs[i]);
            }

            return impl::status::success;
        });
    }

    static std::vector<op_ptr> deep_copy(const std::vector<op_ptr> &ops);
};

#endif
