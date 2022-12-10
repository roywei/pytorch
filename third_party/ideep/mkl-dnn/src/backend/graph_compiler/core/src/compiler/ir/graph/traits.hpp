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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAITS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAITS_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph.hpp"
#include <microkernel/cpu/brgemm_alg_kind.hpp>
namespace sc {

class fusion_manager;
struct brgemm_fusion_register;

namespace op_traits {
struct may_broadcast_t : public virtual op_base_trait_t {
    // returns the input index of the logical tensor that will be broadcast
    // returns -1 when it cannot broadcast
    virtual int get_broadcast_input() const = 0;
    virtual std::vector<int> infer_broadcast_axis() const = 0;
    const std::vector<int> &get_plain_bc_axis() const { return plain_bc_axis_; }

protected:
    std::vector<int> plain_bc_axis_;
};

struct copyable_t : public virtual op_base_trait_t {
    virtual sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr)
            = 0;
};

/**
 * A class is auto-copyable if we can construct a valid copy of the node with
 * the in/out tensors and the attrs of the node. The op name should be in the op
 * registery
 * */
struct auto_copyable_t : public copyable_t {
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
};

/**
 * @brief The util trait template for Ops that is "almost" auto-copyable except
 * that they need to copy the data in the trait.
 *
 * @tparam TArgs op_traits that has copy_from methods
 */
template <typename... TArgs>
struct auto_copyable_with_trait_t : public auto_copyable_t {
    template <typename T>
    static void copy_impl(auto_copyable_with_trait_t *from, sc_op *to) {
        auto pto = dynamic_cast<T *>(to);
        assert(pto);
        auto pfrom = dynamic_cast<T *>(from);
        assert(pfrom);
        pto->copy_from(pfrom);
    }

    template <typename T0, typename... T>
    static void copy_impl(auto_copyable_with_trait_t *from,
            typename std::enable_if<(sizeof...(T) > 0), sc_op *>::type to) {
        copy_impl<T0>(from, to);
        copy_impl<T...>(from, to);
    }
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override {
        auto ret = auto_copyable_t::copy(ins, outs, mgr);
        copy_impl<TArgs...>(this, ret.get());
        return ret;
    }
};

// the OP can be optimized if some of the inputs are constants
struct constant_optimizable_t : public virtual op_base_trait_t {
    // do optimization and return the new optmized op. If no optimization can be
    // applied, return null
    virtual sc_op_ptr constant_optimize(sc_graph_t &graph) = 0;
};

// the part of OP's workload can be computed, e.g. intrisics(brgemm), tensor
// slice.
struct workload_computable_t : public virtual op_base_trait_t {
    using shape_dtype_pair = std::pair<sc_dims, sc_data_type_t>;
    static const size_t read_weight = 1UL;
    static const size_t write_weight = 1UL;
    static constexpr const char *workload_number = "workload_number";
    // compute workload with given input and output tensor pointers, according
    // to read/write times and operator numbers.
    virtual size_t compute_workload(const std::vector<shape_dtype_pair> &ins,
            const std::vector<shape_dtype_pair> &outs)
            = 0;
};

// the OP can accept a fusion manager to do post fusion
struct post_fusion_acceptable_t : public virtual op_base_trait_t {
    virtual ir_module_ptr get_func(context_ptr ctx,
            const std::shared_ptr<fusion_manager> &fuse_mgr,
            const std::string &func_name)
            = 0;
};

// the OP can be fused into brgemm calculation.
struct brgemm_fusion_acceptable_t : public virtual op_base_trait_t {
    static constexpr const char *brgemm_fusion = "brgemm_fusion";
    bool fuse_in_brgemm_ = false;
    brgemm::alg_kind_t alg_kind_ = brgemm::alg_kind_t::alg_kind_undef;
    virtual bool register_brgemm_fusion(const context_ptr &ctx,
            const std::vector<tensor_slice *> &outputs,
            const std::vector<const tensor_slice *> &inputs,
            brgemm_fusion_register &brg_reg)
            = 0;
    void copy_from(brgemm_fusion_acceptable_t *from) {
        fuse_in_brgemm_ = from->fuse_in_brgemm_;
        alg_kind_ = from->alg_kind_;
    }
};

// quantize
struct may_quantize_t : public virtual op_base_trait_t {
    virtual sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) {
        need_compensation_ = false;
        return sc_op_ptr();
    }
    bool should_quantized_ = false;
    bool is_quantized_ = false;
    bool need_compensation_ = true;
};

// the Op may cause batchwisely merged
struct batchwise_shrinkable_t : public virtual op_base_trait_t {
    /** Here, batchwise dims means safety loop ranges E.g.
     * 1. Reduce op:
     *  - Outs: [28,32,56,56]
     *  - Ins: [28,1,56,1]
     *  - Return: [28]
     * 2. Binary Ops:
     *  - Outs: [28,32,56,56]
     *  - Ins: [28,1,56,1] + [28,32,56,56]
     *  - Return: [28,32,56,56]
     * 3. Reorder Ops:
     *  - Outs: [28,16,56,56,2]
     *  - Ins: [28,32,56,56]
     *  - Return: [28,16,56,56]
     * 4. Tensorview Ops:
     *  - Outs: [28,32,56,56]
     *  - Ins: [28,16,2,56,56]
     *  - Return: [28]
     * */

    // this function must ensure all graph tensor of this op is able to be
    // shrinked by shrinkable dims.
    virtual sc_dims get_bwise_fuse_shrink_dims() = 0;

    virtual sc_op_ptr bw_shrinked_copy(
            gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph);

    virtual sc_op_ptr bw_shrinked_copy(gt2gt_map &bw_lt_map,
            sc_graph_t &shrinked_graph, const any_map_t &changed_attr);

    // this function collect shrinked graph tensor map in order to set new plain
    // dims.
    virtual void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map);

    virtual void collect_shrinked_axes_map(
            int bw_size, gt2axes_map &bw_axes_map);

    static graph_tensor_ptr shrink_gt(
            const graph_tensor_ptr &orig_gt, int shrink_offset);

    static void record_shrinked_gt(gt2gt_map &bw_lt_map,
            const graph_tensor_ptr &gt, int shrink_offset);

    static void record_shrinked_gt(gt2gt_map &bw_lt_map,
            const graph_tensor_ptr &gt, const sc_dims &plain_dims);

    static void record_shrinked_axes(
            gt2axes_map &bw_axes_map, const graph_tensor_ptr &gt, int bw_size);

    static void record_shrinked_axes(gt2axes_map &bw_axes_map,
            const graph_tensor_ptr &gt, const std::vector<int> &axes);

    /** this function return shrinkable offset, it should satisfy two
     * conditions:
     *  1. no padding axis. E.g. [16,15] -> [2,2,8,8]  return 2, due to 2*8!=15
     *  2. only touch block_num, rather than any block_size axis.
     * */
    static int get_shrinkable_offset(const graph_tensor_ptr &gt);
};

struct data_compensation_t : public virtual op_base_trait_t {};
struct weight_compensation_t : public virtual op_base_trait_t {};
struct constant_compensation_t : public virtual op_base_trait_t {};

} // namespace op_traits

} // namespace sc

#endif
