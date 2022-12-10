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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_INTRINSICS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_INTRINSICS_HPP

#include <ostream>
#include <string>
#include "sc_expr.hpp"
#include <microkernel/cpu/brgemm_common.hpp>
#include <util/any_map.hpp>
namespace sc {
class ir_visitor_t;
struct intrinsic_handler_t {
    std::string name_;
    virtual void on_initialize(intrin_call_node &node) = 0;
    intrinsic_handler_t(const std::string &name);
    virtual ~intrinsic_handler_t() = default;
};

// the indices of arguments of the brgemm intrinsices
namespace brgemm_args {
constexpr int A = 0;
constexpr int B = 1;
constexpr int C = 2;
constexpr int NUM = 3;
constexpr int M = 4;
constexpr int N = 5;
constexpr int K = 6;
constexpr int LDA = 7;
constexpr int LDB = 8;
constexpr int LDC = 9;
constexpr int STRIDE_A = 10;
constexpr int STRIDE_B = 11;
constexpr int LEN = 12;
// extra +1 for c_buf
constexpr int NUM_ARGS_CPU
        = STRIDE_B + brgemm::postops_data_init_func_nargs + 2;
// extra +1 for c_buf
constexpr int NUM_ARGS_LIST = LEN + brgemm::postops_data_init_func_nargs + 2;

struct cpu_t {
    // use init_update or update
    bool init_;
    bool operator==(const cpu_t &other) const { return init_ == other.init_; }
    bool operator!=(const cpu_t &other) const { return init_ != other.init_; }
};

namespace extra_args_offset {
constexpr int dtypeA = 0;
constexpr int dtypeB = 1;
constexpr int brg_attrs = 2;
constexpr int bd_mask = 3;
constexpr int postops_setting = 4;
constexpr int cache_nargs = postops_setting + 1;
constexpr int postops_data = 5;
constexpr int c_buf = 6;
constexpr int nargs = c_buf + 1;
} // namespace extra_args_offset

struct extra_args_t {
    bool is_cpu_;
    sc_data_type_t dtype_A_ = datatypes::undef; // element dtype of mat A
    sc_data_type_t dtype_B_ = datatypes::undef; // element dtype of mat B
    sc_data_type_t dtype_C_ = datatypes::undef; // element dtype of mat C
    sc_brgemm_attrs_t brg_attrs_; // brgemm attrs
    sc_brgemm_bd_mask_t bd_mask_; // bd mask
    sc_brgemm_postops_setting_t postops_setting_; // post ops setting

    union {
        cpu_t cpu_;
    };
    extra_args_t(const cpu_t &g, sc_data_type_t dtypeA,
            sc_data_type_t dtypeB = datatypes::undef,
            sc_data_type_t dtypeC = datatypes::undef,
            const sc_brgemm_attrs_t &brg_attrs = sc_brgemm_attrs_t(),
            const sc_brgemm_bd_mask_t &bd_mask = sc_brgemm_bd_mask_t(),
            const sc_brgemm_postops_setting_t &brg_postops
            = sc_brgemm_postops_setting_t())
        : is_cpu_(true)
        , dtype_A_(dtypeA)
        , dtype_B_(dtypeB == datatypes::undef ? dtypeA : dtypeB)
        , dtype_C_(dtypeC == datatypes::undef ? dtypeA : dtypeC)
        , brg_attrs_(brg_attrs)
        , bd_mask_(bd_mask)
        , postops_setting_(brg_postops)
        , cpu_(g) {}
    bool operator==(const extra_args_t &other) const {
        return is_cpu_ == other.is_cpu_ && dtype_A_ == other.dtype_A_
                && dtype_B_ == other.dtype_B_ && dtype_C_ == other.dtype_C_
                && brg_attrs_ == other.brg_attrs_ && bd_mask_ == other.bd_mask_
                && postops_setting_ == other.postops_setting_
                && cpu_ == other.cpu_;
    }
    bool operator!=(const extra_args_t &other) const {
        return !(*this == other);
    }
};

extern sc_data_type_t arg_types[NUM_ARGS_CPU];
extern sc_data_type_t list_arg_types[NUM_ARGS_LIST];
} // namespace brgemm_args

intrinsic_handler_t &get_intrinsic_handler(intrin_type intrin);

} // namespace sc

#endif
