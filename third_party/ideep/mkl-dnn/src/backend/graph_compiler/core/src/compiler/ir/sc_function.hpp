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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_FUNCTION_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_FUNCTION_HPP

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "sc_expr.hpp"
#include "sc_stmt.hpp"

namespace sc {

/**
 * The function IR node
 * @param name the function name
 * @param params_ the function parameters. The elements should be var or tensor
 * @param body_ the body of the function
 * @param ret_type_ the return type of the function
 * */
class func_base : public std::enable_shared_from_this<func_base> SC_LEAK_CHECK(
                          func_base) {
public:
    std::string name_;
    std::vector<expr> params_;
    stmt body_;
    sc_data_type_t ret_type_;
    // optional attributes, nullable. In most cases, use attr() to get the
    // attributes
    std::unique_ptr<any_map_t> attr_;
    // returns attr_ if is defined or creates and sets a new any_map_t if not
    // defined
    any_map_t &attr();
    // the function declaration. It has the same prototype of this function
    // will be non-null only when body is not empty
    func_t decl_;

    ~func_base();
    /**
     * Dump the IR node as string to the ostream
     * @param os the output stream
     * */
    void to_string(ostream &os) const;
    func_base(const std::string &name, const std::vector<expr> &params,
            stmt body, sc_data_type_t ret_type);
    /**
     * Does shallow copying copy on this IR node.
     * Makes a new IR node with the same type and the same values of fields.
     * */
    func_t remake() const;

    /**
     * Check if `this` is same as another IR node. May change the internal
     * states of `ctx`
     * @param other the other IR node to compare
     * @param ctx the context of the comparison: how "same" is defined,
     *  the internal states, etc.
     * @return true if the nodes are the same
     * */
    bool equals(const func_c &f, ir_comparer &ctx) const;

    /**
     * Check if `this` is same as another IR node. It will create a new
     * default ir_comparer context to do comparison.
     * @param other the other IR node to compare
     * @return true if the nodes are the same
     * */
    bool equals(const func_c &f) const;
};

// Operator << overload for std::ostream on func_t
SC_INTERNAL_API extern ostream &operator<<(ostream &os, const func_c &e);

// Operator << overload for std::ostream on func_base*
SC_INTERNAL_API extern ostream &operator<<(ostream &os, const func_base *e);

/**
 * Removes redundant loops with parallel attribute. Will reserve the outmost
 * one loop with parallel.
 * @param body the stmts for parallel remove
 * */
void remove_parallel(func_t body);

} // namespace sc

namespace std {
template <>
struct hash<sc::func_t> {
    std::size_t operator()(const sc::func_t &k) const {
        return hash<std::shared_ptr<sc::func_base>>()(k);
    }
};

template <>
struct equal_to<sc::func_t> {
    bool operator()(const sc::func_t &k, const sc::func_t &k2) const {
        return k.ptr_same(k2);
    }
};

} // namespace std

#endif
