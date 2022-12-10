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

#ifndef INTERFACE_TENSOR_HPP
#define INTERFACE_TENSOR_HPP

#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "c_types_map.hpp"
#include "engine.hpp"
#include "logical_tensor.hpp"

struct dnnl_graph_tensor {
public:
    dnnl_graph_tensor() = default;

    dnnl_graph_tensor(const dnnl::graph::impl::logical_tensor_t &lt,
            const dnnl::graph::impl::engine_t *eng, void *handle)
        : tensor_desc_(lt), eng_(eng), data_handle_(handle) {}

    bool is(dnnl::graph::impl::data_type_t dtype) const {
        return dtype == tensor_desc_.data_type;
    }

    template <typename Value>
    typename std::add_pointer<Value>::type get_data_handle() const {
        return is(get_data_type<Value>())
                ? reinterpret_cast<typename std::add_pointer<Value>::type>(
                        data_handle_)
                : nullptr;
    }

    void *get_data_handle() const { return data_handle_; }

    void *get_void_data_handle_if_is(
            dnnl::graph::impl::data_type_t type) const {
        return is(type) ? data_handle_ : nullptr;
    }

    void set_data_handle(void *handle) { data_handle_ = handle; }

    const dnnl::graph::impl::logical_tensor_t &get_logical_tensor() const {
        return tensor_desc_;
    }

    operator bool() const { return data_handle_ != nullptr; }

    const dnnl::graph::impl::engine_t *get_engine() const { return eng_; }

private:
    template <typename T>
    dnnl::graph::impl::data_type_t get_data_type() const {
        if (std::is_same<T, float>::value)
            return dnnl::graph::impl::data_type::f32;
        else if (std::is_same<T, int8_t>::value)
            return dnnl::graph::impl::data_type::s8;
        else if (std::is_same<T, uint8_t>::value)
            return dnnl::graph::impl::data_type::u8;
        else
            return dnnl::graph::impl::data_type::undef;
    }

    dnnl::graph::impl::logical_tensor_t tensor_desc_;
    const dnnl::graph::impl::engine_t *eng_ {nullptr};
    void *data_handle_ {nullptr};
};

#endif
