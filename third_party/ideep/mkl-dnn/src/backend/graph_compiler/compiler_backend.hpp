/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_COMPILER_BACKEND_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "interface/backend.hpp"
#include "utils/pm/pass_manager.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {

class compiler_backend_t : public backend {
    friend class compiler_partition_impl_t;

public:
    static compiler_backend_t &get_singleton() {
        static compiler_backend_t ins("compiler_backend", /*priority*/ 2.f);
        return ins;
    }

    /*! \brief Register defined patterns that can be processed with compiler backend
     */
    impl::pass::pass_registry_t &get_pass_registry() { return pass_registry_; }

    /*! \brief Get the size of logical tensor in the unit of bytes
     */
    size_t get_mem_size(const logical_tensor_t &lt) const override;

    /*! \brief Get the partition that can be processed by compiler backend
     */
    status_t get_partitions(
            graph_t &agraph, partition_policy_t policy) override;

private:
    compiler_backend_t(const std::string &backend_name, float priority)
        : backend(backend_name, priority) {
        bool ret = register_passes();
        if (!ret) {
            throw std::runtime_error(backend_name + " initialize failed");
        }
    };

    bool register_passes();

    impl::pass::pass_registry_t pass_registry_;
};

} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
