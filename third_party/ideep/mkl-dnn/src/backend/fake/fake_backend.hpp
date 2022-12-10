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

#ifndef BACKEND_FAKE_FAKE_BACKEND_HPP
#define BACKEND_FAKE_FAKE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "interface/backend.hpp"
#include "interface/c_types_map.hpp"
#include "interface/logical_tensor.hpp"

#include "utils/compatible.hpp"
#include "utils/pm/pass_manager.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace fake_impl {

class fake_backend_t : public backend {
    friend class fake_partition_impl_t;

public:
    static fake_backend_t &get_singleton() {
        static fake_backend_t ins("fake_backend", /*priority*/ 0.f);
        return ins;
    }

    impl::pass::pass_registry_t &get_pass_registry() { return pass_registry_; }

    size_t get_mem_size(const impl::logical_tensor_t &lt) const override {
        UNUSED(lt);
        return static_cast<size_t>(-1);
    }

    status_t get_partitions(
            impl::graph_t &agraph, impl::partition_policy_t policy) override {
        impl::pass::pass_manager_t pm(get_pass_registry());
        pm.run_passes(agraph, "", policy);
        return status::success;
    }

private:
    fake_backend_t(const std::string &name, float priority);
    bool register_passes();
    impl::pass::pass_registry_t pass_registry_;
};

} // namespace fake_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
