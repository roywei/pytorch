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

#ifndef UNIT_TEST_COMMON_HPP
#define UNIT_TEST_COMMON_HPP

#include <memory>
#include <vector>

#include "interface/engine.hpp"
#include "interface/stream.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
#include "test_thread.hpp"
#endif
namespace impl = dnnl::graph::impl;

#ifdef DNNL_GRAPH_WITH_SYCL
cl::sycl::device &get_device();
cl::sycl::context &get_context();
void *sycl_alloc(size_t n, const void *dev, const void *ctx);
void sycl_free(void *ptr, const void *ctx);
#endif // DNNL_GRAPH_WITH_SYCL

impl::engine_t &get_engine();

impl::stream_t &get_stream();

impl::engine_kind_t get_test_engine_kind();

void set_test_engine_kind(impl::engine_kind_t kind);

namespace test {

#ifdef DNNL_GRAPH_WITH_SYCL
constexpr size_t usm_alignment = 16;
#endif

template <typename T>
class TestAllocator {
public:
    typedef T value_type;

    T *allocate(size_t num_elements) {
        if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
            return reinterpret_cast<T *>(cl::sycl::aligned_alloc(usm_alignment,
                    num_elements * sizeof(T), get_device(), get_context(),
                    cl::sycl::usm::alloc::shared));
#else
            return reinterpret_cast<T *>(malloc(num_elements * sizeof(T)));
#endif
        } else if (get_test_engine_kind() == impl::engine_kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
            return reinterpret_cast<T *>(cl::sycl::aligned_alloc(usm_alignment,
                    num_elements * sizeof(T), get_device(), get_context(),
                    cl::sycl::usm::alloc::shared));
#else
            return nullptr;
#endif
        } else {
            return nullptr;
        }
    }

    void deallocate(T *ptr, size_t) {
        if (!ptr) return;

        if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
            cl::sycl::free(ptr, get_context());
#else
            free(ptr);
#endif
        } else if (get_test_engine_kind() == impl::engine_kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
            cl::sycl::free(ptr, get_context());
#endif
        } else {
        }
    }

    template <typename U>
    struct rebind {
        using other = TestAllocator<U>;
    };
};

template <class T, class U>
bool operator==(const TestAllocator<T> &, const TestAllocator<U> &) {
    return true;
}

template <class T, class U>
bool operator!=(const TestAllocator<T> &, const TestAllocator<U> &) {
    return false;
}

template <typename T>
#ifdef DNNL_GRAPH_WITH_SYCL
using vector = std::vector<T, TestAllocator<T>>;
#else
using vector = std::vector<T>;
#endif // DNNL_GRAPH_WITH_SYCL
} // namespace test

#endif
