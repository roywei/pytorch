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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BODY_GENERATOR_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BODY_GENERATOR_HPP
#include <memory>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/graph/tensor_detail.hpp>
#include <compiler/ir/sc_stmt.hpp>
namespace sc {

class fusion_manager;
struct graph_tensor;

namespace tuner {
struct config_space;
using config_space_ptr = std::unique_ptr<tuner::config_space>;
} // namespace tuner

// fix-me: (lowering) change to config_ptr
using config_ptr2 = std::shared_ptr<void>;

/**
 * The generator base class to generate IR for the body of an Op
 * */
struct body_generator_base_t {
    std::vector<logical_tensor_t> in_tensors_;
    std::vector<logical_tensor_t> out_tensors_;
    /**
     * simply judge the config is valid or not, then we needn't to generate
     * others in graph
     * */
    virtual bool is_valid_config(
            const context_ptr &ctx, const void *config) const {
        return true;
    }
    /**
     * Generates the tensor IR to the current IR builder.
     * @param ctx the context
     * @param config the configuration
     * @param fusion the fusion manager. The generator should push the anchors
     * to the fusion manager
     * @param inputs the input args of the Op
     * @param outputs the output tensors of the Op
     * @param loops the for-loops to be later scheduled by schedule_loops()
     * @return generate status, e.g. success.
     * */
    virtual bool generate(context_ptr ctx, const void *config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const = 0;

    virtual float get_gflop() const = 0;

    sc_data_type_t get_in_dtypes(size_t idx) const {
        return in_tensors_.at(idx).dtype_;
    }

    sc_data_type_t get_out_dtypes(size_t idx) const {
        return out_tensors_.at(idx).dtype_;
    }

    //   std::vector<sc_data_type_t> infer_out_dtypes() const {
    //     if (in_tensors_.size()
    //       && (in_tensors_.at(0).dtype_ == datatypes::u8
    //         || in_tensors_.at(1).dtype_ == datatypes::s8)) {
    //       return {datatypes::s32};
    //     } else {
    //       return {datatypes::f32};
    //     }
    //   }

    /**
     * Returns the type-erased default config. You can use `get()` method in
     * the returned object to get the pointer, which can be used in `generate`
     * */
    virtual std::shared_ptr<void> get_default_config(context_ptr ctx) const = 0;

    virtual void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const = 0;

    virtual ~body_generator_base_t() = default;

    body_generator_base_t(std::vector<logical_tensor_t> &&in_tensors,
            std::vector<logical_tensor_t> &&out_tensors)
        : in_tensors_(std::move(in_tensors))
        , out_tensors_(std::move(out_tensors)) {}

    body_generator_base_t(const std::vector<logical_tensor_t> &in_tensors,
            const std::vector<logical_tensor_t> &out_tensors)
        : in_tensors_(in_tensors), out_tensors_(out_tensors) {}
};

using body_generator_ptr = std::unique_ptr<body_generator_base_t>;

template <typename TConfig>
struct body_generator_t : public body_generator_base_t {
    virtual bool is_valid_config(
            const context_ptr &ctx, const TConfig &config) const {
        return true;
    }
    bool is_valid_config(
            const context_ptr &ctx, const void *config) const override {
        return is_valid_config(ctx, *reinterpret_cast<const TConfig *>(config));
    }
    virtual bool generate(context_ptr ctx, const TConfig &config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const = 0;

    bool generate(context_ptr ctx, const void *config, fusion_manager *fusion,
            const std::vector<expr> &inputs, const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const override {
        return generate(ctx, *reinterpret_cast<const TConfig *>(config), fusion,
                inputs, outputs, loops);
    }

    virtual void schedule_loops(context_ptr ctx, const TConfig &config,
            stmt body, std::vector<for_loop> &fors) const = 0;

    void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const override {
        schedule_loops(
                ctx, *reinterpret_cast<const TConfig *>(config), body, fors);
    }

    body_generator_t(const std::vector<logical_tensor_t> &ins,
            const std::vector<logical_tensor_t> &outs)
        : body_generator_base_t {ins, outs} {}

    body_generator_t(std::vector<logical_tensor_t> &&ins,
            std::vector<logical_tensor_t> &&outs)
        : body_generator_base_t {std::move(ins), std::move(outs)} {}
};

} // namespace sc

#endif
