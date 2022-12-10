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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_JIT_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_JIT_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <common/generic_val.hpp>
#include <compiler/ir/ir_module.hpp>
#include <runtime/context.hpp>

namespace sc {

class jit_module;

// A jitted function that can be called in a module
class SC_API jit_function_t {
public:
    virtual ~jit_function_t() = default;

    virtual std::shared_ptr<jit_module> get_module() const = 0;
    virtual void *get_function_pointer() const = 0;

    /**
     * Calls the generic wrapper function with default stream context. The
     * module must have been compiled with `generate_wrapper=true`.
     * @param args the arguments
     */
    void call_generic_default(generic_val *args) const {
        call_generic(runtime::get_default_stream(), args);
    }

    /**
     * Calls the generic wrapper function with default stream context. The
     * module must have been compiled with `generate_wrapper=true`.
     * @param args the arguments
     */
    virtual void call_generic(
            runtime::stream_t *stream, generic_val *args) const = 0;

    /**
     * Calls the generic wrapper function and specifies a user-defined module
     * data. The module must have been compiled with `generate_wrapper=true`.
     * @param stream the runtime stream context
     * @param module_data the module data buffer. It should hold the module
     * scope vars and tensors
     * @param args the arguments
     */
    virtual void call_generic(runtime::stream_t *stream, void *module_data,
            generic_val *args) const {
        throw std::runtime_error("Not implemeneted");
    }

    virtual void *get_module_data() const { return nullptr; }

    template <typename Ret, typename... Args>
    Ret call_default(Args... args) const {
        return call<Ret>(runtime::get_default_stream(), args...);
    }

    template <typename Ret, typename... Args>
    Ret call(runtime::stream_t *stream, Args... args) const {
        // functype_old is kept for legacy mode for xbyak. Remove this when
        // xbyak JIT is updated to new JIT function interface
        using functype_old = Ret (*)(Args...);
        using functype = Ret (*)(void *, void *, Args...);
        assert(get_function_pointer());
        auto modu_ptr = get_module_data();
        if (modu_ptr) {
            return reinterpret_cast<functype>(get_function_pointer())(
                    stream, modu_ptr, args...);
        }
        return reinterpret_cast<functype_old>(get_function_pointer())(args...);
    }

    using generic_wrapper_t = void (*)(sc::generic_val *);
};

// The result of compiling an ir_module_t
class SC_INTERNAL_API jit_module {
public:
    statics_table_t globals_;
    jit_module() = default;
    jit_module(statics_table_t &&globals) : globals_(std::move(globals)) {}
    virtual void *get_address_of_symbol(const std::string &name) = 0;
    virtual std::shared_ptr<jit_function_t> get_function(
            const std::string &name)
            = 0;
    /// This method only exists to help with debugging.
    virtual std::vector<std::string> get_temp_filenames() const {
        return std::vector<std::string>();
    }
    virtual ~jit_module() = default;
};

class SC_INTERNAL_API general_jit_function_t : public jit_function_t {
    std::shared_ptr<jit_module> module_;
    void *funcptr_;
    void *wrapper_;
    std::string fname_;

public:
    void *get_module_data() const override {
        return module_->globals_.data_.data_;
    }
    general_jit_function_t(
            std::shared_ptr<jit_module> module, void *funcptr, void *wrapper)
        : module_(std::move(module)), funcptr_(funcptr), wrapper_(wrapper) {}
    general_jit_function_t(std::shared_ptr<jit_module> module, void *funcptr,
            void *wrapper, const std::string &name)
        : module_(std::move(module))
        , funcptr_(funcptr)
        , wrapper_(wrapper)
        , fname_(name) {}
    std::shared_ptr<jit_module> get_module() const override { return module_; }
    void *get_function_pointer() const override { return funcptr_; }
    void *get_wrapper_function_pointer() const { return wrapper_; }
    void call_generic(
            runtime::stream_t *stream, generic_val *args) const override;
    void call_generic(runtime::stream_t *stream, void *module_data,
            generic_val *args) const override;
};

// jit interface
class SC_API jit_engine_t {
public:
    context_ptr context_;
    jit_engine_t(context_ptr context) : context_(std::move(context)) {}

    // jit an ir_module_t into a jit_module
    virtual std::shared_ptr<jit_module> make_jit_module(
            const_ir_module_ptr module, bool generate_wrapper)
            = 0;

    /**
     * Generates a executable module and extract the entry function of the
     * ir_module_t
     * @param m module to generate. Must have entry function defined
     * @param generic if true, creates a type-erased wrapper for the
     *  function, users can further call `call_generic` on the
     *  generated executable
     * @return the executable function for the entry function
     * */
    std::shared_ptr<jit_function_t> get_entry_func(
            const ir_module_ptr &m, bool generic = false);
    virtual ~jit_engine_t() = default;

    static std::unique_ptr<jit_engine_t> make(const context_ptr &ctx);
    // negotiate with the JIT engine and get the target machine with as
    // many flags as possible the JIT can support in the user given target
    // machine
    static void set_target_machine(jit_kind kind, target_machine_t &tm);
};
} // namespace sc

#endif
