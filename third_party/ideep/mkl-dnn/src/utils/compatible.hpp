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

#ifndef UTILS_COMPATIBLE_HPP
#define UTILS_COMPATIBLE_HPP

#ifdef DNNL_GRAPH_SUPPORT_CXX17
#include <optional>
#endif
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

class bad_any_cast_t : public std::bad_cast {
public:
    const char *what() const noexcept override { return "bad any_cast"; }
};

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename Ret, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...));

template <typename Ret, typename F, typename Arg, typename... Rest>
Arg first_argument_helper(Ret (F::*)(Arg, Rest...) const);

template <typename F>
decltype(first_argument_helper(&F::operator())) first_argument_helper(F);

template <typename T>
using first_argument = typename std::decay<decltype(
        first_argument_helper(std::declval<T>()))>::type;

// any structure
// now we only use this any struct for the project.
class any_t {
public:
    any_t() = default;
    any_t(any_t &&v) {
        clear();
        avtable_ = v.avtable_;
        v.avtable_ = nullptr;
    }
    any_t(const any_t &v) { avtable_ = v.avtable_; }

    template <typename T,
            typename = enable_if_t<!std::is_same<T, any_t &>::value>>
    any_t(T &&v) {
        clear();
        using value_type = typename std::decay<
                typename std::remove_reference<T>::type>::type;
        avtable_ = std::make_shared<vtable_t<value_type>>(std::forward<T>(v));
    }

    any_t &operator=(const any_t &v) {
        any_t(v).swap(*this);
        return *this;
    }
    any_t &operator=(any_t &&v) {
        v.swap(*this);
        any_t().swap(v);
        return *this;
    }
    template <typename T>
    any_t &operator=(T &&v) {
        any_t(std::forward<T>(v)).swap(*this);
        return *this;
    }

    void clear() {
        if (avtable_) { avtable_ = nullptr; }
    }
    void swap(any_t &v) { std::swap(avtable_, v.avtable_); }
    bool empty() { return avtable_ == nullptr; }
    const std::type_info &type() const {
        return avtable_ ? avtable_->type() : typeid(void);
    }

    template <typename T, typename T1, typename... Args>
    bool match(T defaults, T1 func1, Args &&... args) const {
        using MatchedT = first_argument<T1>;
        if (type() == typeid(MatchedT)) {
            func1(static_cast<vtable_t<MatchedT> *>(avtable_.get())->value_);
            return true;
        }
        return match(std::forward<T>(defaults), std::forward<Args>(args)...);
    }

    template <typename T, typename T1>
    bool match(T defaults, T1 func1) const {
        using MatchedT = first_argument<T1>;
        if (type() == typeid(MatchedT)) {
            func1(static_cast<vtable_t<MatchedT> *>(avtable_.get())->value_);
            return true;
        }
        defaults();
        return false;
    }

private:
    struct any_vtable_t {
        virtual ~any_vtable_t() = default;
        virtual const std::type_info &type() = 0;
        virtual std::shared_ptr<any_vtable_t> get_vtable() = 0;
    };
    template <typename T>
    struct vtable_t : public any_vtable_t {
        vtable_t(const T &value) : value_(value) {}
        vtable_t(T &&value) : value_(std::forward<T>(value)) {}
        vtable_t &operator=(const vtable_t &) = delete;
        const std::type_info &type() override { return typeid(T); }
        std::shared_ptr<any_vtable_t> get_vtable() override {
            return std::make_shared<vtable_t>(value_);
        }
        T value_;
    };

    std::shared_ptr<any_vtable_t> avtable_ = nullptr;

    template <typename T>
    friend T *any_cast(any_t *v);
};

template <typename T>
T *any_cast(any_t *v) {
    using value_type = typename std::remove_cv<T>::type;
    return v && v->type() == typeid(T)
            ? &static_cast<any_t::vtable_t<value_type> *>(v->avtable_.get())
                       ->value_
            : nullptr;
}

template <typename T>
inline const T *any_cast(const any_t *v) {
    return any_cast<T>(const_cast<any_t *>(v));
}

template <typename T>
inline T any_cast(any_t &v) {
    using nonref = typename std::remove_reference<T>::type;
    auto val = any_cast<nonref>(&v);
    if (val) {
        using ref_type = typename std::conditional<std::is_reference<T>::value,
                T, typename std::add_lvalue_reference<T>::type>::type;
        return static_cast<ref_type>(*val);
    } else {
        throw bad_any_cast_t {};
    }
}

template <typename T>
inline T any_cast(const any_t &v) {
    return any_cast<T>(const_cast<any_t &>(v));
}

template <typename T>
inline T any_cast(any_t &&v) {
    static_assert(std::is_rvalue_reference<T &&>::value
                    || std::is_const<
                            typename std::remove_reference<T>::type>::value,
            "should not be used getting non const reference and move object");
    // return any_cast<std::forward<T>>(v);
    return any_cast<T>(v);
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#ifdef DNNL_GRAPH_SUPPORT_CXX17
template <typename T>
using optional = std::optional<T>;
#else
struct nullopt_t {
    enum class _construct { _token };
    explicit constexpr nullopt_t(_construct) {}
};

constexpr const nullopt_t nullopt {nullopt_t::_construct::_token};

template <typename T>
class optional_impl_t {
public:
    optional_impl_t() : is_null_(true) {}
    optional_impl_t(const T &value) {
        is_null_ = false;
        new (&value_) T(value);
    }

    optional_impl_t(const optional_impl_t<T> &opt) {
        is_null_ = opt.is_null_;
        if (!is_null_) { new (&value_) T(opt.value()); }
    }

    optional_impl_t(nullopt_t) : is_null_(true) {}

    ~optional_impl_t() {
        // explicitly deconstructor
        if (!is_null_) { reinterpret_cast<T *>(&value_)->~T(); }
    }

    void swap(optional_impl_t<T> &another) {
        std::swap(value_, another.value_);
        std::swap(is_null_, another.is_null_);
    }

    optional_impl_t<T> &operator=(const optional_impl_t<T> &another) {
        (optional_impl_t<T>(another)).swap(*this);
        return *this;
    }

    optional_impl_t<T> &operator=(const T &value) {
        (optional_impl_t<T>(value)).swap(*this);
        return *this;
    }

    optional_impl_t<T> &operator=(nullopt_t) {
        (optional_impl_t<T>()).swap(*this);
        return *this;
    }

    const T &operator*() const { return *reinterpret_cast<const T *>(&value_); }
    T &operator*() { return *reinterpret_cast<T *>(&value_); }

    const T &value() const {
        if (is_null_) {
            const std::string err_msg = "bad optional access";
            throw std::logic_error(err_msg);
        }
        return *reinterpret_cast<const T *>(&value_);
    }

    bool operator==(const optional_impl_t<T> &other) const {
        return this->is_null_ == other.is_null_
                && (this->is_null_ == true || this->value() == other.value());
    }

    bool has_value() const { return !is_null_; }

private:
    typename std::aligned_storage<sizeof(T), alignof(T)>::type value_;
    bool is_null_;
};

template <typename T>
using optional = optional_impl_t<T>;
#endif // DNNL_GRAPH_SUPPORT_CXX17

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
