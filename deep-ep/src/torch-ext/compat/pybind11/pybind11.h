#pragma once

// kernels-community shim for DeepEP's host sources.
//
// Upstream DeepEP builds `deep_ep._C` as a pybind11 module. Hub kernels are
// Torch op libraries built against Python's limited ABI, so pybind11 cannot be
// used. Apart from the `register_apis` functions (which are compiled out with
// `DEEP_EP_KERNELS` and replaced by `torch-ext/torch_binding.cpp`), upstream
// only uses a few pybind11 *value* types in its function bodies: tuples of
// results, `None`, `cast`, `bytearray`, and `gil_scoped_release`. This header
// provides those on top of `c10::IValue`, so that the vendored sources compile
// unchanged. It must come before Torch's own include directory (which ships
// the real pybind11) on the include path, and the real pybind11 must never be
// included into the same translation unit.
//
// Values that are not plain IValues (DeepEP's `EventHandle` and deferred
// epilogue `std::function`s) are converted by `ivalue_caster` specializations
// in `torch-ext/deep_ep_holders.hpp`, which turn them into Torch custom class
// objects.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/List.h>

namespace pybind11 {

using object = c10::IValue;
using tuple = c10::IValue;

// Primary template: anything an IValue can be constructed from directly.
template <typename T, typename = void>
struct ivalue_caster {
    static c10::IValue cast(const T& value) { return c10::IValue(value); }
};

template <typename T>
c10::IValue cast(T&& value) {
    return ivalue_caster<std::decay_t<T>>::cast(std::forward<T>(value));
}

template <>
struct ivalue_caster<int> {
    static c10::IValue cast(const int& value) { return c10::IValue(static_cast<int64_t>(value)); }
};

template <typename T>
struct ivalue_caster<std::optional<T>> {
    static c10::IValue cast(const std::optional<T>& value) {
        return value.has_value() ? pybind11::cast(*value) : c10::IValue();
    }
};

template <>
struct ivalue_caster<std::vector<int>> {
    static c10::IValue cast(const std::vector<int>& value) {
        return c10::IValue(std::vector<int64_t>(value.begin(), value.end()));
    }
};

template <>
struct ivalue_caster<std::vector<at::Tensor>> {
    static c10::IValue cast(const std::vector<at::Tensor>& value) { return c10::IValue(value); }
};

// Any other vector becomes a generic list (e.g. a list of epilogue hooks).
template <typename T>
struct ivalue_caster<std::vector<T>> {
    static c10::IValue cast(const std::vector<T>& value) {
        c10::impl::GenericList list(c10::AnyType::get());
        list.reserve(value.size());
        for (const auto& item: value)
            list.push_back(pybind11::cast(item));
        return c10::IValue(std::move(list));
    }
};

template <typename... Args>
c10::IValue make_tuple(Args&&... args) {
    return c10::ivalue::Tuple::create(std::vector<c10::IValue>{pybind11::cast(std::forward<Args>(args))...});
}

template <typename... Ts>
struct ivalue_caster<std::tuple<Ts...>> {
    static c10::IValue cast(const std::tuple<Ts...>& value) {
        return std::apply([](const auto&... items) { return pybind11::make_tuple(items...); }, value);
    }
};

inline c10::IValue none() {
    return c10::IValue();
}

class bytearray {
    std::string data;

public:
    bytearray(const char* ptr, const std::size_t& size): data(ptr, size) {}

    template <typename T>
    T cast() const {
        static_assert(std::is_same_v<T, std::string>);
        return data;
    }
};

// Torch releases the GIL before it runs a custom op or a custom class method,
// so there is nothing left to release here.
struct gil_scoped_release {
    gil_scoped_release() = default;
};

}  // namespace pybind11

namespace py = pybind11;
