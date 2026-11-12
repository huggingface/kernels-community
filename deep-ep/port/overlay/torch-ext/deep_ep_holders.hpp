#pragma once

// kernels-community: Torch custom classes that carry DeepEP's stateful C++
// objects across the Python boundary (upstream exposes them as pybind11
// classes). Include this after `utils/event.hpp` and before the buffer
// headers, so the `ivalue_caster` specializations below are visible where
// upstream builds its result tuples.

#include <functional>
#include <memory>
#include <utility>

#include <torch/custom_class.h>

#include "utils/event.hpp"
#include "kernels/comm/api.hpp"

namespace deep_ep::kernels {

// `deep_ep._C.EventHandle`: a CUDA event plus the tensors it keeps alive.
struct EventHandleHolder: torch::CustomClassHolder {
    EventHandle handle;

    EventHandleHolder() = default;
    explicit EventHandleHolder(EventHandle handle): handle(std::move(handle)) {}

    void current_stream_wait() const { handle.current_stream_wait(); }
};

// A deferred epilogue (or wait hook), returned to Python and called later.
struct EpilogueHolder: torch::CustomClassHolder {
    std::function<c10::IValue()> fn;

    explicit EpilogueHolder(std::function<c10::IValue()> fn): fn(std::move(fn)) {}

    c10::IValue run() const { return fn(); }
};

// `deep_ep._C.Context`: a communication context shared by buffers.
struct ContextHolder: torch::CustomClassHolder {
    std::shared_ptr<comm::Context> context;

    explicit ContextHolder(std::shared_ptr<comm::Context> context): context(std::move(context)) {}
};

inline std::optional<EventHandle> unwrap_event(const std::optional<c10::intrusive_ptr<EventHandleHolder>>& event) {
    if (not event.has_value() or not event.value())
        return std::nullopt;
    return event.value()->handle;
}

inline c10::IValue wrap_context(const std::shared_ptr<comm::Context>& context) {
    if (context == nullptr)
        return c10::IValue();
    return c10::IValue(c10::make_intrusive<ContextHolder>(context));
}

}  // namespace deep_ep::kernels

namespace pybind11 {

template <>
struct ivalue_caster<deep_ep::EventHandle> {
    static c10::IValue cast(const deep_ep::EventHandle& value) {
        return c10::IValue(c10::make_intrusive<deep_ep::kernels::EventHandleHolder>(value));
    }
};

template <>
struct ivalue_caster<std::function<c10::IValue()>> {
    static c10::IValue cast(const std::function<c10::IValue()>& value) {
        return c10::IValue(c10::make_intrusive<deep_ep::kernels::EpilogueHolder>(value));
    }
};

}  // namespace pybind11
