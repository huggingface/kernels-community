#include <ATen/core/TensorBody.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <sycl/sycl.hpp>
#include <ATen/core/Array.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <cstdint>
#include <type_traits>
#include <array>
#include <c10/core/ScalarType.h>
#include <c10/xpu/XPUStream.h>
#include <ATen/xpu/XPUContext.h>

constexpr int MAX_DIMS = 12;

struct LoadWithoutCast {
  template <typename scalar_t>
  C10_DEVICE scalar_t load(char* base_ptr, uint32_t offset, int arg) {
    return c10::load(reinterpret_cast<scalar_t*>(base_ptr) + offset);
  }
};

struct StoreWithoutCast {
  template <typename scalar_t>
  C10_DEVICE void store(scalar_t value, char* base_ptr, uint32_t offset, int arg = 0) {
    *(reinterpret_cast<scalar_t*>(base_ptr) + offset) = value;
  }
};

template <template <int i> typename func, int end, int current = 0>
struct static_unroll {
  template <typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current + 1>::with_args(args...);
  }
};

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
  template <typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... args) {}
};

template <int current>
struct multi_outputs_store_helper {
  template <int ntensors, int num_outputs, typename... Args>
  static C10_HOST_DEVICE void apply(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<uint32_t, num_outputs> offsets,
      std::tuple<Args...> ret) {
    using T = typename std::tuple_element<current, std::tuple<Args...>>::type;
    T* to = reinterpret_cast<T*>(data[current]) + offsets[current];
    *to = std::get<current>(ret);
  }
};

template <int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static C10_DEVICE void apply(
      policy_t& self,
      args_t* args,
      offset_t offset,
      loader_t loader,
      int j,
      int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(
        self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int item_work_size, typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;
  StoreWithoutCast storer;
  int item_idx;
  int group_idx;
  int num_items_per_group;
  int group_work_size;

  multi_outputs_unroll(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      int item_idx,
      int group_idx,
      int num_items_per_group)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        item_idx(item_idx),
        group_idx(group_idx),
        num_items_per_group(num_items_per_group),
        group_work_size(item_work_size * num_items_per_group) {}

  inline bool check_inbounds(int item_work_elem) const {
    return (item_idx + item_work_elem * num_items_per_group < remaining);
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int item_idx_ = item_idx;
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      if (item_idx_ >= remaining) {
        return;
      }
      int linear_idx = item_idx_ + group_work_size * group_idx;
      auto offset = input_offset_calculator.get(linear_idx);
      static_unroll<unroll_load_helper, arity>::with_args(
          *this, args, offset, loader, i, num_outputs);
      item_idx_ += num_items_per_group;
    }
  }

  template <typename return_t>
  inline void store(return_t* from) {
    int item_idx_ = item_idx;
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      if (item_idx_ >= this->remaining) {
        return;
      }
      int linear_idx = item_idx_ + group_work_size * group_idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      static_unroll<multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      item_idx_ += num_items_per_group;
    }
  }
};

template <int item_work_size, typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  return_t results[item_work_size];
  args_t args[item_work_size];

  policy.load(args);

#pragma unroll
  for (int i = 0; i < item_work_size; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = std::apply(f, args[i]);
    }
  }

  policy.store(results);
}

template <int num_outputs, typename func_t, typename array_t, typename in_calc_t, typename out_calc_t>
struct UnrolledElementwiseForMultiOutputsKernel {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item_id) const {
    int grpsz = item_id.get_local_range(0);
    int grpid = item_id.get_group(0);
    int lid = item_id.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = multi_outputs_unroll<item_work_size, array_t, in_calc_t, out_calc_t, num_outputs>(
        data_, remaining, ic_, oc_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseForMultiOutputsKernel(int numel, func_t f, array_t data, in_calc_t ic, out_calc_t oc)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
};

template <typename Value>
struct IntDivider {
  IntDivider() = default;
  IntDivider(Value d) : divisor(d) {}

  C10_HOST_DEVICE inline Value div(Value n) const {
    return n / divisor;
  }
  C10_HOST_DEVICE inline Value mod(Value n) const {
    return n % divisor;
  }
  C10_HOST_DEVICE inline auto divmod(Value n) const {
    return std::make_pair(n / divisor, n % divisor);
  }

  Value divisor;
};

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  using stride_t = std::conditional_t<signed_strides, std::make_signed_t<index_t>, index_t>;
  using offset_type = at::detail::Array<stride_t, std::max<int>(NARGS, 1)>;

  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides, const int64_t* element_sizes = nullptr)
      : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i = 0; i < dims; i++) {
      sizes_[i] = IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.first;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.second * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

template <int N>
static OffsetCalculator<N> make_input_offset_calculator(const at::TensorIteratorBase& iter) {
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(const at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

static inline int64_t syclMaxWorkItemsPerSubSlice(at::DeviceIndex dev_id = c10::xpu::getCurrentXPUStream().device_index()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t simd_width = dev_prop->sub_group_sizes[0];
  int64_t eu_count = dev_prop->gpu_eu_count_per_subslice;
  return simd_width * eu_count;
}

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

template <typename ker_t>
static inline void sycl_kernel_submit(int64_t global_range, int64_t local_range, ::sycl::queue q, ker_t ker) {
  q.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(local_range)),
    ker
  );
}

template <int num_outputs, typename func_t, typename array_t, typename in_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t ic,
    out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto ker = UnrolledElementwiseForMultiOutputsKernel<num_outputs, func_t, array_t, in_calc_t, out_calc_t>(N, f, data, ic, oc);
  using ker_t = decltype(ker);

  int wg_sz = syclMaxWorkItemsPerSubSlice();
  int num_wg = ceil_div<int>(N, ker_t::item_work_size * wg_sz);
  sycl_kernel_submit(wg_sz * num_wg, wg_sz, c10::xpu::getCurrentXPUStream().queue(), ker);
}

template <int N>
struct TrivialOffsetCalculator {
  using offset_type = at::detail::Array<uint32_t, std::max<int>(N, 1)>;

  C10_HOST_DEVICE offset_type get(uint32_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < N; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(at::TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  constexpr int num_outputs = std::tuple_size<output_t>::value;
  constexpr int num_inputs = traits::arity;
  constexpr int ntensors = num_outputs + num_inputs;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  if (iter.is_contiguous()) {
    auto input_calc = TrivialOffsetCalculator<num_inputs>();
    auto output_calc = TrivialOffsetCalculator<num_outputs>();
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  } else {
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  }
}

template <typename func_t>
void gpu_kernel_multiple_outputs(at::TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_xpu());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_multiple_outputs(sub_iter, f);
    }
    return;
  }

  gpu_kernel_multiple_outputs_impl(iter, f);
}