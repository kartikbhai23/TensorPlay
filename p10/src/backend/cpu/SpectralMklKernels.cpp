// CPU spectral transforms backed by the transform-plan interface.
//
// A plan describes the signal layout once (precision, sizes, batch distance
// and strides), then the same descriptor runs any number of forward or
// backward transforms.  Transform dimensions are ordered by input stride so
// collapsing the non-transformed axes into one batch axis keeps the data
// contiguous for the plan.

#include "Tensor.h"
#include "DType.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OutWrite.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#if defined(USE_MKL)
#include <mkl_dfti.h>
#include <mkl_version.h>
#endif

namespace tensorplay {
namespace cpu {
namespace {

#if defined(USE_MKL)

struct DftiDescriptorDeleter {
    void operator()(DFTI_DESCRIPTOR* desc) const {
        if (desc != nullptr) DftiFreeDescriptor(&desc);
    }
};

using DftiDescriptor = std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter>;
using SharedDescriptor = std::shared_ptr<DFTI_DESCRIPTOR>;

// Everything that changes the committed plan is part of the key.  Building a
// descriptor is far more expensive than running one transform, so revisiting a
// shape must reuse it.
struct PlanKey {
    std::vector<MKL_LONG> sizes;
    std::vector<MKL_LONG> istrides;
    std::vector<MKL_LONG> ostrides;
    DFTI_CONFIG_VALUE precision;
    DFTI_CONFIG_VALUE signal_type;
    MKL_LONG batch;
    MKL_LONG distance_in;
    MKL_LONG distance_out;
    int64_t normalization;
    bool forward;
    bool complex_input;
    bool complex_output;

    bool operator==(const PlanKey& o) const {
        return sizes == o.sizes && istrides == o.istrides && ostrides == o.ostrides &&
               precision == o.precision && signal_type == o.signal_type &&
               batch == o.batch && distance_in == o.distance_in &&
               distance_out == o.distance_out && normalization == o.normalization &&
               forward == o.forward && complex_input == o.complex_input &&
               complex_output == o.complex_output;
    }
};

class PlanCache {
public:
    SharedDescriptor get(const PlanKey& key) {
        std::lock_guard<std::mutex> guard(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->key == key) {
                entries_.splice(entries_.begin(), entries_, it);
                return it->desc;
            }
        }
        return nullptr;
    }

    void put(const PlanKey& key, SharedDescriptor desc) {
        std::lock_guard<std::mutex> guard(mutex_);
        entries_.push_front({key, std::move(desc)});
        if (entries_.size() > 16) entries_.pop_back();
    }

private:
    struct Entry {
        PlanKey key;
        SharedDescriptor desc;
    };
    std::mutex mutex_;
    std::list<Entry> entries_;
};

PlanCache& plan_cache() {
    // Heap allocation on purpose: tearing the descriptors down during module
    // unload races the runtime teardown that owns them.  The cache is bounded
    // by a handful of entries, so the leak is negligible.
    static PlanCache* cache = new PlanCache();
    return *cache;
}

void check_dfti(MKL_LONG status, const char* what) {
    TP_CHECK(status == DFTI_NO_ERROR, "MKL DFTI ", what, " failed with status ", status);
}

std::vector<int64_t> wrap_mkl_dims(const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t ndim = self.dim();
    std::vector<int64_t> result;
    result.reserve(dims.size());
    for (int64_t dim : dims) {
        TP_CHECK_INDEX(dim >= -ndim && dim < ndim,
                       "Dimension out of range (expected to be in range of [",
                       -ndim, ", ", ndim - 1, "], but got ", dim, ")");
        if (dim < 0) dim += ndim;
        TP_CHECK(std::find(result.begin(), result.end(), dim) == result.end(),
                 "FFT dims must be unique");
        result.push_back(dim);
    }
    return result;
}

DFTI_CONFIG_VALUE precision_of(DType dtype) {
    const ScalarType real = toRealValueType(dtype);
    TP_CHECK(real == ScalarType::Float32 || real == ScalarType::Float64,
             "MKL FFT does not support dtype ", dtype);
    return real == ScalarType::Float32 ? DFTI_SINGLE : DFTI_DOUBLE;
}

void set_scale(DFTI_DESCRIPTOR* desc, int64_t normalization, bool forward,
               int64_t n) {
    if (normalization == 0) return;
    TP_CHECK(normalization == 1 || normalization == 2,
             "Unsupported normalization type ", normalization);
    const double scale = normalization == 1
        ? 1.0 / std::sqrt(static_cast<double>(n))
        : 1.0 / static_cast<double>(n);
    check_dfti(DftiSetValue(desc, forward ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE, scale),
               "set scale");
}

SharedDescriptor plan_mkl_fft(const Tensor& in, const Tensor& out,
                              const std::vector<int64_t>& signal_sizes,
                              bool complex_input, bool complex_output,
                              int64_t normalization, bool forward) {
    const int64_t signal_ndim = static_cast<int64_t>(signal_sizes.size()) - 1;
    const DFTI_CONFIG_VALUE prec = precision_of(isComplexType(in.dtype()) ? in.dtype() : out.dtype());
    const DFTI_CONFIG_VALUE signal_type = forward
        ? (complex_input ? DFTI_COMPLEX : DFTI_REAL)
        : (complex_output ? DFTI_COMPLEX : DFTI_REAL);

    PlanKey key;
    key.precision = prec;
    key.signal_type = signal_type;
    key.batch = static_cast<MKL_LONG>(signal_sizes[0]);
    key.normalization = normalization;
    key.forward = forward;
    key.complex_input = complex_input;
    key.complex_output = complex_output;
    key.sizes.resize(static_cast<size_t>(signal_ndim));
    for (int64_t i = 0; i < signal_ndim; ++i) {
        key.sizes[static_cast<size_t>(i)] =
            static_cast<MKL_LONG>(signal_sizes[static_cast<size_t>(i + 1)]);
    }
    const auto in_strides = in.strides();
    const auto out_strides = out.strides();
    key.distance_in = static_cast<MKL_LONG>(in_strides[0]);
    key.distance_out = static_cast<MKL_LONG>(out_strides[0]);
    // The descriptor's stride arrays are indexed by 1 + signal dimension, where
    // slot 0 is an ignored offset; the batch spacing is a separate setting.
    key.istrides.resize(static_cast<size_t>(signal_ndim) + 1);
    key.ostrides.resize(static_cast<size_t>(signal_ndim) + 1);
    key.istrides[0] = 0;
    key.ostrides[0] = 0;
    for (int64_t i = 1; i <= signal_ndim; ++i) {
        key.istrides[static_cast<size_t>(i)] = static_cast<MKL_LONG>(in_strides[static_cast<size_t>(i)]);
        key.ostrides[static_cast<size_t>(i)] = static_cast<MKL_LONG>(out_strides[static_cast<size_t>(i)]);
    }

    if (SharedDescriptor cached = plan_cache().get(key)) return cached;

    DFTI_DESCRIPTOR* raw = nullptr;
    if (signal_ndim == 1) {
        check_dfti(DftiCreateDescriptor(&raw, prec, signal_type, 1, key.sizes[0]),
                   "create descriptor");
    } else {
        check_dfti(DftiCreateDescriptor(&raw, prec, signal_type,
                                        static_cast<MKL_LONG>(signal_ndim), key.sizes.data()),
                   "create descriptor");
    }
    SharedDescriptor desc(raw, DftiDescriptorDeleter{});

    check_dfti(DftiSetValue(desc.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE), "set placement");
    check_dfti(DftiSetValue(desc.get(), DFTI_NUMBER_OF_TRANSFORMS, key.batch), "set batch size");
    check_dfti(DftiSetValue(desc.get(), DFTI_INPUT_DISTANCE, key.distance_in), "set input distance");
    check_dfti(DftiSetValue(desc.get(), DFTI_OUTPUT_DISTANCE, key.distance_out), "set output distance");
    check_dfti(DftiSetValue(desc.get(), DFTI_INPUT_STRIDES, key.istrides.data()), "set input strides");
    check_dfti(DftiSetValue(desc.get(), DFTI_OUTPUT_STRIDES, key.ostrides.data()), "set output strides");

    if (!complex_input || !complex_output) {
        check_dfti(DftiSetValue(desc.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX),
                   "set storage");
    }

    int64_t n = 1;
    for (int64_t i = 1; i <= signal_ndim; ++i) n *= signal_sizes[static_cast<size_t>(i)];
    set_scale(desc.get(), normalization, forward, n);
    check_dfti(DftiCommitDescriptor(desc.get()), "commit descriptor");
    plan_cache().put(key, desc);
    return desc;
}

void exec_mkl_fft(Tensor& out, const Tensor& self, const std::vector<int64_t>& out_sizes,
                  const std::vector<int64_t>& dims, int64_t normalization, bool forward) {
    if (out.numel() == 0) return;
    const int64_t ndim = self.dim();
    const int64_t signal_ndim = static_cast<int64_t>(dims.size());
    const int64_t batch_dims = ndim - signal_ndim;

    std::vector<int64_t> dim_permute(ndim);
    std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});
    std::vector<bool> transformed(static_cast<size_t>(ndim), false);
    for (int64_t d : dims) transformed[static_cast<size_t>(d)] = true;
    auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
                                    [&](int64_t d) { return !transformed[static_cast<size_t>(d)]; });
    const auto self_strides = self.strides();
    std::sort(dim_permute.begin(), batch_end,
              [&](int64_t a, int64_t b) { return self_strides[static_cast<size_t>(a)] > self_strides[static_cast<size_t>(b)]; });
    std::copy(dims.begin(), dims.end(), batch_end);
    Tensor input = tpx::ops::permute(self, dim_permute);

    std::vector<int64_t> batched_sizes(static_cast<size_t>(signal_ndim) + 1);
    batched_sizes[0] = -1;
    for (int64_t i = batch_dims; i < ndim; ++i) {
        batched_sizes[static_cast<size_t>(i - batch_dims + 1)] = input.size(i);
    }
    input = tpx::ops::reshape(input, batched_sizes);

    const int64_t batch_size = input.size(0);
    std::vector<int64_t> signal_sizes(static_cast<size_t>(signal_ndim) + 1);
    signal_sizes[0] = batch_size;
    for (int64_t i = 0; i < signal_ndim; ++i) {
        const int64_t in_size = input.size(i + 1);
        const int64_t out_size = out_sizes[static_cast<size_t>(dims[static_cast<size_t>(i)])];
        signal_sizes[static_cast<size_t>(i + 1)] = std::max(in_size, out_size);
    }

    std::vector<int64_t> batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
    batched_out_sizes[0] = batch_size;
    for (int64_t i = 0; i < signal_ndim; ++i) {
        batched_out_sizes[static_cast<size_t>(i + 1)] = out_sizes[static_cast<size_t>(dims[static_cast<size_t>(i)])];
    }
    Tensor batched_out = Tensor::empty(batched_out_sizes, out.dtype(), out.device());

    // A zero stride is not a valid DFTI signal layout; materialize a copy.
    for (int64_t stride : input.strides()) {
        if (stride == 0) {
            input = tpx::ops::clone(input);
            break;
        }
    }

    auto desc = plan_mkl_fft(input, batched_out, signal_sizes,
                             isComplexType(input.dtype()), isComplexType(batched_out.dtype()),
                             normalization, forward);
    if (forward) {
        check_dfti(DftiComputeForward(desc.get(), const_cast<void*>(input.data_ptr()),
                                      batched_out.data_ptr()), "compute forward");
    } else {
        check_dfti(DftiComputeBackward(desc.get(), const_cast<void*>(input.data_ptr()),
                                       batched_out.data_ptr()), "compute backward");
    }

    std::vector<int64_t> result_strides(static_cast<size_t>(ndim));
    int64_t batch_numel = 1;
    for (int64_t i = batch_dims - 1; i >= 0; --i) {
        result_strides[static_cast<size_t>(dim_permute[static_cast<size_t>(i)])] =
            batch_numel * batched_out.strides()[0];
        batch_numel *= out_sizes[static_cast<size_t>(dim_permute[static_cast<size_t>(i)])];
    }
    for (int64_t i = batch_dims; i < ndim; ++i) {
        result_strides[static_cast<size_t>(dim_permute[static_cast<size_t>(i)])] =
            batched_out.strides()[static_cast<size_t>(i - batch_dims + 1)];
    }
    out = batched_out.as_strided(out_sizes, result_strides, batched_out.storage_offset());
}

std::vector<int64_t> sort_dims(const Tensor& self, const std::vector<int64_t>& dims,
                               bool exclude_last = false) {
    std::vector<int64_t> sorted(dims.begin(), dims.end());
    const auto strides = self.strides();
    std::sort(sorted.begin(), sorted.end() - (exclude_last ? 1 : 0),
              [&](int64_t a, int64_t b) {
                  return strides[static_cast<size_t>(a)] > strides[static_cast<size_t>(b)];
              });
    return sorted;
}

void check_complex(const Tensor& self) {
    TP_CHECK(self.dtype() == DType::ComplexFloat || self.dtype() == DType::ComplexDouble,
             "Only supports complex dtypes, but found: ", self.dtype());
}

// The plan writes only the first half of a real signal's spectrum.  The rest
// is the conjugate of the mirrored index across every transformed axis.
template <typename T>
void fill_conjugate_half(Tensor& out, const std::vector<int64_t>& dims) {
    using C = std::complex<T>;
    const std::vector<int64_t> sizes = out.sizes().vec();
    const int64_t ndim = static_cast<int64_t>(sizes.size());
    const int64_t last = dims.back();
    const int64_t n = sizes[static_cast<size_t>(last)];
    const int64_t half = n / 2 + 1;
    if (half >= n || out.numel() == 0) return;

    // The spectrum can be a strided view when batch axes surround the
    // transformed ones, so mirror through the layout it actually has.
    const auto strides = out.strides();
    std::vector<int64_t> stride(strides.begin(), strides.end());
    std::vector<bool> mirrored(static_cast<size_t>(ndim), false);
    for (int64_t d : dims) mirrored[static_cast<size_t>(d)] = true;

    C* data = reinterpret_cast<C*>(out.data_ptr());
    const int64_t step = stride[static_cast<size_t>(last)];
    std::vector<int64_t> index(static_cast<size_t>(ndim), 0);
    const int64_t rows = out.numel() / n;
    for (int64_t row = 0; row < rows; ++row) {
        int64_t target = 0;
        int64_t source = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == last) continue;
            const size_t u = static_cast<size_t>(d);
            const int64_t i = index[u];
            target += i * stride[u];
            source += (mirrored[u] && i != 0 ? sizes[u] - i : i) * stride[u];
        }
        for (int64_t k = half; k < n; ++k) {
            data[target + k * step] = std::conj(data[source + (n - k) * step]);
        }
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == last) continue;
            const size_t u = static_cast<size_t>(d);
            if (++index[u] < sizes[u]) break;
            index[u] = 0;
        }
    }
}

Tensor mkl_c2c(const Tensor& self, const std::vector<int64_t>& dim,
               int64_t normalization, bool forward) {
    check_complex(self);
    if (dim.empty()) return tpx::ops::clone(self);
    const auto dims = wrap_mkl_dims(self, dim);
    const auto sorted = sort_dims(self, dims);
    Tensor out(self.sizes().vec(), self.dtype(), self.device());
    exec_mkl_fft(out, self, self.sizes().vec(), sorted, normalization, forward);
    return out;
}

Tensor mkl_r2c(const Tensor& self, const std::vector<int64_t>& dim,
               int64_t normalization, bool onesided) {
    TP_CHECK_TYPE(isFloatingType(self.dtype()),
                  "Only supports floating-point dtypes, but found: ", self.dtype());
    TP_CHECK_NOT_IMPLEMENTED(self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
                             "real-to-complex FFT is not implemented for ", self.dtype(), " on CPU");
    TP_CHECK(!dim.empty(), "_fft_r2c must transform at least one dimension");
    const auto dims = wrap_mkl_dims(self, dim);
    auto out_sizes = self.sizes().vec();
    const int64_t last = dims.back();
    const int64_t half = out_sizes[static_cast<size_t>(last)] / 2 + 1;
    if (onesided) out_sizes[static_cast<size_t>(last)] = half;
    const DType out_dtype = self.dtype() == DType::Float64
        ? DType::ComplexDouble : DType::ComplexFloat;
    const auto sorted = sort_dims(self, dims, /*exclude_last=*/true);
    Tensor out(out_sizes, out_dtype, self.device());
    exec_mkl_fft(out, self, out_sizes, sorted, normalization, /*forward=*/true);
    if (!onesided) {
        if (out.dtype() == DType::ComplexDouble) fill_conjugate_half<double>(out, dims);
        else fill_conjugate_half<float>(out, dims);
    }
    return out;
}

Tensor mkl_c2r(const Tensor& self, const std::vector<int64_t>& dim,
               int64_t normalization, int64_t last_dim_size) {
    check_complex(self);
    TP_CHECK(!dim.empty(), "_fft_c2r must transform at least one dimension");
    TP_CHECK(last_dim_size >= 1, "Invalid number of data points (", last_dim_size, ") specified");
    const auto dims = wrap_mkl_dims(self, dim);
    const int64_t last = dims.back();
    TP_CHECK(self.size(last) >= last_dim_size / 2 + 1,
             "_fft_c2r expects at least ", last_dim_size / 2 + 1,
             " values along dimension ", last, " for a signal of length ", last_dim_size,
             ", but got ", self.size(last));

    Tensor input = self;
    std::vector<int64_t> transform_dims = dims;
    if (transform_dims.size() > 1) {
        const std::vector<int64_t> leading(transform_dims.begin(), transform_dims.end() - 1);
        input = mkl_c2c(self, leading, normalization, /*forward=*/false);
        transform_dims.assign(1, dims.back());
    }
    auto out_sizes = input.sizes().vec();
    out_sizes[static_cast<size_t>(last)] = last_dim_size;
    const DType out_dtype = input.dtype() == DType::ComplexDouble
        ? DType::Float64 : DType::Float32;
    Tensor out(out_sizes, out_dtype, input.device());
    exec_mkl_fft(out, input, out_sizes, transform_dims, normalization, /*forward=*/false);
    return out;
}

#endif  // USE_MKL

}  // namespace

#if defined(USE_MKL)

Tensor _fft_c2c_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, bool forward) {
    return mkl_c2c(self, dim, normalization, forward);
}

Tensor _fft_r2c_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, bool onesided) {
    return mkl_r2c(self, dim, normalization, onesided);
}

Tensor _fft_c2r_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                    int64_t normalization, int64_t last_dim_size) {
    return mkl_c2r(self, dim, normalization, last_dim_size);
}

Tensor& _fft_r2c_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, bool onesided, Tensor& out) {
    return write_out(out, _fft_r2c_cpu(self, dim, normalization, onesided));
}

Tensor& _fft_c2r_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, int64_t last_dim_size, Tensor& out) {
    return write_out(out, _fft_c2r_cpu(self, dim, normalization, last_dim_size));
}

Tensor& _fft_c2c_out_cpu(const Tensor& self, const std::vector<int64_t>& dim,
                         int64_t normalization, bool forward, Tensor& out) {
    return write_out(out, _fft_c2c_cpu(self, dim, normalization, forward));
}

#endif  // USE_MKL

}  // namespace cpu
}  // namespace tensorplay

#if defined(USE_MKL)
TENSORPLAY_LIBRARY_IMPL(CPU, SpectralMklKernels) {
    using namespace tensorplay;
    using namespace tensorplay::cpu;
    m.impl("_fft_r2c", _fft_r2c_cpu);
    m.impl("_fft_r2c.out", _fft_r2c_out_cpu);
    m.impl("_fft_c2r", _fft_c2r_cpu);
    m.impl("_fft_c2r.out", _fft_c2r_out_cpu);
    m.impl("_fft_c2c", _fft_c2c_cpu);
    m.impl("_fft_c2c.out", _fft_c2c_out_cpu);
}
#endif
