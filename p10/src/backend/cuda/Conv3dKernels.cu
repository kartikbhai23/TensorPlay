#include "Tensor.h"
#include "Dispatcher.h"
#include "Convolution.h"
#include "Context.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "CUDNNUtils.h"
#include "Allocator.h"

#include <cuda_runtime.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#ifdef TP_HAS_CUDNN_FRONTEND
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#endif

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

#ifdef USE_CUDNN

inline cudnnDataType_t to_cudnn_data_type(DType d) {
    if (d == DType::Float32) return CUDNN_DATA_FLOAT;
    if (d == DType::Float64) return CUDNN_DATA_DOUBLE;
    if (d == DType::Float16) return CUDNN_DATA_HALF;
    if (d == DType::BFloat16) return CUDNN_DATA_BFLOAT16;
    TP_THROW(NotImplementedError, "cuDNN: only float/double/half/bfloat16 supported");
}

inline cudnnDataType_t to_cudnn_compute_type(DType d) {
    return d == DType::Float64 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
}

inline std::vector<int64_t> expand_param_if_needed(const std::vector<int64_t>& list,
                                                   int64_t n, int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if (list.size() != n) TP_THROW(ValueError, "Parameter size mismatch");
    return list;
}

struct ConvBwdAlgo {
    int algorithm;
    size_t workspace_size;
};

static std::unordered_map<std::string, ConvBwdAlgo> g_conv3d_algo_cache;
static std::mutex g_conv3d_cache_mutex;
static constexpr size_t kConvAutotuneWorkspaceCap = 512ULL * 1024 * 1024;

inline bool conv_autotune_enabled() {
    return tensorplay::globalContext().cudnnBenchmark() &&
           !tensorplay::globalContext().deterministicAlgorithms();
}

inline void key_append(std::string& key, const std::vector<int64_t>& v) {
    for (int64_t d : v) key += ":" + std::to_string(d);
}

template <class XDesc, class WDesc, class CDesc, class YDesc>
ConvBwdAlgo autotune_conv_fwd(
    cudnnHandle_t handle,
    const XDesc& x_desc, const void* x_ptr,
    const WDesc& w_desc, const void* w_ptr,
    const CDesc& conv_desc,
    const YDesc& y_desc, void* y_ptr,
    const Device& device) {
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        kConvAutotuneWorkspaceCap, device);
    cudnnConvolutionFwdAlgoPerf_t perfs[16];
    int returned = 0;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle, x_desc, x_ptr, w_desc, w_ptr, conv_desc, y_desc, y_ptr,
        16, &returned, perfs, workspace.get(), kConvAutotuneWorkspaceCap));
    for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS) {
            return ConvBwdAlgo{static_cast<int>(perfs[i].algo), perfs[i].memory};
        }
    }
    TP_THROW(RuntimeError, "cuDNN: autotune found no forward convolution algorithm");
}

template <class WDesc, class DYDesc, class CDesc, class DXDesc>
ConvBwdAlgo autotune_conv_bwd_data(
    cudnnHandle_t handle,
    const WDesc& w_desc, const void* w_ptr,
    const DYDesc& dy_desc, const void* dy_ptr,
    const CDesc& conv_desc,
    const DXDesc& dx_desc, void* dx_ptr,
    const Device& device) {
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        kConvAutotuneWorkspaceCap, device);
    cudnnConvolutionBwdDataAlgoPerf_t perfs[16];
    int returned = 0;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle, w_desc, w_ptr, dy_desc, dy_ptr, conv_desc, dx_desc, dx_ptr,
        16, &returned, perfs, workspace.get(), kConvAutotuneWorkspaceCap));
    for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS) {
            return ConvBwdAlgo{static_cast<int>(perfs[i].algo), perfs[i].memory};
        }
    }
    TP_THROW(RuntimeError, "cuDNN: autotune found no backward-data convolution algorithm");
}

template <class XDesc, class DYDesc, class CDesc, class DWDesc>
ConvBwdAlgo autotune_conv_bwd_filter(
    cudnnHandle_t handle,
    const XDesc& x_desc, const void* x_ptr,
    const DYDesc& dy_desc, const void* dy_ptr,
    const CDesc& conv_desc,
    const DWDesc& dw_desc, void* dw_ptr,
    const Device& device) {
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        kConvAutotuneWorkspaceCap, device);
    cudnnConvolutionBwdFilterAlgoPerf_t perfs[16];
    int returned = 0;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle, x_desc, x_ptr, dy_desc, dy_ptr, conv_desc, dw_desc, dw_ptr,
        16, &returned, perfs, workspace.get(), kConvAutotuneWorkspaceCap));
    for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS) {
            return ConvBwdAlgo{static_cast<int>(perfs[i].algo), perfs[i].memory};
        }
    }
    TP_THROW(RuntimeError, "cuDNN: autotune found no backward-filter convolution algorithm");
}

template <class Select>
ConvBwdAlgo cached_conv_algo(const std::string& key, Select&& select) {
    {
        std::lock_guard<std::mutex> lock(g_conv3d_cache_mutex);
        auto it = g_conv3d_algo_cache.find(key);
        if (it != g_conv3d_algo_cache.end()) return it->second;
    }
    ConvBwdAlgo entry = select();
    {
        std::lock_guard<std::mutex> lock(g_conv3d_cache_mutex);
        g_conv3d_algo_cache.emplace(key, entry);
    }
    return entry;
}

struct TensorDescNd {
    cudnnTensorDescriptor_t desc;
    TensorDescNd() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDescNd() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }

    void set(const std::vector<int64_t>& sizes, DType dtype) {
        int nbDims = static_cast<int>(sizes.size());
        int dims[8], strides[8];
        int64_t stride = 1;
        for (int i = nbDims - 1; i >= 0; --i) {
            dims[i] = static_cast<int>(sizes[i]);
            strides[i] = static_cast<int>(stride);
            stride *= sizes[i];
        }
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, to_cudnn_data_type(dtype), nbDims,
                                               dims, strides));
    }
    void set(const Tensor& t) { set(t.shape(), t.dtype()); }
};

struct FilterDescNd {
    cudnnFilterDescriptor_t desc;
    FilterDescNd() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~FilterDescNd() { cudnnDestroyFilterDescriptor(desc); }
    operator cudnnFilterDescriptor_t() const { return desc; }

    void set(const Tensor& t) {
        int nbDims = static_cast<int>(t.dim());
        int dims[8];
        for (int i = 0; i < nbDims; ++i) dims[i] = static_cast<int>(t.size(i));
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc, to_cudnn_data_type(t.dtype()),
                                               CUDNN_TENSOR_NCHW, nbDims, dims));
    }
};

struct ConvDescNd {
    cudnnConvolutionDescriptor_t desc;
    ConvDescNd() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ConvDescNd() { cudnnDestroyConvolutionDescriptor(desc); }
    operator cudnnConvolutionDescriptor_t() const { return desc; }

    void set(const std::vector<int64_t>& pads, const std::vector<int64_t>& strides,
             const std::vector<int64_t>& dilations, int64_t groups, DType dtype) {
        int nbDims = static_cast<int>(pads.size());
        int p[3], s[3], d[3];
        for (int i = 0; i < nbDims; ++i) {
            p[i] = static_cast<int>(pads[i]);
            s[i] = static_cast<int>(strides[i]);
            d[i] = static_cast<int>(dilations[i]);
        }
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(desc, nbDims, p, s, d,
                                                    CUDNN_CROSS_CORRELATION,
                                                    to_cudnn_compute_type(dtype)));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, static_cast<int>(groups)));
        cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
        if (dtype == DType::Float16 ||
            (dtype == DType::Float32 && tensorplay::globalContext().allowTF32CuDNN())) {
            math_type = CUDNN_TENSOR_OP_MATH;
        }
        CUDNN_CHECK(cudnnSetConvolutionMathType(desc, math_type));
    }
};

inline void conv3d_add_bias(cudnnHandle_t handle, const Tensor& out, const Tensor& bias) {
    if (!bias.defined() || bias.numel() == 0) return;
    TensorDescNd b_desc;
    b_desc.set(std::vector<int64_t>{1, bias.size(0), 1, 1, 1}, bias.dtype());
    TensorDescNd y_desc;
    y_desc.set(out.shape(), out.dtype());
    float alpha = 1.0f, beta = 1.0f;
    double alpha_d = 1.0, beta_d = 1.0;
    void* alpha_p = &alpha;
    void* beta_p = &beta;
    if (out.dtype() == DType::Float64) {
        alpha_p = &alpha_d;
        beta_p = &beta_d;
    }
    CUDNN_CHECK(cudnnAddTensor(handle, alpha_p, b_desc, bias.data_ptr(),
                               beta_p, y_desc, out.data_ptr()));
}

inline void* conv_alpha_ptr(DType dtype, float& alpha, double& alpha_d) {
    if (dtype == DType::Float64) return &alpha_d;
    return &alpha;
}

#ifdef TP_HAS_CUDNN_FRONTEND
// Byte alignment of a tensor's storage: engines declare the widest alignment
// they require, and one asking for more than the buffer offers is not offered.
inline int64_t conv3d_alignment_of(const void* ptr) {
    int64_t alignment = 1;
    auto address = reinterpret_cast<uintptr_t>(ptr);
    for (; alignment < 32; alignment *= 2) {
        if (address % static_cast<uintptr_t>(alignment * 2)) return alignment;
    }
    return alignment;
}

struct Conv3dFwdKey {
    cudnnDataType_t dtype;
    int64_t N, C, D, H, W, K, kD, kH, kW, groups;
    int64_t Do, Ho, Wo, x_align, w_align, y_align;
    int64_t pd, ph, pw, sd, sh, sw, dd, dh, dw;
    int device;
    int operation;
    bool allow_tf32;
    bool deterministic;
    bool operator==(const Conv3dFwdKey& o) const {
        return dtype == o.dtype && N == o.N && C == o.C && D == o.D && H == o.H &&
               W == o.W && K == o.K && kD == o.kD && kH == o.kH && kW == o.kW &&
               groups == o.groups && Do == o.Do && Ho == o.Ho && Wo == o.Wo &&
               x_align == o.x_align && w_align == o.w_align && y_align == o.y_align &&
               operation == o.operation && pd == o.pd && ph == o.ph && pw == o.pw &&
               sd == o.sd && sh == o.sh && sw == o.sw && dd == o.dd &&
               dh == o.dh && dw == o.dw && device == o.device &&
               allow_tf32 == o.allow_tf32 && deterministic == o.deterministic;
    }
};

struct Conv3dFwdKeyHash {
    size_t operator()(const Conv3dFwdKey& k) const {
        size_t h = std::hash<int64_t>{}(k.N);
        for (auto v : {k.C, k.D, k.H, k.W, k.K, k.kD, k.kH, k.kW, k.groups,
                       k.Do, k.Ho, k.Wo, k.x_align, k.w_align, k.y_align,
                       k.pd, k.ph, k.pw, k.sd, k.sh, k.sw, k.dd, k.dh, k.dw}) {
            h = h * 1000003 ^ std::hash<int64_t>{}(v);
        }
        h = h * 1000003 ^ std::hash<int>{}(static_cast<int>(k.dtype));
        h = h * 1000003 ^ std::hash<int>{}(k.device);
        h = h * 1000003 ^ std::hash<int>{}(k.operation);
        h = h * 1000003 ^ std::hash<int>{}(static_cast<int>(k.allow_tf32));
        h = h * 1000003 ^ std::hash<int>{}(static_cast<int>(k.deterministic));
        return h;
    }
};

// Packed NCDHW operands share a cached plan for each geometry and operation.
inline bool conv3d_run_v8(const Tensor& input, const Tensor& weight,
                              const Tensor& out,
                              const std::vector<int64_t>& stride,
                              const std::vector<int64_t>& padding,
                              const std::vector<int64_t>& dilation,
                              int64_t groups, cudnnBackendDescriptorType_t operation) {
    static std::unordered_map<Conv3dFwdKey, std::shared_ptr<fe::ExecutionPlan>,
                              Conv3dFwdKeyHash> g_conv3d_plan_cache;
    static std::mutex g_conv3d_plan_mutex;

    const cudnnDataType_t dtype = to_cudnn_data_type(input.dtype());
    const cudnnDataType_t compute = to_cudnn_compute_type(input.dtype());
    const bool allow_tf32 = tensorplay::globalContext().allowTF32CuDNN();
    const bool deterministic = tensorplay::globalContext().deterministicAlgorithms();

    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t K = weight.size(0);
    const int64_t kD = weight.size(2), kH = weight.size(3), kW = weight.size(4);
    const int64_t Do = out.size(2), Ho = out.size(3), Wo = out.size(4);

    Conv3dFwdKey key{dtype, N, C, D, H, W, K, kD, kH, kW, groups,
                     Do, Ho, Wo, conv3d_alignment_of(input.data_ptr()),
                     conv3d_alignment_of(weight.data_ptr()), conv3d_alignment_of(out.data_ptr()),
                     padding[0], padding[1], padding[2],
                     stride[0], stride[1], stride[2],
                     dilation[0], dilation[1], dilation[2],
                     static_cast<int>(input.device().index()),
                     static_cast<int>(operation),
                     allow_tf32, deterministic};

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    std::shared_ptr<fe::ExecutionPlan> plan;
    {
        std::lock_guard<std::mutex> lock(g_conv3d_plan_mutex);
        auto it = g_conv3d_plan_cache.find(key);
        if (it != g_conv3d_plan_cache.end()) plan = it->second;
        if (!plan) {
            const std::array<int64_t, 5> x_dim{N, C, D, H, W};
            const std::array<int64_t, 5> w_dim{K, C / groups, kD, kH, kW};
            const std::array<int64_t, 5> y_dim{N, K, Do, Ho, Wo};
            const std::array<int64_t, 5> x_stride{C * D * H * W, D * H * W, H * W, W, 1};
            const std::array<int64_t, 5> w_stride{
                (C / groups) * kD * kH * kW, kD * kH * kW, kH * kW, kW, 1};
            const std::array<int64_t, 5> y_stride{K * Do * Ho * Wo, Do * Ho * Wo,
                                                  Ho * Wo, Wo, 1};
            auto x_desc = fe::TensorBuilder()
                              .setDim(5, x_dim.data())
                              .setStrides(5, x_stride.data())
                              .setId('x')
                              .setAlignment(conv3d_alignment_of(input.data_ptr()))
                              .setDataType(dtype)
                              .build();
            auto w_desc = fe::TensorBuilder()
                              .setDim(5, w_dim.data())
                              .setStrides(5, w_stride.data())
                              .setId('w')
                              .setAlignment(conv3d_alignment_of(weight.data_ptr()))
                              .setDataType(dtype)
                              .build();
            auto y_desc = fe::TensorBuilder()
                              .setDim(5, y_dim.data())
                              .setStrides(5, y_stride.data())
                              .setId('y')
                              .setAlignment(conv3d_alignment_of(out.data_ptr()))
                              .setDataType(dtype)
                              .build();

            int64_t pad[3] = {padding[0], padding[1], padding[2]};
            int64_t strd[3] = {stride[0], stride[1], stride[2]};
            int64_t dil[3] = {dilation[0], dilation[1], dilation[2]};

            auto conv_desc = fe::ConvDescBuilder()
                                 .setComputeType(compute)
                                 .setMathMode(CUDNN_CROSS_CORRELATION)
                                 .setSpatialDimCount(3)
                                 .setSpatialStride(3, strd)
                                 .setPrePadding(3, pad)
                                 .setPostPadding(3, pad)
                                 .setDilation(3, dil)
                                 .build();

            auto builder = fe::OperationBuilder(operation);
            if (operation == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
                builder.setxDesc(x_desc).setwDesc(w_desc).setyDesc(y_desc);
            } else if (operation == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
                builder.setdxDesc(x_desc).setwDesc(w_desc).setdyDesc(y_desc);
            } else {
                builder.setxDesc(x_desc).setdwDesc(w_desc).setdyDesc(y_desc);
            }
            fe::Operation conv_op = builder.setcDesc(conv_desc).build();
            std::array<fe::Operation const*, 1> ops = {&conv_op};
            auto op_graph = fe::OperationGraphBuilder()
                                .setHandle(handle)
                                .setOperationGraph(ops.size(), ops.data())
                                .build();

            // Engines that are non-deterministic, down-convert the inputs, or
            // rely on tensor cores while TF32 is off cannot serve this call.
            auto usable = [&](fe::EngineConfigList& from) {
                fe::EngineConfigList kept;
                auto drop = [&](cudnnBackendDescriptor_t c) {
                    if (deterministic &&
                        fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {
                        return true;
                    }
                    if (fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {
                        return true;
                    }
                    if (dtype == CUDNN_DATA_FLOAT && !allow_tf32 &&
                        fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {
                        return true;
                    }
                    return false;
                };
                fe::filter(from, kept, drop);
                return kept;
            };
            auto first_buildable =
                [&](fe::EngineConfigList& configs) -> std::shared_ptr<fe::ExecutionPlan> {
                for (auto& ec : configs) {
                    try {
                        return std::make_shared<fe::ExecutionPlan>(
                            fe::ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(ec)
                                .build());
                    } catch (...) {
                        // Config cannot be realized on this device; try the next.
                    }
                }
                return nullptr;
            };

            auto heuristics = fe::EngineHeuristicsBuilder()
                                  .setOperationGraph(op_graph)
                                  .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                  .build();
            auto& configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
            auto filtered = usable(configs);
            plan = first_buildable(filtered);
            if (!plan) {
                // The instant list can be tensor-core engines only; the
                // fallback list carries the plain ones a float call without
                // TF32 needs.
                auto fallback = fe::EngineHeuristicsBuilder()
                                    .setOperationGraph(op_graph)
                                    .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                    .build();
                auto& plain = fallback.getEngineConfig(fallback.getEngineConfigCount());
                auto plain_filtered = usable(plain);
                plan = first_buildable(plain_filtered);
            }
            if (!plan) {
                return false;
            }
            g_conv3d_plan_cache[key] = plan;
        }
    }

    const size_t workspace_size = plan->getWorkspaceSize();
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, input.device());
    void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), out.data_ptr()};
    int64_t uids[3] = {'x', 'w', 'y'};
    auto variant_pack = fe::VariantPackBuilder()
                            .setWorkspacePointer(workspace_size ? workspace.get() : nullptr)
                            .setDataPointers(3, data_ptrs)
                            .setUids(3, uids)
                            .build();
    CUDNN_CHECK(cudnnBackendExecute(handle, plan->get_raw_desc(),
                                   variant_pack.get_raw_desc()));
    return true;
}

inline bool conv3d_try_run_v8(const Tensor& input, const Tensor& weight, const Tensor& out,
                              const std::vector<int64_t>& stride,
                              const std::vector<int64_t>& padding,
                              const std::vector<int64_t>& dilation, int64_t groups,
                              cudnnBackendDescriptorType_t operation) {
    try {
        return conv3d_run_v8(input, weight, out, stride, padding, dilation, groups, operation);
    } catch (const fe::cudnnException&) {
        return false;
    }
}
#endif

#endif

}

#if defined(USE_CUDNN) && defined(TP_HAS_CUDNN_FRONTEND)
bool conv3d_backward_data_v8(const Tensor& grad_input, const Tensor& weight,
                             const Tensor& grad_output, const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    if (groups != 1 || !grad_input.numel() || !grad_output.numel() || conv_autotune_enabled()) {
        return false;
    }
    return conv3d_try_run_v8(grad_input, weight, grad_output, stride, padding, dilation, groups,
                             CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
}
#endif

Tensor conv3d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg,
                   const std::vector<int64_t>& dilation_arg, int64_t groups) {
    convolution::check_conv_shapes(input, weight, bias, groups, false);
    convolution::check_conv_geometry(input, weight, stride_arg, padding_arg,
                                     dilation_arg, "conv3d");
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 3, 1);
    auto padding = expand_param_if_needed(padding_arg, 3, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 3, 1);

    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    const int64_t D_in = input_c.size(2), H_in = input_c.size(3), W_in = input_c.size(4);
    const int64_t kD = weight_c.size(2), kH = weight_c.size(3), kW = weight_c.size(4);
    const int64_t D_out = (D_in + 2 * padding[0] - dilation[0] * (kD - 1) - 1) / stride[0] + 1;
    const int64_t H_out = (H_in + 2 * padding[1] - dilation[1] * (kH - 1) - 1) / stride[1] + 1;
    const int64_t W_out = (W_in + 2 * padding[2] - dilation[2] * (kW - 1) - 1) / stride[2] + 1;
    if (D_out <= 0 || H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "conv3d: Calculated output size is too small");

    Tensor out = Tensor::empty({input_c.size(0), weight_c.size(0), D_out, H_out, W_out},
                               input_c.dtype(), input_c.device());
    // An empty batch or filter set has nothing to fill: engines reject
    // zero-sized descriptors, and the empty result is already the answer.
    if (out.numel() == 0) return out;

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
#ifdef TP_HAS_CUDNN_FRONTEND
    // Benchmark mode keeps the enumerating path, whose timed search covers
    // the algorithm space this graph formulation does not name.
    if (!conv_autotune_enabled() &&
        conv3d_try_run_v8(input_c, weight_c, out, stride, padding, dilation, groups,
                          CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)) {
        conv3d_add_bias(handle, out, bias);
        return out;
    }
#endif
    TensorDescNd x_desc; x_desc.set(input_c);
    FilterDescNd w_desc; w_desc.set(weight_c);
    TensorDescNd y_desc; y_desc.set(out);
    ConvDescNd conv_desc;
    conv_desc.set(padding, stride, dilation, groups, input_c.dtype());

    std::string algo_key = "c3fwd:" + std::to_string(static_cast<int>(input_c.dtype())) +
                           ":" + std::to_string(static_cast<int>(input_c.device().index()));
    key_append(algo_key, input_c.shape());
    key_append(algo_key, weight_c.shape());
    key_append(algo_key, padding);
    key_append(algo_key, stride);
    key_append(algo_key, dilation);
    algo_key += ":" + std::to_string(groups);

    ConvBwdAlgo algo_entry = cached_conv_algo(algo_key, [&]() -> ConvBwdAlgo {
        if (conv_autotune_enabled()) {
            return autotune_conv_fwd(handle, x_desc, input_c.data_ptr(),
                                     w_desc, weight_c.data_ptr(), conv_desc,
                                     y_desc, out.data_ptr(), input_c.device());
        }
        cudnnConvolutionFwdAlgoPerf_t perf;
        int returned = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, x_desc, w_desc, conv_desc, y_desc, 1, &returned, &perf));
        if (returned == 0) TP_THROW(RuntimeError, "cuDNN: no forward convolution algorithm");
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, x_desc, w_desc, conv_desc, y_desc, perf.algo, &workspace_size));
        return ConvBwdAlgo{static_cast<int>(perf.algo), workspace_size};
    });

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        algo_entry.workspace_size ? algo_entry.workspace_size : 1, input_c.device());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(input_c.dtype(), alpha, alpha_d);
    void* beta_p = input_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                     : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionForward(handle, alpha_p, x_desc, input_c.data_ptr(),
                                        w_desc, weight_c.data_ptr(), conv_desc,
                                        static_cast<cudnnConvolutionFwdAlgo_t>(algo_entry.algorithm),
                                        workspace.get(), algo_entry.workspace_size, beta_p,
                                        y_desc, out.data_ptr()));
    conv3d_add_bias(handle, out, bias);
    return out;
#else
    TP_THROW(NotImplementedError, "conv3d_cuda requires cuDNN");
#endif
}

Tensor conv3d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                              const std::vector<int64_t>& stride_arg,
                              const std::vector<int64_t>& padding_arg,
                              const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 3, 1);
    auto padding = expand_param_if_needed(padding_arg, 3, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 3, 1);

    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    Tensor grad_input = Tensor::empty_like(input_c, DType::Undefined, input_c.device());
#ifdef TP_HAS_CUDNN_FRONTEND
    if (conv3d_backward_data_v8(grad_input, weight_c, grad_output_c,
                                stride, padding, dilation, groups)) {
        return grad_input;
    }
#endif
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDescNd dy_desc; dy_desc.set(grad_output_c);
    FilterDescNd w_desc; w_desc.set(weight_c);
    TensorDescNd dx_desc; dx_desc.set(input_c);
    ConvDescNd conv_desc;
    conv_desc.set(padding, stride, dilation, groups, input_c.dtype());

    std::string algo_key = "c3gi:" + std::to_string(static_cast<int>(input_c.dtype())) +
                           ":" + std::to_string(static_cast<int>(input_c.device().index()));
    key_append(algo_key, input_c.shape());
    key_append(algo_key, weight_c.shape());
    key_append(algo_key, grad_output_c.shape());
    key_append(algo_key, padding);
    key_append(algo_key, stride);
    key_append(algo_key, dilation);
    algo_key += ":" + std::to_string(groups);

    ConvBwdAlgo algo_entry = cached_conv_algo(algo_key, [&]() -> ConvBwdAlgo {
        if (conv_autotune_enabled()) {
            return autotune_conv_bwd_data(handle, w_desc, weight_c.data_ptr(),
                                           dy_desc, grad_output_c.data_ptr(), conv_desc,
                                           dx_desc, grad_input.data_ptr(), input_c.device());
        }
        cudnnConvolutionBwdDataAlgoPerf_t perf;
        int returned = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle, w_desc, dy_desc, conv_desc, dx_desc, 1, &returned, &perf));
        if (returned == 0) TP_THROW(RuntimeError, "cuDNN: no backward-data convolution algorithm");
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, w_desc, dy_desc, conv_desc, dx_desc, perf.algo, &workspace_size));
        return ConvBwdAlgo{static_cast<int>(perf.algo), workspace_size};
    });

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        algo_entry.workspace_size ? algo_entry.workspace_size : 1, input_c.device());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(input_c.dtype(), alpha, alpha_d);
    void* beta_p = input_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                     : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, alpha_p, w_desc, weight_c.data_ptr(),
                                             dy_desc, grad_output_c.data_ptr(), conv_desc,
                                             static_cast<cudnnConvolutionBwdDataAlgo_t>(algo_entry.algorithm),
                                             workspace.get(), algo_entry.workspace_size,
                                             beta_p, dx_desc, grad_input.data_ptr()));
    return grad_input;
#else
    TP_THROW(NotImplementedError, "conv3d_grad_input_cuda requires cuDNN");
#endif
}

Tensor conv3d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                               const std::vector<int64_t>& stride_arg,
                               const std::vector<int64_t>& padding_arg,
                               const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 3, 1);
    auto padding = expand_param_if_needed(padding_arg, 3, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 3, 1);

    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    Tensor grad_weight = Tensor::empty_like(weight_c, DType::Undefined, weight_c.device());
#ifdef TP_HAS_CUDNN_FRONTEND
    if (groups == 1 && input_c.numel() && grad_output_c.numel() && !conv_autotune_enabled() &&
        conv3d_try_run_v8(input_c, grad_weight, grad_output_c, stride, padding, dilation, groups,
                          CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)) {
        return grad_weight;
    }
#endif
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDescNd x_desc; x_desc.set(input_c);
    TensorDescNd dy_desc; dy_desc.set(grad_output_c);
    FilterDescNd dw_desc; dw_desc.set(weight_c);
    ConvDescNd conv_desc;
    conv_desc.set(padding, stride, dilation, groups, input_c.dtype());

    std::string algo_key = "c3gw:" + std::to_string(static_cast<int>(input_c.dtype())) +
                           ":" + std::to_string(static_cast<int>(input_c.device().index()));
    key_append(algo_key, input_c.shape());
    key_append(algo_key, weight_c.shape());
    key_append(algo_key, grad_output_c.shape());
    key_append(algo_key, padding);
    key_append(algo_key, stride);
    key_append(algo_key, dilation);
    algo_key += ":" + std::to_string(groups);

    ConvBwdAlgo algo_entry = cached_conv_algo(algo_key, [&]() -> ConvBwdAlgo {
        if (conv_autotune_enabled()) {
            return autotune_conv_bwd_filter(handle, x_desc, input_c.data_ptr(),
                                            dy_desc, grad_output_c.data_ptr(), conv_desc,
                                            dw_desc, grad_weight.data_ptr(), input_c.device());
        }
        cudnnConvolutionBwdFilterAlgoPerf_t perf;
        int returned = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle, x_desc, dy_desc, conv_desc, dw_desc, 1, &returned, &perf));
        if (returned == 0) TP_THROW(RuntimeError, "cuDNN: no backward-filter convolution algorithm");
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, x_desc, dy_desc, conv_desc, dw_desc, perf.algo, &workspace_size));
        return ConvBwdAlgo{static_cast<int>(perf.algo), workspace_size};
    });

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        algo_entry.workspace_size ? algo_entry.workspace_size : 1, input_c.device());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(input_c.dtype(), alpha, alpha_d);
    void* beta_p = input_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                     : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, alpha_p, x_desc, input_c.data_ptr(),
                                               dy_desc, grad_output_c.data_ptr(), conv_desc,
                                               static_cast<cudnnConvolutionBwdFilterAlgo_t>(algo_entry.algorithm),
                                               workspace.get(), algo_entry.workspace_size,
                                               beta_p, dw_desc, grad_weight.data_ptr()));
    return grad_weight;
#else
    TP_THROW(NotImplementedError, "conv3d_grad_weight_cuda requires cuDNN");
#endif
}

Tensor conv3d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
#ifdef USE_CUDNN
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    Tensor grad_bias = Tensor::empty({grad_output_c.size(1)}, grad_output_c.dtype(),
                                     grad_output_c.device());

    TensorDescNd dy_desc;
    dy_desc.set(grad_output_c);
    TensorDescNd db_desc;
    db_desc.set(std::vector<int64_t>{1, grad_bias.size(0), 1, 1, 1}, grad_bias.dtype());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(grad_output_c.dtype(), alpha, alpha_d);
    void* beta_p = grad_output_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                           : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, alpha_p, dy_desc,
                                             grad_output_c.data_ptr(), beta_p, db_desc,
                                             grad_bias.data_ptr()));
    return grad_bias;
#else
    TP_THROW(NotImplementedError, "conv3d_grad_bias_cuda requires cuDNN");
#endif
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Conv3dKernels) {
    m.impl("conv3d", conv3d_cuda);
    m.impl("conv3d_grad_input", conv3d_grad_input_cuda);
    m.impl("conv3d_grad_weight", conv3d_grad_weight_cuda);
    m.impl("conv3d_grad_bias", conv3d_grad_bias_cuda);
}

}
}
