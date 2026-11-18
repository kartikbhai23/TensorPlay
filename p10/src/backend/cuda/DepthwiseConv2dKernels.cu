#include "CudaDispatchHelpers.cuh"
#include "Convolution.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// Depthwise / slow convolution spellings.  tp's conv2d/conv3d kernels take
// groups natively; a depthwise convolution has one group per input channel,
// each widened to weight.size(0) / channels outputs.
// ---------------------------------------------------------------------------

Tensor interop__conv_depthwise2d_cuda(const Tensor& self, const Tensor& weight,
                                      const std::vector<int64_t>& kernel_size,
                                      const std::optional<Tensor>& bias,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return ops::conv2d(self, weight, bias, stride, padding, dilation,
                       self.size(-3));
}


Tensor& interop__conv_depthwise2d_out_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, Tensor& out) {
    (void)kernel_size;
    write_out(out, ops::conv2d(self, weight, bias, stride, padding, dilation,
                               self.size(-3)));
    return out;
}

// ---------------------------------------------------------------------------
// Depthwise 2-D convolution backwards.
//
// Every group of a depthwise call holds one input channel and
// weight.size(0) / channels output channels that read only that channel, so
// the backward pass is channels independent single-channel convolutions.
// The two kernels below walk the call once: the data gradient sums the
// filter taps that reach an input site, and the filter gradient reduces each
// tap over the batch and the output sites it was placed at.
//
// The data gradient reads the channel gradient in either order of the two
// 4-D layouts.  Its own result keeps the order the forward input arrived in,
// so a channel-major model stays channel-major end to end, and a channel
// gradient in the other order is repacked once rather than read strided at
// every tap.
// ---------------------------------------------------------------------------

template <typename T> struct DepthwiseMath { using type = T; };
template <> struct DepthwiseMath<tensorplay::Half> { using type = float; };
template <> struct DepthwiseMath<tensorplay::BFloat16> { using type = float; };

constexpr int kDepthwiseThreads = 256;
constexpr int kDepthwiseWarpSize = 32;

int depthwise_blocks(int64_t work, int threads) {
    const int64_t blocks = (work + threads - 1) / threads;
    return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

// A channel-major 4-D tensor keeps the channel dimension innermost:
// (N, C, H, W) stored as N, H, W, C.
bool depthwise_channels_last_4d(const Tensor& t) {
    if (t.dim() != 4) return false;
    const int64_t c = t.size(1), h = t.size(2), w = t.size(3);
    return t.stride(0) == c * h * w && t.stride(1) == 1 &&
           t.stride(2) == w * c && t.stride(3) == c;
}

// One thread per input site.  A site accumulates every filter tap whose
// placement lands on it: the tap row r reads output row
// (h + pad - r * dilation) / stride, and only when that quotient is exact.
// The grid walks one batch item per y index, and x covers that item's sites
// in the order the result is stored (channel innermost or outermost).
template <int KS, int STRIDE, bool CHANNEL_MAJOR, typename T, typename M>
__global__ void depthwise_conv2d_grad_input_kernel(
    int W, int H, int OW, int OH, int R, int S, int channels, int multiplier,
    int sh, int sw, int ph, int pw, int dh, int dw,
    const T* __restrict__ grad_output, const T* __restrict__ weight,
    T* __restrict__ grad_input) {
    const int kh = (KS != 0) ? KS : R;
    const int kw = (KS != 0) ? KS : S;
    const int stride_h = (STRIDE != 0) ? STRIDE : sh;
    const int stride_w = (STRIDE != 0) ? STRIDE : sw;

    const int in_plane = H * W;
    const int out_plane = OH * OW;
    const int out_channels = channels * multiplier;
    const int total = channels * in_plane;
    const int64_t n = blockIdx.y;

    const T* dy_batch = grad_output +
        (CHANNEL_MAJOR ? n * out_plane * out_channels
                       : n * out_channels * out_plane);
    T* dx_batch = grad_input + n * total;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < total;
         j += blockDim.x * gridDim.x) {
        int c, h, w;
        if (CHANNEL_MAJOR) {
            c = j % channels;
            const int rem = j / channels;
            w = rem % W;
            h = rem / W;
        } else {
            c = j / in_plane;
            const int rem = j - c * in_plane;
            h = rem / W;
            w = rem - h * W;
        }
        M acc = M(0);
        for (int m = 0; m < multiplier; ++m) {
            const int k = c * multiplier + m;
            const T* taps = weight + static_cast<int64_t>(k) * R * S;
#pragma unroll
            for (int r = 0; r < kh; ++r) {
                const int hn = h + ph - r * dh;
                if (hn % stride_h != 0) continue;
                const int ho = hn / stride_h;
                if (ho < 0 || ho >= OH) continue;
                const T* tap_row = taps + r * S;
                const int64_t dy_row = CHANNEL_MAJOR
                    ? (static_cast<int64_t>(ho) * OW) * out_channels + k
                    : static_cast<int64_t>(k) * out_plane +
                          static_cast<int64_t>(ho) * OW;
#pragma unroll
                for (int s = 0; s < kw; ++s) {
                    const int wn = w + pw - s * dw;
                    if (wn % stride_w != 0) continue;
                    const int wo = wn / stride_w;
                    if (wo < 0 || wo >= OW) continue;
                    const int64_t dy_index = CHANNEL_MAJOR
                        ? dy_row + static_cast<int64_t>(wo) * out_channels
                        : dy_row + wo;
                    acc += static_cast<M>(tap_row[s]) *
                           static_cast<M>(dy_batch[dy_index]);
                }
            }
        }
        dx_batch[j] = static_cast<T>(acc);
    }
}

// One block per filter tap of one output channel; the block reduces the tap
// over the batch and the output sites.  A warp walks one batch item at a
// time, so the lane stride stays adjacent in memory and the batches beyond
// the warp count are picked up by the same warps in turn.
template <typename T, typename M>
__global__ void depthwise_conv2d_grad_weight_kernel(
    int batch, int channels, int multiplier, int in_plane, int W, int H,
    int out_plane, int OW, int OH, int R, int S, int sh, int sw, int ph,
    int pw, int dh, int dw, const T* __restrict__ grad_output,
    const T* __restrict__ input, T* __restrict__ grad_weight) {
    const int tap = blockIdx.x;
    const int r = tap / S;
    const int s = tap - r * S;
    const int out_channel = blockIdx.y;
    const int in_channel = out_channel / multiplier;

    const int lane = threadIdx.x % kDepthwiseWarpSize;
    const int warp = threadIdx.x / kDepthwiseWarpSize;
    const int warps = blockDim.x / kDepthwiseWarpSize;

    const int out_channels = channels * multiplier;
    const T* dy_batch = grad_output + static_cast<int64_t>(out_channel) * out_plane;
    const T* x_batch = input + static_cast<int64_t>(in_channel) * in_plane;

    M acc = M(0);
    for (int n = warp; n < batch; n += warps) {
        const T* dy = dy_batch + static_cast<int64_t>(n) * out_channels * out_plane;
        const T* x = x_batch + static_cast<int64_t>(n) * channels * in_plane;
        for (int i = lane; i < out_plane; i += kDepthwiseWarpSize) {
            const int oh = i / OW;
            const int ow = i - oh * OW;
            const int hi = oh * sh + r * dh - ph;
            const int wi = ow * sw + s * dw - pw;
            if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
            acc += static_cast<M>(x[static_cast<int64_t>(hi) * W + wi]) *
                   static_cast<M>(dy[i]);
        }
    }

    for (int off = kDepthwiseWarpSize / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ M warp_sums[8];
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        M total = M(0);
        for (int i = 0; i < warps; ++i) total += warp_sums[i];
        grad_weight[static_cast<int64_t>(out_channel) * R * S + tap] =
            static_cast<T>(total);
    }
}

// The shape, dtype and grouping all depthwise backward calls must share; the
// layout is checked by the caller that reads the operands.
bool depthwise_shape_ok(const Tensor& grad_output, const Tensor& input,
                        const Tensor& weight, int64_t groups) {
    if (input.dim() != 4 || weight.dim() != 4 || grad_output.dim() != 4) {
        return false;
    }
    if (groups <= 1 || groups != input.size(1) || weight.size(1) != 1) {
        return false;
    }
    if (weight.size(0) % input.size(1) != 0) return false;
    if (input.size(0) != grad_output.size(0) ||
        weight.size(0) != grad_output.size(1)) {
        return false;
    }
    const DType dtype = input.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float16 &&
        dtype != DType::BFloat16) {
        return false;
    }
    if (weight.dtype() != dtype || grad_output.dtype() != dtype) return false;
    if (!weight.is_contiguous()) return false;
    if (input.numel() == 0 || grad_output.numel() == 0 || weight.numel() == 0) {
        return false;
    }
    // The site loops index with 32-bit arithmetic; refuse a call that would
    // not fit rather than wrap.
    const int64_t batch_sites = input.size(0) * input.size(1) * input.size(2) *
                                input.size(3);
    return batch_sites <= 0x7fffffff && grad_output.numel() <= 0x7fffffff;
}

} // namespace

bool conv2d_depthwise_grad_input_eligible(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    int64_t groups) {
    if (!depthwise_shape_ok(grad_output, input, weight, groups)) return false;
    // The result keeps the input's order, so that order has to be one of the
    // two dense 4-D ones; the gradient is repacked into it when it differs.
    return input.is_contiguous() || depthwise_channels_last_4d(input);
}

bool conv2d_depthwise_grad_weight_eligible(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    int64_t groups) {
    return depthwise_shape_ok(grad_output, input, weight, groups) &&
           input.is_contiguous() && grad_output.is_contiguous();
}

Tensor conv2d_depthwise_grad_input_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    (void)groups;
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(0), R = weight.size(2), S = weight.size(3);
    const int64_t OH = grad_output.size(2), OW = grad_output.size(3);
    const int64_t multiplier = K / C;

    // The result keeps the order the forward input arrived in; a gradient
    // stored in the other order is repacked once, which is far cheaper than
    // reading it strided at every tap.
    const bool channel_major = depthwise_channels_last_4d(input);
    Tensor dy = grad_output;
    const bool dy_in_order = channel_major
        ? grad_output.is_contiguous(MemoryFormat::ChannelsLast)
        : grad_output.is_contiguous();
    if (!dy_in_order) {
        dy = grad_output.contiguous(static_cast<int64_t>(
            channel_major ? MemoryFormat::ChannelsLast
                          : MemoryFormat::Contiguous));
    }

    Tensor grad_input = Tensor::empty_like(input, input.dtype(), input.device());
    const auto stream = getCurrentCUDAStream().stream();
    const int in_plane = static_cast<int>(H * W);
    const dim3 grid(depthwise_blocks(C * in_plane, kDepthwiseThreads),
                    static_cast<unsigned>(N));
    const auto sh = static_cast<int>(stride[0]), sw = static_cast<int>(stride[1]);
    const auto ph = static_cast<int>(padding[0]), pw = static_cast<int>(padding[1]);
    const auto dh = static_cast<int>(dilation[0]), dw = static_cast<int>(dilation[1]);
    const int w_extent = static_cast<int>(W), h_extent = static_cast<int>(H);
    const int ow = static_cast<int>(OW), oh = static_cast<int>(OH);
    const int channels = static_cast<int>(C);
    const int mult = static_cast<int>(multiplier);
    const int rr = static_cast<int>(R), ss = static_cast<int>(S);

    // The tap count and the stride are template parameters: the tap loop
    // unrolls, and a unit stride drops the divisibility test on every tap.
    auto launch = [&](auto taps, auto step) {
        constexpr int KS = decltype(taps)::value;
        constexpr int ST = decltype(step)::value;
        if (channel_major) {
            switch (input.dtype()) {
                case DType::Float32:
                    depthwise_conv2d_grad_input_kernel<KS, ST, true, float, float>
                        <<<grid, kDepthwiseThreads, 0, stream>>>(
                            w_extent, h_extent, ow, oh, rr, ss, channels, mult,
                            sh, sw, ph, pw, dh, dw, dy.data_ptr<float>(),
                            weight.data_ptr<float>(),
                            grad_input.data_ptr<float>());
                    break;
                case DType::Float16:
                    depthwise_conv2d_grad_input_kernel<KS, ST, true,
                                                       tensorplay::Half, float>
                        <<<grid, kDepthwiseThreads, 0, stream>>>(
                            w_extent, h_extent, ow, oh, rr, ss, channels, mult,
                            sh, sw, ph, pw, dh, dw,
                            dy.data_ptr<tensorplay::Half>(),
                            weight.data_ptr<tensorplay::Half>(),
                            grad_input.data_ptr<tensorplay::Half>());
                    break;
                case DType::BFloat16:
                    depthwise_conv2d_grad_input_kernel<KS, ST, true,
                                                       tensorplay::BFloat16, float>
                        <<<grid, kDepthwiseThreads, 0, stream>>>(
                            w_extent, h_extent, ow, oh, rr, ss, channels, mult,
                            sh, sw, ph, pw, dh, dw,
                            dy.data_ptr<tensorplay::BFloat16>(),
                            weight.data_ptr<tensorplay::BFloat16>(),
                            grad_input.data_ptr<tensorplay::BFloat16>());
                    break;
                default:
                    TP_THROW(NotImplementedError, "depthwise conv2d grad_input");
            }
        } else {
            switch (input.dtype()) {
                case DType::Float32:
                    depthwise_conv2d_grad_input_kernel<KS, ST, false, float, float>
                        <<<grid, kDepthwiseThreads, 0, stream>>>(
                            w_extent, h_extent, ow, oh, rr, ss, channels, mult,
                            sh, sw, ph, pw, dh, dw, dy.data_ptr<float>(),
                            weight.data_ptr<float>(),
                            grad_input.data_ptr<float>());
                    break;
                case DType::Float16:
                    depthwise_conv2d_grad_input_kernel<KS, ST, false,
                                                       tensorplay::Half, float>
                        <<<grid, kDepthwiseThreads, 0, stream>>>(
                            w_extent, h_extent, ow, oh, rr, ss, channels, mult,
                            sh, sw, ph, pw, dh, dw,
                            dy.data_ptr<tensorplay::Half>(),
                            weight.data_ptr<tensorplay::Half>(),
                            grad_input.data_ptr<tensorplay::Half>());
                    break;
                case DType::BFloat16:
                    depthwise_conv2d_grad_input_kernel<KS, ST, false,
                                                       tensorplay::BFloat16, float>
                        <<<grid, kDepthwiseThreads, 0, stream>>>(
                            w_extent, h_extent, ow, oh, rr, ss, channels, mult,
                            sh, sw, ph, pw, dh, dw,
                            dy.data_ptr<tensorplay::BFloat16>(),
                            weight.data_ptr<tensorplay::BFloat16>(),
                            grad_input.data_ptr<tensorplay::BFloat16>());
                    break;
                default:
                    TP_THROW(NotImplementedError, "depthwise conv2d grad_input");
            }
        }
    };

    if (rr == 7 && ss == 7 && sh == 1 && sw == 1) {
        launch(std::integral_constant<int, 7>{}, std::integral_constant<int, 1>{});
    } else if (rr == 3 && ss == 3 && sh == 1 && sw == 1) {
        launch(std::integral_constant<int, 3>{}, std::integral_constant<int, 1>{});
    } else if (sh == 1 && sw == 1) {
        launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
    } else {
        launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
    }
    return grad_input;
}

Tensor conv2d_depthwise_grad_weight_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    (void)groups;
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(0), R = weight.size(2), S = weight.size(3);
    const int64_t OH = grad_output.size(2), OW = grad_output.size(3);
    const int64_t multiplier = K / C;

    Tensor grad_weight = Tensor::empty({K, 1, R, S}, weight.dtype(), weight.device());
    const auto stream = getCurrentCUDAStream().stream();
    const int threads = static_cast<int>(
        std::min<int64_t>(N * kDepthwiseWarpSize, kDepthwiseThreads));
    const dim3 grid(static_cast<unsigned>(R * S), static_cast<unsigned>(K));
    const auto sh = static_cast<int>(stride[0]), sw = static_cast<int>(stride[1]);
    const auto ph = static_cast<int>(padding[0]), pw = static_cast<int>(padding[1]);
    const auto dh = static_cast<int>(dilation[0]), dw = static_cast<int>(dilation[1]);
    const int in_plane = static_cast<int>(H * W);
    const int out_plane = static_cast<int>(OH * OW);
    const int channels = static_cast<int>(C);
    const int mult = static_cast<int>(multiplier);
    const int batch = static_cast<int>(N);

    switch (weight.dtype()) {
        case DType::Float32:
            depthwise_conv2d_grad_weight_kernel<float, float>
                <<<grid, threads, 0, stream>>>(
                    batch, channels, mult, in_plane, static_cast<int>(W),
                    static_cast<int>(H), out_plane, static_cast<int>(OW),
                    static_cast<int>(OH), static_cast<int>(R), static_cast<int>(S),
                    sh, sw, ph, pw, dh, dw, grad_output.data_ptr<float>(),
                    input.data_ptr<float>(), grad_weight.data_ptr<float>());
            break;
        case DType::Float16:
            depthwise_conv2d_grad_weight_kernel<tensorplay::Half, float>
                <<<grid, threads, 0, stream>>>(
                    batch, channels, mult, in_plane, static_cast<int>(W),
                    static_cast<int>(H), out_plane, static_cast<int>(OW),
                    static_cast<int>(OH), static_cast<int>(R), static_cast<int>(S),
                    sh, sw, ph, pw, dh, dw, grad_output.data_ptr<tensorplay::Half>(),
                    input.data_ptr<tensorplay::Half>(),
                    grad_weight.data_ptr<tensorplay::Half>());
            break;
        case DType::BFloat16:
            depthwise_conv2d_grad_weight_kernel<tensorplay::BFloat16, float>
                <<<grid, threads, 0, stream>>>(
                    batch, channels, mult, in_plane, static_cast<int>(W),
                    static_cast<int>(H), out_plane, static_cast<int>(OW),
                    static_cast<int>(OH), static_cast<int>(R), static_cast<int>(S),
                    sh, sw, ph, pw, dh, dw,
                    grad_output.data_ptr<tensorplay::BFloat16>(),
                    input.data_ptr<tensorplay::BFloat16>(),
                    grad_weight.data_ptr<tensorplay::BFloat16>());
            break;
        default:
            TP_THROW(NotImplementedError, "depthwise conv2d grad_weight");
    }
    return grad_weight;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, DepthwiseConv2dKernels) {
    // depthwise / slow convolutions
    m.impl("_conv_depthwise2d", interop__conv_depthwise2d_cuda);
    m.impl("_conv_depthwise2d.out", interop__conv_depthwise2d_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
