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
// The filter gradient keeps one accumulator per tap in registers, so callers
// with a wider filter fall back to the general grouped path.
constexpr int kMaxDepthwiseTaps = 64;

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

// One thread per VEC adjacent output columns.  The window of the VEC sites
// spans VEC + kw - 1 consecutive positions of one input row, so a row is read
// once and every tap of every site in the group reuses those values, which
// keeps the load count below the multiply count.  The weights stay in
// registers.  A window that leaves the input takes the guarded route, where
// each site tests its own taps; a group whose sites are strided or dilated
// also takes it, as the covered positions are then not contiguous.
template <int KS, int VEC, typename T, typename M>
__global__ void depthwise_conv2d_forward_kernel(
    int W, int H, int OW, int OH, int channels, int multiplier, int R, int S,
    int sh, int sw, int ph, int pw, int dh, int dw,
    const T* __restrict__ input, const T* __restrict__ weight,
    const T* __restrict__ bias, T* __restrict__ output) {
    const int kh = (KS != 0) ? KS : R;
    const int kw = (KS != 0) ? KS : S;
    const int taps = kh * kw;
    const int out_channel = blockIdx.y;
    const int in_channel = out_channel / multiplier;
    const int in_plane = H * W;
    const int out_plane = OH * OW;
    const int out_channels = channels * multiplier;
    const int cols = (OW + VEC - 1) / VEC;
    // A unit-step, unit-dilation row places the group's taps on consecutive
    // positions, which is what lets one read serve several sites.
    const bool unit_step = sw == 1 && dw == 1;

    M wreg[kMaxDepthwiseTaps];
#pragma unroll
    for (int t = 0; t < taps; ++t) {
        wreg[t] = static_cast<M>(
            weight[static_cast<int64_t>(out_channel) * taps + t]);
    }
    // The bias is one number per output channel, so it starts the
    // accumulator rather than being added to the result afterwards.
    const M bias_value = bias ? static_cast<M>(bias[out_channel]) : M(0);

    const T* x = input +
        (static_cast<int64_t>(blockIdx.z) * channels + in_channel) * in_plane;
    T* y = output +
        (static_cast<int64_t>(blockIdx.z) * out_channels + out_channel) *
            out_plane;

    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < OH * cols;
         g += blockDim.x * gridDim.x) {
        const int oh = g / cols;
        const int ow0 = (g - oh * cols) * VEC;
        M acc[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) acc[j] = bias_value;
#pragma unroll
        for (int r = 0; r < kh; ++r) {
            const int hi = oh * sh + r * dh - ph;
            if (hi < 0 || hi >= H) continue;
            const T* xrow = x + hi * W;
            const int base = ow0 - pw;
            if (KS != 0 && unit_step && base >= 0 &&
                base + VEC + (KS - 2) < W) {
                M xv[VEC + KS];
#pragma unroll
                for (int t = 0; t < VEC + KS - 1; ++t) {
                    xv[t] = static_cast<M>(xrow[base + t]);
                }
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
#pragma unroll
                    for (int s = 0; s < KS; ++s) {
                        acc[j] += wreg[r * KS + s] * xv[j + s];
                    }
                }
            } else if (KS != 0 && unit_step) {
                // The group's window runs past one or both ends of the row,
                // which is the common case when the plane is as wide as the
                // filter.  Reading the whole window once with the outside
                // positions taken as zero still visits each row element a
                // fixed number of times; the per-site route below would read
                // every tap of every site from memory instead.
                M xv[VEC + KS];
#pragma unroll
                for (int t = 0; t < VEC + KS - 1; ++t) {
                    const int i = base + t;
                    xv[t] = (i >= 0 && i < W) ? static_cast<M>(xrow[i]) : M(0);
                }
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
#pragma unroll
                    for (int s = 0; s < KS; ++s) {
                        acc[j] += wreg[r * KS + s] * xv[j + s];
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    const int ow = ow0 + j;
                    if (ow >= OW) continue;
#pragma unroll
                    for (int s = 0; s < kw; ++s) {
                        const int wi = (ow * sw + s * dw) - pw;
                        if (wi < 0 || wi >= W) continue;
                        acc[j] += wreg[r * kw + s] * static_cast<M>(xrow[wi]);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int ow = ow0 + j;
            if (ow < OW) y[oh * OW + ow] = static_cast<T>(acc[j]);
        }
    }
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

// One block per output channel: the block keeps one accumulator per filter
// tap and walks the batches and the output sites.  A site contributes to
// every tap at once, so each operand element is read once per block instead
// of once per tap, and the accumulators are reduced over the block at the
// end.  The tap count is a template parameter so the accumulators stay in
// registers.
template <int KS, typename T, typename M>
__global__ void depthwise_conv2d_grad_weight_kernel(
    int batch, int channels, int multiplier, int H, int W, int OH, int OW,
    int R, int S, int sh, int sw, int ph, int pw, int dh, int dw,
    const T* __restrict__ grad_output, const T* __restrict__ input,
    T* __restrict__ grad_weight) {
    const int kh = (KS != 0) ? KS : R;
    const int kw = (KS != 0) ? KS : S;
    const int taps = kh * kw;
    const int out_channel = blockIdx.x;
    const int in_channel = out_channel / multiplier;
    const int in_plane = H * W;
    const int out_plane = OH * OW;
    const int out_channels = channels * multiplier;
    const int64_t tap_base = static_cast<int64_t>(out_channel) * taps;

    // One accumulator per tap; for a fixed kernel size the loops below unroll
    // to that count, so the array stays in registers.
    M acc[kMaxDepthwiseTaps];
#pragma unroll
    for (int t = 0; t < taps; ++t) acc[t] = M(0);

    for (int n = 0; n < batch; ++n) {
        const T* dy = grad_output +
            (static_cast<int64_t>(n) * out_channels + out_channel) * out_plane;
        const T* x = input +
            (static_cast<int64_t>(n) * channels + in_channel) * in_plane;
        for (int p = threadIdx.x; p < out_plane; p += blockDim.x) {
            const int oh = p / OW;
            const int ow = p - oh * OW;
            const M dyv = static_cast<M>(dy[p]);
            for (int r = 0; r < kh; ++r) {
                const int hi = oh * sh + r * dh - ph;
                if (hi < 0 || hi >= H) continue;
#pragma unroll
                for (int s = 0; s < kw; ++s) {
                    const int wi = ow * sw + s * dw - pw;
                    if (wi < 0 || wi >= W) continue;
                    acc[r * kw + s] +=
                        dyv * static_cast<M>(x[hi * W + wi]);
                }
            }
        }
    }

    const int lane = threadIdx.x % kDepthwiseWarpSize;
    const int warp = threadIdx.x / kDepthwiseWarpSize;
    const int warps = blockDim.x / kDepthwiseWarpSize;
#pragma unroll
    for (int off = kDepthwiseWarpSize / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int t = 0; t < taps; ++t) {
            acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);
        }
    }

    __shared__ M warp_sums[8 * kMaxDepthwiseTaps];
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < taps; ++t) {
            warp_sums[warp * taps + t] = acc[t];
        }
    }
    __syncthreads();
    if (threadIdx.x < taps) {
        M total = M(0);
        for (int w = 0; w < warps; ++w) {
            total += warp_sums[w * taps + threadIdx.x];
        }
        grad_weight[tap_base + threadIdx.x] = static_cast<T>(total);
    }
}

// The grouping, dtype and operand extent every depthwise call must share; the
// layout and the gradient's shape are checked by the callers.
bool depthwise_operands_ok(const Tensor& input, const Tensor& weight,
                           int64_t groups) {
    if (input.dim() != 4 || weight.dim() != 4) return false;
    if (groups <= 1 || groups != input.size(1) || weight.size(1) != 1) {
        return false;
    }
    if (weight.size(0) % input.size(1) != 0) return false;
    const DType dtype = input.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float16 &&
        dtype != DType::BFloat16) {
        return false;
    }
    if (weight.dtype() != dtype) return false;
    if (!weight.is_contiguous()) return false;
    if (input.numel() == 0 || weight.numel() == 0) return false;
    // The site loops index with 32-bit arithmetic; refuse a call that would
    // not fit rather than wrap.
    return input.numel() <= 0x7fffffff;
}

bool depthwise_shape_ok(const Tensor& grad_output, const Tensor& input,
                        const Tensor& weight, int64_t groups) {
    if (!depthwise_operands_ok(input, weight, groups)) return false;
    if (grad_output.dim() != 4) return false;
    if (input.size(0) != grad_output.size(0) ||
        weight.size(0) != grad_output.size(1)) {
        return false;
    }
    if (grad_output.dtype() != input.dtype()) return false;
    if (grad_output.numel() == 0 || grad_output.numel() > 0x7fffffff) {
        return false;
    }
    return true;
}

} // namespace

bool conv2d_depthwise_forward_eligible(const Tensor& input,
                                       const Tensor& weight, int64_t groups) {
    if (weight.size(2) * weight.size(3) > kMaxDepthwiseTaps) return false;
    return depthwise_operands_ok(input, weight, groups) &&
           input.is_contiguous();
}

Tensor conv2d_depthwise_forward_cuda(
    const Tensor& input, const Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups,
    const Tensor& bias, const Tensor& out_provided) {
    (void)groups;
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H = input.size(2), W = input.size(3);
    const int64_t K = weight.size(0), R = weight.size(2), S = weight.size(3);
    const auto sh = static_cast<int>(stride[0]), sw = static_cast<int>(stride[1]);
    const auto ph = static_cast<int>(padding[0]), pw = static_cast<int>(padding[1]);
    const auto dh = static_cast<int>(dilation[0]), dw = static_cast<int>(dilation[1]);
    const int64_t OH = (H + 2 * ph - dh * (R - 1) - 1) / sh + 1;
    const int64_t OW = (W + 2 * pw - dw * (S - 1) - 1) / sw + 1;

    // A buffer the caller handed over is written in place, which spares the
    // allocation and the copy into it; its arrival in the row-major order this
    // kernel writes is what makes it usable.
    Tensor output;
    if (out_provided.defined() && out_provided.dtype() == input.dtype() &&
        out_provided.device() == input.device() &&
        static_cast<std::vector<int64_t>>(out_provided.shape()) ==
            std::vector<int64_t>{N, K, OH, OW} &&
        out_provided.is_contiguous()) {
        output = out_provided;
    } else {
        output = Tensor::empty({N, K, OH, OW}, input.dtype(), input.device());
    }
    const auto stream = getCurrentCUDAStream().stream();
    const int h_extent = static_cast<int>(H), w_extent = static_cast<int>(W);
    const int oh = static_cast<int>(OH), ow = static_cast<int>(OW);
    const int channels = static_cast<int>(C);
    const int mult = static_cast<int>(K / C);
    const int rr = static_cast<int>(R), ss = static_cast<int>(S);
    // A bias of the output's own type, one number per output channel, starts
    // the accumulator; anything else is left to the caller.
    const bool has_bias = bias.defined() && bias.numel() == K &&
                          bias.dtype() == input.dtype() &&
                          bias.is_contiguous();

    // Columns per thread: a group spans one row of the input, so the widest
    // group that still gives every channel a full warp of groups wins.  Small
    // planes keep a narrower group and a smaller block instead.
    const int vec = ow >= 14 ? 7 : (ow >= 4 ? 4 : 2);
    const int group_count = oh * ((ow + vec - 1) / vec);
    const int threads = group_count >= 256 ? 256
                        : (group_count >= 128 ? 128 : 32);
    const dim3 grid(static_cast<unsigned>((group_count + threads - 1) / threads),
                    static_cast<unsigned>(K), static_cast<unsigned>(N));

    auto launch = [&](auto taps, auto vec_ic) {
        constexpr int KS = decltype(taps)::value;
        constexpr int VEC = decltype(vec_ic)::value;
        switch (input.dtype()) {
            case DType::Float32:
                depthwise_conv2d_forward_kernel<KS, VEC, float, float>
                    <<<grid, threads, 0, stream>>>(
                        w_extent, h_extent, ow, oh, channels, mult, rr, ss,
                        sh, sw, ph, pw, dh, dw, input.data_ptr<float>(),
                        weight.data_ptr<float>(),
                        has_bias ? bias.data_ptr<float>() : nullptr,
                        output.data_ptr<float>());
                break;
            case DType::Float16:
                depthwise_conv2d_forward_kernel<KS, VEC, tensorplay::Half, float>
                    <<<grid, threads, 0, stream>>>(
                        w_extent, h_extent, ow, oh, channels, mult, rr, ss,
                        sh, sw, ph, pw, dh, dw, input.data_ptr<tensorplay::Half>(),
                        weight.data_ptr<tensorplay::Half>(),
                        has_bias ? bias.data_ptr<tensorplay::Half>() : nullptr,
                        output.data_ptr<tensorplay::Half>());
                break;
            case DType::BFloat16:
                depthwise_conv2d_forward_kernel<KS, VEC, tensorplay::BFloat16,
                                                float>
                    <<<grid, threads, 0, stream>>>(
                        w_extent, h_extent, ow, oh, channels, mult, rr, ss,
                        sh, sw, ph, pw, dh, dw,
                        input.data_ptr<tensorplay::BFloat16>(),
                        weight.data_ptr<tensorplay::BFloat16>(),
                        has_bias ? bias.data_ptr<tensorplay::BFloat16>()
                                 : nullptr,
                        output.data_ptr<tensorplay::BFloat16>());
                break;
            default:
                TP_THROW(NotImplementedError, "depthwise conv2d forward");
        }
    };

    // The tap count is a template parameter: the tap loop unrolls and the
    // accumulators stay in registers.  Filter sizes the models do not use
    // read it at run time instead.
    auto pick_vec = [&](auto taps) {
        constexpr int KS = decltype(taps)::value;
        if (vec == 7) {
            launch(std::integral_constant<int, KS>{}, std::integral_constant<int, 7>{});
        } else if (vec == 4) {
            launch(std::integral_constant<int, KS>{}, std::integral_constant<int, 4>{});
        } else {
            launch(std::integral_constant<int, KS>{}, std::integral_constant<int, 2>{});
        }
    };

    if (rr == 7 && ss == 7) {
        pick_vec(std::integral_constant<int, 7>{});
    } else if (rr == 5 && ss == 5) {
        pick_vec(std::integral_constant<int, 5>{});
    } else if (rr == 3 && ss == 3) {
        pick_vec(std::integral_constant<int, 3>{});
    } else if (rr == 1 && ss == 1) {
        pick_vec(std::integral_constant<int, 1>{});
    } else {
        launch(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
    }
    return output;
}

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
    if (weight.size(2) * weight.size(3) > kMaxDepthwiseTaps) return false;
    // Each operand is read in whichever of the two dense 4-D orders it already
    // has, so neither has to be repacked for this call.
    return depthwise_shape_ok(grad_output, input, weight, groups) &&
           (input.is_contiguous() || depthwise_channels_last_4d(input)) &&
           (grad_output.is_contiguous() ||
            depthwise_channels_last_4d(grad_output));
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

    // The kernel walks one channel's sites as a dense run, which is how the
    // row-major order stores them.  An operand in the other order is repacked
    // once here rather than read a channel apart at every position, which
    // would cost a separate transaction per element.
    Tensor dy = grad_output.is_contiguous() ? grad_output
                                            : grad_output.contiguous();
    Tensor x = input.is_contiguous() ? input : input.contiguous();

    Tensor grad_weight = Tensor::empty({K, 1, R, S}, weight.dtype(), weight.device());
    const auto stream = getCurrentCUDAStream().stream();
    // One block per output channel, wide enough to keep every lane busy on
    // the output plane when that plane is small.
    const int64_t positions = OH * OW;
    const int threads = static_cast<int>(std::min<int64_t>(
        kDepthwiseThreads, (positions + kDepthwiseWarpSize - 1) /
                               kDepthwiseWarpSize * kDepthwiseWarpSize));
    const dim3 grid(static_cast<unsigned>(K));
    const auto sh = static_cast<int>(stride[0]), sw = static_cast<int>(stride[1]);
    const auto ph = static_cast<int>(padding[0]), pw = static_cast<int>(padding[1]);
    const auto dh = static_cast<int>(dilation[0]), dw = static_cast<int>(dilation[1]);
    const int h_extent = static_cast<int>(H), w_extent = static_cast<int>(W);
    const int oh = static_cast<int>(OH), ow = static_cast<int>(OW);
    const int channels = static_cast<int>(C);
    const int mult = static_cast<int>(multiplier);
    const int batch = static_cast<int>(N);
    const int rr = static_cast<int>(R), ss = static_cast<int>(S);

    auto launch = [&](auto taps) {
        constexpr int KS = decltype(taps)::value;
        switch (weight.dtype()) {
            case DType::Float32:
                depthwise_conv2d_grad_weight_kernel<KS, float, float>
                    <<<grid, threads, 0, stream>>>(
                        batch, channels, mult, h_extent, w_extent, oh, ow,
                        rr, ss, sh, sw, ph, pw, dh, dw,
                        dy.data_ptr<float>(), x.data_ptr<float>(),
                        grad_weight.data_ptr<float>());
                break;
            case DType::Float16:
                depthwise_conv2d_grad_weight_kernel<KS, tensorplay::Half, float>
                    <<<grid, threads, 0, stream>>>(
                        batch, channels, mult, h_extent, w_extent, oh, ow,
                        rr, ss, sh, sw, ph, pw, dh, dw,
                        dy.data_ptr<tensorplay::Half>(),
                        x.data_ptr<tensorplay::Half>(),
                        grad_weight.data_ptr<tensorplay::Half>());
                break;
            case DType::BFloat16:
                depthwise_conv2d_grad_weight_kernel<KS, tensorplay::BFloat16,
                                                    float>
                    <<<grid, threads, 0, stream>>>(
                        batch, channels, mult, h_extent, w_extent, oh, ow,
                        rr, ss, sh, sw, ph, pw, dh, dw,
                        dy.data_ptr<tensorplay::BFloat16>(),
                        x.data_ptr<tensorplay::BFloat16>(),
                        grad_weight.data_ptr<tensorplay::BFloat16>());
                break;
            default:
                TP_THROW(NotImplementedError, "depthwise conv2d grad_weight");
        }
    };

    if (rr == 7 && ss == 7) {
        launch(std::integral_constant<int, 7>{});
    } else if (rr == 5 && ss == 5) {
        launch(std::integral_constant<int, 5>{});
    } else if (rr == 3 && ss == 3) {
        launch(std::integral_constant<int, 3>{});
    } else if (rr == 1 && ss == 1) {
        launch(std::integral_constant<int, 1>{});
    } else {
        launch(std::integral_constant<int, 0>{});
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
