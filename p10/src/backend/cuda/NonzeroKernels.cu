#include "CudaDispatchHelpers.cuh"
#include "Complex.h"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

namespace tensorplay {
namespace cuda {


namespace {

constexpr int kNonzeroThreads = 256;
constexpr int kNonzeroItems = 16;
constexpr int64_t kNonzeroTileSize = kNonzeroThreads * kNonzeroItems;
constexpr int kNonzeroMaxRank = 64;

struct NonzeroShape {
    int64_t sizes[kNonzeroMaxRank];
    int rank;
};

template <typename T>
__global__ void count_nonzero_tiles(const T* input, int64_t count, int64_t* tile_counts) {
    using BlockReduce = cub::BlockReduce<int, kNonzeroThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    const int64_t begin = static_cast<int64_t>(blockIdx.x) * kNonzeroTileSize;
    int found = 0;
    for (int item = 0; item < kNonzeroItems; ++item) {
        const int64_t index = begin + threadIdx.x + item * kNonzeroThreads;
        found += index < count && input[index] != T(0);
    }
    const int total = BlockReduce(storage).Sum(found);
    if (threadIdx.x == 0) tile_counts[blockIdx.x] = total;
}

template <typename T, bool SingleTile>
__global__ void write_nonzero_tiles(
        const T* input, int64_t count, NonzeroShape shape, int64_t limit,
        int tiles, const int64_t* tile_counts, const int64_t* tile_offsets,
        int64_t fill_value, int64_t* out) {
    int64_t base_row = 0;
    int64_t matches = 0;
    const bool has_input_tile = SingleTile || blockIdx.x < tiles;
    if constexpr (!SingleTile) {
        if (tile_offsets != nullptr) {
            matches = tile_offsets[tiles - 1] + tile_counts[tiles - 1];
            if (has_input_tile) base_row = tile_offsets[blockIdx.x];
        } else {
            // A warp can compute both prefixes directly when there are at
            // most 32 tiles, saving a separate offset scan and its scratch.
            __shared__ int64_t prefixes[2];
            const int lane = threadIdx.x;
            if (lane < 32) {
                int64_t total = lane < tiles ? tile_counts[lane] : 0;
                int64_t before = lane < blockIdx.x ? total : 0;
                for (int distance = 16; distance > 0; distance /= 2) {
                    total += __shfl_down_sync(0xffffffffu, total, distance);
                    before += __shfl_down_sync(0xffffffffu, before, distance);
                }
                if (lane == 0) {
                    prefixes[0] = before;
                    prefixes[1] = total;
                }
            }
            __syncthreads();
            base_row = prefixes[0];
            matches = prefixes[1];
        }
    }

    if (has_input_tile && base_row < limit) {
        using BlockScan = cub::BlockScan<int, kNonzeroThreads, cub::BLOCK_SCAN_WARP_SCANS>;
        __shared__ typename BlockScan::TempStorage storage;
        const int64_t begin = static_cast<int64_t>(blockIdx.x) * kNonzeroTileSize +
                              threadIdx.x * kNonzeroItems;
        int flags[kNonzeroItems];
        int ranks[kNonzeroItems];
#pragma unroll
        for (int item = 0; item < kNonzeroItems; ++item) {
            const int64_t index = begin + item;
            flags[item] = index < count && input[index] != T(0);
        }
        int total = 0;
        BlockScan(storage).ExclusiveSum(flags, ranks, total);
        if constexpr (SingleTile) matches = total;
#pragma unroll
        for (int item = 0; item < kNonzeroItems; ++item) {
            const int64_t row = base_row + ranks[item];
            if (!flags[item] || row >= limit) continue;
            int64_t index = begin + item;
            for (int axis = shape.rank - 1; axis >= 0; --axis) {
                const int64_t width = shape.sizes[axis];
                out[axis * limit + row] = index % width;
                index /= width;
            }
        }
    }

    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t row = matches + thread; row < limit; row += stride) {
        for (int axis = 0; axis < shape.rank; ++axis) {
            out[axis * limit + row] = fill_value;
        }
    }
}

template <typename T>
void nonzero_static_typed(const Tensor& input, const NonzeroShape& shape,
                          int64_t limit, int64_t fill_value, Tensor& out) {
    const int64_t count = input.numel();
    const int64_t tiles = (count - 1) / kNonzeroTileSize + 1;
    TP_CHECK(tiles <= std::numeric_limits<int>::max(), "nonzero_static: input is too large");
    const auto stream = getCurrentCUDAStream().stream();
    const T* values = input.const_data_ptr<T>();
    int64_t* indices = out.data_ptr<int64_t>();
    if (tiles == 1 && limit <= kNonzeroTileSize) {
        write_nonzero_tiles<T, true><<<1, kNonzeroThreads, 0, stream>>>(
            values, count, shape, limit, 1, nullptr, nullptr, fill_value, indices);
        checkCuda(cudaGetLastError(), "nonzero_static tile scan");
        return;
    }

    // Prefix offsets assign adjacent output rows to adjacent input tiles.
    // The final offset and count also delimit padding, entirely on device.
    Tensor counts = Tensor::empty({tiles}, DType::Int64, out.device());
    Tensor offsets;
    Tensor scratch;
    int64_t* counts_ptr = counts.data_ptr<int64_t>();
    int64_t* offsets_ptr = nullptr;
    const int tile_grid = static_cast<int>(tiles);
    count_nonzero_tiles<T><<<tile_grid, kNonzeroThreads, 0, stream>>>(values, count, counts_ptr);
    checkCuda(cudaGetLastError(), "nonzero_static tile counts");

    if (tiles > 32) {
        offsets = Tensor::empty({tiles}, DType::Int64, out.device());
        offsets_ptr = offsets.data_ptr<int64_t>();
        size_t scratch_bytes = 0;
        checkCuda(cub::DeviceScan::ExclusiveSum(
            nullptr, scratch_bytes, counts_ptr, offsets_ptr, tile_grid, stream), "nonzero_static scan size");
        scratch = Tensor::empty(
            {static_cast<int64_t>(std::max<size_t>(scratch_bytes, 1))}, DType::UInt8, out.device());
        checkCuda(cub::DeviceScan::ExclusiveSum(
            scratch.data_ptr(), scratch_bytes, counts_ptr, offsets_ptr, tile_grid, stream), "nonzero_static scan");
    }

    const auto pad_grid = static_cast<int>(std::min<int64_t>((limit - 1) / kNonzeroTileSize + 1, 4096));
    const int grid = std::max(tile_grid, pad_grid);
    write_nonzero_tiles<T, false><<<grid, kNonzeroThreads, 0, stream>>>(
        values, count, shape, limit, tile_grid, counts_ptr, offsets_ptr, fill_value, indices);
    checkCuda(cudaGetLastError(), "nonzero_static tile scatter");
}

Tensor interop_nonzero_static_cuda(const Tensor& self, int64_t size, int64_t fill_value) {
    TP_CHECK(size >= 0, "nonzero_static: size must be non-negative");
    const int64_t ndim = self.dim();
    TP_CHECK(ndim <= kNonzeroMaxRank, "nonzero_static: input has too many dimensions");
    OptionalCUDAGuard guard(self.device());
    Tensor result = Tensor::empty({ndim, size}, DType::Int64, self.device()).transpose(0, 1);
    if (result.numel() == 0) return result;
    if (self.numel() == 0) return result.fill_(Scalar(fill_value));

    const Tensor input = self.contiguous();
    NonzeroShape shape{};
    shape.rank = static_cast<int>(ndim);
    const auto sizes = input.sizes();
    std::copy(sizes.begin(), sizes.end(), shape.sizes);
#define TP_NONZERO_STATIC_CASE(ctype, name) \
    case DType::name: nonzero_static_typed<ctype>(input, shape, size, fill_value, result); break;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NONZERO_STATIC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NONZERO_STATIC_CASE)
        TP_NONZERO_STATIC_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_NONZERO_STATIC_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_NONZERO_STATIC_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_NONZERO_STATIC_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default: TP_THROW(TypeError, "nonzero_static: unsupported dtype");
    }
#undef TP_NONZERO_STATIC_CASE
    return result;
}


Tensor& interop_nonzero_static_out_cuda(const Tensor& self, int64_t size,
                                        int64_t fill_value, Tensor& out) {
    if (out.defined()) {
        TP_CHECK_TYPE(out.dtype() == DType::Int64, "nonzero_static: out must be int64");
        TP_CHECK(out.device() == self.device(), "nonzero_static: out must be on the input device");
    }
    write_out(out, interop_nonzero_static_cuda(self, size, fill_value));
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NonzeroKernels) {
    m.impl("nonzero_static", interop_nonzero_static_cuda);
    m.impl("nonzero_static.out", interop_nonzero_static_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
