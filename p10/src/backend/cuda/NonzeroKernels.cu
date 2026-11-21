#include "CudaDispatchHelpers.cuh"
#include "Complex.h"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

#include <numeric>

namespace tensorplay {
namespace cuda {


namespace {

constexpr int kNonzeroThreads = 256;
// Tiles are the unit of parallel work: every block reduces and scans one tile.
// Widths trade metadata for parallelism, so mid-sized inputs use the narrowest
// tile that still keeps the tile table inside the inline prefix scan.
constexpr int kNonzeroItems = 16;
constexpr int kNonzeroSmallItems = 1;
constexpr int kInlineNonzeroScanTiles = 256;
constexpr int64_t kNonzeroTileSize = kNonzeroThreads * kNonzeroItems;
constexpr int64_t kSmallNonzeroCapacity = 16 * 1024;
constexpr int kNonzeroMaxRank = 64;

struct NonzeroShape {
    int64_t sizes[kNonzeroMaxRank];
    int rank;
};

template <typename Index>
__device__ void write_coordinate(Index index, const NonzeroShape& shape,
                                 int64_t rows, int64_t row, int64_t* out) {
    for (int axis = shape.rank - 1; axis > 0; --axis) {
        const auto width = static_cast<Index>(shape.sizes[axis]);
        out[axis * rows + row] = index % width;
        index /= width;
    }
    if (shape.rank != 0) out[row] = index;
}

template <typename T, int Items>
__global__ void count_nonzero_tiles(const T* input, int64_t count, int64_t* tile_counts) {
    using BlockReduce = cub::BlockReduce<int, kNonzeroThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    const int64_t begin = static_cast<int64_t>(blockIdx.x) * (kNonzeroThreads * Items);
    int found = 0;
    for (int item = 0; item < Items; ++item) {
        const int64_t index = begin + threadIdx.x + item * kNonzeroThreads;
        found += index < count && input[index] != T(0);
    }
    const int total = BlockReduce(storage).Sum(found);
    if (threadIdx.x == 0) tile_counts[blockIdx.x] = total;
}

template <typename T, int Items, bool SingleTile, bool ReadCount = false>
__global__ void write_nonzero_tiles(
        const T* input, int64_t count, NonzeroShape shape, int64_t limit,
        int tiles, const int64_t* tile_counts, const int64_t* tile_offsets,
        int64_t fill_value, int64_t* out, int64_t* total_out = nullptr) {
    static_assert(!ReadCount || SingleTile);
    int64_t base_row = 0;
    int64_t matches = 0;
    const bool has_input_tile = SingleTile || blockIdx.x < tiles;
    if constexpr (!SingleTile) {
        if (tile_offsets != nullptr) {
            matches = tile_offsets[tiles - 1];
            if (has_input_tile && blockIdx.x != 0) base_row = tile_offsets[blockIdx.x - 1];
        } else {
            // A warp can compute both prefixes from a small count array,
            // saving a separate offset scan and its scratch.
            __shared__ int64_t prefixes[2];
            const int lane = threadIdx.x;
            if (lane < 32) {
                int64_t total = 0;
                int64_t before = 0;
                for (int tile = lane; tile < tiles; tile += 32) {
                    const int64_t tile_count = tile_counts[tile];
                    total += tile_count;
                    if (static_cast<unsigned>(tile) < blockIdx.x) before += tile_count;
                }
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
        const int64_t begin = static_cast<int64_t>(blockIdx.x) * (kNonzeroThreads * Items) +
                              threadIdx.x * Items;
        int flags[Items];
        int ranks[Items];
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            const int64_t index = begin + item;
            flags[item] = index < count && input[index] != T(0);
        }
        int total = 0;
        BlockScan(storage).ExclusiveSum(flags, ranks, total);
        if constexpr (SingleTile) matches = total;
        if constexpr (ReadCount) {
            if (threadIdx.x == 0) *total_out = total;
            limit = total;
        }
#pragma unroll
        for (int item = 0; item < Items; ++item) {
            const int64_t row = base_row + ranks[item];
            if (!flags[item] || row >= limit) continue;
            const int64_t index = begin + item;
            if (SingleTile || count <= std::numeric_limits<int32_t>::max()) {
                write_coordinate(static_cast<int32_t>(index), shape, limit, row, out);
            } else {
                write_coordinate(index, shape, limit, row, out);
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

struct NonzeroTiles {
    int blocks;
    Tensor counts;
    Tensor offsets;

    const int64_t* offset_data() const {
        return offsets.defined() ? offsets.const_data_ptr<int64_t>() : nullptr;
    }
};

template <typename T, int Items>
NonzeroTiles scan_nonzero_tiles(const Tensor& input, cudaStream_t stream) {
    const int64_t count = input.numel();
    const int64_t tiles = (count - 1) / (kNonzeroThreads * Items) + 1;
    TP_CHECK(tiles <= std::numeric_limits<int>::max(), "nonzero: input is too large");

    // Prefix offsets assign adjacent output rows to adjacent input tiles.
    NonzeroTiles result{static_cast<int>(tiles),
                        Tensor::empty({tiles}, DType::Int64, input.device()), Tensor()};
    auto* counts = result.counts.data_ptr<int64_t>();
    count_nonzero_tiles<T, Items><<<result.blocks, kNonzeroThreads, 0, stream>>>(
        input.const_data_ptr<T>(), count, counts);
    checkCuda(cudaGetLastError(), "nonzero tile counts");

    if (tiles > kInlineNonzeroScanTiles) {
        result.offsets = Tensor::empty({tiles}, DType::Int64, input.device());
        auto* offsets = result.offsets.data_ptr<int64_t>();
        size_t scratch_bytes = 0;
        checkCuda(cub::DeviceScan::InclusiveSum(
            nullptr, scratch_bytes, counts, offsets, result.blocks, stream), "nonzero scan size");
        Tensor scratch = Tensor::empty(
            {static_cast<int64_t>(std::max<size_t>(scratch_bytes, 1))}, DType::UInt8, input.device());
        checkCuda(cub::DeviceScan::InclusiveSum(
            scratch.data_ptr(), scratch_bytes, counts, offsets, result.blocks, stream), "nonzero scan");
    }
    return result;
}

int64_t read_nonzero_count(const int64_t* counts, int count, const CUDAStream& stream) {
    const size_t bytes = static_cast<size_t>(count) * sizeof(int64_t);
    auto host = getPinnedMemoryAllocator()->allocate(bytes);
    checkCuda(cudaMemcpyAsync(host.get(), counts, bytes, cudaMemcpyDeviceToHost, stream.stream()),
              "nonzero count transfer");
    try {
        stream.synchronize();
    } catch (...) {
        recordPinnedStream(host.get(), stream);
        throw;
    }
    const auto* first = static_cast<const int64_t*>(host.get());
    return std::accumulate(first, first + count, int64_t{0});
}

NonzeroShape nonzero_shape(const Tensor& input) {
    TP_CHECK(input.dim() <= kNonzeroMaxRank, "nonzero: input has too many dimensions");
    NonzeroShape shape{};
    shape.rank = static_cast<int>(input.dim());
    const auto sizes = input.sizes();
    std::copy(sizes.begin(), sizes.end(), shape.sizes);
    return shape;
}

Tensor nonzero_result(int64_t matches, int rank, const Device& device) {
    return Tensor::empty_strided({matches, rank}, {1, std::max<int64_t>(matches, 1)},
                                 DType::Int64, device);
}

template <typename T, int Items>
Tensor nonzero_tiled_typed(const Tensor& input, const NonzeroShape& shape,
                           const CUDAStream& stream) {
    const int64_t count = input.numel();
    const NonzeroTiles tiles = scan_nonzero_tiles<T, Items>(input, stream.stream());
    const int64_t* offsets = tiles.offset_data();
    const int64_t* counts = tiles.counts.const_data_ptr<int64_t>();
    const int64_t matches = offsets
        ? read_nonzero_count(offsets + tiles.blocks - 1, 1, stream)
        : read_nonzero_count(counts, tiles.blocks, stream);
    Tensor result = nonzero_result(matches, shape.rank, input.device());
    if (matches != 0) {
        write_nonzero_tiles<T, Items, false><<<tiles.blocks, kNonzeroThreads, 0, stream.stream()>>>(
            input.const_data_ptr<T>(), count, shape, matches, tiles.blocks, counts, offsets, 0,
            result.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "nonzero tile scatter");
    }
    return result;
}

template <typename T>
Tensor nonzero_fused_typed(const Tensor& input, const NonzeroShape& shape,
                           const CUDAStream& stream) {
    const int64_t count = input.numel();
    // One block finishes its scan and publishes the row count in the same pass.
    // Its output columns are already packed with that stride, so the host only
    // reads the count back.
    Tensor buffer = Tensor::empty({count * shape.rank}, DType::Int64, input.device());
    Tensor total = Tensor::empty({1}, DType::Int64, input.device());
    write_nonzero_tiles<T, kNonzeroSmallItems, true, true><<<1, kNonzeroThreads, 0, stream.stream()>>>(
        input.const_data_ptr<T>(), count, shape, count, 1, nullptr, nullptr, 0,
        shape.rank != 0 ? buffer.data_ptr<int64_t>() : nullptr, total.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "nonzero single tile");
    const int64_t matches = read_nonzero_count(total.const_data_ptr<int64_t>(), 1, stream);
    if (matches == 0 || shape.rank == 0) {
        return nonzero_result(matches, shape.rank, input.device());
    }
    return Tensor(buffer.unsafeGetTensorImpl()->storage(), {matches, shape.rank},
                  {1, matches}, DType::Int64);
}

template <typename T>
Tensor nonzero_dynamic_typed(const Tensor& input, const NonzeroShape& shape) {
    const auto stream = getCurrentCUDAStream();
    const int64_t count = input.numel();
    if (count <= kNonzeroThreads && count * shape.rank <= kSmallNonzeroCapacity) {
        return nonzero_fused_typed<T>(input, shape, stream);
    }

    // Fitting the whole input in one tile leaves a single block on a single
    // multiprocessor, and a wide tile leaves most multiprocessors idle while
    // each thread serializes many coordinate decodes. One element per thread
    // keeps both the scan and the scatter spread over the widest grid; the tile
    // table stays proportional to the input only through its 1/256 ratio.
    return nonzero_tiled_typed<T, kNonzeroSmallItems>(input, shape, stream);
}

template <typename T>
void nonzero_static_typed(const Tensor& input, const NonzeroShape& shape,
                          int64_t limit, int64_t fill_value, Tensor& out) {
    const int64_t count = input.numel();
    const auto stream = getCurrentCUDAStream().stream();
    const T* values = input.const_data_ptr<T>();
    int64_t* indices = out.data_ptr<int64_t>();
    if (count <= kNonzeroTileSize && limit <= kNonzeroTileSize) {
        write_nonzero_tiles<T, kNonzeroItems, true><<<1, kNonzeroThreads, 0, stream>>>(
            values, count, shape, limit, 1, nullptr, nullptr, fill_value, indices);
        checkCuda(cudaGetLastError(), "nonzero_static tile scan");
        return;
    }

    const NonzeroTiles tiles = scan_nonzero_tiles<T, kNonzeroItems>(input, stream);
    const auto pad_grid = static_cast<int>(std::min<int64_t>((limit - 1) / kNonzeroTileSize + 1, 4096));
    const int grid = std::max(tiles.blocks, pad_grid);
    write_nonzero_tiles<T, kNonzeroItems, false><<<grid, kNonzeroThreads, 0, stream>>>(
        values, count, shape, limit, tiles.blocks, tiles.counts.const_data_ptr<int64_t>(),
        tiles.offset_data(), fill_value, indices);
    checkCuda(cudaGetLastError(), "nonzero_static tile scatter");
}

template <typename Fn>
decltype(auto) dispatch_nonzero(DType dtype, Fn fn) {
#define TP_NONZERO_CASE(ctype, name) \
    case DType::name: return fn.template operator()<ctype>();
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(TP_NONZERO_CASE)
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NONZERO_CASE
}

Tensor nonzero_cuda(const Tensor& self) {
    OptionalCUDAGuard guard(self.device());
    const NonzeroShape shape = nonzero_shape(self);
    if (self.numel() == 0) return nonzero_result(0, shape.rank, self.device());
    const Tensor input = self.contiguous();
    return dispatch_nonzero(input.dtype(), [&]<typename T>() {
        return nonzero_dynamic_typed<T>(input, shape);
    });
}

Tensor interop_nonzero_static_cuda(const Tensor& self, int64_t size, int64_t fill_value) {
    TP_CHECK(size >= 0, "nonzero_static: size must be non-negative");
    const NonzeroShape shape = nonzero_shape(self);
    OptionalCUDAGuard guard(self.device());
    Tensor result = nonzero_result(size, shape.rank, self.device());
    if (result.numel() == 0) return result;
    if (self.numel() == 0) return result.fill_(Scalar(fill_value));

    const Tensor input = self.contiguous();
    dispatch_nonzero(input.dtype(), [&]<typename T>() {
        nonzero_static_typed<T>(input, shape, size, fill_value, result);
    });
    return result;
}


void check_nonzero_destination(const Tensor& self, const Tensor& out) {
    if (out.defined()) {
        TP_CHECK_TYPE(out.dtype() == DType::Int64, "nonzero: out must be int64");
        TP_CHECK(out.device() == self.device(), "nonzero: out must be on the input device");
    }
}

Tensor& nonzero_out_cuda(const Tensor& self, Tensor& out) {
    check_nonzero_destination(self, out);
    return write_out(out, nonzero_cuda(self));
}

Tensor& interop_nonzero_static_out_cuda(const Tensor& self, int64_t size,
                                        int64_t fill_value, Tensor& out) {
    check_nonzero_destination(self, out);
    return write_out(out, interop_nonzero_static_cuda(self, size, fill_value));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NonzeroKernels) {
    m.impl("nonzero", nonzero_cuda);
    m.impl("nonzero.out", nonzero_out_cuda);
    m.impl("nonzero_static", interop_nonzero_static_cuda);
    m.impl("nonzero_static.out", interop_nonzero_static_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
