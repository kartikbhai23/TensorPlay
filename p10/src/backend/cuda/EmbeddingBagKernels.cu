#include "Tensor.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"

#include <cuda_runtime.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "Atomic.cuh"

namespace tensorplay {
namespace cuda {
namespace {

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    }                                                                          \
  } while (0)

enum BagMode : int64_t { kBagSum = 0, kBagMean = 1, kBagMax = 2 };

constexpr int kWarp = 32;
// Mask value for the warp-sync builtins: the lanes of the hardware wave the
// block covers.  A block of 256 threads fills whole waves on either width, so
// the mask names every lane of them; the 32-thread reduction rows inside the
// wave stay apart through the width argument the shuffles carry.
constexpr unsigned long long kWaveMask = 0xffffffffffffffffull;
constexpr int kBagsPerBlock = 8;
constexpr int kBagSplitRows = 8;
constexpr int kFlatThreads = 256;
constexpr unsigned kMaxGridY = 65535u;

// Half and BFloat16 bags reduce in fp32; fp32/fp64 reduce in their own type.
template <typename T> struct BagAcc { using type = T; };
template <> struct BagAcc<Half> { using type = float; };
template <> struct BagAcc<BFloat16> { using type = float; };

inline unsigned grid_dim(int64_t count, int64_t per_block) {
  const int64_t blocks = (count + per_block - 1) / per_block;
  return static_cast<unsigned>(std::max<int64_t>(blocks, 1));
}

// ---------------------------------------------------------------------------
// Bag layout: [start, end) per bag, plus the index-position -> bag map.
// Positions that no bag covers keep -1 and take no part in either direction.
// ---------------------------------------------------------------------------

// One warp owns one bag: lane 0 reads the bounds, the warp counts the
// non-padding positions of the same range and marks the position-to-bag map,
// and lane 0 publishes the bounds and the size.  The forward route needs all
// of them together, so the map is written here instead of in a launch of its
// own.  The same walk range-checks the indices, so the reduction kernels do
// not have to re-read them for validation.
template <typename IndexT>
__global__ void bag_layout_size_kernel(const IndexT* __restrict__ offsets,
                                       int64_t n_offsets, int64_t num_bags,
                                       int64_t numel, bool include_last_offset,
                                       int64_t padding_idx, int64_t num_rows,
                                       const IndexT* __restrict__ indices,
                                       int64_t* __restrict__ starts,
                                       int64_t* __restrict__ ends,
                                       int64_t* __restrict__ bag_size,
                                       int64_t* __restrict__ offset2bag,
                                       int64_t* __restrict__ cleared_max) {
  const int64_t bag = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (bag >= num_bags) return;

  int64_t start = 0;
  int64_t end = 0;
  if (threadIdx.x == 0) {
    start = static_cast<int64_t>(offsets[bag]);
    end = (bag + 1 < n_offsets) ? static_cast<int64_t>(offsets[bag + 1]) : numel;
    if (!include_last_offset && bag + 1 == num_bags) end = numel;

    if (start < 0) start = 0;
    if (start > numel) start = numel;
    if (end < start) end = start;
    if (end > numel) end = numel;
  }
  start = __shfl_sync(kWaveMask, start, 0, kWarp);
  end = __shfl_sync(kWaveMask, end, 0, kWarp);

  int64_t count = 0;
  bool invalid = false;
  for (int64_t i = start + threadIdx.x; i < end; i += kWarp) {
    const int64_t r = static_cast<int64_t>(indices[i]);
    invalid = invalid || r < 0 || r >= num_rows;
    if (r != padding_idx) ++count;
    offset2bag[i] = bag;
  }
  for (int step = kWarp / 2; step > 0; step >>= 1) {
    count += __shfl_down_sync(kWaveMask, count, step, kWarp);
  }
  assert(__ballot_sync(kWaveMask, invalid) == 0 &&
         "embedding_bag: index out of range in the embedding table");
  if (threadIdx.x == 0) {
    starts[bag] = start;
    ends[bag] = end;
    bag_size[bag] = count;
    if (cleared_max != nullptr) cleared_max[bag] = 0;
  }
}

// ---------------------------------------------------------------------------
// Forward, sum and mean: one block owns one bag and a tile of kWarp features.
// Within the block one warp row owns a strided share of the bag's positions,
// so a long bag is covered by several warps at once; with few bags that is
// what fills the machine.  Every row walks its positions in the same order,
// so the reduction is reproducible: the shares are added back in row order.
// ---------------------------------------------------------------------------

template <typename T, typename AccT, typename IndexT>
__global__ void bag_forward_split_kernel(const T* __restrict__ weight,
                                         int64_t D,
                                         const IndexT* __restrict__ indices,
                                         const T* __restrict__ per_sample_weights,
                                         const int64_t* __restrict__ starts,
                                         const int64_t* __restrict__ ends,
                                         const int64_t* __restrict__ bag_size,
                                         int64_t num_bags, int64_t mode,
                                         int64_t padding_idx,
                                         T* __restrict__ output,
                                         int64_t d_tiles) {
  __shared__ AccT partial[kBagSplitRows * kWarp];

  const int64_t line = static_cast<int64_t>(blockIdx.x);
  const int64_t bag = line / d_tiles;
  if (bag >= num_bags) return;
  const int64_t d = (line % d_tiles) * kWarp + threadIdx.x;
  const int64_t row = threadIdx.y;

  const int64_t s = starts[bag];
  const int64_t e = ends[bag];

  const int64_t rows = blockDim.y;
  AccT acc = 0;
  if (d < D) {
    for (int64_t i = s + row; i < e; i += rows) {
      const int64_t r = static_cast<int64_t>(indices[i]);
      AccT v = static_cast<AccT>(weight[r * D + d]);
      if (per_sample_weights != nullptr) {
        v *= static_cast<AccT>(per_sample_weights[i]);
      }
      acc += (r == padding_idx) ? static_cast<AccT>(0) : v;
    }
  }
  partial[row * kWarp + threadIdx.x] = acc;
  __syncthreads();

  if (row == 0 && d < D) {
    for (int64_t r = 1; r < rows; ++r) {
      acc += partial[r * kWarp + threadIdx.x];
    }
    if (mode == kBagMean) {
      const int64_t count = bag_size[bag];
      if (count > 0) acc /= static_cast<AccT>(count);
    }
    output[bag * D + d] = static_cast<T>(acc);
  }
}

// ---------------------------------------------------------------------------
// Forward, max: one thread owns one (bag, feature) pair.  Feature-major
// threading keeps the weight and output accesses coalesced, and the index
// read is a warp-wide broadcast out of cache.  The selection walks the bag in
// order, so the reported row is the first maximum.
// ---------------------------------------------------------------------------

template <typename T, typename AccT, typename IndexT>
__global__ void bag_max_forward_kernel(const T* __restrict__ weight,
                                       int64_t D,
                                       const IndexT* __restrict__ indices,
                                       const int64_t* __restrict__ starts,
                                       const int64_t* __restrict__ ends,
                                       int64_t num_bags, int64_t padding_idx,
                                       T* __restrict__ output,
                                       int64_t* __restrict__ max_indices) {
  const int64_t bag = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (bag >= num_bags) return;

  const int64_t s = starts[bag];
  const int64_t e = ends[bag];
  const int64_t stride = static_cast<int64_t>(gridDim.y) * blockDim.x;

  for (int64_t d = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
       d < D; d += stride) {
    // Select instead of branch throughout: both the padding test and the
    // running-maximum test vary per lane, so branching on either would
    // serialize the warp over random data.
    AccT best = 0;
    int64_t arg = 0;
    bool first = true;
    for (int64_t i = s; i < e; ++i) {
      const int64_t r = static_cast<int64_t>(indices[i]);
      const AccT v = static_cast<AccT>(weight[r * D + d]);
      const bool keep = r != padding_idx;
      const bool take = keep && (first || v > best);
      best = take ? v : best;
      arg = take ? r : arg;
      first = first && !keep;
    }
    output[bag * D + d] = first ? static_cast<T>(0) : static_cast<T>(best);
    max_indices[bag * D + d] = first ? static_cast<int64_t>(0) : arg;
  }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

template <typename IndexT>
__global__ void bag_counts_kernel(const IndexT* __restrict__ indices,
                                  int64_t numel, int64_t num_weights,
                                  int64_t padding_idx,
                                  int64_t* __restrict__ counts) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  const int64_t r = static_cast<int64_t>(indices[i]);
  assert(r >= 0 && r < num_weights);
  if (r == padding_idx) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(counts + r), 1ULL);
  (void)num_weights;
}

template <typename T, typename AccT, typename IndexT>
__global__ void bag_dense_backward_kernel(const T* __restrict__ grad,
                                          int64_t num_bags, int64_t D,
                                          const IndexT* __restrict__ indices,
                                          int64_t numel,
                                          const int64_t* __restrict__ offset2bag,
                                          const int64_t* __restrict__ bag_size,
                                          const T* __restrict__ per_sample_weights,
                                          const int64_t* __restrict__ counts,
                                          int64_t mode, int64_t padding_idx,
                                          AccT* __restrict__ grad_weight) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (i >= numel) return;

  const int64_t bag = offset2bag[i];
  if (bag < 0 || bag >= num_bags) return;
  const int64_t r = static_cast<int64_t>(indices[i]);
  if (r == padding_idx) return;

  AccT scale = 1;
  if (per_sample_weights != nullptr) {
    scale *= static_cast<AccT>(per_sample_weights[i]);
  }
  if (counts != nullptr) {
    const int64_t c = counts[r];
    if (c > 0) scale /= static_cast<AccT>(c);
  }
  if (mode == kBagMean) {
    const int64_t c = bag_size[bag];
    if (c > 0) scale /= static_cast<AccT>(c);
  }

  const int64_t stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
  for (int64_t d = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
       d < D; d += stride) {
    atomicAdd(grad_weight + r * D + d,
              static_cast<AccT>(grad[bag * D + d]) * scale);
  }
}

template <typename T, typename AccT>
__global__ void bag_dense_backward_max_kernel(const T* __restrict__ grad,
                                              int64_t num_bags, int64_t D,
                                              const int64_t* __restrict__ bag_size,
                                              const int64_t* __restrict__ max_indices,
                                              AccT* __restrict__ grad_weight) {
  const int64_t bag = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (bag >= num_bags) return;
  if (bag_size[bag] == 0) return;

  const int64_t stride = static_cast<int64_t>(gridDim.y) * blockDim.x;
  for (int64_t d = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
       d < D; d += stride) {
    const int64_t r = max_indices[bag * D + d];
    atomicAdd(grad_weight + r * D + d, static_cast<AccT>(grad[bag * D + d]));
  }
}

template <typename T, typename AccT, typename IndexT>
__global__ void bag_psw_backward_kernel(const T* __restrict__ grad,
                                        const T* __restrict__ weight,
                                        int64_t num_bags, int64_t D,
                                        const IndexT* __restrict__ indices,
                                        int64_t numel,
                                        const int64_t* __restrict__ offset2bag,
                                        int64_t padding_idx,
                                        T* __restrict__ output) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (i >= numel) return;

  const int64_t bag = offset2bag[i];
  const int64_t r = static_cast<int64_t>(indices[i]);
  if (bag < 0 || bag >= num_bags || r == padding_idx) {
    if (threadIdx.x == 0) output[i] = static_cast<T>(0);
    return;
  }

  AccT dot = 0;
  for (int64_t d = threadIdx.x; d < D; d += kWarp) {
    dot += static_cast<AccT>(grad[bag * D + d]) * static_cast<AccT>(weight[r * D + d]);
  }
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    dot += __shfl_down_sync(kWaveMask, dot, offset, kWarp);
  }
  if (threadIdx.x == 0) output[i] = static_cast<T>(dot);
}

template <typename T, typename AccT>
__global__ void bag_cast_kernel(int64_t n, const AccT* __restrict__ src,
                                T* __restrict__ dst) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<T>(src[i]);
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

void check_index_dtype(const Tensor& t, const char* what) {
  if (t.dtype() != DType::Int64 && t.dtype() != DType::Int32) {
    TP_THROW(TypeError,
             std::string("embedding_bag: ") + what + " must be Int32 or Int64");
  }
}

void check_float_dtype(DType dtype, const char* what) {
  if (dtype != DType::Float32 && dtype != DType::Float64 &&
      dtype != DType::Float16 && dtype != DType::BFloat16) {
    TP_THROW(TypeError,
             std::string("embedding_bag: ") + what + " must be a floating point tensor");
  }
}

int64_t bag_count(const Tensor& offsets, bool include_last_offset) {
  const int64_t n = offsets.numel();
  if (include_last_offset) {
    if (n < 1) {
      TP_THROW(RuntimeError,
               "embedding_bag: include_last_offset requires at least one offset");
    }
    return n - 1;
  }
  return n;
}

// Materializes starts/ends/offset2bag/bag_size on the device so the operator
// never has to read `offsets` back to the host.  All four live in one int64
// buffer, which the layout kernel writes outright, so it is allocated
// uninitialized.  The bounds stay pointers into it; only the three values the
// caller receives are wrapped as views.  The max-index slots travel in the
// same buffer -- the forward pass fills them in max mode, and the layout warp
// clears them otherwise so the sum and mean results stay zero as specified.
struct BagLayoutBuffers {
  Tensor workspace;
  Tensor offset2bag;
  Tensor bag_size;
  Tensor max_indices;
  // The bounds stay raw: nothing outside the forward launch reads them, and
  // wrapping them as tensors would only cost a view apiece.
  int64_t* starts = nullptr;
  int64_t* ends = nullptr;
};

BagLayoutBuffers build_bag_layout(const Tensor& indices, const Tensor& offsets,
                                  int64_t num_bags, int64_t padding_idx,
                                  bool include_last_offset, bool with_max_indices,
                                  int64_t D, int64_t num_rows, cudaStream_t stream) {
  const int64_t numel = indices.numel();
  const Device device = offsets.device();
  const int64_t max_slots = with_max_indices ? num_bags * D : num_bags;
  const int64_t total = 3 * num_bags + numel + max_slots;

  BagLayoutBuffers buf;
  buf.workspace = Tensor::empty({total}, DType::Int64, device);
  int64_t* base = buf.workspace.data_ptr<int64_t>();
  buf.starts = base;
  buf.ends = base + num_bags;
  int64_t* bag_size = base + 2 * num_bags;
  int64_t* offset2bag = base + 3 * num_bags;
  int64_t* max_slot_ptr = base + 3 * num_bags + numel;
  const auto slot = [&](int64_t offset, int64_t length) {
    return buf.workspace.narrow(0, offset, length);
  };
  buf.bag_size = slot(2 * num_bags, num_bags);
  buf.offset2bag = slot(3 * num_bags, numel);
  buf.max_indices = with_max_indices
      ? slot(3 * num_bags + numel, max_slots).view({num_bags, D})
      : slot(3 * num_bags + numel, max_slots);
  int64_t* cleared_max = with_max_indices ? nullptr : max_slot_ptr;

  // Positions outside every bag keep -1.  Only the head before the first
  // offset and the tail after the last one can stay uncovered on sane input,
  // and a byte fill covers them without a kernel.
  if (numel > 0) {
    CUDA_CHECK(cudaMemsetAsync(offset2bag, 0xFF,
                               static_cast<size_t>(numel) * sizeof(int64_t), stream));
  }
  if (num_bags == 0) return buf;

  const dim3 warp_block(kWarp, kBagsPerBlock);
  const dim3 warp_grid(grid_dim(num_bags, kBagsPerBlock));
  if (offsets.dtype() == DType::Int64) {
    bag_layout_size_kernel<int64_t><<<warp_grid, warp_block, 0, stream>>>(
        offsets.data_ptr<int64_t>(), offsets.numel(), num_bags, numel,
        include_last_offset, padding_idx, num_rows,
        indices.numel() > 0 ? indices.data_ptr<int64_t>() : nullptr,
        buf.starts, buf.ends, bag_size, offset2bag, cleared_max);
  } else {
    bag_layout_size_kernel<int32_t><<<warp_grid, warp_block, 0, stream>>>(
        offsets.data_ptr<int32_t>(), offsets.numel(), num_bags, numel,
        include_last_offset, padding_idx, num_rows,
        indices.numel() > 0 ? indices.data_ptr<int32_t>() : nullptr,
        buf.starts, buf.ends, bag_size, offset2bag, cleared_max);
  }

  return buf;
}

template <typename T, typename IndexT>
void launch_bag_forward(const Tensor& weight, const Tensor& indices,
                        const Tensor& per_sample_weights,
                        const BagLayoutBuffers& layout, int64_t num_bags,
                        int64_t mode, int64_t padding_idx, Tensor& output,
                        Tensor& max_indices, cudaStream_t stream) {
  using acc_t = typename BagAcc<T>::type;
  const int64_t D = weight.size(1);
  const T* weight_ptr = weight.data_ptr<T>();
  const IndexT* index_ptr =
      indices.numel() > 0 ? indices.data_ptr<IndexT>() : nullptr;
  const T* psw_ptr = per_sample_weights.defined() && per_sample_weights.numel() > 0
      ? per_sample_weights.data_ptr<T>() : nullptr;

  if (mode == kBagMax) {
    const dim3 block(kWarp, kBagsPerBlock);
    const dim3 grid(grid_dim(num_bags, kBagsPerBlock),
                    std::min<unsigned>(grid_dim(D, kWarp), kMaxGridY));
    bag_max_forward_kernel<T, acc_t, IndexT><<<grid, block, 0, stream>>>(
        weight_ptr, D, index_ptr,
        layout.starts, layout.ends,
        num_bags, padding_idx, output.data_ptr<T>(),
        max_indices.data_ptr<int64_t>());
    return;
  }

  const int64_t d_tiles = (D + kWarp - 1) / kWarp;
  const int64_t lines = num_bags * d_tiles;
  // Splitting a bag over several warps pays off only when a bag is long
  // enough to keep every warp busy; with many short bags the extra rows would
  // just multiply the thread count past the work.
  const int64_t per_bag = indices.numel() / std::max<int64_t>(num_bags, 1);
  const int rows = per_bag >= 128 ? 8 : per_bag >= 32 ? 4 : per_bag >= 8 ? 2 : 1;
  const dim3 block(kWarp, static_cast<unsigned>(rows));
  bag_forward_split_kernel<T, acc_t, IndexT>
      <<<static_cast<unsigned>(lines), block, 0, stream>>>(
          weight_ptr, D, index_ptr, psw_ptr,
          layout.starts, layout.ends,
          layout.bag_size.data_ptr<int64_t>(), num_bags, mode, padding_idx,
          output.data_ptr<T>(), d_tiles);
}

template <typename T, typename IndexT>
void launch_bag_dense_backward(const Tensor& grad, const Tensor& indices,
                               const Tensor& offset2bag, const Tensor& bag_size,
                               const Tensor& per_sample_weights,
                               const int64_t* counts, int64_t mode,
                               int64_t padding_idx, typename BagAcc<T>::type* accum,
                               cudaStream_t stream) {
  using acc_t = typename BagAcc<T>::type;
  const int64_t num_bags = grad.size(0);
  const int64_t D = grad.size(1);
  const int64_t numel = indices.numel();

  const dim3 block(kWarp, kBagsPerBlock);
  const dim3 grid(grid_dim(numel, kBagsPerBlock),
                  std::min<unsigned>(grid_dim(D, kWarp), kMaxGridY));
  bag_dense_backward_kernel<T, acc_t, IndexT><<<grid, block, 0, stream>>>(
      grad.data_ptr<T>(), num_bags, D, indices.data_ptr<IndexT>(), numel,
      offset2bag.data_ptr<int64_t>(), bag_size.data_ptr<int64_t>(),
      per_sample_weights.defined() && per_sample_weights.numel() > 0
          ? per_sample_weights.data_ptr<T>() : nullptr,
      counts, mode, padding_idx, accum);
}

template <typename T>
void launch_bag_dense_backward_max(const Tensor& grad, const Tensor& bag_size,
                                   const Tensor& max_indices,
                                   typename BagAcc<T>::type* accum,
                                   cudaStream_t stream) {
  using acc_t = typename BagAcc<T>::type;
  const int64_t num_bags = grad.size(0);
  const int64_t D = grad.size(1);

  const dim3 block(kWarp, kBagsPerBlock);
  const dim3 grid(grid_dim(num_bags, kBagsPerBlock),
                  std::min<unsigned>(grid_dim(D, kWarp), kMaxGridY));
  bag_dense_backward_max_kernel<T, acc_t><<<grid, block, 0, stream>>>(
      grad.data_ptr<T>(), num_bags, D, bag_size.data_ptr<int64_t>(),
      max_indices.data_ptr<int64_t>(), accum);
}

template <typename T, typename IndexT>
void launch_bag_psw_backward(const Tensor& grad, const Tensor& weight,
                             const Tensor& indices, const Tensor& offset2bag,
                             int64_t padding_idx, Tensor& output,
                             cudaStream_t stream) {
  using acc_t = typename BagAcc<T>::type;
  const int64_t num_bags = grad.size(0);
  const int64_t D = grad.size(1);
  const int64_t numel = indices.numel();

  const dim3 block(kWarp, kBagsPerBlock);
  const dim3 grid(grid_dim(numel, kBagsPerBlock));
  bag_psw_backward_kernel<T, acc_t, IndexT><<<grid, block, 0, stream>>>(
      grad.data_ptr<T>(), weight.data_ptr<T>(), num_bags, D,
      indices.data_ptr<IndexT>(), numel, offset2bag.data_ptr<int64_t>(),
      padding_idx, output.data_ptr<T>());
}

// Index-type leg of the dispatch; the storage-type leg is the switch in each
// entry point below.
template <typename T>
void launch_bag_forward_idx(DType index, const Tensor& weight, const Tensor& indices,
                            const Tensor& per_sample_weights,
                            const BagLayoutBuffers& layout, int64_t num_bags,
                            int64_t mode, int64_t padding_idx, Tensor& output,
                            Tensor& max_indices, cudaStream_t stream) {
  if (index == DType::Int64) {
    launch_bag_forward<T, int64_t>(weight, indices, per_sample_weights, layout,
                                   num_bags, mode, padding_idx, output,
                                   max_indices, stream);
  } else {
    launch_bag_forward<T, int32_t>(weight, indices, per_sample_weights, layout,
                                   num_bags, mode, padding_idx, output,
                                   max_indices, stream);
  }
}

template <typename T>
void launch_bag_dense_backward_idx(DType index, const Tensor& grad,
                                   const Tensor& indices, const Tensor& offset2bag,
                                   const Tensor& bag_size,
                                   const Tensor& per_sample_weights,
                                   const int64_t* counts, int64_t mode,
                                   int64_t padding_idx,
                                   typename BagAcc<T>::type* accum,
                                   cudaStream_t stream) {
  if (index == DType::Int64) {
    launch_bag_dense_backward<T, int64_t>(grad, indices, offset2bag, bag_size,
                                          per_sample_weights, counts, mode,
                                          padding_idx, accum, stream);
  } else {
    launch_bag_dense_backward<T, int32_t>(grad, indices, offset2bag, bag_size,
                                          per_sample_weights, counts, mode,
                                          padding_idx, accum, stream);
  }
}

template <typename T>
void launch_bag_psw_backward_idx(DType index, const Tensor& grad, const Tensor& weight,
                                 const Tensor& indices, const Tensor& offset2bag,
                                 int64_t padding_idx, Tensor& output,
                                 cudaStream_t stream) {
  if (index == DType::Int64) {
    launch_bag_psw_backward<T, int64_t>(grad, weight, indices, offset2bag,
                                        padding_idx, output, stream);
  } else {
    launch_bag_psw_backward<T, int32_t>(grad, weight, indices, offset2bag,
                                        padding_idx, output, stream);
  }
}

} // namespace

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_cuda(
    const Tensor& weight_arg, const Tensor& indices_arg, const Tensor& offsets_arg,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx) {
  (void)scale_grad_by_freq;
  (void)sparse;

  if (weight_arg.dim() != 2) {
    TP_THROW(RuntimeError, "embedding_bag: weight must be 2-D");
  }
  check_float_dtype(weight_arg.dtype(), "weight");
  check_index_dtype(indices_arg, "indices");
  check_index_dtype(offsets_arg, "offsets");
  if (indices_arg.dim() != 1 || offsets_arg.dim() != 1) {
    TP_THROW(RuntimeError, "embedding_bag: indices and offsets must be 1-D");
  }
  if (mode != kBagSum && mode != kBagMean && mode != kBagMax) {
    TP_THROW(ValueError, "embedding_bag: mode must be 0 (sum), 1 (mean) or 2 (max)");
  }

  Tensor per_sample_weights =
      per_sample_weights_opt.has_value() ? *per_sample_weights_opt : Tensor();
  if (per_sample_weights.defined() && per_sample_weights.numel() > 0) {
    if (mode != kBagSum) {
      TP_THROW(RuntimeError,
               "embedding_bag: per_sample_weights is only supported in sum mode");
    }
    if (per_sample_weights.dim() != 1 ||
        per_sample_weights.numel() != indices_arg.numel()) {
      TP_THROW(RuntimeError,
               "embedding_bag: per_sample_weights must be 1-D with one entry per index");
    }
    if (per_sample_weights.dtype() != weight_arg.dtype()) {
      per_sample_weights = per_sample_weights.to(weight_arg.dtype());
    }
    per_sample_weights = per_sample_weights.contiguous();
  }

  const Tensor weight = weight_arg.contiguous();
  const Tensor indices = indices_arg.contiguous();
  const Tensor offsets = offsets_arg.contiguous();

  const int64_t num_bags = bag_count(offsets, include_last_offset);
  const int64_t D = weight.size(1);
  const cudaStream_t stream = getCurrentCUDAStream().stream();

  BagLayoutBuffers layout = build_bag_layout(
      indices, offsets, num_bags, padding_idx, include_last_offset,
      mode == kBagMax, D, weight.size(0), stream);

  // Every (bag, feature) slot is written by the forward kernel: an empty bag
  // accumulates zero and a max bag without entries publishes zero as well.
  Tensor output = Tensor::empty({num_bags, D}, weight.dtype(), weight.device());
  Tensor max_indices = layout.max_indices;

  if (num_bags > 0 && D > 0) {
    const DType index_dtype = indices.dtype();
    switch (weight.dtype()) {
      case DType::Float32:
        launch_bag_forward_idx<float>(index_dtype, weight, indices, per_sample_weights,
                                      layout, num_bags, mode, padding_idx, output,
                                      max_indices, stream);
        break;
      case DType::Float64:
        launch_bag_forward_idx<double>(index_dtype, weight, indices, per_sample_weights,
                                       layout, num_bags, mode, padding_idx, output,
                                       max_indices, stream);
        break;
      case DType::Float16:
        launch_bag_forward_idx<Half>(index_dtype, weight, indices, per_sample_weights,
                                     layout, num_bags, mode, padding_idx, output,
                                     max_indices, stream);
        break;
      default:
        launch_bag_forward_idx<BFloat16>(index_dtype, weight, indices, per_sample_weights,
                                         layout, num_bags, mode, padding_idx, output,
                                         max_indices, stream);
        break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
  return {output, layout.offset2bag, layout.bag_size, max_indices};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_cuda(
    const Tensor& weight, const Tensor& indices, const Tensor& offsets,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset, int64_t padding_idx) {
  return _embedding_bag_cuda(weight, indices, offsets, scale_grad_by_freq, mode,
                             sparse, per_sample_weights, include_last_offset,
                             padding_idx);
}

Tensor _embedding_bag_dense_backward_cuda(
    const Tensor& grad_arg, const Tensor& indices_arg, const Tensor& offset2bag_arg,
    const Tensor& bag_size_arg, const Tensor& maximum_indices_arg,
    int64_t num_weights, bool scale_grad_by_freq, int64_t mode,
    const std::optional<Tensor>& per_sample_weights_opt, int64_t padding_idx) {
  // Rows are accumulated with atomicAdd, so the summation order varies.
  globalContext().alertNotDeterministic("_embedding_bag_dense_backward_cuda");

  check_float_dtype(grad_arg.dtype(), "grad");
  check_index_dtype(indices_arg, "indices");
  if (grad_arg.dim() != 2) {
    TP_THROW(RuntimeError, "embedding_bag_backward: grad must be 2-D");
  }
  if (num_weights < 0) {
    TP_THROW(ValueError, "embedding_bag_backward: num_weights must be non-negative");
  }

  const Tensor grad = grad_arg.contiguous();
  const Tensor indices = indices_arg.contiguous();
  const Tensor bag_size = bag_size_arg.contiguous();
  const int64_t D = grad.size(1);
  const int64_t numel = indices.numel();

  Tensor grad_weight = Tensor::zeros({num_weights, D}, grad.dtype(), grad.device());
  if (num_weights == 0 || D == 0) return grad_weight;

  const bool low_precision =
      grad.dtype() == DType::Float16 || grad.dtype() == DType::BFloat16;
  Tensor accum = low_precision
      ? Tensor::zeros({num_weights, D}, DType::Float32, grad.device())
      : grad_weight;
  const cudaStream_t stream = getCurrentCUDAStream().stream();

  if (mode == kBagMax) {
    const Tensor max_indices = maximum_indices_arg.contiguous();
    if (max_indices.numel() != grad.size(0) * D) {
      TP_THROW(RuntimeError,
               "embedding_bag_backward: max index buffer does not match the gradient shape");
    }
    switch (grad.dtype()) {
      case DType::Float32:
        launch_bag_dense_backward_max<float>(grad, bag_size, max_indices,
                                             accum.data_ptr<float>(), stream);
        break;
      case DType::Float64:
        launch_bag_dense_backward_max<double>(grad, bag_size, max_indices,
                                              accum.data_ptr<double>(), stream);
        break;
      case DType::Float16:
        launch_bag_dense_backward_max<Half>(grad, bag_size, max_indices,
                                            accum.data_ptr<float>(), stream);
        break;
      default:
        launch_bag_dense_backward_max<BFloat16>(grad, bag_size, max_indices,
                                                accum.data_ptr<float>(), stream);
        break;
    }
  } else {
    if (numel == 0) return grad_weight;
    const Tensor offset2bag = offset2bag_arg.dtype() == DType::Int64
        ? offset2bag_arg.contiguous()
        : offset2bag_arg.to(DType::Int64).contiguous();
    if (offset2bag.numel() != numel) {
      TP_THROW(RuntimeError,
               "embedding_bag_backward: offset2bag must have one entry per index");
    }

    Tensor per_sample_weights =
        per_sample_weights_opt.has_value() ? *per_sample_weights_opt : Tensor();
    if (per_sample_weights.defined() && per_sample_weights.numel() > 0 &&
        per_sample_weights.dtype() != grad.dtype()) {
      per_sample_weights = per_sample_weights.to(grad.dtype());
    }
    if (per_sample_weights.defined() && per_sample_weights.numel() > 0) {
      per_sample_weights = per_sample_weights.contiguous();
    }

    Tensor counts;
    const int64_t* counts_ptr = nullptr;
    if (scale_grad_by_freq) {
      counts = Tensor::zeros({num_weights}, DType::Int64, grad.device());
      const dim3 block(kFlatThreads);
      const dim3 grid(grid_dim(numel, kFlatThreads));
      if (indices.dtype() == DType::Int64) {
        bag_counts_kernel<int64_t><<<grid, block, 0, stream>>>(
            indices.data_ptr<int64_t>(), numel, num_weights, padding_idx,
            counts.data_ptr<int64_t>());
      } else {
        bag_counts_kernel<int32_t><<<grid, block, 0, stream>>>(
            indices.data_ptr<int32_t>(), numel, num_weights, padding_idx,
            counts.data_ptr<int64_t>());
      }
      counts_ptr = counts.data_ptr<int64_t>();
    }

    const DType index_dtype = indices.dtype();
    switch (grad.dtype()) {
      case DType::Float32:
        launch_bag_dense_backward_idx<float>(
            index_dtype, grad, indices, offset2bag, bag_size, per_sample_weights,
            counts_ptr, mode, padding_idx, accum.data_ptr<float>(), stream);
        break;
      case DType::Float64:
        launch_bag_dense_backward_idx<double>(
            index_dtype, grad, indices, offset2bag, bag_size, per_sample_weights,
            counts_ptr, mode, padding_idx, accum.data_ptr<double>(), stream);
        break;
      case DType::Float16:
        launch_bag_dense_backward_idx<Half>(
            index_dtype, grad, indices, offset2bag, bag_size, per_sample_weights,
            counts_ptr, mode, padding_idx, accum.data_ptr<float>(), stream);
        break;
      default:
        launch_bag_dense_backward_idx<BFloat16>(
            index_dtype, grad, indices, offset2bag, bag_size, per_sample_weights,
            counts_ptr, mode, padding_idx, accum.data_ptr<float>(), stream);
        break;
    }
  }

  if (low_precision) {
    const int64_t total = num_weights * D;
    const dim3 block(kFlatThreads);
    const dim3 grid(grid_dim(total, kFlatThreads));
    if (grad.dtype() == DType::Float16) {
      bag_cast_kernel<Half, float><<<grid, block, 0, stream>>>(
          total, accum.data_ptr<float>(), grad_weight.data_ptr<Half>());
    } else {
      bag_cast_kernel<BFloat16, float><<<grid, block, 0, stream>>>(
          total, accum.data_ptr<float>(), grad_weight.data_ptr<BFloat16>());
    }
  }
  CUDA_CHECK(cudaGetLastError());
  return grad_weight;
}

Tensor _embedding_bag_per_sample_weights_backward_cuda(
    const Tensor& grad_arg, const Tensor& weight_arg, const Tensor& indices_arg,
    const Tensor& offsets_arg, const Tensor& offset2bag_arg, int64_t mode,
    int64_t padding_idx) {
  if (mode != kBagSum) {
    TP_THROW(RuntimeError,
             "embedding_bag_backward: per_sample_weights is only supported in sum mode");
  }
  check_float_dtype(grad_arg.dtype(), "grad");
  check_index_dtype(indices_arg, "indices");
  if (grad_arg.dim() != 2 || weight_arg.dim() != 2) {
    TP_THROW(RuntimeError, "embedding_bag_backward: grad and weight must be 2-D");
  }
  if (grad_arg.size(1) != weight_arg.size(1)) {
    TP_THROW(RuntimeError,
             "embedding_bag_backward: grad and weight must agree on the embedding size");
  }

  const Tensor grad = grad_arg.contiguous();
  const Tensor weight = weight_arg.to(grad.dtype()).contiguous();
  const Tensor indices = indices_arg.contiguous();
  const int64_t numel = indices.numel();

  Tensor output = Tensor::zeros({numel}, grad.dtype(), grad.device());
  if (numel == 0) return output;

  const cudaStream_t stream = getCurrentCUDAStream().stream();
  Tensor offset2bag = offset2bag_arg;
  if (offset2bag.numel() == 0) {
    check_index_dtype(offsets_arg, "offsets");
    const Tensor offsets = offsets_arg.contiguous();
    const bool include_last_offset = offsets.numel() == grad.size(0) + 1;
    const int64_t num_bags = bag_count(offsets, include_last_offset);
    offset2bag = build_bag_layout(indices, offsets, num_bags, padding_idx,
                                  include_last_offset, false, 0,
                                  weight.size(0), stream)
                     .offset2bag;
  } else {
    if (offset2bag.numel() != numel) {
      TP_THROW(RuntimeError,
               "embedding_bag_backward: offset2bag must have one entry per index");
    }
    offset2bag = offset2bag.dtype() == DType::Int64
        ? offset2bag.contiguous()
        : offset2bag.to(DType::Int64).contiguous();
  }

  const DType index_dtype = indices.dtype();
  switch (grad.dtype()) {
    case DType::Float32:
      launch_bag_psw_backward_idx<float>(index_dtype, grad, weight, indices,
                                         offset2bag, padding_idx, output, stream);
      break;
    case DType::Float64:
      launch_bag_psw_backward_idx<double>(index_dtype, grad, weight, indices,
                                          offset2bag, padding_idx, output, stream);
      break;
    case DType::Float16:
      launch_bag_psw_backward_idx<Half>(index_dtype, grad, weight, indices,
                                        offset2bag, padding_idx, output, stream);
      break;
    default:
      launch_bag_psw_backward_idx<BFloat16>(index_dtype, grad, weight, indices,
                                            offset2bag, padding_idx, output, stream);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
  return output;
}

// GPU contraction for ragged segment reduction is not wired up yet; the
// registration exists so the schema dispatch table stays complete and the
// failure is an explicit rejection instead of a silent kernel miss.
Tensor _segment_reduce_cuda(const Tensor& /*data*/, const std::string& /*reduce*/,
                            const std::optional<Tensor>& /*lengths*/,
                            const std::optional<Tensor>& /*offsets*/,
                            int64_t /*axis*/,
                            const std::optional<Scalar>& /*initial*/) {
    TP_THROW(NotImplementedError,
             "_segment_reduce: no CUDA kernel yet; run the op on CPU");
}

TENSORPLAY_LIBRARY_IMPL(CUDA, EmbeddingBagKernels) {
  m.impl("_embedding_bag", _embedding_bag_cuda);
  m.impl("_embedding_bag_forward_only", _embedding_bag_forward_only_cuda);
  m.impl("_embedding_bag_dense_backward", _embedding_bag_dense_backward_cuda);
  m.impl("_embedding_bag_per_sample_weights_backward",
         _embedding_bag_per_sample_weights_backward_cuda);
  m.impl("_segment_reduce", _segment_reduce_cuda);
}

} // namespace cuda
} // namespace tensorplay
