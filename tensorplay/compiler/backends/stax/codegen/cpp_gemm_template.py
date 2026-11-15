"""A matrix product written as one prepared kernel.

A product of two tiles is often the largest operation in a region, and what
makes it fast is not the arithmetic but the arrangement: which part of each
operand stays in registers, which part of each is reused, and how the work is
split between threads and between levels of cache.  This template writes that
arrangement out for one shape and one set of types, and folds the work that
reads the result into the same loops.

The kernel is chosen by :func:`create_micro_gemm`, which decides which block
shape fits the host's instruction set, and everything else -- the thread
blocking and the cache blocking -- is decided from the shape and the block.

Weight prepacking and the whole-number paths are not written here: a product of
floating-point tiles with its operands as they lie is the case this covers, and
a product outside it is offered no template rather than a worse one.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from unittest.mock import patch

import sympy

import tensorplay as tp

from .. import config, ir
from ..loops import V
from ..templates.mm_common import mm_args
from ..templates.select_algorithm import DataProcessorTemplateWrapper
from ..utils import has_free_symbols, parallel_num_threads
from ..virtualized import ops
from .cpp import get_export_declaration
from .cpp_micro_gemm import CppMicroGemm, CppMicroGemmFP32Vec, create_micro_gemm
from .cpp_template import CppTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    create_epilogue_with_attr,
    DTYPE_TO_CPP,
    element_size,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)

log = logging.getLogger(__name__)


GEMM_TEMPLATE_INIT_BLOCKING_BASIC_BLOCK = r"""
    constexpr int64_t num_threads = {{num_threads}};
    constexpr int64_t N = {{N}};
    constexpr int64_t K = {{K}};
    constexpr int64_t Mr = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t Nr = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t Kr = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
{%- if is_dynamic_M %}
    const int64_t M = {{kernel.size(GemmOut, 0)}};
    const int64_t Mr_blocks = (M + Mr - 1) / Mr;
{%- else %}
    constexpr int64_t M = {{kernel.size(GemmOut, 0)}};
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
{%- endif %}
"""

GEMM_TEMPLATE_INIT_BLOCKING_EXTENDED = r"""
{%- if is_dynamic_M %}
    {%- if num_threads > 1 %}
    int64_t Mt_blocks, Nt_blocks, Kt_blocks;
    mm_get_thread_blocking(num_threads, {{config.cpp.gemm_max_k_slices}}, M, N, K, Mr, Nr, Kr, Mt_blocks, Nt_blocks, Kt_blocks);
    {%- else %}
    const auto Mt_blocks = Mr_blocks;
    const auto Nt_blocks = Nr_blocks;
    const auto Kt_blocks = Kr_blocks;
    {%- endif %}
    int64_t Mc_blocks, Nc_blocks, Kc_blocks;
    uint32_t L1_cache_size = {{L1_cache_size}};
    uint32_t L2_cache_size = {{L2_cache_size}};
    mm_get_cache_blocking<{{kernel.dtype(X)}}, {{kernel.dtype(W)}}>(
        num_threads,
        M,
        N,
        K,
        Mr,
        Nr,
        Kr,
        Mt_blocks,
        Nt_blocks,
        Kt_blocks,
        Mc_blocks,
        Nc_blocks,
        Kc_blocks,
        L1_cache_size,
        L2_cache_size
    );
    const int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    const int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    const int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    const int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    const int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;
{%- else %}
    constexpr int64_t Mt_blocks = {{template.thread_blocking(num_threads).block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking(num_threads).block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking(num_threads).block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking(num_threads).block_m}};
    constexpr int64_t Nc_blocks = {{template.cache_blocking(num_threads).block_n}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking(num_threads).block_k}};
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    constexpr int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    constexpr int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;
{%- endif %}

    // make sure all partitions are assigned
    {{kernel.assert_function}}(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
"""

GEMM_TEMPLATE_MULTI_THREADS_PARAMS = r"""
const int64_t k_group_id = tid / num_Kt_blocks;
const int64_t k_slice_id = tid % num_Kt_blocks;
const int64_t n_group_id = k_group_id / num_Nt_blocks;
const int64_t n_slice_id = k_group_id % num_Nt_blocks;
const int64_t k_block_start = k_slice_id * Kt_blocks;
const int64_t k_block_end = std::min(k_block_start + Kt_blocks, Kr_blocks);
const int64_t n_block_start = n_slice_id * Nt_blocks;
const int64_t n_block_end = std::min(n_block_start + Nt_blocks, Nr_blocks);
const int64_t m_block_start = std::min(n_group_id * Mt_blocks, Mr_blocks);
const int64_t m_block_end = std::min(m_block_start + Mt_blocks, Mr_blocks);
const int64_t num_Mc_blocks_per_thread = (m_block_end - m_block_start + Mc_blocks - 1) / Mc_blocks;
"""

GEMM_TEMPLATE_SINGLE_THREAD_PARAMS = r"""
constexpr int tid = 0;
constexpr int64_t k_group_id = 0;
constexpr int64_t k_slice_id = 0;
constexpr int64_t n_group_id = 0;
constexpr int64_t n_slice_id = 0;
constexpr int64_t m_block_start = 0;
constexpr int64_t n_block_start = 0;
constexpr int64_t n_block_end = Nr_blocks;
constexpr int64_t k_block_start = 0;
constexpr int64_t k_block_end = Kr_blocks;
{%- if is_dynamic_M %}
const int64_t num_Mc_blocks_per_thread = num_Mc_blocks;
const int64_t m_block_end = Mr_blocks;
{%- else %}
constexpr int64_t num_Mc_blocks_per_thread = num_Mc_blocks;
constexpr int64_t m_block_end = Mr_blocks;
{%- endif %}
"""

GEMM_TEMPLATE_M_LOOP_PARAMS = r"""
const int64_t my_mc_block_id = (mc_block_id + n_slice_id) % num_Mc_blocks_per_thread;
const int64_t mc = m_block_start + my_mc_block_id * Mc_blocks;
const int64_t m_start = mc * Mr;
const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
const int64_t m_size = m_end - m_start;
"""

GEMM_TEMPLATE_N_LOOP_PARAMS = r"""
const int64_t n_start = nc * Nr;
const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
const int64_t n_size = n_end - n_start;
// NB: assume we pad N, nc_block_end won't exceed padded N here.
const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
"""

GEMM_TEMPLATE_MICROKERNEL_DEF = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define(kernel)}}
"""

GEMM_TEMPLATE_STUB_DEF = r"""
    {%- set kernel_args = {"X": X, "W": W, "inp": inp} %}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs={"Y": Y}, aliases=aliases)}}
"""

GEMM_TEMPLATE = r"""
{{ template.codegen_gemm_stub_def() }}
{
    {{ kernel.maybe_codegen_profile(template.get_kernel_prefix_name()) }}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOut, config, L1_cache_size, L2_cache_size, X, W
    ) }}

{%- if maybe_k_slicing %}
    std::unique_ptr<std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[]> local_buf_ptrs;
    const int64_t num_Mc_blocks_per_thread_block = (Mt_blocks + Mc_blocks - 1) / Mc_blocks;
    const int64_t num_Nc_blocks_per_thread_block = (Nt_blocks + Nc_blocks - 1) / Nc_blocks;
    if (num_Kt_blocks > 1) {
        local_buf_ptrs.reset(new std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[
            num_Mt_blocks * num_Nt_blocks *
            num_Mc_blocks_per_thread_block * num_Nc_blocks_per_thread_block *
            num_Kt_blocks
        ]);
    }
{%- endif %}

{%- if num_threads > 1 %}
    {%- set use_dynamic_threads = ((config.cpp.threads < 1) and (num_threads == cpu_count)) or config.cpp.dynamic_threads %}
    {%- if use_dynamic_threads %}
    #pragma omp parallel
    {%- else %}
    #pragma omp parallel num_threads({{num_threads}})
    {%- endif %}
    {
        {{ micro_gemm.codegen_init(kernel) }}
        #pragma omp for schedule(static, 1)
        for (int64_t tid = 0; tid < {{num_threads}}; tid++) {
            {{ template.codegen_multi_threads_params()|indent(12, false) }}
{%- else %}
    {
        {{ template.codegen_single_thread_params(is_dynamic_M)|indent(8, false) }}
        {{ micro_gemm.codegen_init(kernel) }}
{%- endif %}
{%- if use_local_acc %}
    {%- set acc_buf_name = "local_acc_buf" %}
        {{ kernel.define_buffer(acc_buf_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
{%- endif %}
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            {{ template.codegen_m_loop_params()|indent(12, false) }}
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                {{ template.codegen_n_loop_params()|indent(16, false) }}
{%- if use_local_acc %}
    {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf_name) }}
{%- else %}
    {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
{%- set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set acc_slice = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]) %}
{%- set tile_W = kernel.slice_nd(W, [("k_start", "k_end"), ("n_start", "n_start + n_size")]) %}
                        if (kc == k_block_start) {
                            {{ micro_gemm.codegen_call(kernel,
                                                       tile_X,
                                                       tile_W,
                                                       acc_slice,
                                                       accum=False)|indent(28, false)
                            }}
                        } else {
                            {{ micro_gemm.codegen_call(kernel,
                                                       tile_X,
                                                       tile_W,
                                                       acc_slice,
                                                       accum=True)|indent(28, false)
                            }}
                        }
                    }
                }
{%- if maybe_k_slicing %}
                if (num_Kt_blocks > 1) {
                    const int64_t n_cache_block_id = (nc - n_block_start) / Nc_blocks;
                    const int64_t mxn_cache_block_id =
                        ((n_group_id * num_Nt_blocks + n_slice_id) *
                         num_Mc_blocks_per_thread_block + my_mc_block_id) *
                            num_Nc_blocks_per_thread_block +
                        n_cache_block_id;
                    local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks + k_slice_id].reset(
                        {{ kernel.release_buffer(acc_buf_name) }});
                } else
{%- endif %}
                {
{%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- set tile_acc = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
                    {{ kernel.store_output(
                        tile_Y, tile_acc, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
{%- if num_threads > 1 %}
            }
{%- if maybe_k_slicing %}
        if (num_Kt_blocks > 1) {
            #pragma omp for schedule(static, 1)
            for (int64_t tid = 0; tid < {{num_threads}}; tid++) {
                {{ template.codegen_multi_threads_params()|indent(16, false) }}
                for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
                    // We slice M-dim and each thread in the k-slicing group works on a slice
                    const int64_t m_start_unsliced = mc * Mr;
                    const int64_t m_end_unsliced = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
                    const int64_t m_size_unsliced = m_end_unsliced - m_start_unsliced;
                    const int64_t m_slice_size = (m_size_unsliced + num_Kt_blocks - 1) / num_Kt_blocks;
                    const int64_t m_start = std::min(m_start_unsliced + m_slice_size * k_slice_id, m_end_unsliced);
                    const int64_t m_end = std::min(m_start_unsliced + m_slice_size * (k_slice_id + 1), m_end_unsliced);
                    const int64_t m_size = m_end - m_start;
                    const int64_t m_offset = m_start - m_start_unsliced;
                    const int64_t m_cache_block_id = (mc - m_block_start) / Mc_blocks;
                    for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                        const int64_t n_start = nc * Nr;
                        const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
                        const int64_t n_size = n_end - n_start;
                        const int64_t n_cache_block_id = (nc - n_block_start) / Nc_blocks;
                        const int64_t mxn_cache_block_id =
                            ((n_group_id * num_Nt_blocks + n_slice_id) *
                             num_Mc_blocks_per_thread_block + m_cache_block_id) *
                                num_Nc_blocks_per_thread_block +
                            n_cache_block_id;
                        auto {{acc_buf_name}} = local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks].get();
                        for (int64_t other_slice = 1; other_slice < num_Kt_blocks; other_slice++) {
                            auto other_acc = local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks + other_slice].get();
                            for (int64_t m = m_offset; m < m_offset + m_size; m++) {
                                #pragma omp simd
                                for (int64_t n = 0; n < n_size; n++) {
                                    {{acc_buf_name}}[m*Nc_blocks*Nr + n] += other_acc[m*Nc_blocks*Nr + n];
                                }
                            }
                        }
        {%- set tile_Y_m_slice = kernel.slice_nd(Y_2d, [("m_start", "m_start + m_size"), ("n_start", "n_start + n_size")]) %}
        {%- set tile_acc_full = kernel.slice_nd(acc, [("0", "m_size_unsliced"), ("0", "n_size")]) %}
        {%- set tile_acc_m_slice = kernel.slice_nd(tile_acc_full, [("m_offset", "m_offset + m_size"), ()]) %}
                        {{ kernel.store_output(
                            tile_Y_m_slice, tile_acc_m_slice, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                        )|indent(20, false)
                        }}
                    }
                }
            }
        }
{%- endif %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
{%- else %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
{%- endif %}
}
"""



def get_padded_n(n: Any, block_n: int) -> int:
    """The column count a product works in, rounded up to a whole block.

    Stores beyond the real columns are harmless because they are written by
    blocks that also write real columns, and reading a block whose columns are
    all padding would read the buffer past its end -- so a block that would be
    all padding is simply not visited.
    """

    return (int(n) + block_n - 1) // block_n * block_n


def gen_2d_view_of_epilogue_buf(
    Y: ir.Buffer,
    template_buffer: ir.Buffer,
    epilogue_nodes: list[Any],
    reindexers: list[Any],
    default_reindexers: list[Any],
) -> tuple[Any, list[Any]]:
    """The result seen as the two dimensions the product writes.

    The product writes two-dimensional tiles, while the value the folded work
    produces may have more dimensions, or dimensions of the same total size
    arranged otherwise.  What is returned is a two-dimensional view of that
    value together with the index mapping that turns a tile position into the
    position the folded work reads and writes.
    """

    Y_2d: Any = Y
    if (
        Y.get_size() == template_buffer.get_size()
        and Y.get_stride() == template_buffer.get_stride()
    ):
        reindexers.extend(default_reindexers)
        Y_2d = Y
    else:

        def get_reindexer(epilogue_node, default_reindexer=None):
            stride_order = list(
                ir.get_stride_order(
                    V.graph.sizevars.guarding_hints_or_throw(epilogue_node.get_stride())
                )
            )
            fill_order = ir.stride_order2fill_order(stride_order)
            reversed_fill_order = list(reversed(fill_order))
            size_with_stride_ordered_decreasingly = [
                epilogue_node.get_size()[i] for i in reversed_fill_order
            ]
            reshape_reindex = ir.View.dynamic_reshape_indexer(
                size_with_stride_ordered_decreasingly,
                template_buffer.get_size(),
            )
            if default_reindexer:
                reshape_reindex = ir.fuse_reindexing(reshape_reindex, default_reindexer)

            from_stride_ordered_decreasingly_to_epilogue_node_order = [
                (len(stride_order) - 1) - stride_order[i]
                for i in range(len(stride_order))
            ]
            stride_reindex = ir.same_reorder(
                from_stride_ordered_decreasingly_to_epilogue_node_order
            )
            return ir.fuse_reindexing(stride_reindex, reshape_reindex)

        if default_reindexers is None:
            default_reindexers = [None] * len(epilogue_nodes)
        reindexers.extend(
            get_reindexer(epilogue_node, default_reindexer)
            for epilogue_node, default_reindexer in zip(
                epilogue_nodes, default_reindexers
            )
        )
        if isinstance(Y, ir.BaseView):
            storage = ir.StorageBox(Y.unwrap_view())
        else:
            if not isinstance(Y, ir.Buffer):
                raise AssertionError(f"expected ir.Buffer, got {type(Y)}")
            storage = ir.StorageBox(Y)
        Y_2d = ir.ReinterpretView(data=storage, layout=template_buffer.get_layout())
    return Y_2d, reindexers


def _sysfs_cache_size(level: int, kind: str) -> int | None:
    """One cache's size in bytes, as the host describes it."""

    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        entries = os.listdir(base)
    except OSError:
        return None
    for entry in entries:
        if not entry.startswith("index"):
            continue
        directory = os.path.join(base, entry)
        try:
            with open(os.path.join(directory, "level")) as fp:
                if int(fp.read().strip()) != level:
                    continue
            with open(os.path.join(directory, "type")) as fp:
                if fp.read().strip() != kind:
                    continue
            with open(os.path.join(directory, "size")) as fp:
                text = fp.read().strip()
        except (OSError, ValueError):
            continue
        multiplier = 1
        if text.endswith("K"):
            multiplier, text = 1024, text[:-1]
        elif text.endswith("M"):
            multiplier, text = 1024 * 1024, text[:-1]
        try:
            return int(text) * multiplier
        except ValueError:
            continue
    return None


@lru_cache(maxsize=1)
def cpu_cache_sizes() -> tuple[int, int]:
    """The first and second level cache sizes a block is sized against.

    Read from the host where it says so.  Where it does not, a size small
    enough to be safe everywhere is used: these decide how work is split, and a
    split that fits a smaller cache than the host has still computes the same
    result.
    """

    l1 = _sysfs_cache_size(1, "Data") or 32 * 1024
    l2 = _sysfs_cache_size(2, "Unified") or _sysfs_cache_size(2, "Data") or 1024 * 1024
    return l1, l2


class CppGemmTemplate(CppTemplate):
    """A matrix product written as one prepared kernel.

    The operands are read as they lie: no weight is rearranged ahead of time, so
    what is prepared here is the arrangement of the loops rather than of the
    data.  A bias and the work that reads the result are folded into the same
    loops, and the result is written once.
    """

    def __init__(
        self,
        input_nodes: list,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta: Any = 1,
        alpha: Any = 1,
        has_bias: bool = False,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        name: str = "prepared_gemm",
    ) -> None:
        if layout.dtype not in [tp.float32, tp.bfloat16, tp.float16]:
            raise AssertionError(f"Unsupported layout dtype: {layout.dtype}")
        super().__init__(
            name,
            input_nodes,
            layout,
            num_threads,
            epilogue_creator=epilogue_creator,
        )
        self.beta = beta
        self.alpha = alpha
        self.has_bias = has_bias
        self.register_blocking = register_blocking
        self.m, self.n = layout.size[-2:]
        self.k = input_nodes[0].get_size()[-1]
        self.padded_n = get_padded_n(self.n, self.register_blocking.block_n)
        # A shape whose row count is not known while the kernel is written
        # would need the split decided while it runs.  This template decides
        # the split when it is written, so such a shape is not offered it.
        self.is_dynamic_M = has_free_symbols((self.m,))
        if self.is_dynamic_M:
            raise NotImplementedError(
                "the prepared product needs a row count known while it is written"
            )
        self.thread_blocking = self.make_thread_blocking_cache()
        self.cache_blocking = self.make_cache_blocking_cache()

    def make_thread_blocking_cache(self):
        cache = lru_cache()(self._thread_blocking)

        def thread_blocking(num_threads: int) -> GemmBlocking:
            return cache(num_threads)

        return thread_blocking

    def _thread_blocking(self, num_threads: int) -> GemmBlocking:
        """How the blocks of the product are divided among the threads.

        Every thread gets work, the division is as square as it can be so that
        both operands are reused, and only when the rows and columns do not
        fill the threads is the contracted axis also split -- which costs the
        partial results having to be added together afterwards.
        """

        def get_factors(number: int) -> list[int]:
            factors = []
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        def get_blocking(m_factor, n_factor, k_factor, m_blocks, n_blocks, k_blocks):
            thread_block_k = math.ceil(k_blocks / k_factor)
            thread_block_n = math.ceil(n_blocks / n_factor)
            thread_block_m = math.ceil(m_blocks / m_factor)
            return GemmBlocking(thread_block_m, thread_block_n, thread_block_k)

        register_blocking = self.register_blocking
        m_blocks = math.ceil(self.m / register_blocking.block_m)
        n_blocks = math.ceil(self.n / register_blocking.block_n)
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        factors = get_factors(num_threads)
        if not factors:
            raise AssertionError(f"expected at least one factor, got {factors}")

        if config.cpp.gemm_thread_factors is not None:
            factors = [int(i) for i in config.cpp.gemm_thread_factors.split(",")]
            if len(factors) != 3:
                raise AssertionError(
                    f"expected 3 thread factors, got {len(factors)}"
                )
            if math.prod(factors) != self.num_threads:
                raise AssertionError(
                    f"expected factors product == {self.num_threads}, "
                    f"got {math.prod(factors)}"
                )
            return get_blocking(
                factors[0], factors[1], factors[2], m_blocks, n_blocks, k_blocks
            )

        def get_better_blocking(blocking, best_blocking):
            if best_blocking is None:
                return blocking
            block_m_size = blocking.block_m * register_blocking.block_m
            block_n_size = blocking.block_n * register_blocking.block_n
            best_block_m_size = best_blocking.block_m * register_blocking.block_m
            best_block_n_size = best_blocking.block_n * register_blocking.block_n
            if blocking.block_k > best_blocking.block_k:
                return blocking
            if (
                blocking.block_k == best_blocking.block_k
                and block_m_size + block_n_size < best_block_m_size + best_block_n_size
            ):
                return blocking
            return best_blocking

        best_blocking = None
        for n_factor in factors:
            m_factor = num_threads // n_factor
            if n_blocks >= n_factor and m_blocks >= m_factor:
                best_blocking = get_better_blocking(
                    get_blocking(m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks),
                    best_blocking,
                )

        if best_blocking is None:
            for k_factor in factors:
                if k_blocks >= k_factor and (
                    config.cpp.gemm_max_k_slices == 0
                    or k_factor <= config.cpp.gemm_max_k_slices
                ):
                    for n_factor in get_factors(num_threads // k_factor):
                        m_factor = (num_threads // k_factor) // n_factor
                        if n_blocks >= n_factor and m_blocks >= m_factor:
                            best_blocking = get_better_blocking(
                                get_blocking(
                                    m_factor,
                                    n_factor,
                                    k_factor,
                                    m_blocks,
                                    n_blocks,
                                    k_blocks,
                                ),
                                best_blocking,
                            )

        if best_blocking is None:
            for n_factor in factors:
                m_factor = num_threads // n_factor
                if n_blocks >= n_factor or m_blocks >= m_factor:
                    best_blocking = get_better_blocking(
                        get_blocking(
                            m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                        ),
                        best_blocking,
                    )

        if best_blocking is None:
            raise AssertionError("expected a valid thread blocking, got None")
        return best_blocking

    def make_cache_blocking_cache(self):
        cache = lru_cache()(self._cache_blocking)

        def cache_blocking(num_threads: int) -> GemmBlocking:
            return cache(num_threads)

        return cache_blocking

    def _cache_blocking(self, num_threads: int) -> GemmBlocking:
        """How much of each operand one pass over memory takes.

        A block of the second operand is kept in the first level cache and
        reused by several rows of the first, and a block of the first operand is
        kept in the second level cache and reused across columns.  The sizes
        decide how large those blocks are; the ratios below keep half of each
        cache free for the result and for the rest of the program.
        """

        def get_cache_blocking(register_blocking, thread_blocking):
            Mr, Nr, Kr = (
                register_blocking.block_m,
                register_blocking.block_n,
                register_blocking.block_k,
            )
            Mt_blocks = thread_blocking.block_m
            Nt_blocks = thread_blocking.block_n
            Kt_blocks = thread_blocking.block_k

            if config.cpp.gemm_cache_blocking is not None:
                blockings = [int(i) for i in config.cpp.gemm_cache_blocking.split(",")]
                if len(blockings) != 3:
                    raise AssertionError(
                        f"expected 3 cache blockings, got {len(blockings)}"
                    )
                Mc_blocks, Nc_blocks, Kc_blocks = blockings
                return (
                    min(Mc_blocks, Mt_blocks),
                    min(Nc_blocks, Nt_blocks),
                    min(Kc_blocks, Kt_blocks),
                )

            L1_cache_size, L2_cache_size = cpu_cache_sizes()
            L1 = L1_cache_size * 0.8
            L2 = L2_cache_size * 0.5

            num_byte_A = element_size(self.input_nodes[0].get_dtype())
            num_byte_B = element_size(self.input_nodes[1].get_dtype())

            size_cache_B = Kr * Kt_blocks * Nr * num_byte_B
            Kc_blocks = Kt_blocks
            if size_cache_B > L1:
                Kc_blocks = math.floor(L1 / (Kr * Nr * num_byte_B))

            min_Mc_ratio = 2
            min_Mc_blocks = math.ceil(min_Mc_ratio * Mr / Nr)
            if min_Mc_blocks < 1:
                raise AssertionError(
                    f"expected min_Mc_blocks >= 1, got {min_Mc_blocks}"
                )
            Kt_bytes = Kt_blocks * Kr * num_byte_A
            if min_Mc_blocks * Mr * Kt_bytes < L2:
                Mc_blocks = min(Mt_blocks, math.floor(L2 / (Mr * Kt_bytes)))
                Nc_blocks = 1
            else:
                Mc_blocks = Mt_blocks
                Nc_blocks = min(math.ceil(Mc_blocks * Mr / Nr), Nt_blocks)
                Nc_bytes = Nc_blocks * Nr * 4
                Kc_bytes = Kc_blocks * Kr * num_byte_A
                if Mc_blocks * Mr * (Kc_bytes + Nc_bytes) > L2:
                    M_max = (math.sqrt(Kc_bytes * Kc_bytes + 16 * L2) - Kc_bytes) / 8
                    if M_max < Mc_blocks * Mr:
                        Mc_blocks = math.floor(M_max / Mr)
                        Nc_blocks = min(math.ceil(Mc_blocks * Mr / Nr), Nt_blocks)

            return Mc_blocks, Nc_blocks, Kc_blocks

        register_blocking = self.register_blocking
        thread_blocking = self.thread_blocking(num_threads)
        return GemmBlocking(*get_cache_blocking(register_blocking, thread_blocking))

    def log_blockings(self):
        log.debug("Register blocking: %s", self.register_blocking)
        log.debug("Cache blocking: %s", self.cache_blocking(self.num_threads))
        log.debug("Thread blocking: %s", self.thread_blocking(self.num_threads))

    def maybe_k_slicing(self) -> bool:
        """Whether one result is assembled from partial products."""

        if self.num_threads == 1:
            return False
        k_blocks = math.ceil(self.k / self.register_blocking.block_k)
        return k_blocks > self.thread_blocking(self.num_threads).block_k

    @classmethod
    def add_choices(
        cls,
        choices: list,
        layout: ir.Layout,
        input_nodes: list,
        beta: Any = 1,
        alpha: Any = 1,
        has_bias: bool = False,
        input_indices: list[int] | None = None,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
    ):
        """Offer the prepared product as one candidate for this call.

        The caller hands its inputs in the framework's order -- the addend
        first, then the two operands -- while the kernel reads the operands and
        the addend last.  The reordering is applied to the call rather than to
        the kernel, so the kernel is written for the order it asks for.
        """

        if input_indices is None:
            input_indices = list(range(len(input_nodes)))

        deduped_name_to_idx: dict[str, int] = {}
        deduped_input_indices: list[int] = []
        for input_node in input_nodes:
            name = input_node.get_name()
            if name not in deduped_name_to_idx:
                deduped_name_to_idx[name] = len(deduped_name_to_idx)
            deduped_input_indices.append(deduped_name_to_idx[name])

        def indices_for_inputs(inputs):
            if len(inputs) == len(input_nodes):
                return input_indices
            if len(inputs) != len(deduped_name_to_idx):
                raise AssertionError(
                    f"expected {len(deduped_name_to_idx)} inputs, got {len(inputs)}"
                )
            return [deduped_input_indices[idx] for idx in input_indices]

        def reorder_and_filter(inputs, layout_or_out):
            current_input_indices = indices_for_inputs(inputs)
            if has_bias:
                if len(current_input_indices) < 3:
                    raise AssertionError(
                        f"expected >= 3 input indices, got {len(current_input_indices)}"
                    )
                inp_idx, x_idx, w_idx = current_input_indices[:3]
                return [
                    inputs[x_idx],
                    inputs[w_idx],
                    inputs[inp_idx],
                    *[inputs[idx] for idx in current_input_indices[3:]],
                ], layout_or_out
            if len(current_input_indices) < 2:
                raise AssertionError(
                    f"expected >= 2 input indices, got {len(current_input_indices)}"
                )
            return [inputs[idx] for idx in current_input_indices], layout_or_out

        num_threads = parallel_num_threads()
        new_inputs, _ = reorder_and_filter(input_nodes, layout)
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[1])
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[1].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
            use_ref=True,
        )
        if micro_gemm is None:
            raise AssertionError("expected micro_gemm to be created, got None")
        # No weight is rearranged ahead of time, so the operand is read where
        # the kernel expects it rather than from a packed copy.
        def preprocessor(inputs, layout_or_out):
            new_inputs, new_layout = reorder_and_filter(inputs, layout_or_out)
            return new_inputs, new_layout

        def postprocessor(output):
            return output

        template = DataProcessorTemplateWrapper(
            cls,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            name=micro_gemm.__class__.__name__,
        )
        template.maybe_append_choice(choices)
        return template

    @staticmethod
    def get_padded_size(n, block_n, k, should_block_weight):
        return get_padded_n(n, block_n)

    def get_default_reindexers(self, epilogues):
        return [None] * len(epilogues)

    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Any = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[Any] | None = None,
    ) -> dict[str, Any]:
        if len(self.input_nodes) < 2:
            raise AssertionError(
                f"expected >= 2 input nodes, got {len(self.input_nodes)}"
            )

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        inp = self.input_nodes[2] if self.has_bias else None

        template_buffer_has_other_users = None
        if template_buffer_node is not None:
            W = template_buffer_node.inputs[1]
            Y = template_buffer_node
            if flag_template_buffer_has_other_users is None:
                raise AssertionError(
                    "expected flag_template_buffer_has_other_users to be set"
                )
            template_buffer_has_other_users = flag_template_buffer_has_other_users

        template_buffer = Y
        gemm_output_buffer = template_buffer

        epilogues: list[Any] = []
        reindexers: list[Any] = []
        epilogue_creators: list[Callable] = []
        fake_buffers: list[Any] = []
        Y_aliases: set[str] = set()

        use_local_acc = (
            self.layout.dtype != tp.float32
            or template_buffer_has_other_users
            or self.padded_n != self.n
            or self.maybe_k_slicing()
            or bool(epilogue_nodes and epilogue_nodes[-1].get_dtype() != self.layout.dtype)
        )

        if inp is not None and self.beta != 0:
            def _bias_add_epilogue(buf):
                return create_epilogue_with_attr(
                    buf, "bias_add", other=inp, beta=self.beta, dtype=self.layout.dtype
                )

            epilogue_creators.append(_bias_add_epilogue)

        if self.epilogue_creator is not None:
            epilogue_creators.append(self.epilogue_creator)

        def need_copy_from_local_to_global_buffer_epilogue(
            use_local_acc, template_buffer_has_other_users, epilogue_creators
        ):
            if not use_local_acc:
                return False
            if not template_buffer_has_other_users:
                return False
            if epilogue_creators:
                return False
            return True

        if need_copy_from_local_to_global_buffer_epilogue(
            use_local_acc, template_buffer_has_other_users, epilogue_creators
        ):

            def copy_from_local_to_global_buffer_epilogue(input_buffer: ir.Buffer):
                input_loader = input_buffer.make_loader()
                out_dtype = self.layout.dtype

                def copy_inner(index):
                    return ops.to_dtype(input_loader(index), out_dtype)

                return ir.Pointwise(
                    device=input_buffer.get_device_or_error(),
                    dtype=out_dtype,
                    inner_fn=copy_inner,
                    ranges=input_buffer.get_size(),
                )

            epilogue_creators.append(copy_from_local_to_global_buffer_epilogue)

        if epilogue_creators:
            if not isinstance(template_buffer, ir.IRNode):
                raise AssertionError(f"expected ir.IRNode, got {type(template_buffer)}")
            gemm_output_name = f"{template_buffer.get_name()}_GemmOut"
            gemm_output_buffer = ir.Buffer(
                name=gemm_output_name, layout=template_buffer.layout
            )
            current_input_buffer = gemm_output_buffer
            creator_reindexers = self.get_default_reindexers(epilogue_creators)
            for i, creator in enumerate(epilogue_creators):
                if i == len(epilogue_creators) - 1:
                    buffer_name = template_buffer.get_name()
                else:
                    buffer_name = f"{gemm_output_name}_epilogue_{i}"
                epilogues.append(
                    ir.ComputedBuffer(
                        name=buffer_name,
                        layout=template_buffer.layout,
                        data=creator(current_input_buffer),
                    )
                )
                fake_buffers.append(current_input_buffer)
                Y_aliases.add(current_input_buffer.get_name())
                reindexers.append(creator_reindexers[i])
                if i < len(epilogue_creators) - 1:
                    current_input_buffer = ir.Buffer(
                        name=buffer_name, layout=template_buffer.layout
                    )

        if not isinstance(Y, (ir.Buffer, ir.ReinterpretView)):
            raise AssertionError(
                f"expected ir.Buffer or ir.ReinterpretView, got {type(Y)}"
            )
        Y_2d: Any = Y

        if epilogue_nodes:
            if not template_buffer_has_other_users:
                if not isinstance(template_buffer, ir.IRNode):
                    raise AssertionError(
                        f"expected ir.IRNode, got {type(template_buffer)}"
                    )
                Y_aliases.add(template_buffer.get_name())
            epilogues.extend(epilogue_nodes)
            if Y.get_numel() != epilogues[-1].get_numel():
                raise AssertionError(
                    f"expected Y numel == epilogue numel, got {Y.get_numel()} "
                    f"and {epilogues[-1].get_numel()}"
                )
            Y = epilogues[-1]
            if not isinstance(template_buffer, ir.Buffer):
                raise AssertionError(f"expected ir.Buffer, got {type(template_buffer)}")
            Y_2d, reindexers = gen_2d_view_of_epilogue_buf(
                Y,
                template_buffer,
                epilogue_nodes,
                reindexers,
                default_reindexers=self.get_default_reindexers(epilogue_nodes),
            )

        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X.get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X.get_dtype(),
            input2_dtype=W.get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
            use_ref=True,
        )
        if micro_gemm is None:
            raise AssertionError("expected micro_gemm to be created, got None")
        if self.register_blocking != micro_gemm.register_blocking:
            raise AssertionError(
                f"expected register_blocking == {micro_gemm.register_blocking}, "
                f"got {self.register_blocking}"
            )
        self.log_blockings()
        if isinstance(micro_gemm, CppMicroGemmFP32Vec):
            pass

        L1_cache_size, L2_cache_size = cpu_cache_sizes()

        return dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            GemmOut=gemm_output_buffer,
            aliases={alias: Y.get_name() for alias in Y_aliases},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            epilogue_nodes=epilogues,
            reindexers=reindexers,
            Y_2d=Y_2d,
            use_local_acc=use_local_acc,
            maybe_k_slicing=self.maybe_k_slicing(),
            acc_buf_dtype=tp.float32,
            DTYPE_TO_CPP=DTYPE_TO_CPP,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            config=config,
            fake_buffers=fake_buffers,
            cpu_count=os.cpu_count(),
        )

    def get_kernel_prefix_name(self) -> str:
        return f"gemm_m{self.m}n{self.n}k{self.k}"

    def render(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Any = None,
        flag_template_buffer_has_other_users: bool | None = None,
        epilogue_nodes: list[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        options = self.get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
        )
        self.render_options = options

        with contextlib.ExitStack() as stack:
            for buf in options["fake_buffers"]:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            return self._template_from_string(GEMM_TEMPLATE).render(**options)

    def codegen_blocks(
        self,
        num_threads,
        N,
        K,
        micro_gemm,
        is_dynamic_M,
        kernel,
        GemmOut,
        config,
        L1_cache_size,
        L2_cache_size,
        X,
        W,
    ):
        options = dict(
            num_threads=num_threads,
            N=N,
            K=K,
            micro_gemm=micro_gemm,
            is_dynamic_M=is_dynamic_M,
            kernel=kernel,
            GemmOut=GemmOut,
            config=config,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            template=self,
            X=X,
            W=W,
        )
        return self._template_from_string(
            GEMM_TEMPLATE_INIT_BLOCKING_BASIC_BLOCK + GEMM_TEMPLATE_INIT_BLOCKING_EXTENDED
        ).render(options)

    def codegen_microkernel_def(self):
        return self._template_from_string(GEMM_TEMPLATE_MICROKERNEL_DEF).render(
            self.render_options
        )

    def codegen_gemm_stub_def(self):
        microkernel = self.codegen_microkernel_def()
        return microkernel + self._template_from_string(GEMM_TEMPLATE_STUB_DEF).render(
            self.render_options
        )

    def codegen_multi_threads_params(self):
        return self._template_from_string(GEMM_TEMPLATE_MULTI_THREADS_PARAMS).render()

    def codegen_single_thread_params(self, is_dynamic_M):
        return self._template_from_string(GEMM_TEMPLATE_SINGLE_THREAD_PARAMS).render(
            dict(is_dynamic_M=is_dynamic_M)
        )

    def codegen_m_loop_params(self):
        return self._template_from_string(GEMM_TEMPLATE_M_LOOP_PARAMS).render()

    def codegen_n_loop_params(self):
        return self._template_from_string(GEMM_TEMPLATE_N_LOOP_PARAMS).render()

