// Tier-compiled channels-last kernels for grid_sampler_2d (bilinear, nearest
// and bicubic). In channels-last the channel axis
// is the contiguous one, so it is also the vector axis: for every output
// pixel the tap pixels are each loaded for a block of channels with one
// vector load, scaled by the tap weight and accumulated. The per-pixel
// coordinate/weight section is the scalar frame's own arithmetic (same order
// and rounding), so the two frames are bit-identical by construction.

#include "cpu/GridSamplerKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"
#include "../GridSamplerInline.h"
#include "cpu/vec/vec.h"

namespace tensorplay {
namespace cpu {
namespace {

using tensorplay::vec::Vectorized;
using tensorplay::vec::int_same_size_t;
using tensorplay::parallel::parallel_for;
using gridsampler::Interp;
using gridsampler::Pad;

template <typename T>
using Vec = Vectorized<T>;

// ---------------------------------------------------------------------------
// Bilinear
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_bilinear_cl(const T* inp, const T* grid, T* out,
                               int64_t N, int64_t C, int64_t H, int64_t W,
                               int64_t oH, int64_t oW, bool align_corners) {
    // channels-last strides: sN = C*H*W, sC = 1
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N * oH, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t n = task / oH;
            const int64_t h = task - n * oH;
            const T* inp_n = inp + n * C * H * W;
            const T* grid_row = grid + (n * oH + h) * oW * 2;
            T* out_n = out + n * C * out_plane;
            const int64_t row_out = h * oW;
            for (int64_t w = 0; w < oW; ++w) {
                // Coordinate/weight section verbatim from the scalar frame:
                // the pixel centers sit on integers, so floor(x) + 1 - x is
                // exact and the weights come out identical lane for lane.
                const T x = gridsampler::compute_source_index<T>(
                    grid_row[2 * w], W, PADDING, align_corners);
                const T y = gridsampler::compute_source_index<T>(
                    grid_row[2 * w + 1], H, PADDING, align_corners);
                const int64_t ix_nw = static_cast<int64_t>(std::floor(x));
                const int64_t iy_nw = static_cast<int64_t>(std::floor(y));
                const T nw = static_cast<T>(ix_nw + 1 - x) * static_cast<T>(iy_nw + 1 - y);
                const T ne = static_cast<T>(x - ix_nw) * static_cast<T>(iy_nw + 1 - y);
                const T sw = static_cast<T>(ix_nw + 1 - x) * static_cast<T>(y - iy_nw);
                const T se = static_cast<T>(x - ix_nw) * static_cast<T>(y - iy_nw);
                // Per-tap bounds test in every padding mode: an unclipped NaN
                // coordinate casts to an out-of-range index and must
                // contribute nothing, exactly like the scalar frame.
                const bool ok_nw = gridsampler::within_bounds_2d(iy_nw, ix_nw, H, W);
                const bool ok_ne = gridsampler::within_bounds_2d(iy_nw, ix_nw + 1, H, W);
                const bool ok_sw = gridsampler::within_bounds_2d(iy_nw + 1, ix_nw, H, W);
                const bool ok_se = gridsampler::within_bounds_2d(iy_nw + 1, ix_nw + 1, H, W);
                // Accumulate in nw, ne, sw, se order from +0 like the scalar
                // frame; skipped taps contribute nothing at all.
                T* out_px = out_n + (row_out + w) * C;
                int64_t c = 0;
                for (; c + step <= C; c += step) {
                    Vec<T> acc = Vec<T>(T(0));
                    if (ok_nw)
                        acc = acc + Vec<T>::loadu(inp_n + iy_nw * inp_sH + ix_nw * inp_sW + c) * Vec<T>(nw);
                    if (ok_ne)
                        acc = acc + Vec<T>::loadu(inp_n + iy_nw * inp_sH + (ix_nw + 1) * inp_sW + c) * Vec<T>(ne);
                    if (ok_sw)
                        acc = acc + Vec<T>::loadu(inp_n + (iy_nw + 1) * inp_sH + ix_nw * inp_sW + c) * Vec<T>(sw);
                    if (ok_se)
                        acc = acc + Vec<T>::loadu(inp_n + (iy_nw + 1) * inp_sH + (ix_nw + 1) * inp_sW + c) * Vec<T>(se);
                    acc.store(out_px + c);
                }
                for (; c < C; ++c) {
                    T acc = T(0);
                    if (ok_nw) acc += inp_n[iy_nw * inp_sH + ix_nw * inp_sW + c] * nw;
                    if (ok_ne) acc += inp_n[iy_nw * inp_sH + (ix_nw + 1) * inp_sW + c] * ne;
                    if (ok_sw) acc += inp_n[(iy_nw + 1) * inp_sH + ix_nw * inp_sW + c] * sw;
                    if (ok_se) acc += inp_n[(iy_nw + 1) * inp_sH + (ix_nw + 1) * inp_sW + c] * se;
                    out_px[c] = acc;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Nearest
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_nearest_cl(const T* inp, const T* grid, T* out,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t oH, int64_t oW, bool align_corners) {
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N * oH, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t n = task / oH;
            const int64_t h = task - n * oH;
            const T* inp_n = inp + n * C * H * W;
            const T* grid_row = grid + (n * oH + h) * oW * 2;
            T* out_n = out + n * C * out_plane;
            const int64_t row_out = h * oW;
            for (int64_t w = 0; w < oW; ++w) {
                const T x = gridsampler::compute_source_index<T>(
                    grid_row[2 * w], W, PADDING, align_corners);
                const T y = gridsampler::compute_source_index<T>(
                    grid_row[2 * w + 1], H, PADDING, align_corners);
                // round-to-nearest-even, matching the scalar nearbyint.
                const int64_t ix_near = static_cast<int64_t>(std::nearbyint(x));
                const int64_t iy_near = static_cast<int64_t>(std::nearbyint(y));
                T* out_px = out_n + (row_out + w) * C;
                if (gridsampler::within_bounds_2d(iy_near, ix_near, H, W)) {
                    const T* src_px = inp_n + iy_near * inp_sH + ix_near * inp_sW;
                    int64_t c = 0;
                    for (; c + step <= C; c += step)
                        Vec<T>::loadu(src_px + c).store(out_px + c);
                    for (; c < C; ++c)
                        out_px[c] = src_px[c];
                } else {
                    int64_t c = 0;
                    for (; c + step <= C; c += step)
                        Vec<T>(T(0)).store(out_px + c);
                    for (; c < C; ++c)
                        out_px[c] = T(0);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Bicubic
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_bicubic_cl(const T* inp, const T* grid, T* out,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t oH, int64_t oW, bool align_corners) {
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N * oH, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t n = task / oH;
            const int64_t h = task - n * oH;
            const T* inp_n = inp + n * C * H * W;
            const T* grid_row = grid + (n * oH + h) * oW * 2;
            T* out_n = out + n * C * out_plane;
            const int64_t row_out = h * oW;
            for (int64_t w = 0; w < oW; ++w) {
                // Coordinate section verbatim from the scalar frame's bicubic
                // branch: the base coordinate is only unnormalized; every tap
                // is padding-adjusted individually. The 4x4 tap offsets are
                // integer-valued floats and stay integral through the clip
                // and reflection, so the index casts are exact.
                const T x = gridsampler::unnormalize<T>(grid_row[2 * w], W, align_corners);
                const T y = gridsampler::unnormalize<T>(grid_row[2 * w + 1], H, align_corners);
                const T ix_nw = std::floor(x);
                const T iy_nw = std::floor(y);
                const T tx = x - ix_nw;
                const T ty = y - iy_nw;
                T cx[4], cy[4];
                gridsampler::get_cubic_upsampling_coefficients<T>(cx, tx);
                gridsampler::get_cubic_upsampling_coefficients<T>(cy, ty);
                int64_t tap_ix[4], tap_iy[4];
                bool tap_ok[4][4];
                for (int i = 0; i < 4; ++i) {
                    const T tapy = gridsampler::compute_coordinates<T>(
                        iy_nw - 1 + i, H, PADDING, align_corners);
                    tap_iy[i] = static_cast<int64_t>(tapy);
                    for (int k = 0; k < 4; ++k) {
                        const T tapx = gridsampler::compute_coordinates<T>(
                            ix_nw - 1 + k, W, PADDING, align_corners);
                        tap_ix[k] = static_cast<int64_t>(tapx);
                        tap_ok[i][k] =
                            gridsampler::within_bounds_2d(tap_iy[i], tap_ix[k], H, W);
                    }
                }
                // The scalar frame always sums all sixteen weighted terms
                // (out-of-bound taps read as exact zeros), left to right in
                // x-then-y order; the vector form keeps that association.
                T* out_px = out_n + (row_out + w) * C;
                int64_t c = 0;
                for (; c + step <= C; c += step) {
                    Vec<T> rows[4];
                    for (int i = 0; i < 4; ++i) {
                        const int64_t row_base = tap_iy[i] * inp_sH;
                        const Vec<T> v0 = tap_ok[i][0]
                            ? Vec<T>::loadu(inp_n + row_base + tap_ix[0] * inp_sW + c)
                            : Vec<T>(T(0));
                        const Vec<T> v1 = tap_ok[i][1]
                            ? Vec<T>::loadu(inp_n + row_base + tap_ix[1] * inp_sW + c)
                            : Vec<T>(T(0));
                        const Vec<T> v2 = tap_ok[i][2]
                            ? Vec<T>::loadu(inp_n + row_base + tap_ix[2] * inp_sW + c)
                            : Vec<T>(T(0));
                        const Vec<T> v3 = tap_ok[i][3]
                            ? Vec<T>::loadu(inp_n + row_base + tap_ix[3] * inp_sW + c)
                            : Vec<T>(T(0));
                        rows[i] = ((Vec<T>(cx[0]) * v0 + Vec<T>(cx[1]) * v1) +
                                   Vec<T>(cx[2]) * v2) + Vec<T>(cx[3]) * v3;
                    }
                    const Vec<T> res = ((Vec<T>(cy[0]) * rows[0] + Vec<T>(cy[1]) * rows[1]) +
                                        Vec<T>(cy[2]) * rows[2]) + Vec<T>(cy[3]) * rows[3];
                    res.store(out_px + c);
                }
                for (; c < C; ++c) {
                    T rows[4];
                    for (int i = 0; i < 4; ++i) {
                        const int64_t row_base = tap_iy[i] * inp_sH;
                        const T v0 = tap_ok[i][0] ? inp_n[row_base + tap_ix[0] * inp_sW + c] : T(0);
                        const T v1 = tap_ok[i][1] ? inp_n[row_base + tap_ix[1] * inp_sW + c] : T(0);
                        const T v2 = tap_ok[i][2] ? inp_n[row_base + tap_ix[2] * inp_sW + c] : T(0);
                        const T v3 = tap_ok[i][3] ? inp_n[row_base + tap_ix[3] * inp_sW + c] : T(0);
                        rows[i] = (cx[0] * v0 + cx[1] * v1) + cx[2] * v2 + cx[3] * v3;
                    }
                    out_px[c] = (cy[0] * rows[0] + cy[1] * rows[1]) + cy[2] * rows[2] + cy[3] * rows[3];
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Backward (bilinear)
// ---------------------------------------------------------------------------
// Each output pixel scatters into four input pixels, so output pixels can
// collide on the same grad_input element; like the scalar frame, the batch
// axis is the only parallel one, which keeps every element's contribution
// order (h, w) lexicographic and the frames bit-identical. grad_grid needs
// the per-channel sum in ascending channel order, so within a block the
// vector lanes are extracted and summed sequentially.
template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_backward_bilinear_cl(const T* gout, const T* inp, const T* grid,
                                        T* ginp, T* ggrid,
                                        int64_t N, int64_t C, int64_t H, int64_t W,
                                        int64_t oH, int64_t oW, bool align_corners,
                                        bool need_gi, bool need_gg) {
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const T* grid_n = grid + n * out_plane * 2;
            const T* inp_n = inp + n * C * H * W;
            const T* gout_n = gout + n * C * out_plane;
            T* ginp_n = need_gi ? ginp + n * C * H * W : nullptr;
            T* ggrid_n = need_gg ? ggrid + n * out_plane * 2 : nullptr;
            for (int64_t h = 0; h < oH; ++h) {
                for (int64_t w = 0; w < oW; ++w) {
                    const T* grid_px = grid_n + (h * oW + w) * 2;
                    T gix_mult, giy_mult;
                    const T x = gridsampler::compute_source_index_set_grad<T>(
                        grid_px[0], W, PADDING, align_corners, &gix_mult);
                    const T y = gridsampler::compute_source_index_set_grad<T>(
                        grid_px[1], H, PADDING, align_corners, &giy_mult);
                    const int64_t ix_nw = static_cast<int64_t>(std::floor(x));
                    const int64_t iy_nw = static_cast<int64_t>(std::floor(y));
                    const T nw = static_cast<T>(ix_nw + 1 - x) * static_cast<T>(iy_nw + 1 - y);
                    const T ne = static_cast<T>(x - ix_nw) * static_cast<T>(iy_nw + 1 - y);
                    const T sw = static_cast<T>(ix_nw + 1 - x) * static_cast<T>(y - iy_nw);
                    const T se = static_cast<T>(x - ix_nw) * static_cast<T>(y - iy_nw);
                    const bool ok_nw = gridsampler::within_bounds_2d(iy_nw, ix_nw, H, W);
                    const bool ok_ne = gridsampler::within_bounds_2d(iy_nw, ix_nw + 1, H, W);
                    const bool ok_sw = gridsampler::within_bounds_2d(iy_nw + 1, ix_nw, H, W);
                    const bool ok_se = gridsampler::within_bounds_2d(iy_nw + 1, ix_nw + 1, H, W);
                    // Offsets only for in-bounds taps: an unclipped NaN casts
                    // to a wild index whose product would overflow.
                    const int64_t b_nw = ok_nw ? iy_nw * inp_sH + ix_nw * inp_sW : 0;
                    const int64_t b_ne = ok_ne ? iy_nw * inp_sH + (ix_nw + 1) * inp_sW : 0;
                    const int64_t b_sw = ok_sw ? (iy_nw + 1) * inp_sH + ix_nw * inp_sW : 0;
                    const int64_t b_se = ok_se ? (iy_nw + 1) * inp_sH + (ix_nw + 1) * inp_sW : 0;
                    T gix = T(0), giy = T(0);
                    const int64_t go_px = (h * oW + w) * C;
                    int64_t c = 0;
                    for (; c + step <= C; c += step) {
                        const Vec<T> gout_v = Vec<T>::loadu(gout_n + go_px + c);
                        if (need_gi) {
                            if (ok_nw) {
                                Vec<T> g = Vec<T>::loadu(ginp_n + b_nw + c);
                                g = g + Vec<T>(nw) * gout_v;
                                g.store(ginp_n + b_nw + c);
                            }
                            if (ok_ne) {
                                Vec<T> g = Vec<T>::loadu(ginp_n + b_ne + c);
                                g = g + Vec<T>(ne) * gout_v;
                                g.store(ginp_n + b_ne + c);
                            }
                            if (ok_sw) {
                                Vec<T> g = Vec<T>::loadu(ginp_n + b_sw + c);
                                g = g + Vec<T>(sw) * gout_v;
                                g.store(ginp_n + b_sw + c);
                            }
                            if (ok_se) {
                                Vec<T> g = Vec<T>::loadu(ginp_n + b_se + c);
                                g = g + Vec<T>(se) * gout_v;
                                g.store(ginp_n + b_se + c);
                            }
                        }
                        if (need_gg) {
                            alignas(32) T gbuf[step], nwbuf[step], nebuf[step],
                                swbuf[step], sebuf[step];
                            gout_v.store(gbuf);
                            (ok_nw ? Vec<T>::loadu(inp_n + b_nw + c) : Vec<T>(T(0))).store(nwbuf);
                            (ok_ne ? Vec<T>::loadu(inp_n + b_ne + c) : Vec<T>(T(0))).store(nebuf);
                            (ok_sw ? Vec<T>::loadu(inp_n + b_sw + c) : Vec<T>(T(0))).store(swbuf);
                            (ok_se ? Vec<T>::loadu(inp_n + b_se + c) : Vec<T>(T(0))).store(sebuf);
                            // Sequential channel order, tap order nw..se, the
                            // scalar frame's own expressions per lane.
                            for (int64_t j = 0; j < step; ++j) {
                                const T gout_j = gbuf[j];
                                if (ok_nw) {
                                    const T v = nwbuf[j];
                                    gix -= v * (iy_nw + 1 - y) * gout_j;
                                    giy -= v * (ix_nw + 1 - x) * gout_j;
                                }
                                if (ok_ne) {
                                    const T v = nebuf[j];
                                    gix += v * (iy_nw + 1 - y) * gout_j;
                                    giy -= v * (x - ix_nw) * gout_j;
                                }
                                if (ok_sw) {
                                    const T v = swbuf[j];
                                    gix -= v * (y - iy_nw) * gout_j;
                                    giy += v * (ix_nw + 1 - x) * gout_j;
                                }
                                if (ok_se) {
                                    const T v = sebuf[j];
                                    gix += v * (y - iy_nw) * gout_j;
                                    giy += v * (x - ix_nw) * gout_j;
                                }
                            }
                        }
                    }
                    for (; c < C; ++c) {
                        const T gout_c = gout_n[go_px + c];
                        if (need_gi) {
                            if (ok_nw) ginp_n[b_nw + c] += nw * gout_c;
                            if (ok_ne) ginp_n[b_ne + c] += ne * gout_c;
                            if (ok_sw) ginp_n[b_sw + c] += sw * gout_c;
                            if (ok_se) ginp_n[b_se + c] += se * gout_c;
                        }
                        if (need_gg) {
                            if (ok_nw) {
                                const T v = inp_n[b_nw + c];
                                gix -= v * (iy_nw + 1 - y) * gout_c;
                                giy -= v * (ix_nw + 1 - x) * gout_c;
                            }
                            if (ok_ne) {
                                const T v = inp_n[b_ne + c];
                                gix += v * (iy_nw + 1 - y) * gout_c;
                                giy -= v * (x - ix_nw) * gout_c;
                            }
                            if (ok_sw) {
                                const T v = inp_n[b_sw + c];
                                gix -= v * (y - iy_nw) * gout_c;
                                giy += v * (ix_nw + 1 - x) * gout_c;
                            }
                            if (ok_se) {
                                const T v = inp_n[b_se + c];
                                gix += v * (y - iy_nw) * gout_c;
                                giy += v * (x - ix_nw) * gout_c;
                            }
                        }
                    }
                    if (need_gg) {
                        T* ggrid_px = ggrid_n + (h * oW + w) * 2;
                        ggrid_px[0] = gix_mult * gix;
                        ggrid_px[1] = giy_mult * giy;
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Backward (nearest)
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_backward_nearest_cl(const T* gout, const T* inp, const T* grid,
                                       T* ginp, T* ggrid,
                                       int64_t N, int64_t C, int64_t H, int64_t W,
                                       int64_t oH, int64_t oW, bool align_corners,
                                       bool need_gi, bool need_gg) {
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();
    (void)inp;

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const T* grid_n = grid + n * out_plane * 2;
            const T* gout_n = gout + n * C * out_plane;
            T* ginp_n = need_gi ? ginp + n * C * H * W : nullptr;
            T* ggrid_n = need_gg ? ggrid + n * out_plane * 2 : nullptr;
            for (int64_t h = 0; h < oH; ++h) {
                for (int64_t w = 0; w < oW; ++w) {
                    const T* grid_px = grid_n + (h * oW + w) * 2;
                    T gix_mult, giy_mult;
                    const T x = gridsampler::compute_source_index_set_grad<T>(
                        grid_px[0], W, PADDING, align_corners, &gix_mult);
                    const T y = gridsampler::compute_source_index_set_grad<T>(
                        grid_px[1], H, PADDING, align_corners, &giy_mult);
                    const int64_t ix_near = static_cast<int64_t>(std::nearbyint(x));
                    const int64_t iy_near = static_cast<int64_t>(std::nearbyint(y));
                    if (need_gi && gridsampler::within_bounds_2d(iy_near, ix_near, H, W)) {
                        const int64_t b = iy_near * inp_sH + ix_near * inp_sW;
                        const int64_t go_px = (h * oW + w) * C;
                        int64_t c = 0;
                        for (; c + step <= C; c += step) {
                            Vec<T> g = Vec<T>::loadu(ginp_n + b + c) +
                                       Vec<T>::loadu(gout_n + go_px + c);
                            g.store(ginp_n + b + c);
                        }
                        for (; c < C; ++c)
                            ginp_n[b + c] += gout_n[go_px + c];
                    }
                    if (need_gg) {
                        T* ggrid_px = ggrid_n + (h * oW + w) * 2;
                        ggrid_px[0] = T(0);
                        ggrid_px[1] = T(0);
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Backward (bicubic)
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_backward_bicubic_cl(const T* gout, const T* inp, const T* grid,
                                       T* ginp, T* ggrid,
                                       int64_t N, int64_t C, int64_t H, int64_t W,
                                       int64_t oH, int64_t oW, bool align_corners,
                                       bool need_gi, bool need_gg) {
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const T* grid_n = grid + n * out_plane * 2;
            const T* inp_n = inp + n * C * H * W;
            const T* gout_n = gout + n * C * out_plane;
            T* ginp_n = need_gi ? ginp + n * C * H * W : nullptr;
            T* ggrid_n = need_gg ? ggrid + n * out_plane * 2 : nullptr;
            for (int64_t h = 0; h < oH; ++h) {
                for (int64_t w = 0; w < oW; ++w) {
                    const T* grid_px = grid_n + (h * oW + w) * 2;
                    T gix_mult, giy_mult;
                    const T x = gridsampler::unnormalize_set_grad<T>(
                        grid_px[0], W, align_corners, &gix_mult);
                    const T y = gridsampler::unnormalize_set_grad<T>(
                        grid_px[1], H, align_corners, &giy_mult);
                    const T ix_nw = std::floor(x);
                    const T iy_nw = std::floor(y);
                    const T tx = x - ix_nw;
                    const T ty = y - iy_nw;
                    T x_coeffs[4], y_coeffs[4], x_coeffs_grad[4], y_coeffs_grad[4];
                    gridsampler::get_cubic_upsampling_coefficients<T>(x_coeffs, tx);
                    gridsampler::get_cubic_upsampling_coefficients<T>(y_coeffs, ty);
                    gridsampler::get_cubic_coefficients_grad<T>(x_coeffs_grad, tx);
                    gridsampler::get_cubic_coefficients_grad<T>(y_coeffs_grad, ty);
                    // 16 tap positions once per pixel (the tap coordinates do
                    // not depend on the channel); taps are padding-adjusted
                    // individually like the scalar frame.
                    int64_t tap_bx[4], tap_by[4];
                    bool tap_ok[4][4];
                    int64_t tap_base[4][4];
                    for (int i = 0; i < 4; ++i) {
                        const T tapx = gridsampler::compute_coordinates<T>(
                            ix_nw - 1 + i, W, PADDING, align_corners);
                        tap_bx[i] = static_cast<int64_t>(tapx);
                        for (int j = 0; j < 4; ++j) {
                            const T tapy = gridsampler::compute_coordinates<T>(
                                iy_nw - 1 + j, H, PADDING, align_corners);
                            tap_by[j] = static_cast<int64_t>(tapy);
                            tap_ok[i][j] =
                                gridsampler::within_bounds_2d(tap_by[j], tap_bx[i], H, W);
                            tap_base[i][j] = tap_ok[i][j]
                                ? tap_by[j] * inp_sH + tap_bx[i] * inp_sW : 0;
                        }
                    }
                    T gix = T(0), giy = T(0);
                    const int64_t go_px = (h * oW + w) * C;
                    int64_t c = 0;
                    for (; c + step <= C; c += step) {
                        const Vec<T> gout_v = Vec<T>::loadu(gout_n + go_px + c);
                        if (need_gi) {
                            for (int i = 0; i < 4; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    if (!tap_ok[i][j]) continue;
                                    Vec<T> g = Vec<T>::loadu(ginp_n + tap_base[i][j] + c);
                                    g = g + gout_v * Vec<T>(x_coeffs[i]) * Vec<T>(y_coeffs[j]);
                                    g.store(ginp_n + tap_base[i][j] + c);
                                }
                            }
                        }
                        if (need_gg) {
                            alignas(32) T gbuf[step];
                            alignas(32) T vbuf[4][4][step];
                            gout_v.store(gbuf);
                            for (int i = 0; i < 4; ++i)
                                for (int j = 0; j < 4; ++j)
                                    (tap_ok[i][j]
                                         ? Vec<T>::loadu(inp_n + tap_base[i][j] + c)
                                         : Vec<T>(T(0))).store(vbuf[i][j]);
                            // Sequential channel order; every tap term is
                            // subtracted even for out-of-bound taps (their
                            // value is an exact zero), like the scalar frame.
                            for (int64_t k = 0; k < step; ++k) {
                                const T gout_k = gbuf[k];
                                for (int i = 0; i < 4; ++i) {
                                    for (int j = 0; j < 4; ++j) {
                                        const T val = vbuf[i][j][k];
                                        gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gout_k;
                                        giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gout_k;
                                    }
                                }
                            }
                        }
                    }
                    for (; c < C; ++c) {
                        const T gout_c = gout_n[go_px + c];
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                if (need_gi && tap_ok[i][j])
                                    ginp_n[tap_base[i][j] + c] +=
                                        gout_c * x_coeffs[i] * y_coeffs[j];
                                const T val = tap_ok[i][j]
                                    ? inp_n[tap_base[i][j] + c] : T(0);
                                gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gout_c;
                                giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gout_c;
                            }
                        }
                    }
                    if (need_gg) {
                        T* ggrid_px = ggrid_n + (h * oW + w) * 2;
                        ggrid_px[0] = gix_mult * gix;
                        ggrid_px[1] = giy_mult * giy;
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Entry: switch interpolation/padding at run time inside each tier binary.
// ---------------------------------------------------------------------------
template <typename T>
void grid_sample2d_cl_typed(const void* in_, const void* grid_, void* out_,
                            int64_t N, int64_t C, int64_t H, int64_t W,
                            int64_t oH, int64_t oW, int interpolation_mode,
                            int padding_mode, bool align_corners) {
    const T* in = static_cast<const T*>(in_);
    const T* grid = static_cast<const T*>(grid_);
    T* out = static_cast<T*>(out_);
    if (interpolation_mode == Interp::Bilinear) {
        switch (padding_mode) {
            case Pad::Zeros:
                grid_sample2d_bilinear_cl<T, Pad::Zeros>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            case Pad::Border:
                grid_sample2d_bilinear_cl<T, Pad::Border>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            default:
                grid_sample2d_bilinear_cl<T, Pad::Reflection>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
        }
    }
    if (interpolation_mode == Interp::Nearest) {
        switch (padding_mode) {
            case Pad::Zeros:
                grid_sample2d_nearest_cl<T, Pad::Zeros>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            case Pad::Border:
                grid_sample2d_nearest_cl<T, Pad::Border>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            default:
                grid_sample2d_nearest_cl<T, Pad::Reflection>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
        }
    }
    if (interpolation_mode == Interp::Bicubic) {
        switch (padding_mode) {
            case Pad::Zeros:
                grid_sample2d_bicubic_cl<T, Pad::Zeros>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            case Pad::Border:
                grid_sample2d_bicubic_cl<T, Pad::Border>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            default:
                grid_sample2d_bicubic_cl<T, Pad::Reflection>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
        }
    }
    TP_THROW(NotImplementedError, "grid_sampler_2d channels-last: unsupported interpolation mode");
}

void grid_sample2d_cl_impl(const void* in, const void* grid, void* out,
                           int64_t N, int64_t C, int64_t H, int64_t W,
                           int64_t oH, int64_t oW, int interpolation_mode,
                           int padding_mode, int align_corners, int dtype) {
    const bool ac = align_corners != 0;
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            grid_sample2d_cl_typed<float>(in, grid, out, N, C, H, W, oH, oW,
                                          interpolation_mode, padding_mode, ac);
            return;
        case DType::Float64:
            grid_sample2d_cl_typed<double>(in, grid, out, N, C, H, W, oH, oW,
                                           interpolation_mode, padding_mode, ac);
            return;
        default:
            TP_THROW(NotImplementedError, "grid_sampler_2d channels-last: unsupported dtype");
    }
}

template <typename T>
void grid_sample2d_backward_cl_typed(const void* gout_, const void* in_, const void* grid_,
                                     void* ginp_, void* ggrid_,
                                     int64_t N, int64_t C, int64_t H, int64_t W,
                                     int64_t oH, int64_t oW, int interpolation_mode,
                                     int padding_mode, bool align_corners,
                                     bool need_gi, bool need_gg) {
    const T* gout = static_cast<const T*>(gout_);
    const T* in = static_cast<const T*>(in_);
    const T* grid = static_cast<const T*>(grid_);
    T* ginp = static_cast<T*>(ginp_);
    T* ggrid = static_cast<T*>(ggrid_);
#define TP_GS_BWD_CASE(MODE, FN)                                                     \
    switch (padding_mode) {                                                          \
        case Pad::Zeros:                                                             \
            FN<T, Pad::Zeros>(gout, in, grid, ginp, ggrid, N, C, H, W, oH, oW,       \
                              align_corners, need_gi, need_gg);                      \
            return;                                                                  \
        case Pad::Border:                                                            \
            FN<T, Pad::Border>(gout, in, grid, ginp, ggrid, N, C, H, W, oH, oW,      \
                               align_corners, need_gi, need_gg);                     \
            return;                                                                  \
        default:                                                                     \
            FN<T, Pad::Reflection>(gout, in, grid, ginp, ggrid, N, C, H, W, oH, oW,  \
                                   align_corners, need_gi, need_gg);                 \
            return;                                                                  \
    }
    if (interpolation_mode == Interp::Bilinear)
        TP_GS_BWD_CASE(Bilinear, grid_sample2d_backward_bilinear_cl)
    if (interpolation_mode == Interp::Nearest)
        TP_GS_BWD_CASE(Nearest, grid_sample2d_backward_nearest_cl)
    if (interpolation_mode == Interp::Bicubic)
        TP_GS_BWD_CASE(Bicubic, grid_sample2d_backward_bicubic_cl)
#undef TP_GS_BWD_CASE
    TP_THROW(NotImplementedError, "grid_sampler_2d channels-last backward: unsupported interpolation mode");
}

void grid_sample2d_backward_cl_impl(const void* gout, const void* in, const void* grid,
                                    void* ginp, void* ggrid,
                                    int64_t N, int64_t C, int64_t H, int64_t W,
                                    int64_t oH, int64_t oW, int interpolation_mode,
                                    int padding_mode, int align_corners, int dtype,
                                    int need_gi, int need_gg) {
    const bool ac = align_corners != 0;
    const bool gi = need_gi != 0;
    const bool gg = need_gg != 0;
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            grid_sample2d_backward_cl_typed<float>(gout, in, grid, ginp, ggrid, N, C, H, W,
                                                   oH, oW, interpolation_mode, padding_mode,
                                                   ac, gi, gg);
            return;
        case DType::Float64:
            grid_sample2d_backward_cl_typed<double>(gout, in, grid, ginp, ggrid, N, C, H, W,
                                                    oH, oW, interpolation_mode, padding_mode,
                                                    ac, gi, gg);
            return;
        default:
            TP_THROW(NotImplementedError, "grid_sampler_2d channels-last backward: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// Strided-channel (N, C, H, W) frame
// ---------------------------------------------------------------------------
//
// The channel axis is a plane apart, so no contiguous vector runs along it. The
// pixel axis is contiguous, and it carries the vector instead: a block of
// `step` output pixels is resolved once -- source indices, tap weights and
// in-bounds flags -- and the channel walk replays that block for every channel
// with one masked gather per tap.  A block therefore reads its taps from the
// same few lines for all channels, where a per-pixel walk revisits the channel
// planes once per pixel.

template <typename T>
inline void nchw_set_tap(int_same_size_t<T>& off, int_same_size_t<T>& mask,
                         int64_t ih, int64_t iw, int64_t H, int64_t W) {
    if (gridsampler::within_bounds_2d(ih, iw, H, W)) {
        off = static_cast<int_same_size_t<T>>(ih * W + iw);
        mask = static_cast<int_same_size_t<T>>(-1);
    } else {
        off = 0;
        mask = 0;
    }
}

template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_nchw_bilinear(const T* inp, const T* grid, T* out,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t oH, int64_t oW, bool ac) {
    using Vec = Vectorized<T>;
    using iVec = Vectorized<int_same_size_t<T>>;
    using it = int_same_size_t<T>;
    constexpr int64_t S = Vec::size();
    const int64_t spatial = oH * oW;
    const int64_t blocks = (spatial + S - 1) / S;
    const int64_t plane = H * W;

    parallel_for(0, N * blocks, 1, [&](int64_t b0, int64_t b1) {
        alignas(64) it o_nw[S], o_ne[S], o_sw[S], o_se[S];
        alignas(64) it m_nw[S], m_ne[S], m_sw[S], m_se[S];
        alignas(64) T w_nw[S], w_ne[S], w_sw[S], w_se[S];
        for (int64_t task = b0; task < b1; ++task) {
            const int64_t n = task / blocks;
            const int64_t first = (task - n * blocks) * S;
            const int64_t len = std::min<int64_t>(S, spatial - first);
            const T* g = grid + n * spatial * 2 + first * 2;
            // A block that fills the vector needs no zeroed lanes, so the
            // clear pass only runs for the last, partial block.
            for (int64_t k = 0; k < len; ++k) {
                const T x = gridsampler::compute_source_index<T>(g[2 * k], W, PADDING, ac);
                const T y = gridsampler::compute_source_index<T>(g[2 * k + 1], H, PADDING, ac);
                const int64_t iw = static_cast<int64_t>(std::floor(x));
                const int64_t ih = static_cast<int64_t>(std::floor(y));
                const T e = static_cast<T>(iw + 1) - x;
                const T w = x - static_cast<T>(iw);
                const T s = static_cast<T>(ih + 1) - y;
                const T nn = y - static_cast<T>(ih);
                w_nw[k] = e * s;
                w_ne[k] = w * s;
                w_sw[k] = e * nn;
                w_se[k] = w * nn;
                nchw_set_tap<T>(o_nw[k], m_nw[k], ih, iw, H, W);
                nchw_set_tap<T>(o_ne[k], m_ne[k], ih, iw + 1, H, W);
                nchw_set_tap<T>(o_sw[k], m_sw[k], ih + 1, iw, H, W);
                nchw_set_tap<T>(o_se[k], m_se[k], ih + 1, iw + 1, H, W);
            }
            for (int64_t k = len; k < S; ++k) {
                o_nw[k] = o_ne[k] = o_sw[k] = o_se[k] = it(0);
                m_nw[k] = m_ne[k] = m_sw[k] = m_se[k] = it(0);
                w_nw[k] = w_ne[k] = w_sw[k] = w_se[k] = T(0);
            }
            const iVec iv_nw = iVec::loadu(o_nw), iv_ne = iVec::loadu(o_ne);
            const iVec iv_sw = iVec::loadu(o_sw), iv_se = iVec::loadu(o_se);
            const Vec wv_nw = Vec::loadu(w_nw), wv_ne = Vec::loadu(w_ne);
            const Vec wv_sw = Vec::loadu(w_sw), wv_se = Vec::loadu(w_se);
            const T* in_n = inp + n * C * plane;
            T* out_n = out + n * C * spatial + first;
            for (int64_t c = 0; c < C; ++c) {
                const T* p = in_n + c * plane;
                Vec mv_nw = Vec::loadu(static_cast<const void*>(m_nw));
                Vec mv_ne = Vec::loadu(static_cast<const void*>(m_ne));
                Vec mv_sw = Vec::loadu(static_cast<const void*>(m_sw));
                Vec mv_se = Vec::loadu(static_cast<const void*>(m_se));
                Vec acc = mask_gather<sizeof(T)>(Vec(T(0)), p, iv_nw, mv_nw) * wv_nw;
                acc = acc + mask_gather<sizeof(T)>(Vec(T(0)), p, iv_ne, mv_ne) * wv_ne;
                acc = acc + mask_gather<sizeof(T)>(Vec(T(0)), p, iv_sw, mv_sw) * wv_sw;
                acc = acc + mask_gather<sizeof(T)>(Vec(T(0)), p, iv_se, mv_se) * wv_se;
                acc.store(out_n + c * spatial, static_cast<int>(len));
            }
        }
    });
}

template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_nchw_nearest(const T* inp, const T* grid, T* out,
                                int64_t N, int64_t C, int64_t H, int64_t W,
                                int64_t oH, int64_t oW, bool ac) {
    using Vec = Vectorized<T>;
    using iVec = Vectorized<int_same_size_t<T>>;
    using it = int_same_size_t<T>;
    constexpr int64_t S = Vec::size();
    const int64_t spatial = oH * oW;
    const int64_t blocks = (spatial + S - 1) / S;
    const int64_t plane = H * W;

    parallel_for(0, N * blocks, 1, [&](int64_t b0, int64_t b1) {
        alignas(64) it o_px[S], m_px[S];
        for (int64_t task = b0; task < b1; ++task) {
            const int64_t n = task / blocks;
            const int64_t first = (task - n * blocks) * S;
            const int64_t len = std::min<int64_t>(S, spatial - first);
            const T* g = grid + n * spatial * 2 + first * 2;
            for (int64_t k = 0; k < len; ++k) {
                const T x = gridsampler::compute_source_index<T>(g[2 * k], W, PADDING, ac);
                const T y = gridsampler::compute_source_index<T>(g[2 * k + 1], H, PADDING, ac);
                nchw_set_tap<T>(o_px[k], m_px[k],
                                static_cast<int64_t>(std::nearbyint(y)),
                                static_cast<int64_t>(std::nearbyint(x)), H, W);
            }
            for (int64_t k = len; k < S; ++k) {
                o_px[k] = it(0);
                m_px[k] = it(0);
            }
            const iVec iv_px = iVec::loadu(o_px);
            const T* in_n = inp + n * C * plane;
            T* out_n = out + n * C * spatial + first;
            for (int64_t c = 0; c < C; ++c) {
                Vec mv = Vec::loadu(static_cast<const void*>(m_px));
                mask_gather<sizeof(T)>(Vec(T(0)), in_n + c * plane, iv_px, mv)
                    .store(out_n + c * spatial, static_cast<int>(len));
            }
        }
    });
}

template <typename T, int PADDING>
TP_NO_FP_CONTRACT
void grid_sample2d_nchw_bicubic(const T* inp, const T* grid, T* out,
                                int64_t N, int64_t C, int64_t H, int64_t W,
                                int64_t oH, int64_t oW, bool ac) {
    using Vec = Vectorized<T>;
    using iVec = Vectorized<int_same_size_t<T>>;
    using it = int_same_size_t<T>;
    constexpr int64_t S = Vec::size();
    const int64_t spatial = oH * oW;
    const int64_t blocks = (spatial + S - 1) / S;
    const int64_t plane = H * W;

    parallel_for(0, N * blocks, 1, [&](int64_t b0, int64_t b1) {
        alignas(64) it o_tap[16][S], m_tap[16][S];
        alignas(64) T w_h[4][S], w_v[4][S];
        for (int64_t task = b0; task < b1; ++task) {
            const int64_t n = task / blocks;
            const int64_t first = (task - n * blocks) * S;
            const int64_t len = std::min<int64_t>(S, spatial - first);
            const T* g = grid + n * spatial * 2 + first * 2;
            for (int64_t k = 0; k < len; ++k) {
                T hc[4], vc[4];
                // Only the base coordinate is unnormalized; every tap is
                // padding-adjusted on its own, in the scalar frame's order.
                const T x = gridsampler::unnormalize<T>(g[2 * k], W, ac);
                const T y = gridsampler::unnormalize<T>(g[2 * k + 1], H, ac);
                const T fx = std::floor(x);
                const T fy = std::floor(y);
                gridsampler::get_cubic_upsampling_coefficients<T>(hc, x - fx);
                gridsampler::get_cubic_upsampling_coefficients<T>(vc, y - fy);
                for (int i = 0; i < 4; ++i) {
                    w_h[i][k] = hc[i];
                    w_v[i][k] = vc[i];
                }
                for (int j = 0; j < 4; ++j) {
                    for (int i = 0; i < 4; ++i) {
                        const T xc = gridsampler::compute_coordinates<T>(
                            fx - T(1) + T(i), W, PADDING, ac);
                        const T yc = gridsampler::compute_coordinates<T>(
                            fy - T(1) + T(j), H, PADDING, ac);
                        nchw_set_tap<T>(o_tap[j * 4 + i][k], m_tap[j * 4 + i][k],
                                        static_cast<int64_t>(yc),
                                        static_cast<int64_t>(xc), H, W);
                    }
                }
            }
            for (int64_t k = len; k < S; ++k) {
                for (int i = 0; i < 4; ++i) {
                    w_h[i][k] = T(0);
                    w_v[i][k] = T(0);
                }
                for (int t = 0; t < 16; ++t) {
                    o_tap[t][k] = it(0);
                    m_tap[t][k] = it(0);
                }
            }
            iVec iv_tap[16];
            Vec wv_h[4], wv_v[4];
            for (int t = 0; t < 16; ++t) iv_tap[t] = iVec::loadu(o_tap[t]);
            for (int i = 0; i < 4; ++i) {
                wv_h[i] = Vec::loadu(w_h[i]);
                wv_v[i] = Vec::loadu(w_v[i]);
            }
            const T* in_n = inp + n * C * plane;
            T* out_n = out + n * C * spatial + first;
            for (int64_t c = 0; c < C; ++c) {
                const T* p = in_n + c * plane;
                Vec row[4];
                for (int j = 0; j < 4; ++j) {
                    Vec acc = Vec(T(0));
                    for (int i = 0; i < 4; ++i) {
                        Vec mv = Vec::loadu(static_cast<const void*>(m_tap[j * 4 + i]));
                        acc = acc + mask_gather<sizeof(T)>(Vec(T(0)), p, iv_tap[j * 4 + i], mv) * wv_h[i];
                    }
                    row[j] = acc;
                }
                Vec v = row[0] * wv_v[0];
                v = v + row[1] * wv_v[1];
                v = v + row[2] * wv_v[2];
                v = v + row[3] * wv_v[3];
                v.store(out_n + c * spatial, static_cast<int>(len));
            }
        }
    });
}

template <typename T, int PADDING>
void grid_sample2d_nchw_mode(const T* inp, const T* grid, T* out,
                             int64_t N, int64_t C, int64_t H, int64_t W,
                             int64_t oH, int64_t oW, int interpolation_mode,
                             bool ac) {
    switch (interpolation_mode) {
        case Interp::Bilinear:
            grid_sample2d_nchw_bilinear<T, PADDING>(inp, grid, out, N, C, H, W, oH, oW, ac);
            break;
        case Interp::Nearest:
            grid_sample2d_nchw_nearest<T, PADDING>(inp, grid, out, N, C, H, W, oH, oW, ac);
            break;
        default:
            grid_sample2d_nchw_bicubic<T, PADDING>(inp, grid, out, N, C, H, W, oH, oW, ac);
            break;
    }
}

template <typename T>
void grid_sample2d_nchw_typed(const T* inp, const T* grid, T* out,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t oH, int64_t oW, int interpolation_mode,
                              int padding_mode, bool ac) {
    switch (padding_mode) {
        case Pad::Zeros:
            grid_sample2d_nchw_mode<T, Pad::Zeros>(inp, grid, out, N, C, H, W, oH, oW,
                                                   interpolation_mode, ac);
            break;
        case Pad::Border:
            grid_sample2d_nchw_mode<T, Pad::Border>(inp, grid, out, N, C, H, W, oH, oW,
                                                    interpolation_mode, ac);
            break;
        default:
            grid_sample2d_nchw_mode<T, Pad::Reflection>(inp, grid, out, N, C, H, W, oH, oW,
                                                        interpolation_mode, ac);
            break;
    }
}

void grid_sample2d_nchw_impl(const void* in, const void* grid, void* out,
                             int64_t N, int64_t C, int64_t H, int64_t W,
                             int64_t oH, int64_t oW, int interpolation_mode,
                             int padding_mode, int align_corners, int dtype) {
    const bool ac = align_corners != 0;
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            grid_sample2d_nchw_typed<float>(static_cast<const float*>(in),
                                            static_cast<const float*>(grid),
                                            static_cast<float*>(out), N, C, H, W, oH, oW,
                                            interpolation_mode, padding_mode, ac);
            return;
        case DType::Float64:
            grid_sample2d_nchw_typed<double>(static_cast<const double*>(in),
                                             static_cast<const double*>(grid),
                                             static_cast<double*>(out), N, C, H, W, oH, oW,
                                             interpolation_mode, padding_mode, ac);
            return;
        default:
            TP_THROW(NotImplementedError,
                     "grid_sampler_2d strided-channel frame: unsupported dtype");
    }
}

}  // namespace

// DEFAULT/AVX2 tier copies take the REGISTER_DISPATCH slots; the AVX512 tier
// copy uses the opt-in AVX512 slot instead (same as the pooling kernels).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(grid_sample2d_cl_stub, &grid_sample2d_cl_impl);
REGISTER_DISPATCH(grid_sample2d_backward_cl_stub, &grid_sample2d_backward_cl_impl);
REGISTER_DISPATCH(grid_sample2d_nchw_stub, &grid_sample2d_nchw_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(grid_sample2d_cl_stub, &grid_sample2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(grid_sample2d_backward_cl_stub, &grid_sample2d_backward_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(grid_sample2d_nchw_stub, &grid_sample2d_nchw_impl);
#endif

}  // namespace cpu
}  // namespace tensorplay
