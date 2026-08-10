/**
 * @file chromanr.cpp
 * @brief Hue-safe joint chroma denoise in the linear BGR domain.
 *
 * Dr=R-G and Db=B-G are always filtered with the SAME edge weights. This
 * avoids the hue rotation caused by filtering U/V (or Cr/Cb) independently.
 * Rec.601 luminance is kept fixed during reconstruction, saturated colours
 * receive only a light blend, and their hue rotation is capped at about 2
 * degrees. No BGR<->YUV uint8 round trip is involved.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "modules/modules.h"

#define MOD_NAME "chromanr"

namespace {

inline int AbsInt(int value)
{
    return value < 0 ? -value : value;
}

inline int RoundShift8Signed(int value)
{
    return value >= 0 ? (value + 128) >> 8 : -((-value + 128) >> 8);
}

inline int RoundShift4Signed(int value)
{
    return value >= 0 ? (value + 8) >> 4 : -((-value + 8) >> 4);
}

inline int RoundDivSigned(int value, int divisor)
{
    return value >= 0 ? (value + divisor / 2) / divisor
                      : -((-value + divisor / 2) / divisor);
}

inline int Luma601(const int32_t *pixel)
{
    const int b = pixel[0];
    const int g = pixel[1];
    const int r = pixel[2];
    return (29 * b + 150 * g + 77 * r + 128) >> 8;
}

// Fast hue guard. If a filtered saturated colour would rotate farther than
// the configured angle, retain the original chroma vector. This is stricter
// and considerably cheaper than projecting every pixel with 64-bit divides.
inline bool HueWithinLimit(int dr0, int db0, int tan_limit_q8,
                           int dr, int db)
{
    const int64_t dot = static_cast<int64_t>(dr0) * dr +
                        static_cast<int64_t>(db0) * db;
    if (dot <= 0)
        return false;
    const int64_t cross = static_cast<int64_t>(dr0) * db -
                          static_cast<int64_t>(db0) * dr;
    const int64_t abs_cross = cross < 0 ? -cross : cross;
    return abs_cross * 256 <= dot * tan_limit_q8;
}

// Scale all B/G/R offsets from the same luminance by one factor when the
// reconstructed colour leaves the valid gamut. Independent channel clipping
// would rotate hue at saturated colours.
inline void GamutClampPreserveHue(int y, int max_value, int &b, int &g, int &r)
{
    int scale_q8 = 256;
    const int channels[3] = {b, g, r};
    for (int i = 0; i < 3; ++i) {
        const int delta = channels[i] - y;
        if (channels[i] > max_value && delta > 0) {
            scale_q8 = (std::min)(scale_q8,
                static_cast<int>(static_cast<int64_t>(max_value - y) * 256 /
                                 delta));
        } else if (channels[i] < 0 && delta < 0) {
            scale_q8 = (std::min)(scale_q8,
                static_cast<int>(static_cast<int64_t>(y) * 256 / -delta));
        }
    }

    if (scale_q8 < 256) {
        b = y + RoundShift8Signed((b - y) * scale_q8);
        g = y + RoundShift8Signed((g - y) * scale_q8);
        r = y + RoundShift8Signed((r - y) * scale_q8);
    }
    ClipMinMax(b, max_value, 0);
    ClipMinMax(g, max_value, 0);
    ClipMinMax(r, max_value, 0);
}

int ChromaNr(Frame *frame, const IspPrms *isp_prm)
{
    if (frame == nullptr || isp_prm == nullptr) {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    const int width = frame->info.width;
    const int height = frame->info.height;
    const int max_value = frame->info.max_val;
    const ChromaNrPrms &p = isp_prm->chroma_nr_prms;
    const int ty = (std::max)(p.luma_threshold, 1);
    const int tc = (std::max)(p.chroma_threshold, 1);

    const int32_t *src =
        reinterpret_cast<const int32_t *>(frame->data.bgr_s32_i);
    int32_t *dst = reinterpret_cast<int32_t *>(frame->data.bgr_s32_o);

    // Precompute the three scalar planes once. The original implementation
    // recalculated Y/Dr/Db nine times per pixel and spent most of its runtime
    // there. int16 is sufficient for the current 10-bit working range.
    const size_t pixels = static_cast<size_t>(width) * height;
    static thread_local std::vector<int16_t> luma;
    static thread_local std::vector<int16_t> dr_plane;
    static thread_local std::vector<int16_t> db_plane;
    luma.resize(pixels);
    dr_plane.resize(pixels);
    db_plane.resize(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        const int32_t *pixel = src + 3 * i;
        luma[i] = static_cast<int16_t>(Luma601(pixel));
        dr_plane[i] = static_cast<int16_t>(pixel[2] - pixel[1]);
        db_plane[i] = static_cast<int16_t>(pixel[0] - pixel[1]);
    }

    // Saturation-to-strength is a tiny LUT, avoiding a divide for every
    // pixel while keeping the configured smooth transition.
    static thread_local std::vector<int16_t> strength_lut;
    const int max_saturation = 2 * max_value;
    strength_lut.resize(static_cast<size_t>(max_saturation) + 1);
    const int neutral =
        (std::max)(0, (std::min)(p.neutral_strength_q8, 256));
    const int saturated =
        (std::max)(0, (std::min)(p.saturated_strength_q8, 256));
    const int sat_low = (std::max)(p.saturation_low, 0);
    const int sat_high = (std::max)(p.saturation_high, sat_low + 1);
    for (int saturation = 0; saturation <= max_saturation; ++saturation) {
        int strength;
        if (saturation <= sat_low)
            strength = neutral;
        else if (saturation >= sat_high)
            strength = saturated;
        else
            strength = neutral +
                       (saturated - neutral) * (saturation - sat_low) /
                           (sat_high - sat_low);
        strength_lut[saturation] = static_cast<int16_t>(strength);
    }

    // Copy the frame first. Interior pixels that fail the edge/chroma gate or
    // whose rounded result is unchanged then need no reconstruction at all.
    std::copy(src, src + 3 * pixels, dst);

    // Full-resolution 3x3 [1 2 1] x [1 2 1] candidate. Edge awareness is
    // deliberately implemented as one joint accept/reject decision instead
    // of eight per-neighbour branches: it is much cheaper, and Dr/Db still
    // always see exactly the same kernel and gate.
    for (int y_pos = 1; y_pos < height - 1; ++y_pos) {
        const int row = y_pos * width;
        for (int x_pos = 1; x_pos < width - 1; ++x_pos) {
            const int pixel_idx = row + x_pos;

            const int y0 = luma[pixel_idx];
            const int dr0 = dr_plane[pixel_idx];
            const int db0 = db_plane[pixel_idx];
            const int left = pixel_idx - 1;
            const int right = pixel_idx + 1;
            const int up = pixel_idx - width;
            const int down = pixel_idx + width;

            const int up_left = up - 1;
            const int up_right = up + 1;
            const int down_left = down - 1;
            const int down_right = down + 1;

            // Shared 3x3 kernel for Dr and Db. Keep the neighbour-only sum as
            // well: it is a robust local colour base for isolated speckles,
            // while the ordinary filtered candidate retains the centre weight.
            const int neighbor_dr_sum =
                2 * (static_cast<int>(dr_plane[left]) + dr_plane[right] +
                     dr_plane[up] + dr_plane[down]) +
                dr_plane[up_left] + dr_plane[up_right] +
                dr_plane[down_left] + dr_plane[down_right];
            const int neighbor_db_sum =
                2 * (static_cast<int>(db_plane[left]) + db_plane[right] +
                     db_plane[up] + db_plane[down]) +
                db_plane[up_left] + db_plane[up_right] +
                db_plane[down_left] + db_plane[down_right];
            const int filtered_dr = RoundShift4Signed(
                4 * dr0 + neighbor_dr_sum);
            const int filtered_db = RoundShift4Signed(
                4 * db0 + neighbor_db_sum);

            // Special case for the failure mode seen on white walls at high
            // gain. The original chroma-delta guard rejected large isolated
            // deviations, preserving exactly the red/blue speckles we want to
            // remove. Only allow the stronger correction when all surrounding
            // pixels are luma-flat and their common colour is near-neutral.
            const int base_dr = RoundDivSigned(neighbor_dr_sum, 12);
            const int base_db = RoundDivSigned(neighbor_db_sum, 12);
            const int base_chroma = AbsInt(base_dr) + AbsInt(base_db);
            const int outlier = AbsInt(dr0 - base_dr) + AbsInt(db0 - base_db);
            if (base_chroma <= (std::max)(p.neutral_base_chroma, 0) &&
                outlier >= (std::max)(p.neutral_outlier_threshold, 1)) {
                const int neighbours[8] = {
                    left, right, up, down,
                    up_left, up_right, down_left, down_right};
                int luma_min = luma[neighbours[0]];
                int luma_max = luma_min;
                for (int n = 1; n < 8; ++n) {
                    const int value = luma[neighbours[n]];
                    luma_min = (std::min)(luma_min, value);
                    luma_max = (std::max)(luma_max, value);
                }
                const int neighbor_luma_sum =
                    2 * (static_cast<int>(luma[left]) + luma[right] +
                         luma[up] + luma[down]) +
                    luma[up_left] + luma[up_right] +
                    luma[down_left] + luma[down_right];
                const int base_y = (neighbor_luma_sum + 6) / 12;
                const int flat_limit =
                    (std::max)(p.neutral_luma_range, 1);
                if (luma_max - luma_min <= flat_limit &&
                    AbsInt(y0 - base_y) <= flat_limit) {
                    const int alpha_q8 = (std::max)(0, (std::min)(
                        p.neutral_outlier_strength_q8, 256));
                    const int dr = dr0 + RoundShift8Signed(
                        (base_dr - dr0) * alpha_q8);
                    const int db = db0 + RoundShift8Signed(
                        (base_db - db0) * alpha_q8);
                    int g = y0 - RoundShift8Signed(77 * dr + 29 * db);
                    int r = g + dr;
                    int b = g + db;
                    GamutClampPreserveHue(y0, max_value, b, g, r);
                    dst[3 * pixel_idx + 0] = b;
                    dst[3 * pixel_idx + 1] = g;
                    dst[3 * pixel_idx + 2] = r;
                    continue;
                }
            }

            const int max_luma_delta = (std::max)(
                (std::max)(AbsInt(static_cast<int>(luma[left]) - y0),
                           AbsInt(static_cast<int>(luma[right]) - y0)),
                (std::max)(AbsInt(static_cast<int>(luma[up]) - y0),
                           AbsInt(static_cast<int>(luma[down]) - y0)));
            if (max_luma_delta > ty)
                continue;
            if (AbsInt(filtered_dr - dr0) + AbsInt(filtered_db - db0) > tc)
                continue;

            const int saturation = AbsInt(dr0) + AbsInt(db0);
            const int alpha_q8 = strength_lut[
                (std::min)(saturation, max_saturation)];
            int dr = dr0 +
                     RoundShift8Signed((filtered_dr - dr0) * alpha_q8);
            int db = db0 +
                     RoundShift8Signed((filtered_db - db0) * alpha_q8);

            if (dr == dr0 && db == db0)
                continue;
            if (saturation >= sat_low &&
                !HueWithinLimit(dr0, db0,
                                (std::max)(p.max_hue_tan_q8, 0), dr, db))
                continue;

            // Reconstruct BGR while keeping the original Rec.601 luminance.
            int g = y0 - RoundShift8Signed(77 * dr + 29 * db);
            int r = g + dr;
            int b = g + db;
            GamutClampPreserveHue(y0, max_value, b, g, r);
            dst[3 * pixel_idx + 0] = b;
            dst[3 * pixel_idx + 1] = g;
            dst[3 * pixel_idx + 2] = r;
        }
    }

    SwapMem<void>(frame->data.bgr_s32_i, frame->data.bgr_s32_o);
    return 0;
}

}  // namespace

void RegisterChromaNrMod()
{
    IspModule mod;
    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;
    mod.in_domain = ColorDomains::BGR;
    mod.out_domain = ColorDomains::BGR;
    mod.name = MOD_NAME;
    mod.run_function = ChromaNr;
    RegisterIspModule(mod);
}
