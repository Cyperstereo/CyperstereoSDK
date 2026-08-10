/**
 * @file falsecolor.cpp
 * @brief Texture-gated, hue-preserving false-colour suppression.
 *
 * Dense achromatic texture close to the Bayer colour Nyquist limit (fan
 * grilles, fabric weave) can demosaic into coherent cyan/magenta bands that
 * a small chroma denoiser regards as real colour. This module detects that
 * failure mode on a 4x4 statistics grid. A wide local average supplies the
 * stable base colour, then the Dr=R-G and Db=B-G residuals around that base
 * are compressed by one common factor. A warm roof therefore stays warm;
 * only its alternating orange/cyan error is reduced.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "modules/modules.h"

#define MOD_NAME "falsecolor"

namespace {

inline int AbsInt(int value)
{
    return value < 0 ? -value : value;
}

inline int RoundDivSigned(int value, int divisor)
{
    return value >= 0 ? (value + divisor / 2) / divisor
                      : -((-value + divisor / 2) / divisor);
}

inline int RoundShift8Signed(int value)
{
    return value >= 0 ? (value + 128) >> 8 : -((-value + 128) >> 8);
}

inline int Luma601(const int32_t *pixel)
{
    return (29 * pixel[0] + 150 * pixel[1] + 77 * pixel[2] + 128) >> 8;
}

inline void GamutClampPreserveHue(int y, int max_value, int &b, int &g, int &r)
{
    if (b >= 0 && b <= max_value && g >= 0 && g <= max_value &&
        r >= 0 && r <= max_value)
        return;

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
    b = y + RoundShift8Signed((b - y) * scale_q8);
    g = y + RoundShift8Signed((g - y) * scale_q8);
    r = y + RoundShift8Signed((r - y) * scale_q8);
    ClipMinMax(b, max_value, 0);
    ClipMinMax(g, max_value, 0);
    ClipMinMax(r, max_value, 0);
}

int FalseColor(Frame *frame, const IspPrms *isp_prm)
{
    if (frame == nullptr || isp_prm == nullptr) {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    const int width = frame->info.width;
    const int height = frame->info.height;
    const int max_value = frame->info.max_val;
    const FalseColorPrms &p = isp_prm->false_color_prms;
    const int32_t *src =
        reinterpret_cast<const int32_t *>(frame->data.bgr_s32_i);
    int32_t *dst = reinterpret_cast<int32_t *>(frame->data.bgr_s32_o);
    const size_t pixels = static_cast<size_t>(width) * height;
    std::copy(src, src + 3 * pixels, dst);

    static thread_local std::vector<int16_t> luma;
    luma.resize(pixels);
    for (size_t i = 0; i < pixels; ++i)
        luma[i] = static_cast<int16_t>(Luma601(src + 3 * i));

    // Quarter-grid block statistics. A later 3x3 average gives an effective
    // ~12x12 support, large enough to recognise repeated texture instead of
    // treating every ordinary edge as false colour.
    constexpr int kBlock = 4;
    const int q_width = (width + kBlock - 1) / kBlock;
    const int q_height = (height + kBlock - 1) / kBlock;
    const size_t q_pixels = static_cast<size_t>(q_width) * q_height;
    static thread_local std::vector<int16_t> q_y, q_dr, q_db, q_hf;
    static thread_local std::vector<int16_t> base_dr, base_db;
    static thread_local std::vector<uint8_t> attenuation;
    q_y.resize(q_pixels);
    q_dr.resize(q_pixels);
    q_db.resize(q_pixels);
    q_hf.resize(q_pixels);
    base_dr.assign(q_pixels, 0);
    base_db.assign(q_pixels, 0);
    attenuation.assign(q_pixels, 0);

    for (int qy = 0; qy < q_height; ++qy) {
        const int y0 = qy * kBlock;
        const int y1 = (std::min)(y0 + kBlock, height);
        for (int qx = 0; qx < q_width; ++qx) {
            const int x0 = qx * kBlock;
            const int x1 = (std::min)(x0 + kBlock, width);
            int sum_y = 0;
            int sum_dr = 0;
            int sum_db = 0;
            int sum_hf = 0;
            int count = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const int i = y * width + x;
                    const int32_t *pixel = src + 3 * i;
                    const int yy = luma[i];
                    sum_y += yy;
                    sum_dr += pixel[2] - pixel[1];
                    sum_db += pixel[0] - pixel[1];
                    if (x > 0 && x + 1 < width && y > 0 && y + 1 < height) {
                        const int lap = 4 * yy - luma[i - 1] - luma[i + 1] -
                                        luma[i - width] - luma[i + width];
                        sum_hf += (AbsInt(lap) + 2) >> 2;
                    }
                    ++count;
                }
            }
            const int qi = qy * q_width + qx;
            q_y[qi] = static_cast<int16_t>((sum_y + count / 2) / count);
            q_dr[qi] = static_cast<int16_t>(RoundDivSigned(sum_dr, count));
            q_db[qi] = static_cast<int16_t>(RoundDivSigned(sum_db, count));
            q_hf[qi] = static_cast<int16_t>((sum_hf + count / 2) / count);
        }
    }

    const int tex_low = (std::max)(p.texture_low, 0);
    const int tex_high = (std::max)(p.texture_high, tex_low + 1);
    const int neutral_low = (std::max)(p.neutral_chroma_low, 0);
    const int neutral_high =
        (std::max)(p.neutral_chroma_high, neutral_low + 1);
    const int max_attenuation =
        (std::max)(0, (std::min)(p.max_attenuation_q8, 256));

    for (int qy = 0; qy < q_height; ++qy) {
        for (int qx = 0; qx < q_width; ++qx) {
            int sum_y = 0;
            int sum_hf = 0;
            int texture_count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = qy + dy;
                if (ny < 0 || ny >= q_height) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = qx + dx;
                    if (nx < 0 || nx >= q_width) continue;
                    const int ni = ny * q_width + nx;
                    sum_y += q_y[ni];
                    sum_hf += q_hf[ni];
                    ++texture_count;
                }
            }

            const int mean_y =
                (sum_y + texture_count / 2) / texture_count;
            if (mean_y < p.min_luma)
                continue;
            const int texture =
                (sum_hf + texture_count / 2) / texture_count;
            if (texture <= tex_low)
                continue;
            int texture_q8 = texture >= tex_high
                ? 256 : (texture - tex_low) * 256 / (tex_high - tex_low);

            // A 7x7 quarter-grid window (~28x28 full resolution) is wider
            // than the colour-beat period on grilles and roof tiles. It
            // recovers the stable surface colour even when a 12x12 mean is
            // itself orange or cyan.
            int sum_dr = 0;
            int sum_db = 0;
            int chroma_count = 0;
            for (int dy = -3; dy <= 3; ++dy) {
                const int ny = qy + dy;
                if (ny < 0 || ny >= q_height) continue;
                for (int dx = -3; dx <= 3; ++dx) {
                    const int nx = qx + dx;
                    if (nx < 0 || nx >= q_width) continue;
                    const int ni = ny * q_width + nx;
                    sum_dr += q_dr[ni];
                    sum_db += q_db[ni];
                    ++chroma_count;
                }
            }
            const int mean_dr = RoundDivSigned(sum_dr, chroma_count);
            const int mean_db = RoundDivSigned(sum_db, chroma_count);
            const int mean_chroma = AbsInt(mean_dr) + AbsInt(mean_db);
            if (mean_chroma >= neutral_high)
                continue;
            const int neutral_q8 = mean_chroma <= neutral_low
                ? 256 : (neutral_high - mean_chroma) * 256 /
                            (neutral_high - neutral_low);

            int reduce_q8 =
                (max_attenuation * texture_q8 + 128) >> 8;
            reduce_q8 = (reduce_q8 * neutral_q8 + 128) >> 8;
            const int qi = qy * q_width + qx;
            base_dr[qi] = static_cast<int16_t>(mean_dr);
            base_db[qi] = static_cast<int16_t>(mean_db);
            attenuation[qi] =
                static_cast<uint8_t>((std::min)(reduce_q8, 255));
        }
    }

    for (int y = 0; y < height; ++y) {
        const int qy = y / kBlock;
        for (int x = 0; x < width; ++x) {
            const int qi = qy * q_width + x / kBlock;
            const int reduce_q8 = attenuation[qi];
            if (reduce_q8 == 0)
                continue;
            const int i = y * width + x;
            const int32_t *pixel = src + 3 * i;
            const int dr = pixel[2] - pixel[1];
            const int db = pixel[0] - pixel[1];
            const int keep_q8 = 256 - reduce_q8;
            const int mean_dr = base_dr[qi];
            const int mean_db = base_db[qi];
            const int new_dr = mean_dr +
                RoundShift8Signed((dr - mean_dr) * keep_q8);
            const int new_db = mean_db +
                RoundShift8Signed((db - mean_db) * keep_q8);
            const int yy = luma[i];
            int g = yy - RoundShift8Signed(77 * new_dr + 29 * new_db);
            int r = g + new_dr;
            int b = g + new_db;
            GamutClampPreserveHue(yy, max_value, b, g, r);
            dst[3 * i + 0] = b;
            dst[3 * i + 1] = g;
            dst[3 * i + 2] = r;
        }
    }

    SwapMem<void>(frame->data.bgr_s32_i, frame->data.bgr_s32_o);
    return 0;
}

}  // namespace

void RegisterFalseColorMod()
{
    IspModule mod;
    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;
    mod.in_domain = ColorDomains::BGR;
    mod.out_domain = ColorDomains::BGR;
    mod.name = MOD_NAME;
    mod.run_function = FalseColor;
    RegisterIspModule(mod);
}
