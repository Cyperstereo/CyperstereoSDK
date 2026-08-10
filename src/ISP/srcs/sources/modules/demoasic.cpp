/**
 * @file demoasic.cpp
 * @brief Direction-aware Bayer demosaic with colour-difference interpolation.
 *
 * Green at red/blue sites follows the lower Hamilton-Adams directional
 * gradient. R and B are then reconstructed from R-G/B-G differences, which
 * tracks luminance edges substantially better than independent bilinear
 * channel interpolation and reduces zippering/false colour at fine detail.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "modules/modules.h"

#define MOD_NAME "demoasic"

namespace {

inline int AbsInt(int value)
{
    return value < 0 ? -value : value;
}

inline int Average2Signed(int a, int b)
{
    const int sum = a + b;
    return sum >= 0 ? (sum + 1) >> 1 : -((-sum + 1) >> 1);
}

inline int Average4Signed(int a, int b, int c, int d)
{
    const int sum = a + b + c + d;
    return sum >= 0 ? (sum + 2) >> 2 : -((-sum + 2) >> 2);
}

inline PixelCfaTypes PixelType(CfaTypes cfa, int x, int y)
{
    return kPixelCfaLut[static_cast<int>(cfa)][x & 1][y & 1];
}

int Demoasic(Frame *frame, const IspPrms *isp_prm)
{
    if (frame == nullptr || isp_prm == nullptr) {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    const int width = frame->info.width;
    const int height = frame->info.height;
    const int max_value = frame->info.max_val;
    const CfaTypes cfa = frame->info.cfa;
    const int32_t *raw =
        reinterpret_cast<const int32_t *>(frame->data.raw_s32_i);
    int32_t *bgr = reinterpret_cast<int32_t *>(frame->data.bgr_s32_o);

    if (width < 3 || height < 3) {
        for (int i = 0; i < width * height; ++i) {
            bgr[3 * i + 0] = raw[i];
            bgr[3 * i + 1] = raw[i];
            bgr[3 * i + 2] = raw[i];
        }
        SwapMem<void>(frame->data.bgr_s32_o, frame->data.bgr_s32_i);
        return 0;
    }

    const size_t pixels = static_cast<size_t>(width) * height;
    static thread_local std::vector<int32_t> green;
    green.resize(pixels);

    // First reconstruct green. At R/B sites, compare horizontal and vertical
    // gradients including the same-colour samples two pixels away.
    for (int y = 0; y < height; ++y) {
        const int row = y * width;
        for (int x = 0; x < width; ++x) {
            const int i = row + x;
            const PixelCfaTypes type = PixelType(cfa, x, y);
            if (type == PixelCfaTypes::GR || type == PixelCfaTypes::GB) {
                green[i] = raw[i];
                continue;
            }

            if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
                const int center = raw[i];
                const int gh = (raw[i - 1] + raw[i + 1] + 1) >> 1;
                const int gv = (raw[i - width] + raw[i + width] + 1) >> 1;
                const int grad_h = AbsInt(raw[i - 1] - raw[i + 1]) +
                    AbsInt(2 * center - raw[i - 2] - raw[i + 2]);
                const int grad_v = AbsInt(raw[i - width] - raw[i + width]) +
                    AbsInt(2 * center - raw[i - 2 * width] -
                           raw[i + 2 * width]);
                if (grad_h < grad_v)
                    green[i] = gh;
                else if (grad_v < grad_h)
                    green[i] = gv;
                else
                    green[i] = (gh + gv + 1) >> 1;
            } else {
                int sum = 0;
                int count = 0;
                if (x > 0) { sum += raw[i - 1]; ++count; }
                if (x + 1 < width) { sum += raw[i + 1]; ++count; }
                if (y > 0) { sum += raw[i - width]; ++count; }
                if (y + 1 < height) { sum += raw[i + width]; ++count; }
                green[i] = count > 0 ? (sum + count / 2) / count : raw[i];
            }
            ClipMinMax(green[i], max_value, 0);
        }
    }

    // Then reconstruct red/blue through colour differences to the already
    // edge-aware green plane. The same green edge is shared by both chroma
    // components, preventing the independent-channel phase errors of the
    // previous bilinear implementation.
    for (int y = 1; y < height - 1; ++y) {
        const int row = y * width;
        for (int x = 1; x < width - 1; ++x) {
            const int i = row + x;
            const int g = green[i];
            int r = g;
            int b = g;
            const PixelCfaTypes type = PixelType(cfa, x, y);

            if (type == PixelCfaTypes::R) {
                r = raw[i];
                b = g + Average4Signed(
                    raw[i - width - 1] - green[i - width - 1],
                    raw[i - width + 1] - green[i - width + 1],
                    raw[i + width - 1] - green[i + width - 1],
                    raw[i + width + 1] - green[i + width + 1]);
            } else if (type == PixelCfaTypes::B) {
                b = raw[i];
                r = g + Average4Signed(
                    raw[i - width - 1] - green[i - width - 1],
                    raw[i - width + 1] - green[i - width + 1],
                    raw[i + width - 1] - green[i + width - 1],
                    raw[i + width + 1] - green[i + width + 1]);
            } else if (type == PixelCfaTypes::GR) {
                b = g + Average2Signed(raw[i - 1] - green[i - 1],
                                       raw[i + 1] - green[i + 1]);
                r = g + Average2Signed(raw[i - width] - green[i - width],
                                       raw[i + width] - green[i + width]);
            } else {  // PixelCfaTypes::GB
                r = g + Average2Signed(raw[i - 1] - green[i - 1],
                                       raw[i + 1] - green[i + 1]);
                b = g + Average2Signed(raw[i - width] - green[i - width],
                                       raw[i + width] - green[i + width]);
            }

            ClipMinMax(b, max_value, 0);
            ClipMinMax(r, max_value, 0);
            bgr[3 * i + 0] = b;
            bgr[3 * i + 1] = g;
            bgr[3 * i + 2] = r;
        }
    }

    // Match OpenCV's practical border policy: replicate the nearest valid
    // demosaiced row/column instead of leaving a coloured CFA border.
    for (int x = 1; x < width - 1; ++x) {
        for (int c = 0; c < 3; ++c) {
            bgr[3 * x + c] = bgr[3 * (width + x) + c];
            bgr[3 * ((height - 1) * width + x) + c] =
                bgr[3 * ((height - 2) * width + x) + c];
        }
    }
    for (int y = 0; y < height; ++y) {
        const int row = y * width;
        for (int c = 0; c < 3; ++c) {
            bgr[3 * row + c] = bgr[3 * (row + 1) + c];
            bgr[3 * (row + width - 1) + c] =
                bgr[3 * (row + width - 2) + c];
        }
    }

    SwapMem<void>(frame->data.bgr_s32_o, frame->data.bgr_s32_i);
    return 0;
}

}  // namespace

void RegisterDemoasicMod()
{
    IspModule mod;
    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;
    mod.in_domain = ColorDomains::RAW;
    mod.out_domain = ColorDomains::BGR;
    mod.name = MOD_NAME;
    mod.run_function = Demoasic;
    RegisterIspModule(mod);
}
