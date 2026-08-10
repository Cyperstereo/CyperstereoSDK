/**
 * @file rgbgamma.cpp
 * @author joker.mao (joker_mao@163.com)
 * @brief
 * @version 0.1
 * @date 2023-08-10
 *
 * Copyright (c) of ADAS_EYES 2023
 *
 */

#include <algorithm>

#include "modules/modules.h"

#define MOD_NAME "rgbgamma"

static int RgbGamma(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }
    int pixel_idx = 0;

    const auto &gamma_prm = isp_prm->rgb_gamma;

    float step_coff = (float)(gamma_prm.nums - 1) / (1 << gamma_prm.in_bits);
    float out_max = (1 << gamma_prm.out_bits) - 1;

    int32_t *bgr_i = reinterpret_cast<int32_t *>(frame->data.bgr_s32_i);
    int32_t *bgr_o = reinterpret_cast<int32_t *>(frame->data.bgr_s32_o);

    FOR_ITER(h, frame->info.height)
    {
        FOR_ITER(w, frame->info.width)
        {
            pixel_idx = h * frame->info.width + w;

            const int b = bgr_i[3 * pixel_idx + 0];
            const int g = bgr_i[3 * pixel_idx + 1];
            const int r = bgr_i[3 * pixel_idx + 2];
            // Apply the LUT to Rec.601 luminance, then multiply every channel
            // by the SAME gain.  Per-channel gamma changes B:G:R in dark,
            // saturated pixels; this preserves their hue and saturation.
            const int y = (29 * b + 150 * g + 77 * r) >> 8;
            if (y <= 0) {
                bgr_o[3 * pixel_idx + 0] = 0;
                bgr_o[3 * pixel_idx + 1] = 0;
                bgr_o[3 * pixel_idx + 2] = 0;
                continue;
            }

            const float curve_id_f = y * step_coff;
            const int curve_id = (std::min)(
                (std::max)(static_cast<int>(curve_id_f), 0),
                gamma_prm.nums - 2);
            const float scale =
                (curve_id_f - curve_id) *
                    (gamma_prm.curve[curve_id + 1] - gamma_prm.curve[curve_id]) +
                gamma_prm.curve[curve_id];
            float gain = out_max * scale / y;

            // Avoid clipping one channel independently of the others: that
            // would create a hue shift.  It also protects noisy near-saturated
            // pixels from further amplification.
            const int max_color = (std::max)(b, (std::max)(g, r));
            if (max_color > 0)
                gain = (std::min)(gain, out_max / max_color);

            FOR_ITER(color_idx, 3)
            {
                int color = static_cast<int>(
                    bgr_i[3 * pixel_idx + color_idx] * gain + 0.5f);
                ClipMinMax(color, static_cast<int>(out_max), 0);
                bgr_o[3 * pixel_idx + color_idx] = color;
            }
        }
    }

    SwapMem<void>(frame->data.bgr_s32_i, frame->data.bgr_s32_o);

    return 0;
}

void RegisterRgbGammaMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;

    mod.in_domain = ColorDomains::BGR;
    mod.out_domain = ColorDomains::BGR;

    mod.name = MOD_NAME;
    mod.run_function = RgbGamma;

    RegisterIspModule(mod);
}
