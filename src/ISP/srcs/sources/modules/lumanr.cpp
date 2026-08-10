/**
 * @file lumanr.cpp
 * @brief Gain-adaptive guided luminance denoise in linear BGR.
 *
 * The module never converts through 8-bit YUV. It filters only BT.601 luma
 * and applies one common additive delta to B, G and R, so Dr=R-G and Db=B-G
 * are bit-exact unless the common delta has to be reduced to stay in gamut.
 */

#include <algorithm>
#include <cstdint>

#include <opencv2/imgproc/imgproc.hpp>

#include "modules/modules.h"

#define MOD_NAME "lumanr"

namespace {

inline int Luma601(const int32_t *pixel)
{
    return (29 * pixel[0] + 150 * pixel[1] + 77 * pixel[2] + 128) >> 8;
}

inline int RoundShift8Signed(int value)
{
    return value >= 0 ? (value + 128) >> 8 : -((-value + 128) >> 8);
}

int LumaNr(Frame *frame, const IspPrms *isp_prm)
{
    if (frame == nullptr || isp_prm == nullptr) {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    const int width = frame->info.width;
    const int height = frame->info.height;
    const int max_value = frame->info.max_val;
    const LumaNrPrms &p = isp_prm->luma_nr_prms;
    const float eps = static_cast<float>((std::max)(p.eps, 1));
    const int strength =
        (std::max)(0, (std::min)(p.strength_q8, 256));

    const int32_t *src =
        reinterpret_cast<const int32_t *>(frame->data.bgr_s32_i);
    int32_t *dst = reinterpret_cast<int32_t *>(frame->data.bgr_s32_o);

    // Per-camera ISP calls run on separate worker threads. Reusing these Mats
    // avoids allocations on every frame while keeping the module thread-safe.
    static thread_local cv::Mat y16, yq16, yq32, sq32;
    static thread_local cv::Mat mean_i, mean_ii;
    static thread_local cv::Mat a_q12, b_q4, mean_a_q12, mean_b_q4;
    y16.create(height, width, CV_16U);
    for (int y = 0; y < height; ++y) {
        uint16_t *dy = y16.ptr<uint16_t>(y);
        const int32_t *s = src + static_cast<size_t>(y) * width * 3;
        for (int x = 0; x < width; ++x)
            dy[x] = static_cast<uint16_t>(Luma601(s + 3 * x));
    }

    // A 4x4 area decimation is both the antialias prefilter and the guided
    // statistics source. It avoids a full-resolution 16-bit Gaussian pass;
    // the following 3x3 box still provides about 12x12 full-res support.
    const cv::Size quarter((width + 3) / 4, (height + 3) / 4);
    cv::resize(y16, yq16, quarter, 0.0, 0.0, cv::INTER_AREA);
    yq16.convertTo(yq32, CV_32F);

    const cv::Size stats_window(3, 3);
    cv::boxFilter(yq32, mean_i, CV_32F, stats_window, cv::Point(-1, -1),
                  true, cv::BORDER_REFLECT);
    cv::multiply(yq32, yq32, sq32);
    cv::boxFilter(sq32, mean_ii, CV_32F, stats_window, cv::Point(-1, -1),
                  true, cv::BORDER_REFLECT);

    a_q12.create(quarter, CV_16U);
    b_q4.create(quarter, CV_16U);
    for (int y = 0; y < quarter.height; ++y) {
        const float *mi = mean_i.ptr<float>(y);
        const float *mii = mean_ii.ptr<float>(y);
        uint16_t *pa = a_q12.ptr<uint16_t>(y);
        uint16_t *pb = b_q4.ptr<uint16_t>(y);
        for (int x = 0; x < quarter.width; ++x) {
            float variance = mii[x] - mi[x] * mi[x];
            if (variance < 0.0f) variance = 0.0f;
            const float coeff = variance / (variance + eps);
            pa[x] = static_cast<uint16_t>(coeff * 4096.0f + 0.5f);
            pb[x] = static_cast<uint16_t>(
                (1.0f - coeff) * mi[x] * 16.0f + 0.5f);
        }
    }
    cv::boxFilter(a_q12, mean_a_q12, -1, stats_window, cv::Point(-1, -1),
                  true, cv::BORDER_REFLECT);
    cv::boxFilter(b_q4, mean_b_q4, -1, stats_window, cv::Point(-1, -1),
                  true, cv::BORDER_REFLECT);

    for (int y = 0; y < height; ++y) {
        const int qy = (std::min)(y >> 2, quarter.height - 1);
        const uint16_t *pa = mean_a_q12.ptr<uint16_t>(qy);
        const uint16_t *pb = mean_b_q4.ptr<uint16_t>(qy);
        const uint16_t *py = y16.ptr<uint16_t>(y);
        const int32_t *s = src + static_cast<size_t>(y) * width * 3;
        int32_t *d = dst + static_cast<size_t>(y) * width * 3;
        for (int x = 0; x < width; ++x) {
            const int qx = (std::min)(x >> 2, quarter.width - 1);
            const int y0 = py[x];
            int yf = (static_cast<int>(pa[qx]) * y0 +
                      (static_cast<int>(pb[qx]) << 8) + 2048) >> 12;
            yf = (std::max)(0, (std::min)(yf, max_value));
            int delta = RoundShift8Signed((yf - y0) * strength);

            const int blue = s[3 * x + 0];
            const int green = s[3 * x + 1];
            const int red = s[3 * x + 2];
            const int channel_min =
                (std::min)(blue, (std::min)(green, red));
            const int channel_max =
                (std::max)(blue, (std::max)(green, red));
            // Restrict the one shared delta instead of clipping channels
            // independently. Channel differences, and therefore hue, remain
            // unchanged even next to saturated highlights or deep shadows.
            delta = (std::max)(-channel_min,
                               (std::min)(delta, max_value - channel_max));
            d[3 * x + 0] = blue + delta;
            d[3 * x + 1] = green + delta;
            d[3 * x + 2] = red + delta;
        }
    }

    SwapMem<void>(frame->data.bgr_s32_i, frame->data.bgr_s32_o);
    return 0;
}

}  // namespace

void RegisterLumaNrMod()
{
    IspModule mod;
    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;
    mod.in_domain = ColorDomains::BGR;
    mod.out_domain = ColorDomains::BGR;
    mod.name = MOD_NAME;
    mod.run_function = LumaNr;
    RegisterIspModule(mod);
}
