/**
 * @file bayernr.cpp
 * @brief Light edge-preserving noise reduction on Bayer same-colour samples.
 *
 * Samples two pixels away are on the same CFA sub-lattice (including the two
 * distinct green phases). Only neighbours close to the centre sample take
 * part, and the result is blended lightly. The module therefore reduces RAW
 * read/shot noise before WB and demosaic amplify it without mixing colours.
 */

#include <algorithm>
#include <cstdint>

#include "modules/modules.h"

#define MOD_NAME "bayernr"

namespace {

inline int AbsInt(int value)
{
    return value < 0 ? -value : value;
}

inline int RoundShift8Signed(int value)
{
    return value >= 0 ? (value + 128) >> 8 : -((-value + 128) >> 8);
}

static constexpr int kReciprocalQ16[9] = {
    0, 65536, 32768, 21845, 16384, 13107, 10923, 9362, 8192};

int BayerNr(Frame *frame, const IspPrms *isp_prm)
{
    if (frame == nullptr || isp_prm == nullptr) {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    const int width = frame->info.width;
    const int height = frame->info.height;
    const int max_value = frame->info.max_val;
    const BayerNrPrms &p = isp_prm->bayer_nr_prms;
    const int center_weight =
        (std::max)(1, (std::min)(p.center_weight, 4));
    const int strength = (std::max)(0, (std::min)(p.strength_q8, 256));

    const int32_t *src =
        reinterpret_cast<const int32_t *>(frame->data.raw_s32_i);
    int32_t *dst = reinterpret_cast<int32_t *>(frame->data.raw_s32_o);
    const size_t pixels = static_cast<size_t>(width) * height;
    std::copy(src, src + pixels, dst);

    // +/-2 stays on exactly the same R/Gr/Gb/B sub-lattice.
    for (int y = 2; y < height - 2; ++y) {
        const int row = y * width;
        for (int x = 2; x < width - 2; ++x) {
            const int i = row + x;
            const int center = src[i];
            int threshold = p.threshold_base +
                            ((center * p.threshold_signal_q8 + 128) >> 8);
            threshold = (std::max)(threshold, 1);

            int sum = center_weight * center;
            int weight = center_weight;
            int sample = src[i - 2];
            int accept = AbsInt(sample - center) <= threshold;
            sum += sample * accept;
            weight += accept;
            sample = src[i + 2];
            accept = AbsInt(sample - center) <= threshold;
            sum += sample * accept;
            weight += accept;
            sample = src[i - 2 * width];
            accept = AbsInt(sample - center) <= threshold;
            sum += sample * accept;
            weight += accept;
            sample = src[i + 2 * width];
            accept = AbsInt(sample - center) <= threshold;
            sum += sample * accept;
            weight += accept;

            if (weight == center_weight)
                continue;
            const int filtered = static_cast<int>(
                (static_cast<int64_t>(sum) * kReciprocalQ16[weight] + 32768) >>
                16);
            int value = center +
                        RoundShift8Signed((filtered - center) * strength);
            ClipMinMax(value, max_value, 0);
            dst[i] = value;
        }
    }

    SwapMem<void>(frame->data.raw_s32_i, frame->data.raw_s32_o);
    return 0;
}

}  // namespace

void RegisterBayerNrMod()
{
    IspModule mod;
    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;
    mod.in_domain = ColorDomains::RAW;
    mod.out_domain = ColorDomains::RAW;
    mod.name = MOD_NAME;
    mod.run_function = BayerNr;
    RegisterIspModule(mod);
}
