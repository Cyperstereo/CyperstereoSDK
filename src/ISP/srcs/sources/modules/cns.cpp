/**
 * @file cns.cpp
 * @author joker.mao (joker_mao@163.com)
 * @brief chrome noise signal filter
 * @version 0.1
 * @date 2023-08-27
 * Copyright (c) of ADAS_EYES 2023
 */

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "modules/modules.h"

#define MOD_NAME "cns"

// Chroma (U/V) median denoise. Median is the right tool for chroma noise:
// it removes the isolated colored specks the CCM amplifies without bleeding
// hue across edges the way a box/gaussian blur would.
//
// Radius: CYPERSTEREO_CNS_RADIUS (1..3, default 1 => 3x3). Larger removes
// heavier chroma noise at higher cost. Radius 1 uses a branch-free 9-element
// median network (no per-pixel std::sort), ~5-8x faster than the original.

// Median of 9 via the classic 19-comparison min/max network (Smith 1996).
static inline uint8_t Median9(uint8_t *p) {
#define PIX_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PIX_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PIX_SORT(a, b)                 \
  do {                                 \
    const uint8_t mn = PIX_MIN(a, b);  \
    const uint8_t mx = PIX_MAX(a, b);  \
    a = mn;                            \
    b = mx;                            \
  } while (0)
  PIX_SORT(p[1], p[2]); PIX_SORT(p[4], p[5]); PIX_SORT(p[7], p[8]);
  PIX_SORT(p[0], p[1]); PIX_SORT(p[3], p[4]); PIX_SORT(p[6], p[7]);
  PIX_SORT(p[1], p[2]); PIX_SORT(p[4], p[5]); PIX_SORT(p[7], p[8]);
  PIX_SORT(p[0], p[3]); PIX_SORT(p[5], p[8]); PIX_SORT(p[4], p[7]);
  PIX_SORT(p[3], p[6]); PIX_SORT(p[1], p[4]); PIX_SORT(p[2], p[5]);
  PIX_SORT(p[4], p[7]); PIX_SORT(p[4], p[2]); PIX_SORT(p[6], p[4]);
  PIX_SORT(p[4], p[2]);
  return p[4];
#undef PIX_SORT
#undef PIX_MAX
#undef PIX_MIN
}

static int CnsRadius() {
  static const int r = [] {
    if (const char *e = std::getenv("CYPERSTEREO_CNS_RADIUS")) {
      const int v = std::atoi(e);
      if (v >= 1 && v <= 3) return v;
    }
    return 1;
  }();
  return r;
}

static int Cns(Frame *frame, const IspPrms *isp_prm)
{
    if ((frame == nullptr) || (isp_prm == nullptr))
    {
        LOG(ERROR) << "input prms is null";
        return -1;
    }

    uint8_t *u_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.u);
    uint8_t *v_i = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_i.v);
    uint8_t *u_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.u);
    uint8_t *v_o = reinterpret_cast<uint8_t *>(frame->data.yuv_u8_o.v);

    const int width = frame->info.width;
    const int height = frame->info.height;
    const int r = CnsRadius();

    if (r == 1) {
        // Fast 3x3 path: branch-free median-of-9 network.
        uint8_t u[9], v[9];
        FOR_ITER(ih, height)
        {
            FOR_ITER(iw, width)
            {
                const int pixel_idx = ih * width + iw;
                if (iw < 1 || iw >= width - 1 || ih < 1 || ih >= height - 1) {
                    u_o[pixel_idx] = u_i[pixel_idx];
                    v_o[pixel_idx] = v_i[pixel_idx];
                    continue;
                }
                int s = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int row = (ih + dy) * width + iw;
                    u[s] = u_i[row - 1]; v[s] = v_i[row - 1]; ++s;
                    u[s] = u_i[row];     v[s] = v_i[row];     ++s;
                    u[s] = u_i[row + 1]; v[s] = v_i[row + 1]; ++s;
                }
                u_o[pixel_idx] = Median9(u);
                v_o[pixel_idx] = Median9(v);
            }
        }
    } else {
        // General (2r+1)x(2r+1) path: partial-selection median via nth_element.
        const int win = (2 * r + 1) * (2 * r + 1);
        const int center = win >> 1;
        std::vector<uint8_t> u(win), v(win);
        FOR_ITER(ih, height)
        {
            FOR_ITER(iw, width)
            {
                const int pixel_idx = ih * width + iw;
                if (iw < r || iw >= width - r || ih < r || ih >= height - r) {
                    u_o[pixel_idx] = u_i[pixel_idx];
                    v_o[pixel_idx] = v_i[pixel_idx];
                    continue;
                }
                int s = 0;
                for (int dy = -r; dy <= r; ++dy) {
                    const int row = (ih + dy) * width + iw;
                    for (int dx = -r; dx <= r; ++dx) {
                        u[s] = u_i[row + dx];
                        v[s] = v_i[row + dx];
                        ++s;
                    }
                }
                std::nth_element(u.begin(), u.begin() + center, u.end());
                std::nth_element(v.begin(), v.begin() + center, v.end());
                u_o[pixel_idx] = u[center];
                v_o[pixel_idx] = v[center];
            }
        }
    }

    SwapMem<void>(frame->data.yuv_u8_i.u, frame->data.yuv_u8_o.u);
    SwapMem<void>(frame->data.yuv_u8_i.v, frame->data.yuv_u8_o.v);

    return 0;
}

void RegisterCnsMod()
{
    IspModule mod;

    mod.in_type = DataPtrTypes::TYPE_INT32;
    mod.out_type = DataPtrTypes::TYPE_INT32;

    mod.in_domain = ColorDomains::YUV;
    mod.out_domain = ColorDomains::YUV;

    mod.name = MOD_NAME;
    mod.run_function = Cns;

    RegisterIspModule(mod);
}
