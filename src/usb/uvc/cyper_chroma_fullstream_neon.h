// Exact AArch64 chroma streaming kernel. The caller supplies plane pointers,
// strides, outputs, and persistent scratch; no OpenCV types are used here.
#ifndef CYPER_CHROMA_FULLSTREAM_NEON_H_
#define CYPER_CHROMA_FULLSTREAM_NEON_H_

#include <arm_neon.h>
#include <stdint.h>

namespace cyper_chroma_proto {

struct Scratch {
  // Sizes in elements for an input W x H:
  //   gauss_h_*: 8*W uint16_t, box_h_*: 8*(W/2) uint16_t,
  //   down_*: W/2 uint8_t.
  uint16_t *gauss_h_cr;
  uint16_t *gauss_h_cb;
  uint16_t *box_h_cr;
  uint16_t *box_h_cb;
  uint8_t *down_cr;
  uint8_t *down_cb;
};

static inline int Reflect101(int p, int length) {
  if (length == 1) return 0;
  while ((unsigned)p >= (unsigned)length)
    p = p < 0 ? -p : 2 * length - p - 2;
  return p;
}

static inline int Reflect(int p, int length) {
  if (length == 1) return 0;
  while ((unsigned)p >= (unsigned)length)
    p = p < 0 ? -p - 1 : 2 * length - p - 1;
  return p;
}

// Exact [1 4 6 4 1] horizontal numerator with BORDER_REFLECT_101.
static inline void Gauss5HRow(const uint8_t *source, uint16_t *destination,
                              int width) {
  const auto at = [&](int x) {
    return (int)source[Reflect101(x, width)];
  };
  for (int x = 0; x < 2; ++x)
    destination[x] = (uint16_t)(at(x - 2) + 4 * at(x - 1) +
                                6 * at(x) + 4 * at(x + 1) + at(x + 2));

  int x = 2;
  const uint8x8_t six = vdup_n_u8(6);
  for (; x + 18 <= width; x += 16) {
    const uint8x16_t m2 = vld1q_u8(source + x - 2);
    const uint8x16_t m1 = vld1q_u8(source + x - 1);
    const uint8x16_t c0 = vld1q_u8(source + x);
    const uint8x16_t p1 = vld1q_u8(source + x + 1);
    const uint8x16_t p2 = vld1q_u8(source + x + 2);
    uint16x8_t lo = vaddl_u8(vget_low_u8(m2), vget_low_u8(p2));
    uint16x8_t hi = vaddl_u8(vget_high_u8(m2), vget_high_u8(p2));
    lo = vmlaq_n_u16(lo,
                     vaddl_u8(vget_low_u8(m1), vget_low_u8(p1)), 4);
    hi = vmlaq_n_u16(hi,
                     vaddl_u8(vget_high_u8(m1), vget_high_u8(p1)), 4);
    lo = vmlal_u8(lo, vget_low_u8(c0), six);
    hi = vmlal_u8(hi, vget_high_u8(c0), six);
    vst1q_u16(destination + x, lo);
    vst1q_u16(destination + x + 8, hi);
  }
  for (; x < width - 2; ++x)
    destination[x] = (uint16_t)(source[x - 2] + 4 * source[x - 1] +
                                6 * source[x] + 4 * source[x + 1] +
                                source[x + 2]);
  for (; x < width; ++x)
    destination[x] = (uint16_t)(at(x - 2) + 4 * at(x - 1) +
                                6 * at(x) + 4 * at(x + 1) + at(x + 2));
}

// Generate two adjacent Gaussian rows together.  Six horizontal numerator
// rows are loaded once; the separate implementation loads ten rows.  All
// biased numerators are <=65408, hence u16 arithmetic is exact.
static inline void Gauss5VPair8(const uint16_t *r0, const uint16_t *r1,
                                const uint16_t *r2, const uint16_t *r3,
                                const uint16_t *r4, const uint16_t *r5, int x,
                                uint8x8_t &top, uint8x8_t &bottom) {
  const uint16x8_t a0 = vld1q_u16(r0 + x);
  const uint16x8_t a1 = vld1q_u16(r1 + x);
  const uint16x8_t a2 = vld1q_u16(r2 + x);
  const uint16x8_t a3 = vld1q_u16(r3 + x);
  const uint16x8_t a4 = vld1q_u16(r4 + x);
  const uint16x8_t a5 = vld1q_u16(r5 + x);
  uint16x8_t sum_top = vaddq_u16(a0, a4);
  sum_top = vmlaq_n_u16(sum_top, vaddq_u16(a1, a3), 4);
  sum_top = vmlaq_n_u16(sum_top, a2, 6);
  uint16x8_t sum_bottom = vaddq_u16(a1, a5);
  sum_bottom = vmlaq_n_u16(sum_bottom, vaddq_u16(a2, a4), 4);
  sum_bottom = vmlaq_n_u16(sum_bottom, a3, 6);
  const uint16x8_t round = vdupq_n_u16(128);
  top = vmovn_u16(vshrq_n_u16(vaddq_u16(sum_top, round), 8));
  bottom = vmovn_u16(vshrq_n_u16(vaddq_u16(sum_bottom, round), 8));
}

static inline uint8_t Gauss5VScalar(const uint16_t *const rows[5], int x) {
  const unsigned value = rows[0][x] + 4u * rows[1][x] +
                         6u * rows[2][x] + 4u * rows[3][x] + rows[4][x];
  return (uint8_t)((value + 128) >> 8);
}

// Seven-pixel horizontal box numerator, BORDER_REFLECT, u8 -> u16.
static inline void Box7HRow(const uint8_t *source, uint16_t *destination,
                            int width) {
  int x = 0;
  for (; x < 3 && x < width; ++x) {
    unsigned sum = 0;
    for (int k = -3; k <= 3; ++k) sum += source[Reflect(x + k, width)];
    destination[x] = (uint16_t)sum;
  }
  for (; x + 19 <= width; x += 16) {
    const uint8x16_t m3 = vld1q_u8(source + x - 3);
    const uint8x16_t m2 = vld1q_u8(source + x - 2);
    const uint8x16_t m1 = vld1q_u8(source + x - 1);
    const uint8x16_t c0 = vld1q_u8(source + x);
    const uint8x16_t p1 = vld1q_u8(source + x + 1);
    const uint8x16_t p2 = vld1q_u8(source + x + 2);
    const uint8x16_t p3 = vld1q_u8(source + x + 3);
    uint16x8_t lo = vaddl_u8(vget_low_u8(m3), vget_low_u8(p3));
    uint16x8_t hi = vaddl_u8(vget_high_u8(m3), vget_high_u8(p3));
    lo = vaddq_u16(lo,
                  vaddl_u8(vget_low_u8(m2), vget_low_u8(p2)));
    hi = vaddq_u16(hi,
                  vaddl_u8(vget_high_u8(m2), vget_high_u8(p2)));
    lo = vaddq_u16(lo,
                  vaddl_u8(vget_low_u8(m1), vget_low_u8(p1)));
    hi = vaddq_u16(hi,
                  vaddl_u8(vget_high_u8(m1), vget_high_u8(p1)));
    lo = vaddw_u8(lo, vget_low_u8(c0));
    hi = vaddw_u8(hi, vget_high_u8(c0));
    vst1q_u16(destination + x, lo);
    vst1q_u16(destination + x + 8, hi);
  }
  for (; x < width; ++x) {
    unsigned sum = 0;
    for (int k = -3; k <= 3; ++k) sum += source[Reflect(x + k, width)];
    destination[x] = (uint16_t)sum;
  }
}

// For n<=12519, floor(n/49) == (n*2675)>>17.  Here n is the seven-row
// numerator plus the round-to-nearest bias 24, so this is exactly
// OpenCV's saturate_cast<uchar>(sum/49.f) for every reachable input.
static inline uint8x8_t Divide49Rounded8(uint16x8_t sum) {
  sum = vaddq_u16(sum, vdupq_n_u16(24));
  const uint32x4_t lo = vmull_n_u16(vget_low_u16(sum), 2675);
  const uint32x4_t hi = vmull_n_u16(vget_high_u16(sum), 2675);
  return vmovn_u16(vcombine_u16(
      vmovn_u32(vshrq_n_u32(lo, 17)),
      vmovn_u32(vshrq_n_u32(hi, 17))));
}

static inline void EmitBox7Row(int output_y, int qheight, int qwidth,
                               uint16_t *ring_cr, uint16_t *ring_cb,
                               uint8_t *output_cr, uint8_t *output_cb) {
  const uint16_t *cr[7];
  const uint16_t *cb[7];
  for (int k = -3; k <= 3; ++k) {
    const int row = Reflect(output_y + k, qheight) & 7;
    cr[k + 3] = ring_cr + (uint64_t)row * qwidth;
    cb[k + 3] = ring_cb + (uint64_t)row * qwidth;
  }
  int x = 0;
  for (; x + 8 <= qwidth; x += 8) {
    uint16x8_t sum_cr = vaddq_u16(vld1q_u16(cr[0] + x),
                                  vld1q_u16(cr[1] + x));
    uint16x8_t sum_cb = vaddq_u16(vld1q_u16(cb[0] + x),
                                  vld1q_u16(cb[1] + x));
    for (int k = 2; k < 7; ++k) {
      sum_cr = vaddq_u16(sum_cr, vld1q_u16(cr[k] + x));
      sum_cb = vaddq_u16(sum_cb, vld1q_u16(cb[k] + x));
    }
    vst1_u8(output_cr + x, Divide49Rounded8(sum_cr));
    vst1_u8(output_cb + x, Divide49Rounded8(sum_cb));
  }
  for (; x < qwidth; ++x) {
    unsigned sum_cr = 0, sum_cb = 0;
    for (int k = 0; k < 7; ++k) {
      sum_cr += cr[k][x];
      sum_cb += cb[k][x];
    }
    output_cr[x] = (uint8_t)((sum_cr + 24) / 49);
    output_cb[x] = (uint8_t)((sum_cb + 24) / 49);
  }
}

// Returns 0 on success. Dimensions must be even and at least 8x8.
inline int ChromaGaussAreaBox7Neon(
    const uint8_t *source_cr, uint64_t source_cr_step,
    const uint8_t *source_cb, uint64_t source_cb_step,
    uint8_t *gauss_cr, uint64_t gauss_cr_step,
    uint8_t *gauss_cb, uint64_t gauss_cb_step,
    uint8_t *base_cr, uint64_t base_cr_step,
    uint8_t *base_cb, uint64_t base_cb_step,
    int width, int height, Scratch scratch) {
  if (!source_cr || !source_cb || !gauss_cr || !gauss_cb || !base_cr ||
      !base_cb || !scratch.gauss_h_cr || !scratch.gauss_h_cb ||
      !scratch.box_h_cr || !scratch.box_h_cb || !scratch.down_cr ||
      !scratch.down_cb || width < 8 || height < 8 || (width & 1) ||
      (height & 1))
    return -1;
  const int qwidth = width / 2;
  const int qheight = height / 2;
  const auto ghcr = [&](int row) {
    return scratch.gauss_h_cr + (uint64_t)(row & 7) * width;
  };
  const auto ghcb = [&](int row) {
    return scratch.gauss_h_cb + (uint64_t)(row & 7) * width;
  };
  const auto bhcr = [&](int row) {
    return scratch.box_h_cr + (uint64_t)(row & 7) * qwidth;
  };
  const auto bhcb = [&](int row) {
    return scratch.box_h_cb + (uint64_t)(row & 7) * qwidth;
  };

  int next_horizontal = 0;
  for (int quarter_y = 0; quarter_y < qheight; ++quarter_y) {
    const int y0 = 2 * quarter_y;
    const int y1 = y0 + 1;
    const int through = y1 + 2 < height ? y1 + 2 : height - 1;
    for (; next_horizontal <= through; ++next_horizontal) {
      Gauss5HRow(source_cr + (uint64_t)next_horizontal * source_cr_step,
                 ghcr(next_horizontal), width);
      Gauss5HRow(source_cb + (uint64_t)next_horizontal * source_cb_step,
                 ghcb(next_horizontal), width);
    }

    const uint16_t *cr_rows[6];
    const uint16_t *cb_rows[6];
    for (int k = -2; k <= 3; ++k) {
      const int row = Reflect101(y0 + k, height);
      cr_rows[k + 2] = ghcr(row);
      cb_rows[k + 2] = ghcb(row);
    }
    uint8_t *gcr0 = gauss_cr + (uint64_t)y0 * gauss_cr_step;
    uint8_t *gcr1 = gauss_cr + (uint64_t)y1 * gauss_cr_step;
    uint8_t *gcb0 = gauss_cb + (uint64_t)y0 * gauss_cb_step;
    uint8_t *gcb1 = gauss_cb + (uint64_t)y1 * gauss_cb_step;

    int x = 0, xq = 0;
    for (; x + 16 <= width; x += 16, xq += 8) {
      uint8x8_t cr00, cr10, cr01, cr11;
      uint8x8_t cb00, cb10, cb01, cb11;
      Gauss5VPair8(cr_rows[0], cr_rows[1], cr_rows[2], cr_rows[3],
                   cr_rows[4], cr_rows[5], x, cr00, cr10);
      Gauss5VPair8(cr_rows[0], cr_rows[1], cr_rows[2], cr_rows[3],
                   cr_rows[4], cr_rows[5], x + 8, cr01, cr11);
      Gauss5VPair8(cb_rows[0], cb_rows[1], cb_rows[2], cb_rows[3],
                   cb_rows[4], cb_rows[5], x, cb00, cb10);
      Gauss5VPair8(cb_rows[0], cb_rows[1], cb_rows[2], cb_rows[3],
                   cb_rows[4], cb_rows[5], x + 8, cb01, cb11);
      const uint8x16_t cr_top = vcombine_u8(cr00, cr01);
      const uint8x16_t cr_bottom = vcombine_u8(cr10, cr11);
      const uint8x16_t cb_top = vcombine_u8(cb00, cb01);
      const uint8x16_t cb_bottom = vcombine_u8(cb10, cb11);
      vst1q_u8(gcr0 + x, cr_top);
      vst1q_u8(gcr1 + x, cr_bottom);
      vst1q_u8(gcb0 + x, cb_top);
      vst1q_u8(gcb1 + x, cb_bottom);
      vst1_u8(scratch.down_cr + xq,
              vrshrn_n_u16(vaddq_u16(vpaddlq_u8(cr_top),
                                     vpaddlq_u8(cr_bottom)), 2));
      vst1_u8(scratch.down_cb + xq,
              vrshrn_n_u16(vaddq_u16(vpaddlq_u8(cb_top),
                                     vpaddlq_u8(cb_bottom)), 2));
    }
    const uint16_t *cr_top_rows[5] = {cr_rows[0], cr_rows[1], cr_rows[2],
                                      cr_rows[3], cr_rows[4]};
    const uint16_t *cr_bottom_rows[5] = {cr_rows[1], cr_rows[2], cr_rows[3],
                                         cr_rows[4], cr_rows[5]};
    const uint16_t *cb_top_rows[5] = {cb_rows[0], cb_rows[1], cb_rows[2],
                                      cb_rows[3], cb_rows[4]};
    const uint16_t *cb_bottom_rows[5] = {cb_rows[1], cb_rows[2], cb_rows[3],
                                         cb_rows[4], cb_rows[5]};
    for (; x + 1 < width; x += 2, ++xq) {
      const uint8_t cr00s = Gauss5VScalar(cr_top_rows, x);
      const uint8_t cr01s = Gauss5VScalar(cr_top_rows, x + 1);
      const uint8_t cr10s = Gauss5VScalar(cr_bottom_rows, x);
      const uint8_t cr11s = Gauss5VScalar(cr_bottom_rows, x + 1);
      const uint8_t cb00s = Gauss5VScalar(cb_top_rows, x);
      const uint8_t cb01s = Gauss5VScalar(cb_top_rows, x + 1);
      const uint8_t cb10s = Gauss5VScalar(cb_bottom_rows, x);
      const uint8_t cb11s = Gauss5VScalar(cb_bottom_rows, x + 1);
      gcr0[x] = cr00s; gcr0[x + 1] = cr01s;
      gcr1[x] = cr10s; gcr1[x + 1] = cr11s;
      gcb0[x] = cb00s; gcb0[x + 1] = cb01s;
      gcb1[x] = cb10s; gcb1[x + 1] = cb11s;
      scratch.down_cr[xq] =
          (uint8_t)((cr00s + cr01s + cr10s + cr11s + 2) >> 2);
      scratch.down_cb[xq] =
          (uint8_t)((cb00s + cb01s + cb10s + cb11s + 2) >> 2);
    }

    Box7HRow(scratch.down_cr, bhcr(quarter_y), qwidth);
    Box7HRow(scratch.down_cb, bhcb(quarter_y), qwidth);
    if (quarter_y >= 3) {
      const int output_y = quarter_y - 3;
      EmitBox7Row(output_y, qheight, qwidth,
                  scratch.box_h_cr, scratch.box_h_cb,
                  base_cr + (uint64_t)output_y * base_cr_step,
                  base_cb + (uint64_t)output_y * base_cb_step);
    }
  }
  int output_y = qheight > 3 ? qheight - 3 : 0;
  for (; output_y < qheight; ++output_y)
    EmitBox7Row(output_y, qheight, qwidth,
                scratch.box_h_cr, scratch.box_h_cb,
                base_cr + (uint64_t)output_y * base_cr_step,
                base_cb + (uint64_t)output_y * base_cb_step);
  return 0;
}

}  // namespace cyper_chroma_proto

#endif  // CYPER_CHROMA_FULLSTREAM_NEON_H_
