#ifndef CYPER_CHROMA_FULLSTREAM_AVX2_H_
#define CYPER_CHROMA_FULLSTREAM_AVX2_H_

#include <immintrin.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define CYPER_CHROMA_AVX2_TARGET __attribute__((target("avx2")))
#else
#define CYPER_CHROMA_AVX2_TARGET
#endif

namespace cyper_chroma_x86_proto {

struct Scratch {
  uint16_t *gauss_h_cr;
  uint16_t *gauss_h_cb;
  uint16_t *box_h_cr;
  uint16_t *box_h_cb;
  uint8_t *down_cr;
  uint8_t *down_cb;
};

CYPER_CHROMA_AVX2_TARGET static inline int Reflect101(int p, int length) {
  if (length == 1) return 0;
  while ((unsigned)p >= (unsigned)length)
    p = p < 0 ? -p : 2 * length - p - 2;
  return p;
}

CYPER_CHROMA_AVX2_TARGET static inline int Reflect(int p, int length) {
  if (length == 1) return 0;
  while ((unsigned)p >= (unsigned)length)
    p = p < 0 ? -p - 1 : 2 * length - p - 1;
  return p;
}

CYPER_CHROMA_AVX2_TARGET static inline __m256i Gauss5H16(const uint8_t *source) {
  const __m256i m2 = _mm256_cvtepu8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(source - 2)));
  const __m256i m1 = _mm256_cvtepu8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(source - 1)));
  const __m256i c0 = _mm256_cvtepu8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(source)));
  const __m256i p1 = _mm256_cvtepu8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 1)));
  const __m256i p2 = _mm256_cvtepu8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(source + 2)));
  __m256i value = _mm256_add_epi16(m2, p2);
  value = _mm256_add_epi16(
      value, _mm256_slli_epi16(_mm256_add_epi16(m1, p1), 2));
  value = _mm256_add_epi16(
      value, _mm256_add_epi16(_mm256_slli_epi16(c0, 2),
                              _mm256_slli_epi16(c0, 1)));
  return value;
}

CYPER_CHROMA_AVX2_TARGET static inline void Gauss5HRow(const uint8_t *source, uint16_t *destination,
                              int width) {
  const auto at = [&](int x) {
    return (int)source[Reflect101(x, width)];
  };
  for (int x = 0; x < 2; ++x)
    destination[x] = (uint16_t)(at(x - 2) + 4 * at(x - 1) +
                                6 * at(x) + 4 * at(x + 1) + at(x + 2));
  int x = 2;
  for (; x + 18 <= width; x += 16) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination + x),
                        Gauss5H16(source + x));
  }
  for (; x < width - 2; ++x)
    destination[x] = (uint16_t)(source[x - 2] + 4 * source[x - 1] +
                                6 * source[x] + 4 * source[x + 1] +
                                source[x + 2]);
  for (; x < width; ++x)
    destination[x] = (uint16_t)(at(x - 2) + 4 * at(x - 1) +
                                6 * at(x) + 4 * at(x + 1) + at(x + 2));
}

CYPER_CHROMA_AVX2_TARGET static inline __m128i Pack16U16ToU8(__m256i value) {
  return _mm_packus_epi16(_mm256_castsi256_si128(value),
                          _mm256_extracti128_si256(value, 1));
}

CYPER_CHROMA_AVX2_TARGET static inline void Gauss5VPair16(
    const uint16_t *r0, const uint16_t *r1, const uint16_t *r2,
    const uint16_t *r3, const uint16_t *r4, const uint16_t *r5, int x,
    __m128i &top, __m128i &bottom) {
  const __m256i a0 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(r0 + x));
  const __m256i a1 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(r1 + x));
  const __m256i a2 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(r2 + x));
  const __m256i a3 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(r3 + x));
  const __m256i a4 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(r4 + x));
  const __m256i a5 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(r5 + x));
  __m256i sum_top = _mm256_add_epi16(a0, a4);
  sum_top = _mm256_add_epi16(
      sum_top, _mm256_slli_epi16(_mm256_add_epi16(a1, a3), 2));
  sum_top = _mm256_add_epi16(
      sum_top, _mm256_add_epi16(_mm256_slli_epi16(a2, 2),
                                _mm256_slli_epi16(a2, 1)));
  __m256i sum_bottom = _mm256_add_epi16(a1, a5);
  sum_bottom = _mm256_add_epi16(
      sum_bottom, _mm256_slli_epi16(_mm256_add_epi16(a2, a4), 2));
  sum_bottom = _mm256_add_epi16(
      sum_bottom, _mm256_add_epi16(_mm256_slli_epi16(a3, 2),
                                   _mm256_slli_epi16(a3, 1)));
  const __m256i round = _mm256_set1_epi16(128);
  top = Pack16U16ToU8(
      _mm256_srli_epi16(_mm256_add_epi16(sum_top, round), 8));
  bottom = Pack16U16ToU8(
      _mm256_srli_epi16(_mm256_add_epi16(sum_bottom, round), 8));
}

CYPER_CHROMA_AVX2_TARGET static inline uint8_t Gauss5VScalar(const uint16_t *const rows[5], int x) {
  const unsigned value = rows[0][x] + 4u * rows[1][x] +
                         6u * rows[2][x] + 4u * rows[3][x] + rows[4][x];
  return (uint8_t)((value + 128) >> 8);
}

CYPER_CHROMA_AVX2_TARGET static inline void Down2x2Store8(__m128i top, __m128i bottom,
                                 uint8_t *destination) {
  const __m128i pair_ones = _mm_set1_epi16(0x0101);
  __m128i sum = _mm_add_epi16(_mm_maddubs_epi16(top, pair_ones),
                              _mm_maddubs_epi16(bottom, pair_ones));
  sum = _mm_srli_epi16(_mm_add_epi16(sum, _mm_set1_epi16(2)), 2);
  const __m128i packed = _mm_packus_epi16(sum, _mm_setzero_si128());
  _mm_storel_epi64(reinterpret_cast<__m128i *>(destination), packed);
}

CYPER_CHROMA_AVX2_TARGET static inline __m256i Box7H16(const uint8_t *source) {
  __m256i sum = _mm256_add_epi16(
      _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(source - 3))),
      _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(source + 3))));
  sum = _mm256_add_epi16(
      sum, _mm256_add_epi16(
          _mm256_cvtepu8_epi16(_mm_loadu_si128(
              reinterpret_cast<const __m128i *>(source - 2))),
          _mm256_cvtepu8_epi16(_mm_loadu_si128(
              reinterpret_cast<const __m128i *>(source + 2)))));
  sum = _mm256_add_epi16(
      sum, _mm256_add_epi16(
          _mm256_cvtepu8_epi16(_mm_loadu_si128(
              reinterpret_cast<const __m128i *>(source - 1))),
          _mm256_cvtepu8_epi16(_mm_loadu_si128(
              reinterpret_cast<const __m128i *>(source + 1)))));
  return _mm256_add_epi16(
      sum, _mm256_cvtepu8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(source))));
}

CYPER_CHROMA_AVX2_TARGET static inline void Box7HRow(const uint8_t *source, uint16_t *destination,
                            int width) {
  int x = 0;
  for (; x < 3 && x < width; ++x) {
    unsigned sum = 0;
    for (int k = -3; k <= 3; ++k) sum += source[Reflect(x + k, width)];
    destination[x] = (uint16_t)sum;
  }
  for (; x + 19 <= width; x += 16) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(destination + x),
                        Box7H16(source + x));
  }
  for (; x < width; ++x) {
    unsigned sum = 0;
    for (int k = -3; k <= 3; ++k) sum += source[Reflect(x + k, width)];
    destination[x] = (uint16_t)sum;
  }
}

CYPER_CHROMA_AVX2_TARGET static inline __m128i Divide49Rounded16(__m256i sum) {
  sum = _mm256_add_epi16(sum, _mm256_set1_epi16(24));
  // floor((n * 2675) / 2^17) = floor(high16(n*2675) / 2).
  return Pack16U16ToU8(
      _mm256_srli_epi16(_mm256_mulhi_epu16(sum,
                                           _mm256_set1_epi16(2675)), 1));
}

CYPER_CHROMA_AVX2_TARGET static inline void EmitBox7Row(int output_y, int qheight, int qwidth,
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
  for (; x + 16 <= qwidth; x += 16) {
    __m256i sum_cr = _mm256_add_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cr[0] + x)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cr[1] + x)));
    __m256i sum_cb = _mm256_add_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cb[0] + x)),
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cb[1] + x)));
    for (int k = 2; k < 7; ++k) {
      sum_cr = _mm256_add_epi16(
          sum_cr, _mm256_loadu_si256(
                      reinterpret_cast<const __m256i *>(cr[k] + x)));
      sum_cb = _mm256_add_epi16(
          sum_cb, _mm256_loadu_si256(
                      reinterpret_cast<const __m256i *>(cb[k] + x)));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output_cr + x),
                     Divide49Rounded16(sum_cr));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output_cb + x),
                     Divide49Rounded16(sum_cb));
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

CYPER_CHROMA_AVX2_TARGET inline int ChromaGaussAreaBox7Avx2(
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
      (height & 1)) return -1;
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
    for (; x + 32 <= width; x += 32, xq += 16) {
      __m128i cr00, cr10, cr01, cr11;
      __m128i cb00, cb10, cb01, cb11;
      Gauss5VPair16(cr_rows[0], cr_rows[1], cr_rows[2], cr_rows[3],
                    cr_rows[4], cr_rows[5], x, cr00, cr10);
      Gauss5VPair16(cr_rows[0], cr_rows[1], cr_rows[2], cr_rows[3],
                    cr_rows[4], cr_rows[5], x + 16, cr01, cr11);
      Gauss5VPair16(cb_rows[0], cb_rows[1], cb_rows[2], cb_rows[3],
                    cb_rows[4], cb_rows[5], x, cb00, cb10);
      Gauss5VPair16(cb_rows[0], cb_rows[1], cb_rows[2], cb_rows[3],
                    cb_rows[4], cb_rows[5], x + 16, cb01, cb11);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcr0 + x), cr00);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcr0 + x + 16), cr01);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcr1 + x), cr10);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcr1 + x + 16), cr11);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcb0 + x), cb00);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcb0 + x + 16), cb01);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcb1 + x), cb10);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(gcb1 + x + 16), cb11);
      Down2x2Store8(cr00, cr10, scratch.down_cr + xq);
      Down2x2Store8(cr01, cr11, scratch.down_cr + xq + 8);
      Down2x2Store8(cb00, cb10, scratch.down_cb + xq);
      Down2x2Store8(cb01, cb11, scratch.down_cb + xq + 8);
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

}  // namespace cyper_chroma_x86_proto

#undef CYPER_CHROMA_AVX2_TARGET

#endif
