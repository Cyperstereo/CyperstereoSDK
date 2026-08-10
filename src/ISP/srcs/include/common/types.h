#ifndef ADAS_ISP_TYPE_H
#define ADAS_ISP_TYPE_H

#include <stdint.h>
#include <string>

enum class CfaTypes
{
    RGGB,
    GRBG,
    BGGR,
    GBRG,

    CFA_MAX
};

enum class PixelCfaTypes
{
    GR,
    GB,
    R,
    B
};

enum class ColorDomains
{
    RAW,
    BGR,
    YUV,
    HSV,
};

enum class DataPtrTypes
{
    TYPE_INT8,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_UINT32,
    TYPE_FLOAT32,
    TYPE_MAX
};

enum class RawDataTypes
{
    RAW8 = 0x22,
    RAW10 = 0x2a,
    RAW12 = 0x2C,
    RAW14 = 0x2D,
    RAW16 = 0x2E,
    RAW20 = 0x2F,
    RAW24 = 0x30,
};

enum class YuvTypes
{
    YUV422_8,
    YUV422_10,
};

struct ImageInfo
{
    /* data */
    int width;
    int height;

    int bpp;
    int max_val;

    bool mipi_packed;

    CfaTypes cfa;
    ColorDomains domain;
    RawDataTypes dt;
    YuvTypes yuv_type;
};

struct ImageMem
{
    /* all data is double frame, enable pipeline in gpu */
    void *raw_u8_i;
    void *raw_u16_i;
    void *raw_u16_o;
    void *raw_s32_i;
    void *raw_s32_o;
    void *bgr_s32_i;
    void *bgr_s32_o;
    void *bgr_u8_o;
    struct YuvMem
    {
        void *y;
        void *u;
        void *v;
    } yuv_f32_i, yuv_f32_o, yuv_u8_i, yuv_u8_o;

    // ltm
    // void *y_log_f32_;
};

static constexpr int kCfaNums = static_cast<int>(CfaTypes::CFA_MAX);
static constexpr PixelCfaTypes kPixelCfaLut[kCfaNums][2][2] = {
    {
        {PixelCfaTypes::R, PixelCfaTypes::GR},
        {PixelCfaTypes::GB, PixelCfaTypes::B},
    },
    {
        {PixelCfaTypes::GR, PixelCfaTypes::R},
        {PixelCfaTypes::B, PixelCfaTypes::GB},
    },
    {
        {PixelCfaTypes::B, PixelCfaTypes::GB},
        {PixelCfaTypes::GR, PixelCfaTypes::R},
    },
    {
        {PixelCfaTypes::GB, PixelCfaTypes::B},
        {PixelCfaTypes::R, PixelCfaTypes::GR},
    }};
/**
 * @brief for depwl use
 */
#define MAX_PWL_NUMS 24
#define MAX_GAMMA_NUMS 21
struct DePwlPrms
{
    int pwl_nums;
    bool pedestal_before_pwl;
    int pedestal;
    int x_cood[MAX_PWL_NUMS];
    int y_cood[MAX_PWL_NUMS];
    float slope[MAX_PWL_NUMS];
};

struct CcmPrms
{
    float ccm[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};
};

struct WbGain
{
    // R G G B
    float d65_gain[4];
    float d50_gain[4];
    float f11_gain[4];
    float f12_gain[4];
};

struct GammmaCurve
{
    // 0-1
    int nums;
    int in_bits;
    int out_bits;
    float curve[MAX_GAMMA_NUMS];
};

static constexpr int kMaxLtmKenerlSize = 30;
struct LtmPrms
{
    int kernel_size = 9;
    float gauss_kernel[kMaxLtmKenerlSize][kMaxLtmKenerlSize];
    float range_kernel[kMaxLtmKenerlSize][kMaxLtmKenerlSize];
    float range_sigma = 0.4f;
    float space_sigma = 10;
    float constrast = 136;
    int in_bits = 10;
    int out_bits = 8;
};

struct SaturationPrms
{
    float rotate_angle;
};

struct ContrastPrms
{
    float ratio;
};

struct SharpenPrms
{
    float ratio;
};


static constexpr int kLscMeshBoxHNums = 10;
static constexpr int kLscMeshBoxVNums = 9;
static constexpr int kLscMeshPointHNums = kLscMeshBoxHNums + 1;
static constexpr int kLscMeshPointVNums = kLscMeshBoxVNums + 1;

struct LscPrms
{
    float mesh_r[kLscMeshPointVNums][kLscMeshPointHNums];
    float mesh_gr[kLscMeshPointVNums][kLscMeshPointHNums];
    float mesh_gb[kLscMeshPointVNums][kLscMeshPointHNums];
    float mesh_b[kLscMeshPointVNums][kLscMeshPointHNums];

    LscPrms()
    {
        for (int idy = 0; idy < kLscMeshPointVNums; ++idy) {
            for (int idx = 0; idx < kLscMeshPointHNums; ++idx) {
                mesh_r[idy][idx] = 1;
                mesh_gr[idy][idx] = 1;
                mesh_gb[idy][idx] = 1;
                mesh_b[idy][idx] = 1;
            }
        }
    }
};

enum class DpcMode { 
    MEAN,
    GRADIENT
}; 
struct DpcPrms
{
    int thres = 30;
    DpcMode mode = DpcMode::GRADIENT;
};

// Light, edge-preserving noise reduction on each Bayer colour sub-lattice.
// The Cyperstereo bridge expands RAW8 by <<2, so 16 working-range units are
// four original sensor DN. Strengths and the signal-dependent slope are Q8.
struct BayerNrPrms
{
    int threshold_base = 16;
    int threshold_signal_q8 = 4;
    int center_weight = 4;
    int strength_q8 = 128;
};

// Edge-preserving luminance NR in the linear BGR working domain. Guided
// statistics are evaluated on a quarter-resolution grid; eps is therefore a
// variance in the 10-bit working range. The filtered luma delta is added to
// B/G/R equally, preserving both R-G and B-G exactly.
struct LumaNrPrms
{
    int eps = 128;
    int strength_q8 = 64;
};

// Joint chroma denoise in the linear BGR domain. Thresholds are expressed in
// the ISP working range (10-bit for CyperstereoSDK-develop). Strengths are Q8
// so the per-pixel implementation stays deterministic and inexpensive.
struct ChromaNrPrms
{
    int luma_threshold = 24;
    int chroma_threshold = 64;
    int saturation_low = 48;
    int saturation_high = 192;
    int neutral_strength_q8 = 144;   // 0.5625 in smooth/near-neutral areas
    int saturated_strength_q8 = 40;  // 0.15625 on strongly coloured areas
    int max_hue_tan_q8 = 9;          // tan(about 2 degrees) * 256
    // Isolated-colour branch: when the eight neighbours are luma-flat and
    // near-neutral, pull a centre chroma outlier toward their common Dr/Db
    // base. Both opponent components use the same strength, so hue is not
    // rotated by independent channel decisions.
    int neutral_base_chroma = 72;
    int neutral_outlier_threshold = 48;
    int neutral_luma_range = 40;
    int neutral_outlier_strength_q8 = 192;  // 0.75 toward neighbour base
};

// False-colour suppression for dense, locally low-saturation texture. A wide
// neighbourhood estimates its stable base colour; the two opponent-colour
// residuals around that base always use one common attenuation factor.
struct FalseColorPrms
{
    int texture_low = 12;
    int texture_high = 32;
    int neutral_chroma_low = 64;
    int neutral_chroma_high = 192;
    int min_luma = 24;
    int max_attenuation_q8 = 224;
};

#endif
