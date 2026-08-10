# -*- coding: utf-8 -*-
"""CCM calibration from a 24-patch ColorChecker capture.

The SDK pipeline is: BLC -> WB -> demosaic -> CCM (linear) -> tone curve.
This tool solves the 3x3 CCM by least squares against the ColorChecker
Classic reference values. The solver needs LINEAR camera data, so the
capture MUST be taken with the tone/color stages disabled:

Capture procedure
-----------------
1. Run the capture program with the pipeline forced linear (PowerShell):
       $env:CYPERSTEREO_GAMMA = "1.0"
       $env:CYPERSTEREO_CCM = "off"
       $env:CYPERSTEREO_SATURATION = "1.0"
       .\\capture_image_imu.exe
2. Light the checker evenly (no glare, no strong color cast around it --
   the auto white balance should settle on the room, not on the chart).
   The chart should fill roughly 1/4 .. 1/2 of the frame, near the image
   center (fisheye corners are too distorted for the flat-grid sampler).
3. Save one frame losslessly (PNG/BMP -- NOT JPEG).
4.     python ccm_calibrate.py capture.png
5. Hold the chart in the classic orientation (brown "dark skin" patch at
   top-left, white patch at bottom-left) and click the CENTERS of the four
   CORNER PATCHES in this order:
       1. dark skin  (top-left, brown)
       2. bluish green (top-right)
       3. black      (bottom-right)
       4. white      (bottom-left)
6. Copy the printed CYPERSTEREO_CCM value into your environment.

Dependencies: pip install numpy opencv-python
"""
import sys

import cv2
import numpy as np

# ColorChecker Classic (post-Nov-2014 nominal sRGB 8-bit values, BabelColor
# averages), row-major from dark skin (patch 1) to black (patch 24).
REF_SRGB = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
    [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108],
    [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
    [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121],
    [85, 85, 85], [52, 52, 52],
], dtype=np.float64)

WHITE_IDX = 18  # patch 19


def srgb_to_linear(v):
    v = v / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def auto_corners(img):
    """Estimate the 4 corner patch centers from local texture variance."""
    h, w = img.shape[:2]
    crop = img[int(h * 0.15) : int(h * 0.85), int(w * 0.15) : int(w * 0.85)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = 15
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    std = np.sqrt(np.maximum(m2 - m * m, 0))
    ys, xs = np.where(std > 8)
    if len(xs) < 100:
        sys.exit("auto-detect failed: no chart-like texture in the center crop")
    x0 = xs.min() + int(w * 0.15)
    x1 = xs.max() + int(w * 0.15)
    y0 = ys.min() + int(h * 0.15)
    y1 = ys.max() + int(h * 0.15)
    mx = int((x1 - x0) * 0.08)
    my = int((y1 - y0) * 0.08)
    return np.array([
        [x0 + mx, y0 + my],  # TL dark skin
        [x1 - mx, y0 + my],  # TR bluish green
        [x1 - mx, y1 - my],  # BR black
        [x0 + mx, y1 - my],  # BL white
    ], dtype=np.float64)


def parse_corners_arg(s):
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 8:
        sys.exit("--corners needs 8 numbers: tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y")
    vals = [float(p) for p in parts]
    return np.array([
        [vals[0], vals[1]],
        [vals[2], vals[3]],
        [vals[4], vals[5]],
        [vals[6], vals[7]],
    ], dtype=np.float64)


def pick_corners(img):
    """Interactive: click the 4 corner patch centers (see module docstring)."""
    pts = []
    h, w = img.shape[:2]
    # Upscale for easier clicking; map mouse coords back to original pixels.
    scale = max(1.0, min(2.5, 1600.0 / max(w, h)))
    disp_w, disp_h = int(w * scale), int(h * scale)
    disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    marker_r = max(10, int(12 * scale))
    win = "CCM calibrate: click 4 corner patch centers (Esc=cancel)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            ox, oy = x / scale, y / scale
            pts.append((ox, oy))
            cv2.circle(disp, (x, y), marker_r, (0, 0, 255), max(2, marker_r // 5))
            cv2.imshow(win, disp)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.imshow(win, disp)
    cv2.setMouseCallback(win, on_mouse)
    print("Window opened at %dx%d (scale %.2fx). Click 4 corner patch centers:"
          % (disp_w, disp_h, scale))
    print("  1 TL dark skin  2 TR bluish green  3 BR black  4 BL white")
    while len(pts) < 4:
        if cv2.waitKey(30) == 27:
            cv2.destroyWindow(win)
            sys.exit("aborted")
    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.float64)


def patch_centers(corners):
    """Bilinear 6x4 grid of patch centers from the 4 clicked corner centers.

    corners: [TL(dark skin), TR(bluish green), BR(black), BL(white)]
    """
    tl, tr, br, bl = corners
    centers = []
    for r in range(4):
        v = r / 3.0
        left = tl + (bl - tl) * v
        right = tr + (br - tr) * v
        for c in range(6):
            u = c / 5.0
            centers.append(left + (right - left) * u)
    return np.array(centers)  # 24 x 2, row-major = REF_SRGB order


def sample_patches(img, centers):
    """Median BGR inside a small box at each patch center -> N x 3 RGB."""
    pitch = np.linalg.norm(centers[1] - centers[0])
    h = max(3, int(pitch * 0.18))
    vals = []
    for cx, cy in centers:
        x0, x1 = int(cx - h), int(cx + h + 1)
        y0, y1 = int(cy - h), int(cy + h + 1)
        roi = img[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3).astype(np.float64)
        med = np.median(roi, axis=0)
        vals.append(med[::-1])  # BGR -> RGB
    return np.array(vals)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    img_path = sys.argv[1]
    mode = "interactive"
    corners_arg = None
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--auto":
            mode = "auto"
        elif sys.argv[i] == "--corners" and i + 1 < len(sys.argv):
            mode = "corners"
            corners_arg = sys.argv[i + 1]
            i += 1
        else:
            sys.exit("unknown arg: %s\n\n%s" % (sys.argv[i], __doc__))
        i += 1

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("cannot read " + img_path)

    if mode == "auto":
        corners = auto_corners(img)
        print("auto-detected corners (TL,TR,BR,BL):")
        for idx, (x, y) in enumerate(corners):
            print("  %d: (%.0f, %.0f)" % (idx + 1, x, y))
    elif mode == "corners":
        corners = parse_corners_arg(corners_arg)
    else:
        corners = pick_corners(img)
    centers = patch_centers(corners)
    cam = sample_patches(img, centers)  # linear camera RGB, 0..~239

    ref_lin = srgb_to_linear(REF_SRGB) * 255.0

    # Exposure normalization: scale the reference so its white patch matches
    # the captured white patch (green channel), removing the exposure /
    # full-scale-239 mismatch from the fit.
    scale = cam[WHITE_IDX, 1] / ref_lin[WHITE_IDX, 1]
    ref = ref_lin * scale

    # Drop clipped patches (any camera channel at/above 250: the ratio
    # information is destroyed by clipping).
    ok = np.all(cam < 250.0, axis=1)
    if ok.sum() < 12:
        sys.exit("too many clipped patches (%d usable) - reduce exposure "
                 "or dim the light" % ok.sum())

    # Least squares: ref ~= cam @ M^T  ->  M is the R-major CCM.
    M, *_ = np.linalg.lstsq(cam[ok], ref[ok], rcond=None)
    M = M.T

    # Preserve neutrals exactly (gray in = gray out, keeps WB): rows sum 1.
    M = M / M.sum(axis=1, keepdims=True)

    fit = cam @ M.T
    err = np.abs(fit[ok] - ref[ok]).mean()
    print("\npatches used: %d/24, mean abs fit error: %.2f (linear 8-bit)"
          % (ok.sum(), err))
    print("\nCCM (R-major rows applied to [R,G,B]):")
    for row in M:
        print("  [%8.4f %8.4f %8.4f]" % tuple(row))

    flat = ",".join("%.4f" % v for v in M.flatten())
    print("\nPowerShell:\n  $env:CYPERSTEREO_CCM = \"%s\"" % flat)
    print("\ncmd:\n  set CYPERSTEREO_CCM=%s" % flat)
    print("\nbash:\n  export CYPERSTEREO_CCM=\"%s\"" % flat)


if __name__ == "__main__":
    main()
