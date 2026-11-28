# Digital twin twin_gs (global shutter)

Simulator example `twin_gs`: 1520 base-IMU poses over 7.6 s at 200 Hz, 2 camera(s), 1 IMU(s), apriltag board 6 x 6.

```python
from mvis.simulation.examples import load_example, simulate
ex = load_example("twin_gs")
data = simulate(ex, seed=0)   # cams, t_imu, w_m, a_m, aux_imus
```

## Source

Made from the digital-twin regression fixture `tests/data/twins/twin_gs.npz` (global-shutter stereo fisheye, 470 Hz IMU, 8 s window).

| Camera | Model | Image | Rate (Hz) | Shutter |
|---|---|---|---|---|
| cam0 | equidistant | 1920 x 1080 | 29.94 | global |
| cam1 | equidistant | 1920 x 1080 | 29.94 | global |

Base IMU: 470.59 Hz, model calyx.

## How it was made

Exported from the shipped fixture `tests/data/twins/twin_gs.npz` by maintainer tooling (not part of the package).
The twin's truth (cameras, IMU, board pose and shape, IMU-filled trajectory) is copied as it is. Camera and IMU rates are the medians of the recording's stamps; the simulator samples them uniformly. The noise is the twin's: the corner sigma and the IMU densities that reproduce the real residuals.

The truth is not a real device calibration: it is moved off the solve it came from by the fixed seeded perturbation of `mvis.simulation.twin.perturb` (`truth.perturbation` in `setup.yaml` has the seed, the name it is mixed with and the half-widths: up to 1 % focal, 4 px principal point, 5 % of each distortion coefficient, 10 mrad rotation and 5 mm lever per axis, 1 ms td, 0.5 ms t_ga).

`trajectory.txt` samples the truth spline at 200 Hz, time shifted to start at 0; `load_example` fits a 5 ms B-spline through it.

## Licence

GPL-3.0-only, as the regression fixture it comes from (the repository licence).
