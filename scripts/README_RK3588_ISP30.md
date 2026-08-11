### RK3588 ISP30 capability probe

`scripts/probe_rk3588_isp30.sh` collects the board evidence needed before
attempting the four-camera offline-RAW/FakeCamera POC. It reports:

- media-controller topology and ISP `rawrd`/`mainpath` entities;
- V4L2 RAW8 and UYVY/NV16/YUYV/NV12 format advertisements;
- matching RKAIQ/RawStream headers, libraries, symbols and FakeCamera IQ files;
- DMA-BUF heaps, RGA and Mali/OpenCL availability and node permissions;
- processes that may already own an ISP or camera graph.

The probe is deliberately read-only. It does not create media links, set a
format or control, start a stream, load a kernel module, write sysfs/debugfs,
allocate DMA-BUFs, or run `rkisp_demo`/`rkaiq_3A_server`. `media-ctl`,
`v4l2-ctl`, and `clinfo` are invoked only for capability queries and are
time-limited when `timeout` is installed.

Run it as the same unprivileged account that will run the capture process:

```sh
./scripts/probe_rk3588_isp30.sh | tee /tmp/rk3588-isp30-probe.log
```

Do not use `sudo` merely to hide permission failures. A missing command,
missing node, `EACCES`, `EBUSY`, or query timeout is printed as diagnostic
evidence, and the remaining independent checks continue. The script always
exits with status 0 by design, so its exit status is **not** a capability PASS
signal. Review the final checklist and attach the complete log when reporting
results.

Useful but optional packages are the distribution packages that provide
`media-ctl`/`v4l2-ctl`, `clinfo`, `pkg-config`, and `nm`. The script never
installs them.
