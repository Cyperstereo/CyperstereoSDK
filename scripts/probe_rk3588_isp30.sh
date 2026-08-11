#!/usr/bin/env bash
# Read-only RK3588 ISP30 / accelerator capability probe.
#
# This script intentionally performs no media links, format/control changes,
# stream starts, module loads, sysfs/debugfs writes, or device allocations.
# Some read-only ioctl queries may fail with EBUSY, EACCES, or ETIMEDOUT; those
# failures are printed as evidence and never "fixed" by this script.

set -u
export LC_ALL=C

section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Bound device queries when GNU/BusyBox timeout is available.  Do not enable
# `set -e` or `pipefail`: an unavailable optional tool/node must not prevent
# later independent probes from running.
probe_capture() {
  local seconds="$1"
  shift
  if have timeout; then
    timeout "$seconds" "$@" 2>&1
  else
    "$@" 2>&1
  fi
}

show_node_access() {
  local n="$1" mode=""
  [ -r "$n" ] && mode="${mode}r" || mode="${mode}-"
  [ -w "$n" ] && mode="${mode}w" || mode="${mode}-"
  printf '%-36s access=%s  ' "$n" "$mode"
  if ! stat -Lc 'type=%F mode=%A owner=%U:%G major_minor=%t:%T' \
      "$n" 2>/dev/null; then
    printf 'stat=unavailable\n'
  fi
}

section "identity"
date -Is 2>/dev/null || date
uname -a
printf 'uid: '; id
if [ -r /proc/device-tree/compatible ]; then
  printf 'device-tree compatible: '
  tr '\000' ' ' </proc/device-tree/compatible
  printf '\n'
fi
if have lscpu; then
  lscpu | sed -n '/^Architecture:/p;/^Model name:/p;/^CPU(s):/p;/^On-line CPU(s) list:/p'
fi

section "kernel drivers and command availability"
for c in media-ctl v4l2-ctl yavta rkaiq_3A_server rkisp_demo clinfo \
         pkg-config ldconfig nm timeout; do
  if have "$c"; then
    printf '%-20s %s\n' "$c" "$(command -v "$c")"
  else
    printf '%-20s MISSING\n' "$c"
  fi
done
printf '\nloaded modules (read-only /proc/modules):\n'
if [ -r /proc/modules ]; then
  awk 'tolower($1) ~ /(rkisp|rkcif|rga|mali|panthor|panfrost|dma_heap|videobuf)/ {print}' /proc/modules
else
  printf 'unavailable\n'
fi

section "media-controller topology"
media_count=0
isp_models=0
for media in /dev/media[0-9]*; do
  [ -e "$media" ] || continue
  media_count=$((media_count + 1))
  show_node_access "$media"
  if have media-ctl; then
    topo="$(probe_capture 5 media-ctl -p -d "$media")"
    query_rc=$?
    printf '%s\n' "$topo"
    if [ "$query_rc" -ne 0 ]; then
      printf 'QUERY_FAILED rc=%d node=%s (continuing)\n' "$query_rc" "$media"
    fi
    if printf '%s\n' "$topo" | grep -Eiq 'rkisp|rockchip.*isp|isp[0-9]'; then
      isp_models=$((isp_models + 1))
      printf '%s\n' "$topo" | grep -Ei 'driver|model|rkisp|rawrd|raw.*read|mainpath|selfpath|rawwr|stats|params' || true
    fi
  else
    printf 'media-ctl missing: cannot prove links/entities for this node\n'
  fi
done
printf '\nsummary: media_nodes=%d media_topologies_with_isp_tokens=%d\n' "$media_count" "$isp_models"
printf 'PASS requires two independently addressable ISP30 pipelines and raw-read + output path entities.\n'

section "V4L2 node names and advertised formats"
for sysnode in /sys/class/video4linux/video*; do
  [ -e "$sysnode" ] || continue
  dev="/dev/$(basename "$sysnode")"
  name="$(cat "$sysnode/name" 2>/dev/null || printf '?')"
  printf '\n%s name=%s\n' "$dev" "$name"
  show_node_access "$dev"
  case "$name" in
    *rkisp*|*ISP*|*isp*|*rawrd*|*mainpath*|*selfpath*|*scale*)
      if have v4l2-ctl; then
        v4l2_info="$(probe_capture 5 v4l2-ctl -d "$dev" --all)"
        query_rc=$?
        printf '%s\n' "$v4l2_info" | sed -n '1,80p'
        if [ "$query_rc" -ne 0 ]; then
          printf 'QUERY_FAILED rc=%d node=%s operation=all (continuing)\n' \
            "$query_rc" "$dev"
        fi
        printf '%s\n' '-- formats --'
        v4l2_formats="$(probe_capture 5 v4l2-ctl -d "$dev" --list-formats-ext)"
        query_rc=$?
        printf '%s\n' "$v4l2_formats"
        if [ "$query_rc" -ne 0 ]; then
          printf 'QUERY_FAILED rc=%d node=%s operation=formats (continuing)\n' \
            "$query_rc" "$dev"
        fi
      else
        printf 'v4l2-ctl missing: cannot query capabilities/formats\n'
      fi
      ;;
  esac
done
printf '\nRequired input evidence: 1280x1024 RAW8 and both physical Bayer phases used by the cameras.\n'
printf 'Preferred output order: direct UYVY; otherwise NV16/YUYV plus a measured repack; NV12 is last choice.\n'

section "RKAIQ / RawStream userspace"
if have ldconfig; then
  ldconfig -p 2>/dev/null | grep -Ei 'rkaiq|rk_aiq|rawstream|rockit|librga|OpenCL' || true
fi
for pc in rkaiq libRkAiq librga OpenCL; do
  if have pkg-config && pkg-config --exists "$pc" 2>/dev/null; then
    printf 'pkg-config %-12s version=%s cflags=%s libs=%s\n' \
      "$pc" "$(pkg-config --modversion "$pc")" \
      "$(pkg-config --cflags "$pc")" "$(pkg-config --libs "$pc")"
  fi
done
printf '\nheaders (bounded standard vendor locations):\n'
find /usr/include /usr/local/include /opt /vendor/usr/include \
  -maxdepth 6 -type f \
  \( -name 'rk_aiq_user_api2_sysctl.h' -o -name 'rk_aiq_user_api_sysctl.h' \
     -o -iname '*rawstream*.h' -o -name 'im2d.h' -o -name 'rga.h' \) \
  -print 2>/dev/null | sort -u | head -n 200
printf '\nIQ files mentioning FakeCamera:\n'
find /etc /usr/share /oem /vendor -maxdepth 6 -type f \
  \( -iname 'FakeCamera*.json' -o -iname 'FakeCamera*.xml' -o -iname 'FakeCamera*.bin' \) \
  -print 2>/dev/null | sort -u | head -n 100
printf '\nRKAIQ symbols exposed by installed shared libraries:\n'
for lib in /usr/lib/aarch64-linux-gnu/lib*rkaiq*.so* /usr/lib/lib*rkaiq*.so* \
           /usr/local/lib/lib*rkaiq*.so* /vendor/lib64/lib*rkaiq*.so*; do
  [ -r "$lib" ] || continue
  printf '%s\n' "-- $lib"
  if have nm; then
    nm -D "$lib" 2>/dev/null | grep -E \
      'sysctl_(prepareRkRaw|enqueueRkRawBuf|registRkRawCb|rawReproc|preInit_rkrawstream|init|start|stop)' || true
  else
    printf 'nm missing\n'
  fi
done

section "DMA-BUF heaps"
dma_nodes=0
for node in /dev/dma_heap/* /dev/dma-heap/*; do
  [ -e "$node" ] || continue
  dma_nodes=$((dma_nodes + 1))
  show_node_access "$node"
done
printf 'dma_heap_nodes=%d (write permission is needed later for allocation; this probe allocates nothing)\n' "$dma_nodes"
if [ -r /proc/config.gz ] && have zcat; then
  zcat /proc/config.gz 2>/dev/null | grep -E '^CONFIG_(DMABUF_HEAPS|DMABUF_HEAPS_SYSTEM|DMA_SHARED_BUFFER)=' || true
elif [ -r "/boot/config-$(uname -r)" ]; then
  grep -E '^CONFIG_(DMABUF_HEAPS|DMABUF_HEAPS_SYSTEM|DMA_SHARED_BUFFER)=' "/boot/config-$(uname -r)" || true
fi

section "RGA"
for node in /dev/rga /dev/rga2 /dev/rga3; do
  [ -e "$node" ] && show_node_access "$node"
done
for f in /sys/kernel/debug/rkrga/driver_version; do
  if [ -r "$f" ]; then
    printf '%s: ' "$f"; head -n 20 "$f"
  fi
done
if have pkg-config && pkg-config --exists librga 2>/dev/null; then
  printf 'librga pkg-config version=%s\n' "$(pkg-config --modversion librga)"
fi

section "OpenCL / Mali"
for node in /dev/mali0 /dev/dri/renderD*; do
  [ -e "$node" ] && show_node_access "$node"
done
printf 'ICD files:\n'
for icd in /etc/OpenCL/vendors/*.icd /usr/share/OpenCL/vendors/*.icd; do
  [ -r "$icd" ] || continue
  printf '%s: ' "$icd"; tr '\n' ' ' <"$icd"; printf '\n'
done
if have clinfo; then
  clinfo_text="$(probe_capture 10 clinfo)"
  query_rc=$?
  printf '%s\n' "$clinfo_text" | grep -E \
    'Platform Name|Platform Version|Device Name|Device Version|Driver Version|Max compute units|Global memory size|cl_arm_import_memory|cl_khr_external_memory|cl_khr_image2d_from_buffer' || true
  if [ "$query_rc" -ne 0 ]; then
    printf 'QUERY_FAILED rc=%d operation=clinfo (continuing)\n' "$query_rc"
  fi
else
  printf 'clinfo missing; libOpenCL alone does not prove a usable Mali platform\n'
fi

section "process conflicts (read-only snapshot)"
ps -eo pid,psr,comm,args 2>/dev/null | grep -E \
  '[r]kaiq|[r]kisp|[c]amera_engine|[c]apture_image_imu|[r]ga' || true

section "verdict checklist"
printf '%s\n' \
  '[ ] exactly two usable ISP30 raw-read pipelines identified by media model/entity names' \
  '[ ] RAW8 1280x1024 accepted at the raw-read input (not inferred from source code)' \
  '[ ] direct UYVY or lossless 4:2:2 output route advertised' \
  '[ ] board-matching RKAIQ headers and library expose offline-RAW APIs' \
  '[ ] four independent AIQ states can bind/time-multiplex over two ISPs, or fixed manual IQ is accepted' \
  '[ ] writable dma-heap and an explicit cache-coherency strategy exist' \
  '[ ] IQ file is calibrated for SC136HGS at 1280x1024 and both CFA phases' \
  '[ ] no camera_engine/rkaiq process already owns the selected graph'
printf '\nNo PASS is implied merely by finding /dev/media* or librkaiq.so.\n'
printf 'Probe complete: failures above are diagnostic; no device state was changed.\n'
exit 0
