#!/bin/bash
# Enables the RK3588 ISP30 cores and their virtual ISPs (rkisp0-vir0..2,
# rkisp1-vir0..2), which the SDK's hardware ISP backend replays RAW frames
# through. Most RK3588 images ship them disabled. Installs a device-tree
# overlay for the Orange Pi / Armbian boot script; takes effect after a reboot.
#
#   sudo ./scripts/setup_rk3588_hw_isp.sh           # install, then reboot
#   sudo ./scripts/setup_rk3588_hw_isp.sh --remove  # undo, then reboot
set -euo pipefail

NAME=cyperstereo-rkisp
DIR=/boot/overlay-user
ENV_FILE=""
for f in /boot/orangepiEnv.txt /boot/armbianEnv.txt; do
  [ -f "$f" ] && { ENV_FILE=$f; break; }
done

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0 $*" >&2
  exit 1
fi
if ! tr -d '\0' < /proc/device-tree/compatible | grep -q 'rockchip,rk3588'; then
  echo "this is not an RK3588" >&2
  exit 1
fi
if [ -z "$ENV_FILE" ]; then
  echo "no /boot/orangepiEnv.txt or /boot/armbianEnv.txt: add the overlay" \
       "printed by '$0 --print' to the bootloader by hand" >&2
  [ "${1:-}" = "--print" ] || exit 1
fi

overlay() {
  local nodes="/rkisp@fdcb0000 /iommu@fdcb7f00 /rkisp@fdcc0000 /iommu@fdcc7f00"
  local i n
  for i in 0 1; do for n in 0 1 2; do nodes="$nodes /rkisp$i-vir$n"; done; done
  echo "/dts-v1/;"
  echo "/plugin/;"
  echo "/ {"
  i=0
  for n in $nodes; do
    echo "  fragment@$i { target-path = \"$n\"; __overlay__ { status = \"okay\"; }; };"
    i=$((i + 1))
  done
  echo "};"
}

# Rewrites the user_overlays= line of the boot environment with or without
# this overlay.
set_user_overlays() {
  local add=$1 current list="" o
  current=$(sed -n 's/^user_overlays=//p' "$ENV_FILE" | tail -n 1)
  for o in $current; do [ "$o" != "$NAME" ] && list="$list $o"; done
  [ "$add" = 1 ] && list="$list $NAME"
  list=${list# }
  sed -i '/^user_overlays=/d' "$ENV_FILE"
  [ -n "$list" ] && echo "user_overlays=$list" >> "$ENV_FILE"
  return 0
}

case "${1:-}" in
  --print)
    overlay
    ;;
  --remove)
    rm -f "$DIR/$NAME.dtbo"
    set_user_overlays 0
    echo "removed $NAME; reboot to disable the ISP virtual devices"
    ;;
  "")
    command -v dtc >/dev/null || { echo "dtc not found: apt install device-tree-compiler" >&2; exit 1; }
    mkdir -p "$DIR"
    overlay | dtc -q -@ -I dts -O dtb -o "$DIR/$NAME.dtbo" -
    set_user_overlays 1
    echo "installed $DIR/$NAME.dtbo ($ENV_FILE: $(grep '^user_overlays=' "$ENV_FILE"))"
    echo "reboot, then 'ls /sys/bus/platform/drivers/rkisp/' lists the virtual ISPs"
    ;;
  *)
    echo "usage: $0 [--remove|--print]" >&2
    exit 1
    ;;
esac
