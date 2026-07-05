#!/usr/bin/env bash
#
# ============================================================================
#  One-shot setup for Orange Pi 5 / RK3588 camera-capture boards
# ============================================================================
# Consolidates every host-side measure that was validated against
# "failed: v4l2 get stream time out" / frame drops with the Cyper FX3 UVC
# camera (04b4:00f9), so a fresh board can be provisioned with ONE command:
#
#   1. U-Boot kernel parameters (/boot/orangepiEnv.txt or armbianEnv.txt)
#        - strips literal quotes from extraargs (quotes break kernel arg
#          parsing: everything after the first param is silently ignored!)
#        - cma=512M                        DMA contiguous memory for capture
#        - usbcore.quirks=04b4:00f9:g     delay-init quirk for the camera
#        - uvcvideo.nodrop=1              keep partial frames instead of drop
#        - uvcvideo.timeout=5000          longer streaming ctrl timeout
#        - uvcvideo.quirks=0x100          PROBE_DEF quirk
#        - iommu.passthrough=1            lower USB DMA latency
#        - pcie_aspm=off                  no PCIe power-state latency spikes
#        - usbcore.autosuspend=-1         never autosuspend USB devices
#        - systemd.unified_cgroup_hierarchy=0
#          boot with cgroup v1. Rockchip kernels have RT_GROUP_SCHED=y and
#          cgroup v2 lacks cpu.rt_runtime_us, making SCHED_FIFO fail with
#          EPERM even for root. v1 restores it (the SDK capture threads use
#          real-time priorities, see thread_priority.h).
#
#   2. Runtime tuning, applied now AND persisted via ONE systemd service
#      (camera-low-latency.service):
#        - disable deep CPU idle states (keep WFI): ARM equivalent of x86
#          max_cstate=1; removes ~220us wakeup latency of "cpu-sleep"
#        - performance cpufreq governor on all clusters (ondemand leaves the
#          A76 cores at 600MHz and ramps too slowly for per-frame bursts)
#        - pin the camera xHCI IRQ to a dedicated A76 big core (CPU4)
#        - grant RT bandwidth to user.slice (500ms/s) and system.slice
#          (400ms/s) so SCHED_FIFO works from SSH sessions and services
#          (requires the cgroup v1 boot param above + reboot)
#
# The script is idempotent (safe to re-run) and migrates away the older
# split services (disable-cpu-deep-idle / rk3588-low-latency) if present.
#
# Usage:
#   sudo bash setup_orangepi_camera.sh              # install everything
#   sudo bash setup_orangepi_camera.sh --revert     # undo everything
#
# A REBOOT is required after the first install for the boot parameters
# (and thus RT priorities) to take effect. The script never reboots by
# itself.
#
# Verify after reboot:
#   cat /proc/cmdline                    # params present, no quotes
#   sudo chrt -f 82 true && echo RT_OK   # SCHED_FIFO works
#   cat /sys/module/uvcvideo/parameters/nodrop        # 1
#   cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable   # 1
#   cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor  # performance
# ============================================================================
set -euo pipefail

SERVICE_NAME=camera-low-latency
SERVICE=/etc/systemd/system/${SERVICE_NAME}.service
HELPER=/usr/local/sbin/${SERVICE_NAME}-apply.sh
XHCI_IRQ_CPU=4          # first A76 big core
RT_USER_US=500000       # RT bandwidth for user.slice   (per 1s period)
RT_SYSTEM_US=400000     # RT bandwidth for system.slice (root total: 950000)

BOOT_PARAMS=(
    cma=512M
    usbcore.quirks=04b4:00f9:g
    uvcvideo.nodrop=1
    uvcvideo.timeout=5000
    uvcvideo.quirks=0x100
    iommu.passthrough=1
    pcie_aspm=off
    usbcore.autosuspend=-1
    systemd.unified_cgroup_hierarchy=0
)

# Old split services this script supersedes.
LEGACY_SERVICES=(disable-cpu-deep-idle.service rk3588-low-latency.service)
LEGACY_HELPERS=(/usr/local/sbin/rk3588-low-latency-apply.sh)

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (use sudo)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Boot env file handling (U-Boot; there is no GRUB on these boards)
# ---------------------------------------------------------------------------
find_env_file() {
    for f in /boot/orangepiEnv.txt /boot/armbianEnv.txt; do
        [[ -f "$f" ]] && { echo "$f"; return 0; }
    done
    return 1
}

# merge_boot_params <add|remove>
merge_boot_params() {
    local mode="$1"
    local env_file
    if ! env_file="$(find_env_file)"; then
        echo "ERROR: no /boot/orangepiEnv.txt or /boot/armbianEnv.txt found." >&2
        echo "       This board may use extlinux (/boot/extlinux/extlinux.conf)." >&2
        exit 1
    fi
    echo "Boot env file: $env_file"

    cp -a "$env_file" "${env_file}.bak.$(date +%Y%m%d-%H%M%S)"

    grep -qE '^extraargs=' "$env_file" || echo 'extraargs=' >> "$env_file"

    local raw current new changed=0
    raw="$(grep -E '^extraargs=' "$env_file" | head -n1)"
    current="$(sed -E 's/^extraargs=//; s/^"(.*)"$/\1/' <<<"$raw")"
    [[ "$raw" == *'"'* ]] && { changed=1; echo "  [fix]  removing literal quotes from extraargs"; }

    new="$current"
    local p key
    for p in "${BOOT_PARAMS[@]}"; do
        key="${p%%=*}"
        if [[ "$mode" == "remove" ]]; then
            if grep -qE "(^| )${key}=[^ ]*( |$)" <<<"$new"; then
                new="$(sed -E "s/(^| )${key}=[^ ]*( |$)/ /g; s/^ +//; s/ +$//" <<<"$new")"
                echo "  [del]  $p"
                changed=1
            fi
            continue
        fi
        if grep -qE "(^| )${key}=[^ ]*( |$)" <<<"$new"; then
            local exist
            exist="$(grep -oE "(^| )${key}=[^ ]*" <<<"$new" | tail -n1 | sed -E 's/^ //')"
            if [[ "$exist" == "$p" ]]; then
                echo "  [skip] $p already present"
                continue
            fi
            new="$(sed -E "s/(^| )${key}=[^ ]*( |$)/ /g; s/^ +//; s/ +$//" <<<"$new")"
        fi
        new="${new:+$new }$p"
        echo "  [add]  $p"
        changed=1
    done

    if [[ $changed -eq 1 ]]; then
        local tmp
        tmp="$(mktemp)"
        awk -v new="$new" '/^extraargs=/ { print "extraargs=" new; next } { print }' \
            "$env_file" > "$tmp"
        chown --reference="$env_file" "$tmp"
        chmod --reference="$env_file" "$tmp"
        mv "$tmp" "$env_file"
        echo "  extraargs = $new"
        NEED_REBOOT=1
    else
        echo "  extraargs already up to date"
    fi
}

# ---------------------------------------------------------------------------
# Runtime tuning (also the body of the persistence helper)
# ---------------------------------------------------------------------------
write_helper() {
    cat > "$HELPER" <<EOF
#!/bin/sh
# Applied at boot by ${SERVICE_NAME}.service (generated by setup_orangepi_camera.sh)

# 1. Disable deep CPU idle states, keep state0 (WFI).
for s in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]*/disable; do
    [ -e "\$s" ] && echo 1 > "\$s"
done

# 2. Performance governor on all clusters.
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    echo performance > "\$p"
done

# 3. Camera xHCI IRQ -> dedicated big core.
irq=\$(awk -F: '/xhci-hcd:usb1/ {gsub(/ /,"",\$1); print \$1; exit}' /proc/interrupts)
[ -n "\$irq" ] && echo ${XHCI_IRQ_CPU} > "/proc/irq/\$irq/smp_affinity_list"

# 4. RT bandwidth for slices (RT_GROUP_SCHED + cgroup v1; no-op on v2).
cg=/sys/fs/cgroup/cpu,cpuacct
[ -d "\$cg" ] || cg=/sys/fs/cgroup/cpu
if [ -f "\$cg/cpu.rt_runtime_us" ]; then
    [ -f "\$cg/user.slice/cpu.rt_runtime_us" ]   && echo ${RT_USER_US}   > "\$cg/user.slice/cpu.rt_runtime_us"
    [ -f "\$cg/system.slice/cpu.rt_runtime_us" ] && echo ${RT_SYSTEM_US} > "\$cg/system.slice/cpu.rt_runtime_us"
fi
exit 0
EOF
    chmod 755 "$HELPER"
}

install_service() {
    write_helper
    cat > "$SERVICE" <<EOF
[Unit]
Description=Camera capture low-latency tuning (idle states, governor, IRQ affinity, RT bandwidth)
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$HELPER

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service" >/dev/null
    echo "Installed $SERVICE (+ $HELPER)"
}

remove_legacy() {
    local s
    for s in "${LEGACY_SERVICES[@]}"; do
        if [[ -f "/etc/systemd/system/$s" ]]; then
            systemctl disable --now "$s" 2>/dev/null || true
            rm -f "/etc/systemd/system/$s"
            echo "  removed legacy $s"
        fi
    done
    rm -f "${LEGACY_HELPERS[@]}"
    systemctl daemon-reload
}

revert_runtime() {
    for s in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]*/disable; do
        [ -e "$s" ] && echo 0 > "$s"
    done
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        echo ondemand > "$p"
    done
    local cg=/sys/fs/cgroup/cpu,cpuacct
    [[ -d "$cg" ]] || cg=/sys/fs/cgroup/cpu
    if [[ -f "$cg/cpu.rt_runtime_us" ]]; then
        [[ -f "$cg/user.slice/cpu.rt_runtime_us" ]]   && echo 0 > "$cg/user.slice/cpu.rt_runtime_us"
        [[ -f "$cg/system.slice/cpu.rt_runtime_us" ]] && echo 0 > "$cg/system.slice/cpu.rt_runtime_us"
    fi
    echo "Runtime tuning reverted (idle states, governor, RT bandwidth)."
}

# ---------------------------------------------------------------------------
NEED_REBOOT=0

case "${1:-}" in
  --revert)
      echo "== Reverting boot parameters =="
      merge_boot_params remove
      echo
      echo "== Reverting runtime tuning =="
      revert_runtime
      if [[ -f "$SERVICE" ]]; then
          systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
          rm -f "$SERVICE" "$HELPER"
          systemctl daemon-reload
          echo "Removed $SERVICE"
      fi
      remove_legacy
      echo
      echo "Done. Reboot to restore the original kernel command line."
      exit 0
      ;;
  "")
      ;;
  *)
      echo "Usage: sudo bash $0 [--revert]" >&2
      exit 1
      ;;
esac

echo "== 1/4 Boot parameters (U-Boot env) =="
merge_boot_params add
echo
echo "== 2/4 Persistence service =="
remove_legacy
install_service
echo
echo "== 3/4 Applying runtime tuning now =="
"$HELPER"
echo "  done (idle states off, governor=performance, IRQ pinned, RT bandwidth set)"
echo
echo "== 4/4 Summary =="
echo "  governor : $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "  cpu-sleep: disable=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state1/disable 2>/dev/null || echo n/a)"
irq=$(awk -F: '/xhci-hcd:usb1/ {gsub(/ /,"",$1); print $1; exit}' /proc/interrupts || true)
[[ -n "${irq:-}" ]] && echo "  xhci irq : $irq -> cpu $(cat /proc/irq/$irq/smp_affinity_list)"
if chrt -f 50 true 2>/dev/null; then
    echo "  RT sched : working"
else
    echo "  RT sched : NOT yet available (expected before the first reboot)"
fi
echo
if [[ $NEED_REBOOT -eq 1 ]]; then
    echo ">>> REBOOT REQUIRED for boot parameters (cgroup v1 / uvcvideo / RT). <<<"
    echo ">>> Run: sudo reboot                                                <<<"
else
    echo "Boot parameters unchanged; no reboot needed."
fi
