#!/usr/bin/env bash
#
# ============================================================================
#  One-shot setup for NVIDIA Jetson Orin (JetPack 5 / L4T R35) camera hosts
# ============================================================================
# Orin port of setup_orangepi_camera.sh: every host-side measure validated
# against "failed: v4l2 get stream time out" / frame drops with the Cyper FX3
# UVC camera (04b4:00f9), adapted to the Jetson boot flow and CPU topology.
#
#   1. Kernel parameters (/boot/extlinux/extlinux.conf, every active APPEND
#      line, so whichever LABEL is DEFAULT -- e.g. JetsonIO -- gets them)
#        - cma=512M                        DMA contiguous memory for capture
#        - usbcore.quirks=04b4:00f9:g     delay-init quirk for the camera
#        - uvcvideo.nodrop=1              keep partial frames instead of drop
#        - uvcvideo.timeout=5000          longer streaming ctrl timeout
#        - uvcvideo.quirks=0x100          PROBE_DEF quirk
#        - pcie_aspm=off                  no PCIe power-state latency spikes
#        - usbcore.autosuspend=-1         never autosuspend USB devices
#        - systemd.unified_cgroup_hierarchy=0
#          only added when the cgroup-v1 cpu controller (cpu.rt_runtime_us)
#          is missing. L4T R35 already boots hybrid with cpu,cpuacct on v1,
#          which is all RT_GROUP_SCHED needs for SCHED_FIFO.
#        - iommu.passthrough=1            ONLY with --iommu-passthrough.
#          Tegra234 builds use ARM_SMMU_DISABLE_BYPASS_BY_DEFAULT; display,
#          GPU and PCIe clients rely on the SMMU, so this is opt-in.
#      uvcvideo is a loadable module on L4T, so its options are also written
#      to /etc/modprobe.d and applied to the live module when loaded.
#
#   2. Runtime tuning, applied now AND persisted via ONE systemd service
#      (orin-v4l2-low-latency.service):
#        - disable deep CPU idle states (keep WFI). Orin's "c7" state has a
#          5 ms exit latency, far worse than the RK3588 cpu-sleep state
#        - performance cpufreq governor on all clusters (schedutil default)
#        - pin ALL xHCI IRQs (tegra-xusb: xhci-hcd:usbN + 3610000.xhci) to a
#          dedicated core, CPU3 by default (XHCI_IRQ_CPU=N to override)
#        - RESERVE that core: it is removed from every cpuset group that
#          also has other CPUs (e.g. system_rest of ros_isolate.sh), every
#          thread still allowed on it loses only that bit (per-thread, so
#          SDK/ROS pinning is preserved), and a systemd system.conf.d
#          drop-in starts PID1 without it after reboot
#        - kernel.sched_rt_runtime_us=-1: no RT throttling (the default
#          950ms/1s budget pauses ALL SCHED_FIFO threads for 50ms)
#        - USB runtime PM off for every device (power/control=on)
#        - RT bandwidth for user.slice (500ms/s) and system.slice (400ms/s)
#          so SCHED_FIFO works from SSH sessions and services
#        - rtprio limit for the invoking user (limits.d) so the SDK capture
#          threads can request SCHED_FIFO without running as root
#
# Compatible with ros_cgroup_setup.service: this service is ordered after it,
# so the cpuset groups exist before the reserved core is removed from them.
# It is NOT ordered after multi-user.target or nx-max-performance.service.
# Both of those are After=multi-user.target while also WantedBy= it, and
# systemd 245 deletes that start job to break the ordering cycle. Waiting
# on nx-max-performance would pull this service into the same cycle, so the
# governor and idle settings would silently never be applied after reboot.
#
# Usage:
#   sudo bash setup_orin_v4l2_config.sh                     # install
#   sudo bash setup_orin_v4l2_config.sh --iommu-passthrough # + iommu param
#   sudo bash setup_orin_v4l2_config.sh --revert            # undo everything
#        bash setup_orin_v4l2_config.sh --dry-run           # show changes
#        bash setup_orin_v4l2_config.sh --status            # verify
#
# The script is idempotent. A REBOOT is required after the first install for
# the kernel parameters to take effect; the script never reboots by itself.
# ============================================================================
set -euo pipefail

SERVICE_NAME=orin-v4l2-low-latency
SERVICE=/etc/systemd/system/${SERVICE_NAME}.service
HELPER=/usr/local/sbin/${SERVICE_NAME}-apply.sh
STATE_DIR=/var/lib/${SERVICE_NAME}
AFFINITY_DROPIN=/etc/systemd/system.conf.d/10-orin-v4l2-cpuaffinity.conf
MODPROBE_CONF=/etc/modprobe.d/orin-v4l2-uvcvideo.conf
LIMITS_CONF=/etc/security/limits.d/99-orin-v4l2-rtprio.conf
EXTLINUX=${EXTLINUX:-/boot/extlinux/extlinux.conf}
CAMERA_USB_ID=04b4:00f9
XHCI_IRQ_CPU=${XHCI_IRQ_CPU:-3}
RT_USER_US=500000       # RT bandwidth for user.slice   (per 1s period)
RT_SYSTEM_US=400000     # RT bandwidth for system.slice (root total: 950000)
RT_USER=${SUDO_USER:-${USER:-root}}
DEFAULT_GOVERNOR=schedutil

NCPU=$(nproc --all)
if (( XHCI_IRQ_CPU < 0 || XHCI_IRQ_CPU >= NCPU )); then
    echo "ERROR: XHCI_IRQ_CPU=$XHCI_IRQ_CPU outside 0..$((NCPU - 1))." >&2
    exit 1
fi
ALL_MASK=$(( (1 << NCPU) - 1 ))
RESERVED_BIT=$(( 1 << XHCI_IRQ_CPU ))
NONRESERVED_MASK=$(printf '%x' $(( ALL_MASK & ~RESERVED_BIT )))
NONRESERVED_CPUS=$(for ((i = 0; i < NCPU; i++)); do
    (( i == XHCI_IRQ_CPU )) || printf '%s ' "$i"; done)
NONRESERVED_CPUS=${NONRESERVED_CPUS% }

BOOT_PARAMS=(
    cma=512M
    usbcore.quirks=04b4:00f9:g
    uvcvideo.nodrop=1
    uvcvideo.timeout=5000
    uvcvideo.quirks=0x100
    pcie_aspm=off
    usbcore.autosuspend=-1
)
# Removed by --revert even if they were added conditionally.
MANAGED_OPTIONAL_PARAMS=(systemd.unified_cgroup_hierarchy=0 iommu.passthrough=1)

CG_CPU=/sys/fs/cgroup/cpu,cpuacct
[[ -d "$CG_CPU" ]] || CG_CPU=/sys/fs/cgroup/cpu
CG_CPUSET=/sys/fs/cgroup/cpuset

# ---------------------------------------------------------------------------
# Kernel command line (extlinux)
# ---------------------------------------------------------------------------
# manage_args <add|remove> <label> <args>; result in NEW_ARGS
manage_args() {
    local mode="$1" label="$2" p key w exact
    local -a words kept
    read -r -a words <<<"$3"
    for p in "${PARAMS_IN_USE[@]}"; do
        key="${p%%=*}"
        exact=0
        kept=()
        for w in "${words[@]}"; do
            if [[ "$w" == "$key="* ]]; then
                [[ "$w" == "$p" ]] && exact=$((exact + 1))
                continue
            fi
            kept+=("$w")
        done
        local found=$(( ${#words[@]} - ${#kept[@]} ))
        if [[ "$mode" == "remove" ]]; then
            (( found )) && echo "  [del]  [$label] ${key}=..."
            words=("${kept[@]}")
            continue
        fi
        if (( found == 1 && exact == 1 )); then
            echo "  [skip] [$label] $p already present"
            continue
        fi
        words=("${kept[@]}" "$p")
        echo "  [add]  [$label] $p"
    done
    NEW_ARGS="${words[*]}"
}

# merge_boot_params <add|remove> <file>; sets BOOT_CHANGED
merge_boot_params() {
    local mode="$1" file="$2" out line label="" default="" n=0
    out="$(mktemp)"
    BOOT_CHANGED=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^[[:space:]]*DEFAULT[[:space:]]+([^[:space:]]+) ]]; then
            default="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^[[:space:]]*LABEL[[:space:]]+([^[:space:]]+) ]]; then
            label="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^([[:space:]]*APPEND[[:space:]]+)(.*)$ ]]; then
            local prefix="${BASH_REMATCH[1]}" args="${BASH_REMATCH[2]}"
            manage_args "$mode" "${label:-?}" "$args"
            [[ "$NEW_ARGS" != "$args" ]] && BOOT_CHANGED=1
            line="${prefix}${NEW_ARGS}"
            n=$((n + 1))
        fi
        printf '%s\n' "$line" >> "$out"
    done < "$file"
    if (( n == 0 )); then
        rm -f "$out"
        echo "ERROR: no active APPEND line in $file" >&2
        exit 1
    fi
    echo "  DEFAULT label: ${default:-<none>} ($n active APPEND line(s) processed)"
    MERGED_FILE="$out"
}

select_params() {
    PARAMS_IN_USE=("${BOOT_PARAMS[@]}")
    if [[ "${1:-}" == "remove" ]]; then
        PARAMS_IN_USE+=("${MANAGED_OPTIONAL_PARAMS[@]}")
        return
    fi
    if [[ ! -f "$CG_CPU/cpu.rt_runtime_us" ]]; then
        PARAMS_IN_USE+=(systemd.unified_cgroup_hierarchy=0)
    fi
    (( WITH_IOMMU )) && PARAMS_IN_USE+=(iommu.passthrough=1)
    return 0
}

apply_boot_params() {
    local mode="$1"
    [[ -f "$EXTLINUX" ]] || { echo "ERROR: $EXTLINUX not found." >&2; exit 1; }
    echo "Boot config: $EXTLINUX"
    select_params "$mode"
    merge_boot_params "$mode" "$EXTLINUX"
    if (( BOOT_CHANGED )); then
        cp -a "$EXTLINUX" "${EXTLINUX}.bak.orin-v4l2.$(date +%Y%m%d-%H%M%S)"
        chown --reference="$EXTLINUX" "$MERGED_FILE"
        chmod --reference="$EXTLINUX" "$MERGED_FILE"
        mv "$MERGED_FILE" "$EXTLINUX"
        sync
        echo "  extlinux.conf updated (backup kept next to it)"
        NEED_REBOOT=1
    else
        rm -f "$MERGED_FILE"
        echo "  kernel parameters already up to date"
    fi
}

write_modprobe_conf() {
    cat > "$MODPROBE_CONF" <<EOF
# Generated by setup_orin_v4l2_config.sh (mirrors the kernel cmdline).
options uvcvideo nodrop=1 timeout=5000 quirks=0x100
EOF
    echo "Installed $MODPROBE_CONF"
}

write_limits_conf() {
    [[ "$RT_USER" == "root" ]] && return 0
    cat > "$LIMITS_CONF" <<EOF
# Generated by setup_orin_v4l2_config.sh: SCHED_FIFO for the camera SDK user.
$RT_USER  -  rtprio   99
$RT_USER  -  memlock  unlimited
EOF
    echo "Installed $LIMITS_CONF (user $RT_USER, effective at next login)"
}

# ---------------------------------------------------------------------------
# Runtime tuning (also the body of the persistence helper)
# ---------------------------------------------------------------------------
write_helper() {
    mkdir -p "$STATE_DIR"
    cat > "$HELPER" <<EOF
#!/bin/bash
# ${SERVICE_NAME}-apply.sh v1 (generated by setup_orin_v4l2_config.sh)
# Applied at boot by ${SERVICE_NAME}.service; safe to re-run any time.
R=${XHCI_IRQ_CPU}
RBIT=${RESERVED_BIT}
STATE_DIR=${STATE_DIR}

# 1. Disable deep CPU idle states, keep state0 (WFI).
for s in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]*/disable; do
    [ -e "\$s" ] && echo 1 > "\$s"
done

# 2. Performance governor on all clusters.
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    echo performance > "\$p" 2>/dev/null
done

# 3. ALL xHCI IRQs -> dedicated core CPU\$R.
for irq in \$(awk -F: '/xhci/ {gsub(/ /,"",\$1); print \$1}' /proc/interrupts); do
    echo \$R > "/proc/irq/\$irq/smp_affinity_list" 2>/dev/null
done

# 4. No global RT throttling.
sysctl -qw kernel.sched_rt_runtime_us=-1

# 5. RT bandwidth for slices (RT_GROUP_SCHED refuses SCHED_FIFO from a
#    group whose own rt_runtime is 0, even with the global limit off).
cg=${CG_CPU}
if [ -f "\$cg/cpu.rt_runtime_us" ]; then
    [ -f "\$cg/user.slice/cpu.rt_runtime_us" ]   && echo ${RT_USER_US}   > "\$cg/user.slice/cpu.rt_runtime_us"
    [ -f "\$cg/system.slice/cpu.rt_runtime_us" ] && echo ${RT_SYSTEM_US} > "\$cg/system.slice/cpu.rt_runtime_us"
fi

# 6. USB never autosuspends.
for f in /sys/bus/usb/devices/*/power/control; do
    echo on > "\$f" 2>/dev/null
done

# 7. uvcvideo options on the live module (cmdline/modprobe.d cover reloads).
m=/sys/module/uvcvideo/parameters
if [ -d "\$m" ]; then
    echo 1    > "\$m/nodrop"  2>/dev/null
    echo 5000 > "\$m/timeout" 2>/dev/null
    echo 256  > "\$m/quirks"  2>/dev/null
fi

# 8. Take CPU\$R out of every cpuset group that also owns other CPUs
#    (ros_isolate.sh system_rest = 0-4).  The original list is saved once
#    for --revert.  Changing cpuset.cpus resets member task masks.
expand() {
    echo "\$1" | tr ',' '\n' | while IFS=- read a b; do
        [ -n "\$a" ] && seq "\$a" "\${b:-\$a}"
    done
}
mkdir -p "\$STATE_DIR"
for f in ${CG_CPUSET}/*/cpuset.cpus; do
    [ -f "\$f" ] || continue
    cur=\$(cat "\$f")
    cpus=\$(expand "\$cur")
    echo "\$cpus" | grep -qx "\$R" || continue
    rest=\$(echo "\$cpus" | grep -vx "\$R" | paste -sd, -)
    [ -n "\$rest" ] || continue
    g=\$(basename "\$(dirname "\$f")")
    [ -f "\$STATE_DIR/cpuset.\$g" ] || echo "\$cur" > "\$STATE_DIR/cpuset.\$g"
    echo "\$rest" > "\$f" 2>/dev/null
done

# 9. Remaining threads (root cpuset, kthreads): clear only the CPU\$R bit,
#    per thread, so explicit pinning elsewhere survives.  Threads allowed
#    ONLY on CPU\$R (per-cpu kthreads) are left alone.
for t in /proc/[0-9]*/task/[0-9]*; do
    tid=\${t##*/}
    m=\$(taskset -p "\$tid" 2>/dev/null | awk '{print \$NF}')
    [ -n "\$m" ] || continue
    cur=\$((0x\$m))
    [ \$((cur & RBIT)) -ne 0 ] || continue
    new=\$((cur & ~RBIT))
    [ \$new -ne 0 ] || continue
    taskset -p "\$(printf '%x' \$new)" "\$tid" >/dev/null 2>&1
done

exit 0
EOF
    chmod 755 "$HELPER"
}

write_affinity_dropin() {
    mkdir -p "$(dirname "$AFFINITY_DROPIN")"
    cat > "$AFFINITY_DROPIN" <<EOF
# Generated by setup_orin_v4l2_config.sh: reserve CPU${XHCI_IRQ_CPU} for the
# camera xHCI IRQ.  PID1 and everything it spawns start on the other cores.
[Manager]
CPUAffinity=${NONRESERVED_CPUS}
EOF
    echo "Installed $AFFINITY_DROPIN (takes effect for new processes after reboot)"
}

install_service() {
    write_helper
    write_affinity_dropin
    write_modprobe_conf
    write_limits_conf
    cat > "$SERVICE" <<EOF
[Unit]
Description=Orin V4L2 camera low-latency tuning (idle, governor, xHCI IRQ, reserved core, RT, USB PM)
# After ros_cgroup_setup only. Do not add multi-user.target or
# nx-max-performance.service here: both are ordered after multi-user.target
# and also wanted by it, and systemd 245 drops that job as an ordering cycle.
After=ros_cgroup_setup.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$HELPER

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "${SERVICE_NAME}.service" >/dev/null
    echo "Installed and started $SERVICE (+ $HELPER)"
}

revert_runtime() {
    local s p f g t tid m cur
    for s in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]*/disable; do
        [[ -e "$s" ]] && echo 0 > "$s"
    done
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        echo "$DEFAULT_GOVERNOR" > "$p" 2>/dev/null || true
    done
    sysctl -qw kernel.sched_rt_runtime_us=950000
    for f in /sys/bus/usb/devices/*/power/control; do
        echo auto > "$f" 2>/dev/null || true
    done
    for irq in $(awk -F: '/xhci/ {gsub(/ /,"",$1); print $1}' /proc/interrupts); do
        echo "0-$((NCPU - 1))" > "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || true
    done
    if [[ -f "$CG_CPU/cpu.rt_runtime_us" ]]; then
        [[ -f "$CG_CPU/user.slice/cpu.rt_runtime_us" ]]   && echo 0 > "$CG_CPU/user.slice/cpu.rt_runtime_us"
        [[ -f "$CG_CPU/system.slice/cpu.rt_runtime_us" ]] && echo 0 > "$CG_CPU/system.slice/cpu.rt_runtime_us"
    fi
    for f in "$STATE_DIR"/cpuset.*; do
        [[ -f "$f" ]] || continue
        g="${f##*/cpuset.}"
        [[ -f "$CG_CPUSET/$g/cpuset.cpus" ]] && cat "$f" > "$CG_CPUSET/$g/cpuset.cpus" 2>/dev/null || true
        echo "  cpuset $g restored to $(cat "$f")"
    done
    for t in /proc/[0-9]*/task/[0-9]*; do
        tid=${t##*/}
        m=$(taskset -p "$tid" 2>/dev/null | awk '{print $NF}') || continue
        [[ -n "$m" ]] || continue
        cur=$((16#$m))
        (( cur == (ALL_MASK & ~RESERVED_BIT) )) || continue
        taskset -p "$(printf '%x' "$ALL_MASK")" "$tid" >/dev/null 2>&1 || true
    done
    rm -rf "$STATE_DIR"
    rm -f "$AFFINITY_DROPIN" "$MODPROBE_CONF" "$LIMITS_CONF"
    echo "Runtime tuning reverted (idle, governor=$DEFAULT_GOVERNOR, RT throttling, USB PM, IRQs, core reservation)."
    if systemctl is-enabled nx-max-performance.service >/dev/null 2>&1; then
        echo "  note: nx-max-performance.service is enabled and sets performance again at boot."
    fi
}

# ---------------------------------------------------------------------------
# Read-only reporting
# ---------------------------------------------------------------------------
show_status() {
    local p irq cmdline
    cmdline=" $(cat /proc/cmdline) "
    echo "== Kernel cmdline =="
    for p in "${BOOT_PARAMS[@]}" "${MANAGED_OPTIONAL_PARAMS[@]}"; do
        if [[ "$cmdline" == *" $p "* ]]; then
            echo "  [ok]   $p"
        else
            echo "  [--]   $p"
        fi
    done
    echo "== Runtime =="
    echo "  governor : $(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort -u | paste -sd' ' -)"
    echo "  c7 idle  : disable=$(cat /sys/devices/system/cpu/cpu*/cpuidle/state1/disable 2>/dev/null | sort -u | paste -sd' ' -)"
    for irq in $(awk -F: '/xhci/ {gsub(/ /,"",$1); print $1}' /proc/interrupts); do
        echo "  xhci irq : $irq $(awk -v i="$irq:" '$1==i {print $NF}' /proc/interrupts) -> cpu $(cat /proc/irq/$irq/smp_affinity_list)"
    done
    echo "  rt limit : $(sysctl -n kernel.sched_rt_runtime_us) (-1 = throttling off)"
    echo "  slice rt : user=$(cat "$CG_CPU/user.slice/cpu.rt_runtime_us" 2>/dev/null || echo n/a)" \
         "system=$(cat "$CG_CPU/system.slice/cpu.rt_runtime_us" 2>/dev/null || echo n/a)"
    echo "  pid1 mask: $(taskset -p 1 | awk '{print $NF}') (${NONRESERVED_MASK} = CPU${XHCI_IRQ_CPU} reserved)"
    for p in "$CG_CPUSET"/*/cpuset.cpus; do
        [[ -f "$p" ]] && echo "  cpuset   : $(basename "$(dirname "$p")") = $(cat "$p")"
    done
    echo "  usb pm   : $(cat /sys/bus/usb/devices/*/power/control 2>/dev/null | sort | uniq -c | paste -sd' ' -)"
    if [[ -d /sys/module/uvcvideo/parameters ]]; then
        echo "  uvcvideo : nodrop=$(cat /sys/module/uvcvideo/parameters/nodrop)" \
             "timeout=$(cat /sys/module/uvcvideo/parameters/timeout)" \
             "quirks=$(cat /sys/module/uvcvideo/parameters/quirks)"
    else
        echo "  uvcvideo : module not loaded"
    fi
    echo "  camera   : $(lsusb -d "$CAMERA_USB_ID" 2>/dev/null || true)"
    if chrt -f 50 true 2>/dev/null; then
        echo "  RT sched : working for $(id -un)"
    else
        echo "  RT sched : NOT available for $(id -un)"
    fi
    echo "  service  : $(systemctl is-enabled "${SERVICE_NAME}.service" 2>/dev/null || echo not-installed)"
}

dry_run() {
    echo "== Kernel parameters (no files are written) =="
    select_params add
    merge_boot_params add "$EXTLINUX"
    diff -u "$EXTLINUX" "$MERGED_FILE" || true
    rm -f "$MERGED_FILE"
    echo
    echo "== Would install =="
    echo "  $SERVICE"
    echo "  $HELPER  (xHCI IRQ + reserved core: CPU${XHCI_IRQ_CPU})"
    echo "  $AFFINITY_DROPIN  (CPUAffinity=${NONRESERVED_CPUS})"
    echo "  $MODPROBE_CONF"
    [[ "$RT_USER" != "root" ]] && echo "  $LIMITS_CONF  (user $RT_USER)"
    return 0
}

# ---------------------------------------------------------------------------
NEED_REBOOT=0
WITH_IOMMU=0
ACTION=install
for arg in "$@"; do
    case "$arg" in
        --revert)            ACTION=revert ;;
        --status)            ACTION=status ;;
        --dry-run)           ACTION=dry-run ;;
        --iommu-passthrough) WITH_IOMMU=1 ;;
        *) echo "Usage: sudo bash $0 [--iommu-passthrough] [--revert] | bash $0 --status|--dry-run" >&2
           exit 1 ;;
    esac
done

case "$ACTION" in
    status)  show_status; exit 0 ;;
    dry-run) dry_run; exit 0 ;;
esac

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (use sudo)." >&2
    exit 1
fi

if [[ "$ACTION" == "revert" ]]; then
    echo "== Reverting kernel parameters =="
    apply_boot_params remove
    echo
    echo "== Reverting runtime tuning =="
    if [[ -f "$SERVICE" ]]; then
        systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
    fi
    revert_runtime
    rm -f "$SERVICE" "$HELPER"
    systemctl daemon-reload
    echo
    echo "Done. Reboot to restore the original kernel command line."
    exit 0
fi

echo "== 1/4 Kernel parameters (extlinux) =="
apply_boot_params add
echo
echo "== 2/4 Persistence service =="
install_service
echo
echo "== 3/4 Applying runtime tuning now =="
"$HELPER"
echo "  done (idle off, governor=performance, xHCI IRQs -> CPU${XHCI_IRQ_CPU}," \
     "CPU${XHCI_IRQ_CPU} reserved, RT throttling off, USB PM off)"
echo
echo "== 4/4 Summary =="
show_status
echo
if [[ $NEED_REBOOT -eq 1 ]]; then
    echo ">>> REBOOT REQUIRED for kernel parameters (cma / usbcore / uvcvideo). <<<"
    echo ">>> Run: sudo reboot                                                 <<<"
else
    echo "Kernel parameters unchanged; no reboot needed."
fi
