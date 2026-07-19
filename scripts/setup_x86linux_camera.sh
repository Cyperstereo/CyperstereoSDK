#!/usr/bin/env bash
#
# ============================================================================
#  One-shot setup for x86 Linux camera-capture hosts  (v2)
# ============================================================================
# Applies every host-side measure validated against Cyper FX3 UVC camera
# (04b4:00f9) stream stalls, frame drops and timestamp anomalies.
#
# Three layers (v1 had only the first two; v2 adds the third after a 374 ms
# capture-thread starvation on a GRUB-only-tuned host produced a 566 ms /
# 17-frame outage -- FIFO threads can only be starved like that with RT
# throttling active and an untuned scheduler):
#
#   1. GRUB kernel parameters (/etc/default/grub, idempotent):
#        - pcie_aspm=off
#        - intel_idle.max_cstate=1
#        - processor.max_cstate=1
#        - usbcore.autosuspend=-1
#        - uvcvideo.nodrop=1          deliver partial frames (SDK filters bad
#                                       ones) instead of silent kernel drop
#        - uvcvideo.quirks=0x100        PROBE_DEF quirk (match Orange Pi)
#
#   2. Runtime uvcvideo module parameters (immediate, no reboot):
#        - echo 1 > /sys/module/uvcvideo/parameters/nodrop
#        - echo 0x100 > /sys/module/uvcvideo/parameters/quirks
#      quirks affects newly probed devices -- replug the camera (or reboot)
#      after the first install so quirks takes effect on the current device.
#
#   3. Runtime scheduling/power tuning, applied now AND persisted via ONE
#      systemd service (camera-low-latency.service), mirroring the tuning
#      validated on the Orange Pi 5 Plus (setup_orangepi_camera.sh):
#        - kernel.sched_rt_runtime_us=-1: no RT throttling.  The default
#          950ms/1s budget force-idles ALL SCHED_FIFO threads (the SDK
#          capture thread runs FIFO 82) once the budget is spent -- the
#          only way a FIFO-82 thread loses the CPU to normal tasks
#        - performance cpufreq governor on all policies (powersave ramps
#          too slowly for per-frame 5 MB bursts)
#        - deep C-state guard: disable cpuidle states with exit latency
#          >20 us (belt-and-suspenders on top of the GRUB max_cstate=1)
#        - pin ALL xHCI IRQs to the highest-numbered CPU, and RESERVE that
#          CPU's whole physical core (both HT siblings): existing tasks are
#          swept off it (taskset) and a systemd system.conf.d drop-in makes
#          PID1 start new processes off it after reboot.  Only the xHCI
#          hardirq + uvcvideo URB work run there
#        - USB runtime PM off for every device (power/control=on)
#        - irqbalance disabled if active (it would re-shuffle the pinned
#          IRQ within minutes); recorded and re-enabled on --revert
#        - cgroup v1 RT-slice grants (no-op on cgroup v2 / Ubuntu default,
#          kept for parity with vendor kernels that use RT_GROUP_SCHED)
#
# The script is idempotent (safe to re-run).
#
# Usage:
#   sudo bash scripts/setup_x86linux_camera.sh              # all + reboot
#   sudo bash scripts/setup_x86linux_camera.sh --no-reboot  # all, no reboot
#   sudo bash scripts/setup_x86linux_camera.sh --now        # runtime only
#                                                   (uvc params + tuning +
#                                                    service; GRUB untouched)
#   sudo bash scripts/setup_x86linux_camera.sh --revert     # undo runtime
#                                                   layer + service (GRUB is
#                                                   restored from its .bak)
#
# Verify:
#   cat /proc/cmdline | tr ' ' '\n' | grep uvcvideo
#   cat /sys/module/uvcvideo/parameters/nodrop        # 1
#   cat /sys/module/uvcvideo/parameters/quirks        # 256 (0x100)
#   sysctl -n kernel.sched_rt_runtime_us              # -1
#   cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor # performance
#   cat /proc/irq/$(awk -F: '/xhci/{gsub(/ /,"",$1);print $1;exit}' \
#       /proc/interrupts)/smp_affinity_list           # last CPU
#   taskset -p 1                                      # mask without reserved core
# ============================================================================
set -euo pipefail

GRUB_FILE="/etc/default/grub"
NODROP_SYSFS=/sys/module/uvcvideo/parameters/nodrop
QUIRKS_SYSFS=/sys/module/uvcvideo/parameters/quirks

SERVICE_NAME=camera-low-latency
SERVICE=/etc/systemd/system/${SERVICE_NAME}.service
HELPER=/usr/local/sbin/${SERVICE_NAME}-apply.sh
AFFINITY_DROPIN=/etc/systemd/system.conf.d/10-camera-cpuaffinity.conf
IRQBALANCE_MARKER=/var/lib/${SERVICE_NAME}.irqbalance-was-active
RT_USER_US=500000       # cgroup v1 RT bandwidth for user.slice   (parity w/ Pi)
RT_SYSTEM_US=400000     # cgroup v1 RT bandwidth for system.slice

GRUB_PARAMS=(
    pcie_aspm=off
    intel_idle.max_cstate=1
    processor.max_cstate=1
    usbcore.autosuspend=-1
    uvcvideo.nodrop=1
    uvcvideo.quirks=0x100
)

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (use sudo)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# CPU topology: reserve the whole physical core containing the last CPU.
# On SMT machines a single reserved logical CPU still shares execution
# resources with its hyper-thread sibling, so both siblings are reserved
# (e.g. i7-7700HQ: CPUs 3,7 = core 3).  On no-SMT / hybrid E-cores the
# sibling list collapses to the CPU itself.  Fewer than 4 CPUs: reservation
# is skipped (IRQ pinning alone still helps).
# ---------------------------------------------------------------------------
NCPU="$(nproc --all)"
IRQ_CPU=$((NCPU - 1))
RESERVATION_ENABLED=1
RESERVED_LIST=""
NONRESERVED_LIST=""
NONRESERVED_MASK=""
ALL_MASK_HEX=""

detect_topology() {
    ALL_MASK_HEX="$(printf '%x' $(( (1 << NCPU) - 1 )))"
    if (( NCPU < 4 || NCPU > 32 )); then
        RESERVATION_ENABLED=0
        return
    fi
    local sib_file="/sys/devices/system/cpu/cpu${IRQ_CPU}/topology/thread_siblings"
    local reserved_mask
    if [[ -r "$sib_file" ]]; then
        reserved_mask=$((16#$(tr -d ',\n' < "$sib_file")))
    else
        reserved_mask=$((1 << IRQ_CPU))
    fi
    local nonres_mask=$(( ((1 << NCPU) - 1) & ~reserved_mask ))
    NONRESERVED_MASK="$(printf '%x' "$nonres_mask")"
    local i
    for ((i = 0; i < NCPU; i++)); do
        if (( (reserved_mask >> i) & 1 )); then
            RESERVED_LIST="${RESERVED_LIST:+$RESERVED_LIST,}$i"
        else
            NONRESERVED_LIST="${NONRESERVED_LIST:+$NONRESERVED_LIST }$i"
        fi
    done
}

# ---------------------------------------------------------------------------
# Layer 2: runtime uvcvideo module parameters
# ---------------------------------------------------------------------------
apply_runtime_uvc() {
    if [[ ! -d /sys/module/uvcvideo ]]; then
        echo "WARN: uvcvideo module not loaded; runtime nodrop/quirks skipped."
        echo "      Plug in the camera or run: sudo modprobe uvcvideo"
        return 0
    fi

    if [[ -w "$NODROP_SYSFS" ]]; then
        echo 1 > "$NODROP_SYSFS"
        echo "  [runtime] nodrop=$(cat "$NODROP_SYSFS")"
    else
        echo "WARN: cannot write $NODROP_SYSFS" >&2
    fi

    if [[ -w "$QUIRKS_SYSFS" ]]; then
        echo 0x100 > "$QUIRKS_SYSFS"
        echo "  [runtime] quirks=$(cat "$QUIRKS_SYSFS") (0x100; replug camera to apply to current device)"
    else
        echo "WARN: cannot write $QUIRKS_SYSFS" >&2
    fi
}

# ---------------------------------------------------------------------------
# Layer 1: GRUB kernel parameters
# ---------------------------------------------------------------------------
merge_grub_params() {
    if [[ ! -f "$GRUB_FILE" ]]; then
        echo "ERROR: $GRUB_FILE not found." >&2
        exit 1
    fi

    TS="$(date +%Y%m%d-%H%M%S)"
    BACKUP="${GRUB_FILE}.bak.${TS}"
    cp -a "$GRUB_FILE" "$BACKUP"
    echo "Backed up $GRUB_FILE -> $BACKUP"

    if ! grep -qE '^GRUB_CMDLINE_LINUX=' "$GRUB_FILE"; then
        echo 'GRUB_CMDLINE_LINUX=""' >> "$GRUB_FILE"
    fi

    CURRENT="$(grep -E '^GRUB_CMDLINE_LINUX=' "$GRUB_FILE" | sed -E 's/^GRUB_CMDLINE_LINUX=//; s/^"(.*)"$/\1/')"

    CHANGED=0
    NEW="$CURRENT"
    local p key
    for p in "${GRUB_PARAMS[@]}"; do
        key="${p%%=*}"
        if [[ "$key" == "$p" ]]; then
            if grep -qE "(^| )${p}( |$)" <<<"$NEW"; then
                echo "  [skip] $p already present"
                continue
            fi
        else
            if grep -qE "(^| )${key}=[^ ]*( |$)" <<<"$NEW"; then
                local exist
                exist="$(grep -oE "(^| )${key}=[^ ]*" <<<"$NEW" | tail -n1 | sed -E 's/^ //')"
                if [[ "$exist" == "$p" ]]; then
                    echo "  [skip] $p already present"
                    continue
                fi
                NEW="$(echo "$NEW" | sed -E "s/(^| )${key}=[^ ]*( |$)/ /g; s/^ +//; s/ +$//")"
            fi
        fi
        if [[ -z "$NEW" ]]; then
            NEW="$p"
        else
            NEW="$NEW $p"
        fi
        echo "  [add]  $p"
        CHANGED=1
    done

    if [[ $CHANGED -eq 0 ]]; then
        echo "GRUB_CMDLINE_LINUX already up to date."
        echo "Current value: \"$CURRENT\""
    else
        TMP="$(mktemp)"
        awk -v new="$NEW" '
            /^GRUB_CMDLINE_LINUX=/ { print "GRUB_CMDLINE_LINUX=\"" new "\""; next }
            { print }
        ' "$GRUB_FILE" > "$TMP"
        chown --reference="$GRUB_FILE" "$TMP"
        chmod --reference="$GRUB_FILE" "$TMP"
        mv "$TMP" "$GRUB_FILE"
        echo "Updated GRUB_CMDLINE_LINUX to: \"$NEW\""
    fi

    echo
    echo "--- /etc/default/grub (CMDLINE lines) ---"
    grep -nE '^GRUB_CMDLINE_LINUX' "$GRUB_FILE"
    echo "-----------------------------------------"

    echo
    echo "Running update-grub..."
    if command -v update-grub >/dev/null 2>&1; then
        update-grub
    else
        grub2-mkconfig -o /boot/grub2/grub.cfg 2>/dev/null \
            || grub-mkconfig -o /boot/grub/grub.cfg
    fi
    echo "update-grub done."
}

# ---------------------------------------------------------------------------
# Layer 3: scheduling/power tuning (helper body = what runs at every boot)
# ---------------------------------------------------------------------------
write_helper() {
    cat > "$HELPER" <<EOF
#!/bin/sh
# ${SERVICE_NAME}-apply.sh (x86 v2, generated by setup_x86linux_camera.sh)
# Applied at boot by ${SERVICE_NAME}.service; safe to re-run any time.

# 1. No RT throttling: the default 950ms/1s budget force-idles ALL
#    SCHED_FIFO threads once spent -- the only way the FIFO-82 capture
#    thread can be starved by normal tasks (observed: 374 ms starvation
#    -> 566 ms / 17-frame outage).
sysctl -qw kernel.sched_rt_runtime_us=-1

# 2. Performance governor on all policies (powersave ramps too slowly for
#    per-frame 5 MB burst work).
for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    [ -w "\$p" ] && echo performance > "\$p"
done

# 3. Deep C-state guard: GRUB max_cstate=1 is the primary control; this
#    catches kernels booted without it.  Keep states with <=20 us exit
#    latency (POLL/C1), disable deeper ones.
for d in /sys/devices/system/cpu/cpu*/cpuidle/state*; do
    [ -f "\$d/latency" ] || continue
    if [ "\$(cat "\$d/latency")" -gt 20 ]; then
        echo 1 > "\$d/disable" 2>/dev/null
    fi
done

# 4. ALL xHCI IRQs -> CPU${IRQ_CPU} (reserved core; robust if the camera
#    moves to another port of the same controller).
for irq in \$(awk -F: '/xhci/ {gsub(/ /,"",\$1); print \$1}' /proc/interrupts); do
    echo ${IRQ_CPU} > "/proc/irq/\$irq/smp_affinity_list" 2>/dev/null
done

# 5. USB never autosuspends (cmdline usbcore.autosuspend=-1 covers the
#    delay; control=on removes runtime PM entirely).
for f in /sys/bus/usb/devices/*/power/control; do
    echo on > "\$f" 2>/dev/null
done

# 6. cgroup v1 RT-slice grants (no-op on cgroup v2 / kernels without
#    RT_GROUP_SCHED; kept for parity with the Orange Pi vendor kernel).
cg=/sys/fs/cgroup/cpu,cpuacct
[ -d "\$cg" ] || cg=/sys/fs/cgroup/cpu
if [ -f "\$cg/cpu.rt_runtime_us" ]; then
    [ -f "\$cg/user.slice/cpu.rt_runtime_us" ]   && echo ${RT_USER_US}   > "\$cg/user.slice/cpu.rt_runtime_us"
    [ -f "\$cg/system.slice/cpu.rt_runtime_us" ] && echo ${RT_SYSTEM_US} > "\$cg/system.slice/cpu.rt_runtime_us"
fi
EOF

    if [[ $RESERVATION_ENABLED -eq 1 ]]; then
        cat >> "$HELPER" <<EOF

# 7. Keep ALL existing userspace + movable kthreads off the reserved core
#    (CPUs ${RESERVED_LIST}: the physical core handling the xHCI hardirq,
#    both HT siblings) so URB completion never waits behind a scheduled
#    task.  Per-cpu kernel threads refuse the change (PF_NO_SETAFFINITY),
#    which is correct.  New processes inherit the mask from PID1 (systemd
#    CPUAffinity drop-in installed alongside this helper).
for p in \$(ps -e -o pid=); do
    taskset -a -p ${NONRESERVED_MASK} "\$p" >/dev/null 2>&1
done
EOF
    fi

    echo "exit 0" >> "$HELPER"
    chmod 755 "$HELPER"
}

write_affinity_dropin() {
    [[ $RESERVATION_ENABLED -eq 1 ]] || return 0
    mkdir -p "$(dirname "$AFFINITY_DROPIN")"
    cat > "$AFFINITY_DROPIN" <<EOF
# Generated by setup_x86linux_camera.sh: reserve CPUs ${RESERVED_LIST}
# (one whole physical core) for the camera xHCI IRQ.  PID1 and everything
# it spawns start on the other CPUs; combined with the taskset sweep in
# ${SERVICE_NAME}-apply.sh this keeps the reserved core free of tasks.
[Manager]
CPUAffinity=${NONRESERVED_LIST}
EOF
    echo "Installed $AFFINITY_DROPIN (takes effect for new processes after reboot)"
}

disable_irqbalance() {
    if systemctl is-active --quiet irqbalance 2>/dev/null; then
        systemctl disable --now irqbalance
        touch "$IRQBALANCE_MARKER"
        echo "  irqbalance was active: disabled (it would re-shuffle the pinned IRQ)"
    fi
}

install_service() {
    write_helper
    write_affinity_dropin
    cat > "$SERVICE" <<EOF
[Unit]
Description=Camera capture low-latency tuning (RT throttling, governor, C-states, IRQ affinity, reserved core, USB PM)
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

revert_runtime() {
    sysctl -qw kernel.sched_rt_runtime_us=950000
    local p f d
    for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        [[ -w "$p" ]] && echo powersave > "$p" || true
    done
    for d in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        [[ -e "$d" ]] && echo 0 > "$d" || true
    done
    for f in /sys/bus/usb/devices/*/power/control; do
        echo auto > "$f" 2>/dev/null || true
    done
    for p in $(ps -e -o pid=); do
        taskset -a -p "$ALL_MASK_HEX" "$p" >/dev/null 2>&1 || true
    done
    rm -f "$AFFINITY_DROPIN"
    if [[ -f "$IRQBALANCE_MARKER" ]]; then
        systemctl enable --now irqbalance 2>/dev/null || true
        rm -f "$IRQBALANCE_MARKER"
        echo "  irqbalance re-enabled"
    fi
    echo "Runtime tuning reverted (RT throttling, governor, C-states, USB PM, core reservation)."
}

print_summary() {
    echo "  rt limit : $(sysctl -n kernel.sched_rt_runtime_us) (-1 = throttling off)"
    echo "  governor : $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
    local irq
    for irq in $(awk -F: '/xhci/ {gsub(/ /,"",$1); print $1}' /proc/interrupts); do
        echo "  xhci irq : $irq -> cpu $(cat "/proc/irq/$irq/smp_affinity_list")"
    done
    if [[ $RESERVATION_ENABLED -eq 1 ]]; then
        echo "  reserved : CPUs ${RESERVED_LIST} (IRQ core), others=${NONRESERVED_LIST}"
        echo "  pid1 mask: $(taskset -p 1 | awk '{print $NF}') (expect ${NONRESERVED_MASK})"
    else
        echo "  reserved : (skipped: $NCPU CPUs)"
    fi
    if chrt -f 50 true 2>/dev/null; then
        echo "  RT sched : working"
    else
        echo "  RT sched : NOT available -- check kernel/cgroup config"
    fi
}

# ---------------------------------------------------------------------------
detect_topology

REBOOT=1
RUNTIME_ONLY=0
case "${1:-}" in
    --revert)
        echo "== Reverting runtime tuning =="
        revert_runtime
        if [[ -f "$SERVICE" ]]; then
            systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
            rm -f "$SERVICE" "$HELPER"
            systemctl daemon-reload
            echo "Removed $SERVICE"
        fi
        echo
        echo "GRUB was NOT touched; restore from ${GRUB_FILE}.bak.* + update-grub"
        echo "if you also want the boot parameters gone."
        exit 0
        ;;
    --no-reboot) REBOOT=0 ;;
    --now) RUNTIME_ONLY=1; REBOOT=0 ;;
    "") ;;
    -h|--help)
        sed -n '2,72p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "ERROR: unknown option: ${1} (try --help)" >&2
        exit 1
        ;;
esac

echo "=== x86 camera setup (v2) ==="

echo "[1/3] Runtime uvcvideo parameters"
apply_runtime_uvc
echo

echo "[2/3] Scheduling/power tuning + persistence service"
disable_irqbalance
install_service
"$HELPER"
echo "  applied (RT throttling off, governor=performance, IRQs -> CPU${IRQ_CPU}," \
     "core {${RESERVED_LIST:-none}} reserved, USB PM off)"
echo

if [[ $RUNTIME_ONLY -eq 1 ]]; then
    echo "[3/3] GRUB kernel parameters: SKIPPED (--now)"
else
    echo "[3/3] GRUB kernel parameters"
    merge_grub_params
fi

echo
echo "=== verify ==="
if [[ -r "$NODROP_SYSFS" ]]; then
    echo "  nodrop=$(cat "$NODROP_SYSFS")"
else
    echo "  nodrop=(module not loaded)"
fi
if [[ -r "$QUIRKS_SYSFS" ]]; then
    echo "  quirks=$(cat "$QUIRKS_SYSFS")"
else
    echo "  quirks=(module not loaded)"
fi
grep -oE 'uvcvideo\.[^ ]+' /proc/cmdline 2>/dev/null || true
print_summary

if [[ $RUNTIME_ONLY -eq 1 ]]; then
    echo
    echo "Runtime-only mode (--now). GRUB untouched; replug the camera if"
    echo "quirks did not change. Reserved-core PID1 affinity needs a reboot"
    echo "to cover new processes (existing ones were swept already)."
    exit 0
fi

if [[ $REBOOT -eq 1 ]]; then
    echo
    echo "Rebooting in 3 seconds so GRUB params take effect... (Ctrl-C to cancel)"
    sleep 3
    reboot
else
    echo
    echo "Skipping reboot (--no-reboot). Reboot manually to apply GRUB params"
    echo "and the PID1 CPUAffinity drop-in. Runtime tuning is already active."
fi
