#!/usr/bin/env bash
#
# Permanently apply low-latency / stability kernel parameters to GRUB.
#
# Adds the following to GRUB_CMDLINE_LINUX (idempotent — safe to re-run):
#   pcie_aspm=off
#   intel_idle.max_cstate=1
#   processor.max_cstate=1
#   usbcore.autosuspend=-1
#
# Then runs `sudo update-grub` and reboots.
#
# Usage:
#   sudo bash scripts/apply_grub_tuning.sh          # apply + reboot
#   sudo bash scripts/apply_grub_tuning.sh --no-reboot   # apply only
#
set -euo pipefail

GRUB_FILE="/etc/default/grub"
PARAMS=(pcie_aspm=off intel_idle.max_cstate=1 processor.max_cstate=1 usbcore.autosuspend=-1)

# --- pre-flight checks ---------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (use sudo)." >&2
    exit 1
fi

if [[ ! -f "$GRUB_FILE" ]]; then
    echo "ERROR: $GRUB_FILE not found." >&2
    exit 1
fi

REBOOT=1
if [[ "${1:-}" == "--no-reboot" ]]; then
    REBOOT=0
fi

# --- backup --------------------------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP="${GRUB_FILE}.bak.${TS}"
cp -a "$GRUB_FILE" "$BACKUP"
echo "Backed up $GRUB_FILE -> $BACKUP"

# --- ensure GRUB_CMDLINE_LINUX line exists -------------------------------------
if ! grep -qE '^GRUB_CMDLINE_LINUX=' "$GRUB_FILE"; then
    echo 'GRUB_CMDLINE_LINUX=""' >> "$GRUB_FILE"
fi

# --- read current value, strip surrounding quotes ------------------------------
CURRENT="$(grep -E '^GRUB_CMDLINE_LINUX=' "$GRUB_FILE" | sed -E 's/^GRUB_CMDLINE_LINUX=//; s/^"(.*)"$/\1/')"

# --- merge missing parameters --------------------------------------------------
CHANGED=0
NEW="$CURRENT"
for p in "${PARAMS[@]}"; do
    # Match the param as a whole token (key=value or key=anything).
    # For params of the form key=value, replace any existing key=* token.
    key="${p%%=*}"
    if [[ "$key" == "$p" ]]; then
        # bare flag — skip if already present as a whole token
        if grep -qE "(^| )${p}( |$)" <<<"$NEW"; then
            echo "  [skip] $p already present"
            continue
        fi
    else
        # key=value — remove any existing key=* token first to avoid duplicates
        if grep -qE "(^| )${key}=[^ ]*( |$)" <<<"$NEW"; then
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
    echo "All requested parameters already present in GRUB_CMDLINE_LINUX."
    echo "Current value: \"$CURRENT\""
else
    # Write back. Use a temp file + mv for atomicity, preserving perms.
    TMP="$(mktemp)"
    awk -v new="$NEW" '
        /^GRUB_CMDLINE_LINUX=/ { print "GRUB_CMDLINE_LINUX=\"" new "\""; next }
        { print }
    ' "$GRUB_FILE" > "$TMP"
    # preserve ownership/mode
    chown --reference="$GRUB_FILE" "$TMP"
    chmod --reference="$GRUB_FILE" "$TMP"
    mv "$TMP" "$GRUB_FILE"
    echo "Updated GRUB_CMDLINE_LINUX to: \"$NEW\""
fi

# --- show resulting relevant lines ---------------------------------------------
echo
echo "--- /etc/default/grub (CMDLINE lines) ---"
grep -nE '^GRUB_CMDLINE_LINUX' "$GRUB_FILE"
echo "-----------------------------------------"

# --- update grub ---------------------------------------------------------------
echo
echo "Running update-grub..."
if command -v update-grub >/dev/null 2>&1; then
    update-grub
else
    # fallback for non-Debian distros
    grub2-mkconfig -o /boot/grub2/grub.cfg 2>/dev/null \
        || grub-mkconfig -o /boot/grub/grub.cfg
fi
echo "update-grub done."

# --- reboot --------------------------------------------------------------------
if [[ $REBOOT -eq 1 ]]; then
    echo
    echo "Rebooting in 3 seconds... (Ctrl-C to cancel)"
    sleep 3
    reboot
else
    echo
    echo "Skipping reboot (--no-reboot). Reboot manually to apply."
fi
