# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""System hardware information recorded in calibration reports for reproducibility.

Auto-detects CPU, cores, RAM, platform. No external dependencies
(psutil optional). Used by `mvis.analysis.report_shell` to stamp each
report with the machine it was solved on.
"""

import logging
import multiprocessing
import os
import platform
import subprocess

logger = logging.getLogger(__name__)


def get_system_info() -> dict:
    """Auto-detect hardware and platform information.

    Returns dict with keys:
        platform, cpu_name, cpu_cores_logical, cpu_cores_physical,
        ram_gb, python_version, gtsam_threads
    """
    info = {
        "platform": platform.system().lower(),
        "platform_version": platform.version(),
        "cpu_name": _get_cpu_name(),
        "cpu_cores_logical": multiprocessing.cpu_count(),
        "cpu_cores_physical": _get_physical_cores(),
        "ram_gb": _get_ram_gb(),
        "python_version": platform.python_version(),
        "gtsam_threads": _get_gtsam_threads(),
    }
    return info


def get_system_summary() -> str:
    """One-line summary for report headers."""
    info = get_system_info()
    return (f"{info['platform'].capitalize()}, "
            f"{info['cpu_cores_physical']} cores "
            f"({info['cpu_cores_logical']} threads), "
            f"{info['ram_gb']:.0f} GB RAM, "
            f"Python {info['python_version']}")


def _get_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system().lower()
    try:
        if system == "linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        elif system == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        elif system == "windows":
            return platform.processor()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _get_physical_cores() -> int:
    """Get physical core count (not hyperthreaded)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
    except ImportError:
        pass

    system = platform.system().lower()
    try:
        if system == "linux":
            # Count unique physical cores from /proc/cpuinfo
            cores = set()
            with open("/proc/cpuinfo") as f:
                phys_id = core_id = None
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    elif line.strip() == "" and phys_id is not None:
                        cores.add((phys_id, core_id))
                        phys_id = core_id = None
            if cores:
                return len(cores)
        elif system == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return int(result.stdout.strip())
    except Exception:
        pass

    # Fallback: assume half of logical cores
    return max(1, multiprocessing.cpu_count() // 2)


def _get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    system = platform.system().lower()
    try:
        if system == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        elif system == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _get_gtsam_threads() -> int:
    """Get number of threads GTSAM uses (TBB)."""
    try:
        # GTSAM uses TBB; check TBB thread count
        tbb_threads = os.environ.get("TBB_NUM_THREADS")
        if tbb_threads:
            return int(tbb_threads)
    except (ValueError, TypeError):
        pass
    # Default: GTSAM uses all available cores
    return multiprocessing.cpu_count()
