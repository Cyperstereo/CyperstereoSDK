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
"""Shared HTML report shell.

The look-and-feel primitives shared by the camera-only report (html_report.py) and
the cam-IMU report (cam_imu_report.py): branding/style, the CSS template + builder,
the page JS, the TOC / system-line / grade-badge helpers, the base64 + plotly embed
helpers, and a `render_page()` wrapper that emits the header/nav/footer/buttons
chrome. Extracted verbatim from html_report.py so the camera-only report is
byte-identical; both reports now import from here (one source of truth).
"""

import base64
import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

from mvis import __version__

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go


def _fig_to_base64(fig, bg_color: str = "#FFFFFF", dpi: int = 120) -> str:
    fig.patch.set_facecolor(bg_color)
    for ax in fig.get_axes():
        ax.set_facecolor(bg_color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg_color)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _image_to_base64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf).decode("utf-8")


def _plotly_to_html(fig) -> str:
    # include_plotlyjs=False: the figure div carries data only; render_page
    # inlines the plotly.js bundle ONCE per page (self-contained — the old
    # "cdn" mode left every 3D plot blank offline / behind a strict CSP).
    meta = fig.layout.meta
    spec = meta.get("mvis_progress") if isinstance(meta, dict) else None
    if not spec:
        return plotly.io.to_html(fig, full_html=False, include_plotlyjs=False)
    # trajectory figure (plot_3d.add_progress): add the calibration progress
    # strip; the spec moves to the strip's script instead of the figure JSON
    fig = go.Figure(fig)
    fig.layout.meta = None
    div_id = "mvis-plot-" + uuid.uuid4().hex[:8]
    plot = plotly.io.to_html(fig, full_html=False, include_plotlyjs=False,
                             div_id=div_id, config={"responsive": True})
    last = len(spec["xyz"]) - 1
    views = spec.get("views") or []
    if any(v.get("panel") for v in views):
        multi = " is-multi" if len(views) > 1 else ""
        plot = (f'<div class="mvis-3d{multi}"><div class="mvis-3d-plot">{plot}</div>'
                f'{_panel_html(div_id, views, last)}</div>')
    picker = ""
    if len(views) > 1:
        opts = "".join(f'<option value="{k}">{v["name"]}</option>'
                       for k, v in enumerate(views))
        picker = (f'<select class="mvis-progress-cam" aria-label="Camera shown '
                  f'on the trajectory">{opts}</select>')
    strip = (
        f'<div class="mvis-progress" data-plot="{div_id}">'
        '<button type="button">&#9654; Play</button>'
        f'{picker}'
        f'<input type="range" min="0" max="{last}" value="{last}" step="1" '
        'aria-label="Calibration progress">'
        f'<span class="mvis-progress-label">{_progress_label(spec, last)}</span>'
        '</div>')
    return (plot + strip + f"<script>{_PROGRESS_JS}"
            f"mvisProgress({json.dumps(div_id)},{json.dumps(spec)});</script>")


def _panel_html(div_id, views, last) -> str:
    """Side panel: one camera-view + image-coverage card per camera (filled by
    _PROGRESS_JS); the card of the camera on the trajectory is highlighted."""
    cards = []
    for k, v in enumerate(views):
        p = v.get("panel")
        if not p:
            continue
        src = f'data:image/jpeg;base64,{p["jpg"][-1]}' if p["jpg"] else ""
        multi = len(views) > 1
        img = (f'<img class="mvis-cam" alt="{v["name"]} frame at the current pose" '
               f'src="{src}">')
        cov = (f'<div class="mvis-cov-wrap" style="aspect-ratio:{p["w"]}/{p["h"]}">'
               '<canvas class="mvis-cov-base"></canvas><canvas class="mvis-cov-top"></canvas>'
               '</div>')
        cam_note = '<div class="mvis-panel-note mvis-cam-note"></div>'
        cov_note = (f'<div class="mvis-panel-note mvis-cov-note">{p["cov"][last]:.0f} % '
                    f'of the image &middot; {p["n"][last]:,} corners</div>')
        if multi:   # compact: view | coverage side by side, notes below
            body = (f'<div class="mvis-panel-title"><span class="mvis-swatch" '
                    f'style="background:{v["color"]}"></span>{v["name"]}</div>'
                    + img + cov + cam_note + cov_note)
        else:
            body = ('<div class="mvis-panel-title">Camera view</div>' + img + cam_note
                    + '<div class="mvis-panel-title">Image coverage so far</div>'
                    + cov + cov_note)
        cards.append(
            f'<div class="mvis-card{" is-sel" if k == 0 else ""}" data-view="{k}"'
            + (f' role="button" tabindex="0" title="Show {v["name"]} on the trajectory"'
               if multi else "") + '>' + body + '</div>')
    grid = views[0]["panel"]["grid"] if views and views[0].get("panel") else (8, 6)
    covis = ('<div class="mvis-panel-note mvis-covis-note"></div>'
             if len(views) > 1 else "")
    return (f'<aside class="mvis-panel{" is-multi" if len(views) > 1 else ""}" '
            f'data-plot="{div_id}">'
            + ('<div class="mvis-panel-title">Cameras &middot; view | coverage</div>'
               if len(views) > 1 else "")
            + "".join(cards) + covis
            + '<div class="mvis-panel-note">coverage: dots = detected corners up to '
            'now; amber = this frame; shaded = covered cells '
            f'({grid[0]}&times;{grid[1]} grid)'
            + ('; click a camera to show it on the trajectory' if len(views) > 1 else '')
            + '</div></aside>')


def _progress_label(spec, i) -> str:
    """Initial strip label; mirrors label() in _PROGRESS_JS."""
    n, t = len(spec["xyz"]), spec["t"]
    pct = 100.0 * (t[i] / t[-1] if t and t[-1] > 0 else i / (n - 1))
    when = f"{t[i]:.1f} / {t[-1]:.1f} s &middot; " if t else ""
    return f"{pct:.0f} % &middot; {when}{spec['unit']} {i + 1} / {n}"


# Play/scrub along the trajectory: restyles the "covered" + "current" traces
# that plot_3d.add_progress appended, the pose glyph of the selected camera
# view, its side-panel cards and the board co-visibility colours. Play spans
# ~12 s for any run length and advances in run time when timestamps are known.
_PROGRESS_JS = """window.mvisProgress = window.mvisProgress || function (id, s) {
  var gd = document.getElementById(id);
  var strip = document.querySelector('.mvis-progress[data-plot="' + id + '"]');
  if (!gd || !strip) return;
  var btn = strip.querySelector('button'), rng = strip.querySelector('input');
  var lab = strip.querySelector('.mvis-progress-label');
  var pick = strip.querySelector('.mvis-progress-cam');
  var n = s.xyz.length, t = s.t, cur = n - 1, playing = false, t0, i0;
  var X = s.xyz.map(function (p) { return p[0]; });
  var Y = s.xyz.map(function (p) { return p[1]; });
  var Z = s.xyz.map(function (p) { return p[2]; });
  var V = s.views || [], sel = 0;
  function label(i) {
    var pct = 100 * (t && t[n - 1] > 0 ? t[i] / t[n - 1] : i / (n - 1));
    var when = t ? t[i].toFixed(1) + ' / ' + t[n - 1].toFixed(1) + ' s \\u00b7 ' : '';
    return Math.round(pct) + ' % \\u00b7 ' + when + s.unit + ' ' + (i + 1) + ' / ' + n;
  }
  // per-view panel corner pixels (plot_3d._pack_uv: int16 + offsets)
  V.forEach(function (v) {
    var P = v.panel;
    if (!P || !P.uvb) return;
    var bin = atob(P.uvb), u8 = new Uint8Array(bin.length);
    for (var b = 0; b < bin.length; b++) u8[b] = bin.charCodeAt(b);
    P.uvArr = new Int16Array(u8.buffer);
  });
  function uvAt(P, i) { return P.uvArr.subarray(P.uvo[i], P.uvo[i + 1]); }
  // per-view frustum image bytes (plot_3d.add_progress views[k].img)
  V.forEach(function (v) {
    if (!v.img) return;
    var bin = atob(v.img);
    v.pix = new Uint8Array(bin.length);
    for (var b = 0; b < bin.length; b++) v.pix[b] = bin.charCodeAt(b);
  });
  // pose glyph (mirrors plot_3d._glyph_xyz / _image_grid / _cam_pose)
  var HW = 0.6;
  function mul(A, B) {
    var C = [];
    for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++)
      C.push(A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]);
    return C;
  }
  function at(p, R, v, k) {
    return [0, 1, 2].map(function (r) {
      return p[r] + k * (R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2]);
    });
  }
  function camPose(i) {
    var p = s.xyz[i], R = s.R[i], v = V[sel];
    if (!v || !v.cam) return {p: p, R: R};
    return {p: at(p, R, v.cam.slice(9, 12), 1), R: mul(R, v.cam.slice(0, 9))};
  }
  function nearest(idx, i) {            // position in sorted idx closest to i
    var lo = 0, hi = idx.length - 1;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (idx[mid] < i) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(idx[lo - 1] - i) < Math.abs(idx[lo] - i)) lo--;
    return lo;
  }
  function image(i) {
    var v = V[sel], iw = v.iw, ih = v.ih, HH = v.hh;
    var off = (v.img_idx ? nearest(v.img_idx, i) : i) * iw * ih;
    var C = camPose(i), gx = [], gy = [], gz = [], gc = [];
    for (var r = 0; r < ih; r++) {
      var rx = [], ry = [], rz = [], rc = [];
      for (var c = 0; c < iw; c++) {
        var q = at(C.p, C.R, [-HW + 2 * HW * c / (iw - 1), -HH + 2 * HH * r / (ih - 1), 1], s.fr);
        rx.push(q[0]); ry.push(q[1]); rz.push(q[2]); rc.push(v.pix[off + r * iw + c]);
      }
      gx.push(rx); gy.push(ry); gz.push(rz); gc.push(rc);
    }
    Plotly.restyle(gd, {x: [gx], y: [gy], z: [gz], surfacecolor: [gc]}, [s.glyph.image]);
  }
  function glyph(i, xs, ys, zs, idx) {
    var g = s.glyph, p = s.xyz[i], C = camPose(i), HH = V[sel] ? V[sel].hh : 0.4;
    var CORNERS = [[-HW, -HH, 1], [HW, -HH, 1], [HW, HH, 1], [-HW, HH, 1]];
    var UP = [[-0.2, -HH - 0.05, 1], [0, -HH - 0.22, 1], [0.2, -HH - 0.05, 1]];
    function put(pts, trace) {
      xs.push(pts.map(function (q) { return q && q[0]; }));
      ys.push(pts.map(function (q) { return q && q[1]; }));
      zs.push(pts.map(function (q) { return q && q[2]; }));
      idx.push(trace);
    }
    if (g.frustum !== undefined) {
      var c = CORNERS.map(function (v) { return at(C.p, C.R, v, s.fr); });
      var u = UP.map(function (v) { return at(C.p, C.R, v, s.fr); });
      var f = [];
      c.forEach(function (q) { f.push(C.p, q, null); });
      put(f.concat([c[0], c[1], c[2], c[3], c[0], null, u[0], u[1], u[2]]), g.frustum);
      if (g.plane !== undefined) put(c, g.plane);
    }
    if (g.ray !== undefined) {           // optical axis -> plane n.x = d (plot_3d._ray_end)
      var nn = s.plane, dz = [C.R[2], C.R[5], C.R[8]], end = C.p;
      var den = nn[0] * dz[0] + nn[1] * dz[1] + nn[2] * dz[2];
      var num = nn[3] - (nn[0] * C.p[0] + nn[1] * C.p[1] + nn[2] * C.p[2]);
      if (Math.abs(den) > 1e-9 && num / den > 0) end = at(C.p, [1, 0, 0, 0, 1, 0, 0, 0, 1], dz, num / den);
      put([C.p, end], g.ray);
    }
    ['x', 'y', 'z'].forEach(function (a, k) {
      var e = [0, 0, 0]; e[k] = 1;
      put([p, at(p, s.R[i], e, s.ax)], g[a]);
    });
  }
  // board co-visibility (mirrors plot_3d._covis_style)
  var covisNote = document.querySelector('.mvis-panel[data-plot="' + id + '"] .mvis-covis-note');
  function covis(i) {
    if (!s.covis) return;
    var cs = s.covis, col = cs.color.slice(), size = cs.size.slice(), seen = {}, both = 0, any = 0;
    V.forEach(function (v, k) {
      if (!v.ids) return;
      v.ids[i].forEach(function (pid) { (seen[pid] = seen[pid] || []).push(k); });
    });
    Object.keys(seen).forEach(function (pid) {
      var ks = seen[pid];
      if (pid < 0 || pid >= col.length) return;
      any++;
      if (ks.length > 1) { both++; col[pid] = cs.both; size[pid] = 6; }
      else { col[pid] = V[ks[0]].color; size[pid] = 4.5; }
    });
    Plotly.restyle(gd, {'marker.color': [col], 'marker.size': [size]}, [cs.trace]);
    if (covisNote) covisNote.innerHTML = '<span class="mvis-swatch" style="background:' + cs.both
      + '"></span>co-visible now: <b>' + both + '</b> of ' + any + ' detected corners seen by '
      + '\\u2265 2 cameras';
  }
  // side panel cards (pose_images.pose_panel): camera view + image coverage
  var aside = document.querySelector('.mvis-panel[data-plot="' + id + '"]');
  var cards = aside ? Array.prototype.slice.call(aside.querySelectorAll('.mvis-card')) : [];
  cards.forEach(function (el) {
    var v = V[+el.dataset.view];
    v.el = {cam: el.querySelector('.mvis-cam'), camNote: el.querySelector('.mvis-cam-note'),
            covNote: el.querySelector('.mvis-cov-note'),
            base: el.querySelector('.mvis-cov-base'), top: el.querySelector('.mvis-cov-top'),
            card: el};
    v.drawn = -1;
    if (V.length > 1) {
      el.addEventListener('click', function () { select(+el.dataset.view); });
      el.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(+el.dataset.view); }
      });
    }
  });
  function fit(cv, P) {
    var dpr = Math.min(2, window.devicePixelRatio || 1), w = cv.clientWidth;
    cv.width = Math.max(1, Math.round(w * dpr));
    cv.height = Math.max(1, Math.round(w * P.h / P.w * dpr));
    var g = cv.getContext('2d'); g.setTransform(cv.width / P.w, 0, 0, cv.height / P.h, 0, 0);
    return g;
  }
  function card(v, i) {
    var P = v.panel, E = v.el;
    if (!P || !E) return;
    if (P.jpg.length) E.cam.src = 'data:image/jpeg;base64,' + P.jpg[nearest(P.jpg_idx, i)];
    E.camNote.textContent = s.unit + ' ' + (i + 1) + (t ? ' \\u00b7 t = ' + t[i].toFixed(1) + ' s' : '')
      + ' \\u00b7 ' + (uvAt(P, i).length ? (uvAt(P, i).length / 2) + ' corners' : 'no detection used here');
    var gx = P.grid[0], gy = P.grid[1], cw = P.w / gx, ch = P.h / gy, g;
    if (i < v.drawn || v.drawn < 0 || !E.base.width) {   // rewind / first draw: start over
      g = fit(E.base, P); g.fillStyle = '#0F172A'; g.fillRect(0, 0, P.w, P.h);
      g.strokeStyle = 'rgba(148,163,184,0.25)'; g.lineWidth = P.w / 400;
      for (var a = 1; a < gx; a++) { g.beginPath(); g.moveTo(a * cw, 0); g.lineTo(a * cw, P.h); g.stroke(); }
      for (var b = 1; b < gy; b++) { g.beginPath(); g.moveTo(0, b * ch); g.lineTo(P.w, b * ch); g.stroke(); }
      v.seen = {}; v.drawn = -1;
    } else {
      g = E.base.getContext('2d');
    }
    var r = Math.max(1.5, P.w / 260);
    for (var k = v.drawn + 1; k <= i; k++) {
      var uv = uvAt(P, k);
      for (var j = 0; j < uv.length; j += 2) {
        var cx = Math.min(gx - 1, Math.floor(uv[j] / cw)), cy = Math.min(gy - 1, Math.floor(uv[j + 1] / ch));
        if (!v.seen[cx + ',' + cy]) {
          v.seen[cx + ',' + cy] = 1;
          g.fillStyle = 'rgba(56,189,248,0.16)'; g.fillRect(cx * cw, cy * ch, cw, ch);
        }
        g.fillStyle = 'rgba(125,211,252,0.55)';
        g.fillRect(uv[j] - r / 2, uv[j + 1] - r / 2, r, r);
      }
    }
    v.drawn = i;
    var o = fit(E.top, P), cu = uvAt(P, i), rr = Math.max(3, P.w / 110);
    o.fillStyle = '#F59E0B';
    for (var q = 0; q < cu.length; q += 2) { o.beginPath(); o.arc(cu[q], cu[q + 1], rr / 2, 0, 6.2832); o.fill(); }
    E.covNote.textContent = Math.round(P.cov[i]) + ' % of the image \\u00b7 ' + P.n[i].toLocaleString() + ' corners';
  }
  function show(i) {
    cur = Math.max(0, Math.min(n - 1, i));
    rng.value = cur; lab.textContent = label(cur);
    var xs = [X.slice(0, cur + 1), [X[cur]]], ys = [Y.slice(0, cur + 1), [Y[cur]]];
    var zs = [Z.slice(0, cur + 1), [Z[cur]]], idx = [s.done, s.cur];
    if (s.glyph) glyph(cur, xs, ys, zs, idx);
    Plotly.restyle(gd, {x: xs, y: ys, z: zs}, idx);
    if (s.glyph && s.glyph.image !== undefined && V[sel] && V[sel].pix) image(cur);
    covis(cur);
    V.forEach(function (v) { card(v, cur); });
  }
  function select(k) {                 // camera shown on the trajectory
    sel = k;
    if (pick) pick.value = String(k);
    cards.forEach(function (el) { el.classList.toggle('is-sel', +el.dataset.view === k); });
    var g = s.glyph, col = V[k].color, tr = [];
    if (g.frustum !== undefined) tr.push(g.frustum);
    if (g.ray !== undefined) tr.push(g.ray);
    if (tr.length) Plotly.restyle(gd, {'line.color': col}, tr);
    if (g.plane !== undefined) Plotly.restyle(gd, {color: col}, [g.plane]);
    if (g.image !== undefined) Plotly.restyle(gd, {visible: !!V[k].pix}, [g.image]);
    show(cur);
  }
  function stop() { playing = false; btn.innerHTML = '&#9654; Play'; }
  function step(ts) {
    if (!playing) return;
    if (t0 === undefined) t0 = ts;
    var f = (ts - t0) / 12000, i = i0;
    if (t && t[n - 1] > 0) {
      var target = t[i0] + f * t[n - 1];
      while (i < n - 1 && t[i + 1] <= target) i++;
    } else {
      i = i0 + Math.floor(f * (n - 1));
    }
    if (i >= n - 1) { show(n - 1); stop(); return; }
    if (i !== cur) show(i);
    requestAnimationFrame(step);
  }
  btn.onclick = function () {
    if (playing) { stop(); return; }
    i0 = cur >= n - 1 ? 0 : cur; t0 = undefined; playing = true;
    btn.innerHTML = '&#10074;&#10074; Pause';
    show(i0); requestAnimationFrame(step);
  };
  rng.oninput = function () { stop(); show(+rng.value); };
  if (pick) pick.onchange = function () { select(+pick.value); };
  if (cards.length) {
    V.forEach(function (v) { card(v, n - 1); });
    window.addEventListener('resize', function () {
      V.forEach(function (v) { v.drawn = -1; card(v, cur); });
    });
  }
  if (covisNote && s.covis) covis(n - 1);
};
"""


def _plotly_js_inline() -> str:
    """The full plotly.js bundle as an inline <script> (version-matched to the
    plotly.py that serialized the figures — required for bdata decoding)."""
    from plotly.offline import get_plotlyjs
    return "<script>" + get_plotlyjs() + "</script>"


@dataclass
class ReportStyle:
    """Tunable parameters for the HTML report appearance."""

    # --- Branding ---
    brand_name: str = "OpenMVIS"
    version: str = f"v{__version__}"      # the installed package, so every report names the release that wrote it
    tagline: str = "Open-source multi-visual-inertial sensor calibration"
    author_name: str = "Yulin Yang"
    author_email: str = "contact@openmvis.com"
    author_url: str = "https://openmvis.com"

    # --- Section alternating backgrounds ---
    section_bg: str = "#FFFFFF"          # odd sections
    section_alt_bg: str = "#EDF2F7"      # even sections (blue-grey)
    section_alt_accent: str = "#94A3B8"  # left border accent on alt sections
    # Dark mode overrides
    section_alt_bg_dark: str = "#162032"
    section_alt_accent_dark: str = "#475569"

    # --- Header gradient ---
    header_gradient: str = "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)"
    header_gradient_dark: str = "linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%)"
    header_accent: str = "linear-gradient(90deg, #3b82f6, #06b6d4)"

    # --- Plot ---
    plot_dpi: int = 120


# Singleton — import and modify before calling generate_html_report()
# to customize the report appearance.
REPORT_STYLE = ReportStyle()


def _build_css(s: ReportStyle) -> str:
    """Build the CSS string from ReportStyle parameters."""
    return _CSS_TEMPLATE.replace(
        "/*ALT_BG*/", s.section_alt_bg,
    ).replace(
        "/*ALT_ACCENT*/", s.section_alt_accent,
    ).replace(
        "/*ALT_BG_DARK*/", s.section_alt_bg_dark,
    ).replace(
        "/*ALT_ACCENT_DARK*/", s.section_alt_accent_dark,
    ).replace(
        "/*HEADER_GRADIENT*/", s.header_gradient,
    ).replace(
        "/*HEADER_GRADIENT_DARK*/", s.header_gradient_dark,
    ).replace(
        "/*HEADER_ACCENT*/", s.header_accent,
    )


_CSS_TEMPLATE = """
:root {
    --primary:       #2563EB;
    --primary-dark:  #1E40AF;
    --primary-light: #DBEAFE;
    --primary-bg:    #EFF6FF;
    --text-primary:  #1F2937;
    --text-secondary:#6B7280;
    --text-muted:    #9CA3AF;
    --border:        #E5E7EB;
    --border-dark:   #D1D5DB;
    --bg-page:       #F9FAFB;
    --bg-card:       #FFFFFF;
    --bg-code:       #F3F4F6;
    --success:       #059669;
    --success-bg:    #D1FAE5;
    --warning:       #D97706;
    --warning-bg:    #FEF3C7;
    --danger:        #DC2626;
    --danger-bg:     #FEE2E2;
}

/* --- Base --- */
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.65;
    color: var(--text-primary);
    background: var(--bg-page);
    max-width: 1440px;      /* was 1200: room for the 3D plots + side panel */
    margin: 0 auto;
    padding: 24px 32px;
    -webkit-font-smoothing: antialiased;
}

/* --- Typography --- */
h1 { font-size: 28px; font-weight: 700; letter-spacing: -0.02em; color: var(--primary-dark);
     margin: 0 0 4px; border: none; padding: 0; }
h2 { font-size: 21px; font-weight: 600; letter-spacing: -0.01em; color: var(--primary-dark);
     margin-top: 32px; margin-bottom: 16px; border-bottom: 2px solid var(--border);
     padding-bottom: 8px; }
h2:first-child { margin-top: 0; }
h3 { font-size: 17px; font-weight: 600; color: var(--text-primary); margin-top: 24px; margin-bottom: 12px; }
h4 { font-size: 15px; font-weight: 600; color: var(--text-secondary); margin-top: 16px; margin-bottom: 8px; }
p { margin: 8px 0; }
code {
    background: var(--bg-code); padding: 2px 6px; border-radius: 4px;
    font-size: 13px; font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
}

/* --- Section cards (alternating backgrounds) --- */
.section {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 28px 32px;
    margin: 20px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid var(--border);
}
.section.section-alt {
    background: /*ALT_BG*/;
    border-left: 3px solid /*ALT_ACCENT*/;
}
@media (prefers-color-scheme: dark) {
    .section.section-alt { background: /*ALT_BG_DARK*/; border-left-color: /*ALT_ACCENT_DARK*/; }
}

/* --- Tables (borderless / minimal) --- */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 14px;
}
th {
    background: transparent;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    padding: 10px 14px;
    border-bottom: 2px solid var(--border-dark);
    text-align: left;
}
td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}
tr:last-child td { border-bottom: none; }
tbody tr { transition: background 0.15s ease; }
tbody tr:hover { background: var(--primary-bg); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }

/* --- Stat boxes --- */
.stat-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 16px; }
.stat-box {
    display: inline-flex; flex-direction: column; align-items: center;
    background: linear-gradient(135deg, var(--primary-bg) 0%, var(--bg-card) 100%);
    border: 1px solid var(--primary-light);
    border-radius: 10px; padding: 16px 24px; min-width: 120px; text-align: center;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stat-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
}
.stat-box .value { font-size: 26px; font-weight: 700; color: var(--primary-dark);
                   font-variant-numeric: tabular-nums; }
.stat-box .label { font-size: 11px; font-weight: 500; text-transform: uppercase;
                   letter-spacing: 0.05em; color: var(--text-secondary); margin-top: 4px; }

/* --- Plots --- */
.plot-container { margin: 20px 0; text-align: center; }
.plot-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; align-items: center; margin: 20px 0; }
.plot-row .plot-container { margin: 0; }
@media (max-width: 800px) { .plot-row { grid-template-columns: 1fr; } }
.plot-container img { max-width: 100%; border-radius: 6px; border: 1px solid var(--border);
                      box-shadow: 0 1px 2px rgba(0,0,0,0.04); }

/* --- Calibration progress strip (3D trajectory plots) --- */
.mvis-progress {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    margin: 0 0 20px; padding: 8px 12px;
    border: 1px solid var(--border); border-radius: 6px; background: var(--bg-card);
}
.mvis-progress button {
    font: inherit; font-size: 13px; padding: 3px 12px; cursor: pointer;
    color: var(--text-primary); background: var(--bg-page);
    border: 1px solid var(--border); border-radius: 4px;
}
.mvis-progress button:hover { border-color: var(--primary); }
.mvis-progress input[type=range] { flex: 1 1 240px; min-width: 0; accent-color: var(--primary); }
.mvis-progress-label {
    font-size: 13px; color: var(--text-secondary);
    font-variant-numeric: tabular-nums; min-width: 22ch;
}

/* --- 3D plot + side panel (camera view, image coverage) --- */
.mvis-3d { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 12px;
           align-items: start; text-align: left; }
@media (max-width: 960px) { .mvis-3d { grid-template-columns: minmax(0, 1fr); } }
.mvis-3d-plot { min-width: 0; }
.mvis-panel { border: 1px solid var(--border); border-radius: 6px; background: var(--bg-card);
              padding: 10px 12px; display: grid; gap: 6px; }
.mvis-panel-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.06em; color: var(--text-secondary); margin-top: 4px; }
.mvis-cam { width: 100%; display: block; border-radius: 4px; background: #0F172A; }
.mvis-cov-wrap { position: relative; width: 100%; border-radius: 4px; overflow: hidden; }
.mvis-cov-wrap canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.mvis-panel-note { font-size: 12px; color: var(--text-muted); font-variant-numeric: tabular-nums;
                   line-height: 1.4; }
.mvis-card { display: grid; gap: 6px; padding: 6px; border-radius: 6px;
             border: 1px solid transparent; }
.mvis-3d.is-multi { grid-template-columns: minmax(0, 1fr) 360px; }
@media (max-width: 960px) { .mvis-3d.is-multi { grid-template-columns: minmax(0, 1fr); } }
.mvis-panel.is-multi { max-height: 820px; overflow-y: auto; align-content: start; }
.mvis-panel.is-multi .mvis-card { grid-template-columns: 1fr 1fr; align-items: center; }
.mvis-panel.is-multi .mvis-card > .mvis-panel-title,
.mvis-panel.is-multi .mvis-card > .mvis-panel-note { grid-column: 1 / -1; }
.mvis-card[role=button] { cursor: pointer; }
.mvis-card[role=button]:hover { border-color: var(--border); }
.mvis-panel .mvis-card.is-sel[role=button] { border-color: var(--primary); background: var(--primary-bg); }
.mvis-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
               margin-right: 6px; vertical-align: -1px; }
.mvis-progress select { font: inherit; font-size: 13px; padding: 3px 8px; color: var(--text-primary);
                        background: var(--bg-page); border: 1px solid var(--border); border-radius: 4px; }

/* --- Copyable verbatim blocks (config, result files) --- */
.copy-wrap { position: relative; }
.copy-pre { background: var(--bg-code); border: 1px solid var(--border); border-radius: 6px;
            padding: 14px; overflow-x: auto; font-size: 13px; line-height: 1.45; margin: 0;
            max-height: 520px; }
.copy-btn { position: absolute; top: 8px; right: 8px; font: inherit; font-size: 12px;
            padding: 2px 10px; cursor: pointer; color: var(--text-secondary);
            background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; }
.copy-btn:hover { border-color: var(--primary); color: var(--primary); }

/* --- Image grids --- */
.img-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px;
}
.img-grid img {
    width: 100%; border: 1px solid var(--border); border-radius: 6px;
    transition: transform 0.15s ease;
}
.img-grid img:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }

/* --- Report header (dark gradient) --- */
.report-header {
    background: /*HEADER_GRADIENT*/;
    color: #fff;
    text-align: center;
    padding: 48px 32px 40px;
    margin: -24px -32px 24px;
    border-radius: 0 0 16px 16px;
}
.report-header h1 { color: #f1f5f9; font-size: 32px; }
.report-header h1.brand { display: inline-flex; align-items: center; gap: 14px; }
.report-header .brand-logo svg { width: 46px; height: 46px; display: block; border-radius: 12px; box-shadow: 0 0 0 1px rgba(148, 163, 184, .35), 0 4px 14px rgba(0, 0, 0, .35); }
.report-header .subtitle { font-size: 16px; color: #94a3b8; font-weight: 400; }
.report-header .meta-row {
    display: flex; justify-content: center; gap: 24px; margin-top: 12px;
    font-size: 13px; color: #64748b;
}
.report-header::after {
    content: ''; display: block; width: 80px; height: 3px;
    background: /*HEADER_ACCENT*/;
    margin: 24px auto 0; border-radius: 2px;
}

/* --- Report footer --- */
.report-footer {
    text-align: center; padding: 40px 0 32px; margin-top: 48px;
    border-top: 2px solid var(--border); font-size: 13px; color: var(--text-secondary);
    line-height: 1.8;
}
.report-footer .footer-brand {
    font-size: 15px; font-weight: 600; color: var(--primary-dark);
    letter-spacing: -0.01em; margin-bottom: 4px;
}
.report-footer .footer-info {
    font-size: 12px; color: var(--text-muted);
}
.report-footer .footer-info a {
    color: var(--primary); text-decoration: none;
}
.report-footer .footer-info a:hover { text-decoration: underline; }

/* --- Meta text --- */
.meta { color: var(--text-muted); font-size: 13px; }
.note { color: var(--text-secondary); font-size: 13px; font-style: italic; }
.desc { font-size: 13px; color: var(--text-secondary); margin-bottom: 12px; line-height: 1.6; }

.nt { text-transform: none; }
ul.acc { margin: 4px 0 8px; padding-left: 20px; font-size: 13.5px; }
/* --- Camera colour chip and graded table values --- */
.chip { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px;
    vertical-align: 1px; }
.g { padding: 1px 6px; border-radius: 5px; }
.g-good { background: rgba(22, 163, 74, .10); }
.g-marginal { background: rgba(234, 179, 8, .18); }
.g-poor { background: rgba(220, 38, 38, .14); }

/* --- Per-camera tabs (cam_tabs): without JS every panel shows, stacked --- */
.cam-tabs .tab-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 22px 0 12px;
    padding: 10px 12px; border: 1px solid var(--primary-light); border-radius: 10px; background: var(--primary-bg); }
.cam-tabs .tab-bar .tab-hint { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;
    color: var(--primary-dark); margin-right: 4px; }
.cam-tabs .tab-bar button { font: inherit; font-size: 13.5px; font-weight: 600; padding: 7px 16px; cursor: pointer;
    border: 1px solid var(--primary-light); border-radius: 999px; background: var(--bg-card); color: var(--primary-dark);
    box-shadow: 0 1px 2px rgba(15, 23, 42, .08); transition: background .15s, color .15s, box-shadow .15s, transform .1s; }
.cam-tabs .tab-bar button:hover { border-color: var(--primary); box-shadow: 0 2px 6px rgba(37, 99, 235, .25);
    transform: translateY(-1px); }
.cam-tabs .tab-bar button.on { background: var(--primary); border-color: var(--primary); color: #fff;
    box-shadow: 0 2px 8px rgba(37, 99, 235, .35); }
.cam-tabs .tab-bar button.on .chip { box-shadow: 0 0 0 2px #fff; }
.cam-tabs.js .cam-panel { display: none; }
.cam-tabs.js .cam-panel.on { display: block; }

/* --- Summary: one-line run facts under the tiles --- */
.facts { display: flex; flex-wrap: wrap; gap: 6px 28px; margin: 0 0 18px; font-size: 13px;
    color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.facts b { font-weight: 500; color: var(--text-muted); margin-right: 6px; text-transform: uppercase;
    letter-spacing: .04em; font-size: 11px; }

/* --- Summary: calibration parameters by sensor (param_folds) --- */
.param-folds h3 { display: flex; align-items: center; gap: 12px; }
.fold-all { font: inherit; font-size: 12.5px; font-weight: 600; padding: 5px 14px; border-radius: 999px; cursor: pointer;
    border: 1px solid var(--primary); background: var(--primary); color: #fff;
    box-shadow: 0 2px 6px rgba(37, 99, 235, .3); transition: box-shadow .15s, transform .1s; }
.fold-all:hover { box-shadow: 0 3px 10px rgba(37, 99, 235, .4); transform: translateY(-1px); }
details.param-fold { margin: 8px 0; }
details.param-fold > summary { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
.param-key { color: var(--text-secondary); font-size: 12.5px; font-variant-numeric: tabular-nums; }
.param-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px 24px; }
.param-group h5 { margin: 0 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--primary-dark); }
.param-group.wide { grid-column: 1 / -1; }
table.kv { margin: 0; width: 100%; }
table.kv tr:last-child th { border-bottom: none; }
table.kv th { text-align: left; font-weight: 500; text-transform: none; letter-spacing: normal;
    color: var(--text-secondary); font-size: 13px; padding: 5px 10px; width: 46%; border-bottom: 1px solid var(--border); }
table.kv td { padding: 5px 10px; font-size: 13px; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pm { color: var(--text-secondary); font-size: .92em; }
.v3 { display: inline-grid; grid-template-columns: repeat(3, minmax(64px, auto)); column-gap: 12px; justify-items: end; }
.v3 > span { display: flex; flex-direction: column; align-items: flex-end; line-height: 1.25; }
.v3 small.pm { font-size: 11px; }

/* --- Collapsible appendix --- */
details.appendix { border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; }
details.appendix > summary {
    padding: 12px 16px; cursor: pointer; font-weight: 600; color: var(--primary-dark);
    background: var(--primary-bg); border-radius: 8px; list-style: none;
    user-select: none;
}
details.appendix > summary::-webkit-details-marker { display: none; }
details.appendix > summary::before {
    content: '\\25B6'; display: inline-block; margin-right: 8px; font-size: 12px;
    transition: transform 0.2s;
}
details.appendix[open] > summary::before { transform: rotate(90deg); }
details.appendix[open] > summary { border-radius: 8px 8px 0 0; border-bottom: 1px solid var(--border); }
details.appendix > .appendix-content { padding: 16px; }

/* --- Inline appendix items --- */
.appendix-img { max-width: 100%; border: 1px solid var(--border); margin: 8px 0; border-radius: 4px; }
.appendix-item {
    display: inline-block; margin: 6px; text-align: center; vertical-align: top;
}
.appendix-item img {
    max-width: 280px; max-height: 210px; border: 1px solid var(--border-dark); border-radius: 4px;
}
.appendix-item .caption { font-size: 12px; margin-top: 4px; color: var(--text-secondary); }

/* --- Pill badges for grades --- */
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
}
.badge-good     { background: var(--success-bg); color: var(--success); }
.badge-marginal { background: var(--warning-bg); color: var(--warning); }
.badge-poor     { background: var(--danger-bg);  color: var(--danger); }

/* --- TOC sidebar --- */
#toc {
    position: fixed; left: 0; top: 0; width: 240px; height: 100vh;
    overflow-y: auto; padding: 24px 16px; background: var(--bg-card);
    border-right: 1px solid var(--border); font-size: 13px; z-index: 100;
    transform: translateX(-100%); transition: transform 0.25s ease;
}
#toc.open { transform: translateX(0); }
#toc .toc-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 16px;
}
#toc a {
    display: block; padding: 6px 10px; margin: 2px 0; color: var(--text-secondary);
    text-decoration: none; border-radius: 5px; line-height: 1.4;
    transition: background 0.15s, color 0.15s; font-size: 13px;
}
#toc a:hover { background: var(--primary-bg); color: var(--primary-dark); }
#toc a.active { background: var(--primary-light); color: var(--primary-dark); font-weight: 600; }
@media (min-width: 1500px) {
    #toc { transform: translateX(0); }
    body { margin-left: 260px; }
    #toc-toggle { display: none !important; }
}

/* --- Floating buttons --- */
#toc-toggle {
    position: fixed; bottom: 24px; left: 24px; width: 44px; height: 44px;
    border-radius: 50%; background: var(--primary); color: white; border: none;
    font-size: 20px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    z-index: 101; display: flex; align-items: center; justify-content: center;
}
#toc-toggle:hover { background: var(--primary-dark); }
#scroll-top {
    position: fixed; bottom: 24px; right: 24px; width: 40px; height: 40px;
    border-radius: 50%; background: var(--bg-card); border: 1px solid var(--border);
    color: var(--text-secondary); font-size: 18px; cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1); opacity: 0; transition: opacity 0.2s; z-index: 99;
}
#scroll-top.visible { opacity: 1; }
#scroll-top:hover { background: var(--primary-bg); color: var(--primary-dark); }

/* --- Dark mode (auto, respects OS preference) --- */
@media (prefers-color-scheme: dark) {
    :root {
        --primary:       #3b82f6;
        --primary-dark:  #60a5fa;
        --primary-light: #1e3a5f;
        --primary-bg:    #172033;
        --text-primary:  #e2e8f0;
        --text-secondary:#94a3b8;
        --text-muted:    #64748b;
        --border:        #334155;
        --border-dark:   #475569;
        --bg-page:       #0f172a;
        --bg-card:       #1e293b;
        --bg-code:       #334155;
        --success:       #34d399;
        --success-bg:    #064e3b;
        --warning:       #fbbf24;
        --warning-bg:    #78350f;
        --danger:        #f87171;
        --danger-bg:     #7f1d1d;
    }
    .report-header {
        background: /*HEADER_GRADIENT_DARK*/;
    }
    .plot-container img { border-color: var(--border); }
    #toc {
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border-color: var(--border);
    }
    #scroll-top { background: var(--bg-card); border-color: var(--border); color: var(--text-secondary); }
    .stat-box { border-color: var(--border); }
    details.appendix { border-color: var(--border); }
    details.appendix > summary { background: var(--primary-bg); color: var(--primary-dark); }
}

/* --- Print --- */
@media print {
    body { background: white !important; color: black !important;
           max-width: none; margin: 0; padding: 0; font-size: 11pt; }
    #toc, #toc-toggle, #scroll-top { display: none !important; }
    .section { box-shadow: none; border: 1px solid #ddd; break-inside: avoid;
               page-break-inside: avoid; margin: 12px 0; padding: 16px;
               background: white !important; }
    .report-header {
        background: white !important; color: black !important;
        margin: 0; padding: 24px 0 16px; border-radius: 0;
        border-bottom: 2px solid #333;
    }
    .report-header h1 { color: black !important; }
    .report-header .subtitle { color: #666 !important; }
    .report-header .meta-row span { color: #999 !important; }
    .report-footer { border-top: 1px solid #ccc; }
    .report-footer .footer-brand { color: black !important; }
    .report-footer .footer-info a { color: #333 !important; }
    .section.section-alt { background: #f5f5f5 !important; border-left-color: #999 !important; }
    .report-header::after { background: #333 !important; }
    .plot-container img { max-width: 100%; box-shadow: none; border: 1px solid #ddd; }
    .stat-box { border: 1px solid #ddd; background: #f5f5f5 !important; }
    .stat-box .value { color: black !important; }
    h1, h2, h3 { color: black !important; }
    h2 { page-break-after: avoid; }
    a { color: inherit; text-decoration: none; }
    /* long sections may break; their tables and figures stay whole */
    .section { break-inside: auto; page-break-inside: auto; }
    table, .plot-container, .param-group, .stat-row { break-inside: avoid; page-break-inside: avoid; }
    .fold-all, .copy-btn, .cam-tabs .tab-bar { display: none !important; }
    .cam-tabs.js .cam-panel { display: block !important; }
    details > summary::before { display: none; }
    tbody tr:hover { background: transparent !important; }
    .badge { border: 1px solid #ccc; }
    .js-plotly-plot .modebar { display: none !important; }
}
"""


_JS = """
// --- Printing: open every fold (parameters, appendix, result files) and restore it afterwards ---
(function() {
    var closed = [];
    window.addEventListener('beforeprint', function() {
        closed = Array.prototype.filter.call(document.querySelectorAll('details'), function(d) { return !d.open; });
        closed.forEach(function(d) { d.open = true; });
    });
    window.addEventListener('afterprint', function() { closed.forEach(function(d) { d.open = false; }); closed = []; });
})();
// --- Expand all / Collapse all on the Summary's parameter folds ---
document.addEventListener('click', function(e) {
    var b = e.target.closest && e.target.closest('.fold-all');
    if (!b) return;
    var box = b.closest('.param-folds'); if (!box) return;
    var folds = box.querySelectorAll('details.param-fold');
    var open = b.textContent.indexOf('Expand') === 0;
    folds.forEach(function(d) { d.open = open; });
    b.textContent = open ? 'Collapse all' : 'Expand all';
});
// --- Copy buttons on copy_block() code blocks ---
document.addEventListener('click', function(e) {
    var b = e.target.closest && e.target.closest('.copy-btn');
    if (!b) return;
    var pre = b.parentNode.querySelector('pre');
    if (!pre || !navigator.clipboard) return;
    navigator.clipboard.writeText(pre.textContent).then(function() {
        b.textContent = 'Copied';
        setTimeout(function() { b.textContent = 'Copy'; }, 1500);
    });
});
document.addEventListener('DOMContentLoaded', function() {
    // --- Per-camera tabs: a tab bar over .cam-panel blocks; TOC links / #hash open their tab ---
    document.querySelectorAll('.cam-tabs').forEach(function(box) {
        var panels = Array.prototype.slice.call(box.querySelectorAll(':scope > .cam-panel'));
        if (panels.length < 2) return;
        var bar = document.createElement('div'); bar.className = 'tab-bar';
        var hint = document.createElement('span'); hint.className = 'tab-hint';
        hint.textContent = 'Select a camera:'; bar.appendChild(hint);
        function show(i) {
            panels.forEach(function(p, k) { p.classList.toggle('on', k === i); });
            Array.prototype.forEach.call(bar.querySelectorAll('button'), function(b, k) { b.classList.toggle('on', k === i); });
        }
        panels.forEach(function(p, i) {
            var b = document.createElement('button'); b.type = 'button';
            b.innerHTML = (p.querySelector('h3 .chip') ? p.querySelector('h3 .chip').outerHTML : '') + p.dataset.label;
            b.addEventListener('click', function() { show(i); });
            bar.appendChild(b);
        });
        box.insertBefore(bar, box.firstChild); box.classList.add('js'); show(0);
        function fromHash() {
            var i = panels.findIndex(function(p) { return '#' + p.id === location.hash; });
            if (i >= 0) { show(i); bar.scrollIntoView({block: 'start'}); }
        }
        window.addEventListener('hashchange', fromHash); fromHash();
    });
    // --- Numeric columns: right-aligned, tabular digits, header aligned with its numbers ---
    var NUM = /^[-+]?[\d.,]+(e[-+]?\d+)?( ?(±|\/) ?[-+]?[\d.,]+(e[-+]?\d+)?)*( ?(px|mm|ms|rad|rad\/s|m\/s²|s|°|%|kB|MB))?$/i;
    document.querySelectorAll('table').forEach(function(t) {
        var rows = Array.prototype.slice.call(t.rows);
        if (rows.length < 2) return;
        var ncol = Math.max.apply(null, rows.map(function(r) { return r.cells.length; }));
        for (var c = 0; c < ncol; c++) {
            var cells = [], blanks = [], head = null, ok = true;
            rows.forEach(function(r) {
                var x = r.cells[c]; if (!x) return;
                if (x.tagName === 'TH') { if (!head) head = x; return; }
                var v = x.textContent.replace(/\u2212/g, '-').trim();
                if (v === '' || v === '—' || v === '-') { blanks.push(x); return; }
                if (!NUM.test(v)) ok = false; else cells.push(x);
            });
            if (ok && cells.length) cells = cells.concat(blanks);
            if (ok && cells.length) {
                cells.forEach(function(x) { x.classList.add('num'); });
                if (head && head.parentNode === rows[0]) head.classList.add('num');
            }
        }
    });
    // --- Smooth scroll for TOC links ---
    document.querySelectorAll('#toc a').forEach(function(a) {
        a.addEventListener('click', function(e) {
            e.preventDefault();
            var target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                document.getElementById('toc').classList.remove('open');
            }
        });
    });

    // --- Active TOC highlight via IntersectionObserver ---
    var sections = document.querySelectorAll('.section[id]');
    var tocLinks = document.querySelectorAll('#toc a');
    if (sections.length && tocLinks.length) {
        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    tocLinks.forEach(function(l) { l.classList.remove('active'); });
                    var active = document.querySelector('#toc a[href=\"#' + entry.target.id + '\"]');
                    if (active) active.classList.add('active');
                }
            });
        }, { rootMargin: '-20% 0px -70% 0px' });
        sections.forEach(function(s) { observer.observe(s); });
    }

    // --- TOC toggle (mobile / narrow screens) ---
    var tocToggle = document.getElementById('toc-toggle');
    if (tocToggle) {
        tocToggle.addEventListener('click', function() {
            document.getElementById('toc').classList.toggle('open');
        });
    }

    // --- Scroll-to-top button ---
    var scrollBtn = document.getElementById('scroll-top');
    if (scrollBtn) {
        window.addEventListener('scroll', function() {
            scrollBtn.classList.toggle('visible', window.scrollY > 400);
        });
        scrollBtn.addEventListener('click', function() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
});
"""


def _get_system_line() -> str:
    """One-line system info for report footer."""
    try:
        from mvis.core.system_info import get_system_summary
        return get_system_summary()
    except Exception:
        return ""


def _build_toc(toc_entries: list[tuple[str, str]]) -> str:
    """Build the <nav id="toc"> HTML from a list of (id, label) tuples."""
    links = []
    for sec_id, label in toc_entries:
        links.append(f'<a href="#{sec_id}">{label}</a>')
    return (
        '<nav id="toc">\n'
        '<div class="toc-title">Contents</div>\n'
        + "\n".join(links)
        + "\n</nav>"
    )


# One look for every figure in the reports: font, sizes, light grid, no top/right spines, and the
# camera colours of the 3D views (views.spec.COLORS) as the colour cycle.
REPORT_RC = {
    "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#94A3B8",
    "axes.grid": True, "grid.color": "#CBD5E1", "grid.alpha": 0.5, "grid.linewidth": 0.6,
    "axes.prop_cycle": None,  # set below from the views' camera colours
    "figure.dpi": 100, "savefig.bbox": "tight",
}


def report_plot_style(fn):
    """Run a report generator under REPORT_RC, so every figure it draws shares the look."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        import matplotlib.pyplot as plt
        from cycler import cycler
        from mvis.analysis.views.spec import COLORS
        rc = dict(REPORT_RC, **{"axes.prop_cycle": cycler(color=COLORS)})
        with plt.rc_context(rc):
            return fn(*a, **k)
    return run


def cam_label(c: int) -> str:
    """'camN' with its colour chip (the 3D views' camera colour), for a table's first column."""
    from mvis.analysis.views.spec import COLORS
    return f"<span class='chip' style='background:{COLORS[int(c) % len(COLORS)]}'></span>cam{int(c)}"


def graded(text: str, grade) -> str:
    """A table value tinted by its display grade (good / marginal / poor); plain when ungraded."""
    return f"<span class='g g-{grade}'>{text}</span>" if grade else text


# The OpenMVIS logo, as on openmvis.com (mvis_pages site/src/assets/logo-placeholder.svg): inline, so a
# report stays one self-contained file; also the page's favicon.
LOGO_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" '
            'aria-label="OpenMVIS logo"><rect x="2" y="2" width="60" height="60" rx="12" fill="#0f172a"/>'
            '<circle cx="22" cy="32" r="9" fill="none" stroke="#38bdf8" stroke-width="3"/>'
            '<circle cx="22" cy="32" r="3" fill="#38bdf8"/>'
            '<rect x="36" y="22" width="20" height="20" rx="3" fill="none" stroke="#bae6fd" stroke-width="2.5"/>'
            '<line x1="36" y1="32" x2="56" y2="32" stroke="#bae6fd" stroke-width="2"/>'
            '<line x1="46" y1="22" x2="46" y2="42" stroke="#bae6fd" stroke-width="2"/></svg>')


def _logo_favicon() -> str:
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode()).decode()


def folded(summary: str, body: str) -> str:
    """A closed-by-default fold in the report's appendix style."""
    return (f'<details class="appendix"><summary>{summary}</summary>'
            f'<div class="appendix-content">{body}</div></details>')


def copy_block(text: str) -> str:
    """Verbatim text in a <pre> with a Copy button (handled in _JS)."""
    import html as _html
    return ('<div class="copy-wrap"><button type="button" class="copy-btn">Copy</button>'
            '<pre class="copy-pre">' + _html.escape(text) + '</pre></div>')


def config_fold(text: str) -> str:
    """The run configuration verbatim, folded, with a Copy button."""
    n = text.count("\n") + (0 if text.endswith("\n") else 1)
    return folded(f"Run configuration &middot; YAML, {n} lines, copy-paste ready "
                  "(click to expand)", copy_block(text))


def result_files_html(paths, max_bytes: int = 256 * 1024) -> str:
    """The files a run wrote, verbatim, one fold each (missing ones skipped).

    Files over `max_bytes` (e.g. a full yaml with per-frame poses) show their
    head only, labelled as such; the path points at the complete file."""
    import html as _html
    import os
    parts = []
    for path in paths or []:
        try:
            with open(path) as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        size = len(text.encode())
        kind = os.path.splitext(path)[1].lstrip(".").upper() or "text"
        note = ""
        if size > max_bytes:
            head = text.encode()[:max_bytes].decode(errors="ignore")
            head = head[:head.rfind("\n") + 1] or head
            note = (f' &middot; first {head.count(chr(10)):,} of '
                    f'{text.count(chr(10)):,} lines shown')
            text = head
        parts.append(folded(
            f"<code>{_html.escape(os.path.basename(path))}</code> &middot; {kind}, "
            f"{size / 1024:.1f} kB{note} (click to expand)",
            f'<p class="note">{_html.escape(os.path.abspath(path))}</p>'
            + copy_block(text)))
    if not parts:
        return ""
    return ("<p>Files this run wrote, verbatim. Expand one to read or copy it.</p>"
            + "".join(parts))


GRADE_SYM = {"good": "&#10003;", "marginal": "&#9888;", "poor": "&#10007;"}


def _grade_cell(grade: str) -> str:
    """Return a <td> with a pill badge for the grade."""
    css_class = f"badge-{grade}" if grade in GRADE_SYM else ""
    sym = GRADE_SYM.get(grade, "?")
    return f'<td><span class="badge {css_class}">{sym} {grade}</span></td>'


def badge(grade) -> str:
    """Pill badge for a display grade (good / marginal / poor); "" for none."""
    if not grade:
        return ""
    return f'<span class="badge badge-{grade}">{GRADE_SYM[grade]} {grade}</span>'


def stat_box(value, label, grade=None) -> str:
    """One summary tile: the value, its label and an optional grade badge."""
    extra = f'<div style="margin-top:4px">{badge(grade)}</div>' if grade else ""
    return (f'<div class="stat-box"><div class="value">{value}</div>'
            f'<div class="label">{label}</div>{extra}</div>')


def html_table(headers, rows) -> str:
    """A plain ``<table>``: one header row, then one row per sequence of cells."""
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def render_page(title, toc_entries, sections, style=REPORT_STYLE):
    """Assemble a full self-contained HTML page from rendered section strings.

    `sections` is a list of HTML fragments (each typically a `<div class="section">
    ... </div>`); `toc_entries` is a list of (sec_id, label) tuples. Emits the same
    header / TOC nav / footer / toggle-button chrome as the camera-only report.
    """
    toc_html = _build_toc(toc_entries)
    content = "\n".join(sections)
    # self-contained plotly: inline the JS bundle once iff any section holds a
    # plotly div (adds ~4.5 MB only when 3D/interactive plots are present)
    plotly_js = _plotly_js_inline() if "plotly-graph-div" in content else ""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    css = _build_css(style)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{style.brand_name} &mdash; {title}</title>
<link rel="icon" type="image/svg+xml" href="{_logo_favicon()}">
<style>{css}</style>
{plotly_js}
</head>
<body>
<div class="report-header">
    <h1 class="brand"><span class="brand-logo">{LOGO_SVG}</span>{style.brand_name}</h1>
    <div class="subtitle">{title}</div>
    <div class="meta-row">
        <span>Generated {now_str}</span>
        <span>&middot;</span>
        <span>{style.brand_name} {style.version}</span>
    </div>
</div>
{toc_html}
{content}
<div class="report-footer">
    <div class="footer-brand">{style.brand_name}</div>
    <div class="footer-info">
        {style.tagline}
        &nbsp;&middot;&nbsp;
        <a href="mailto:{style.author_email}">{style.author_email}</a>
        &nbsp;&middot;&nbsp;
        <a href="{style.author_url}">{style.author_url.replace('https://', '')}</a>
        <br>{style.version} &middot; {now_str} &middot; {_get_system_line()}
    </div>
</div>
<button id="toc-toggle" title="Table of contents">&#9776;</button>
<button id="scroll-top" title="Scroll to top">&#8593;</button>
<script>{_JS}</script>
</body>
</html>"""


# Public aliases (stable names for importers; the private names above are the
# verbatim-extracted originals kept so html_report.py imports them unchanged).
fig_to_base64 = _fig_to_base64
image_to_base64 = _image_to_base64
plotly_to_html = _plotly_to_html
build_css = _build_css
build_toc = _build_toc
get_system_line = _get_system_line
grade_cell = _grade_cell
