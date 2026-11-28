// OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
// Copyright (C) 2026 Yulin Yang
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
// SPDX-License-Identifier: GPL-3.0-only
/* MVIS report views: the calibration process player and the solved rig, on a plain 2D canvas.
   Data: mvis.analysis.views.spec (v2). Markup: a div.mvv with data-mvv="scene" or "rig" (and data-mode="report"
   or "site") holding its spec in a JSON script element. Shared by the MVIS reports and the openmvis.com pages. */
(function () {
  if (window.mvisViews) return;
  var VERSION = 2;
  var COMPACT = 5;   // this many cameras or more: the compact camera column
  // ---------- maths ----------
  function R9(r) { return [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]]; }
  function mv(R, v) { return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2], R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2], R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]; }
  function mtv(R, v) { return [R[0][0]*v[0]+R[1][0]*v[1]+R[2][0]*v[2], R[0][1]*v[0]+R[1][1]*v[1]+R[2][1]*v[2], R[0][2]*v[0]+R[1][2]*v[1]+R[2][2]*v[2]]; }
  function mm(A, B) { return A.map(function (r) { return [0, 1, 2].map(function (j) { return r[0]*B[0][j]+r[1]*B[1][j]+r[2]*B[2][j]; }); }); }
  function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
  function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
  function scl(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }
  function dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
  function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
  function unit(a) { return scl(a, 1 / (norm(a) || 1)); }
  function col(R, i) { return [R[0][i], R[1][i], R[2][i]]; }
  function mean(P) { var s = [0, 0, 0]; P.forEach(function (p) { s = add(s, p); }); return scl(s, 1 / Math.max(P.length, 1)); }
  function sgn(v, n) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(n); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  // ---------- theme ----------
  function tok(el, n) { return getComputedStyle(el).getPropertyValue('--mv-' + n).trim() || '#888'; }
  function rgbOf(c) { var x = document.createElement('canvas').getContext('2d'); x.fillStyle = '#000'; x.fillStyle = c; var s = x.fillStyle;
    if (s[0] === '#') { var n = parseInt(s.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; } var m = s.match(/[\d.]+/g) || [0, 0, 0]; return [+m[0], +m[1], +m[2]]; }
  function dark(el) { var c = rgbOf(tok(el, 'card')); return (0.299*c[0] + 0.587*c[1] + 0.114*c[2]) < 110; }
  function alpha(c, a) { var r = rgbOf(c); return 'rgba(' + r[0] + ',' + r[1] + ',' + r[2] + ',' + a + ')'; }
  // ---------- shared interaction ----------
  function orbit(cv, st, redraw) {
    var drag = null, pts = {}, pinch = null;
    cv.addEventListener('pointerdown', function (e) { pts[e.pointerId] = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); st.moved = 0;
      var ids = Object.keys(pts); if (ids.length === 2) { var a = pts[ids[0]], b = pts[ids[1]]; pinch = [Math.hypot(a[0] - b[0], a[1] - b[1]), st.zoom]; drag = null; return; }
      drag = [e.clientX, e.clientY, st.yaw, st.pitch]; cv.style.cursor = 'grabbing'; });
    cv.addEventListener('pointermove', function (e) { if (pts[e.pointerId]) pts[e.pointerId] = [e.clientX, e.clientY];
      if (pinch) { var ids = Object.keys(pts); if (ids.length === 2) { var a = pts[ids[0]], b = pts[ids[1]]; st.zoom = Math.max(0.4, Math.min(6, pinch[1] * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch[0])); st.moved += 5; redraw(); } return; }
      if (!drag) return; st.moved += 1; st.yaw = drag[2] + (e.clientX - drag[0]) * 0.008;
      st.pitch = Math.max(-1.55, Math.min(1.55, drag[3] + (e.clientY - drag[1]) * 0.008)); redraw(); });
    var up = function (e) { delete pts[e.pointerId]; if (Object.keys(pts).length < 2) pinch = null; drag = null; cv.style.cursor = 'grab'; };
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    // wheel zooms with Ctrl (or a pinch on a trackpad) or in full screen; otherwise the page scrolls as usual
    cv.addEventListener('wheel', function (e) { if (!e.ctrlKey && !cv.closest('.mv-fs')) return; e.preventDefault(); st.zoom *= e.deltaY < 0 ? 1.1 : 1 / 1.1; redraw(); }, {passive: false});
  }
  function fullscreen(box, btn, redraw) {
    btn.onclick = function () { var on = box.classList.toggle('mv-fs'); btn.textContent = on ? 'Done' : 'Full screen'; document.documentElement.style.overflow = on ? 'hidden' : ''; redraw(); };
    addEventListener('keydown', function (e) { if (e.key === 'Escape' && box.classList.contains('mv-fs')) btn.click(); });
  }
  function setup(cv) { var r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; cv.width = Math.max(1, r.width * d); cv.height = Math.max(1, r.height * d);
    var g = cv.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, r.width, r.height); g.lineCap = 'round'; return [g, r.width, r.height]; }
  function line(g, a, b, c, w, dash) { g.strokeStyle = c; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.setLineDash([]); }
  function disc(g, p, r, c, ring) { g.fillStyle = c; g.beginPath(); g.arc(p[0], p[1], r, 0, 7); g.fill(); if (ring) { g.strokeStyle = ring; g.lineWidth = 1.2; g.stroke(); } }
  function poly(g, P) { g.beginPath(); P.forEach(function (p, i) { i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }); g.closePath(); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function btn(label, pressed) { var b = el('button', null, label); b.type = 'button'; if (pressed != null) b.setAttribute('aria-pressed', pressed); return b; }

  // =====================================================================================
  // Calibration process
  // =====================================================================================
  function Scene(root, S, opts) {
    if (S.v !== VERSION) { root.textContent = 'This view needs a newer player (spec v' + S.v + ').'; return; }
    var mode = (opts && opts.mode) || root.dataset.mode || 'report';
    var N = S.xyz.length, P = S.xyz, RS = S.R.map(R9), T = S.t;
    var cams = S.cams.map(function (c) {
      var o = Object.assign({}, c); o.Rm = R9(c.R);
      if (c.uvb) { var b = atob(c.uvb), u8 = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) u8[i] = b.charCodeAt(i); o.uv = new Int16Array(u8.buffer); }
      o.imgs = {}; return o;
    });
    var B = S.board, L = S.landmarks || [];
    var pts = P.slice(); if (B) pts = pts.concat(B.outline); else pts = pts.concat(L);
    var center = mean(pts), extent = 1e-3;
    pts.forEach(function (p) { extent = Math.max(extent, norm(sub(p, center))); });
    if (!B && L.length) { var dists = L.map(function (p) { return norm(sub(p, center)); }).sort(function (a, b) { return a - b; }); extent = Math.max(dists[Math.floor(dists.length * 0.9)] || extent, 1e-3); }
    var bn = B ? unit(B.normal) : null;
    if (bn && dot(bn, sub(mean(P), B.center)) < 0) bn = scl(bn, -1);
    var face0 = bn ? Math.atan2(-bn[0], -bn[1]) : -0.9;
    var PRE = {iso: [-0.9, 0.42], top: [-0.9, Math.PI / 2 - 1e-3], front: [face0, 0.05], side: [face0 + Math.PI / 2, 0.05]};
    var st = {yaw: PRE.iso[0], pitch: PRE.iso[1], zoom: 1, moved: 0}, cur = N - 1, sel = 0, playing = false, speed = 1;
    var showAxes = mode === 'report', showImg = true;
    // ---- DOM
    var box = el('div', 'mv-box'), main = el('div', 'mv-main'), stage = el('div', 'mv-stage'), cv = el('canvas'), tools = el('div', 'mv-bar'), readout = el('div', 'mv-readout');
    cv.setAttribute('aria-label', 'Calibration process in 3D'); cv.tabIndex = 0;
    var vb = {}; ['3D', 'Top', 'Front', 'Side'].forEach(function (n, i) { var k = ['iso', 'top', 'front', 'side'][i]; vb[k] = btn(n, i === 0); tools.appendChild(vb[k]); });
    var axesB = btn('Axes', showAxes), imgB = btn('Image', true), fsB = btn('Full screen'); fsB.className = 'mv-gap';
    tools.appendChild(axesB); tools.appendChild(imgB); tools.appendChild(fsB);
    stage.appendChild(cv); stage.appendChild(readout); main.appendChild(stage); main.appendChild(tools);
    var compact = cams.length >= COMPACT;   // two columns of camera images; coverage only as a number
    var side = el('div', 'mv-side' + (compact ? ' mv-compact' : '')), list = el('div', 'mv-list');
    side.appendChild(el('div', 'mv-hd', compact ? 'Cameras · view (' + cams.length + ')' : 'Cameras · view | coverage')); side.appendChild(list);
    var cards = cams.map(function (c, i) {
      var d = el('div', 'mv-cam'); d.setAttribute('aria-current', i === sel);
      var tag = c.jpg ? '' : '<span class="mv-tag">drawn, no frame</span>';
      d.innerHTML = '<b><i style="background:' + c.color + '"></i>' + esc(c.name) + tag + '</b><div class="mv-pair"><canvas></canvas><canvas></canvas></div><small></small>';
      d.onclick = function () { pick(i); }; list.appendChild(d); if (compact) d.title = c.name;
      var cs = d.querySelectorAll('canvas');
      return {el: d, view: cs[0], cov: cs[1], note: d.querySelector('small'), covAt: -1, covCv: null, cells: null, nAll: 0};
    });
    var covis = el('div', 'mv-covis'); side.appendChild(covis);
    var strip = el('div', 'mv-strip'), play = btn('▶︎ Play'), camSel = el('select'), spB = btn('1×'), rng = el('input'), lab = el('span', 'mv-lab');
    camSel.setAttribute('aria-label', 'Camera shown'); cams.forEach(function (c, i) { var o = el('option', null, esc(c.name)); o.value = i; camSel.appendChild(o); });
    spB.setAttribute('aria-label', 'Playback speed'); rng.type = 'range'; rng.min = 0; rng.max = N - 1; rng.step = 1; rng.value = cur; rng.setAttribute('aria-label', 'Calibration progress');
    [play, camSel, spB, rng, lab].forEach(function (x) { strip.appendChild(x); });
    if (cams.length < 2) camSel.style.display = 'none';
    box.appendChild(main); box.appendChild(side); box.appendChild(strip);
    var chips = el('div', 'mv-chips', (B ? '<span><i style="background:var(--mv-board)"></i>Board corners</span>' : '<span><i style="background:var(--mv-muted)"></i>Landmarks</span>')
      + '<span><i class="mv-ln"></i><span class="mv-pathlab">Rig path, darker is later</span></span><span><i style="background:var(--mv-start)"></i>Start</span>'
      + '<span><i style="background:var(--mv-cur)"></i>Current position</span><span><i style="background:var(--mv-accent);border-radius:2px"></i>Selected camera, its image'
      + (B ? ' and its ray to the board' : '') + '</span>');
    root.appendChild(box); root.appendChild(chips);
    // ---- helpers
    function camPose(ci, k) { var c = cams[ci]; return {R: mm(RS[k], c.Rm), t: add(mv(RS[k], c.p), P[k])}; }
    function uvAt(c, k) { return c.uv ? c.uv.subarray(c.uvo[k], c.uvo[k + 1]) : null; }
    function nearestJpg(c, k) { if (!c.jpg) return -1; var a = c.jpg_idx, lo = 0, hi = a.length - 1; while (lo < hi) { var m = (lo + hi) >> 1; if (a[m] < k) lo = m + 1; else hi = m; }
      if (lo > 0 && Math.abs(a[lo - 1] - k) <= Math.abs(a[lo] - k)) lo -= 1; return lo; }
    function image(c, j) { var im = c.imgs[j]; if (im) return im.complete && im.naturalWidth ? im : null;
      im = new Image(); im.onload = function () { drawAll(); }; im.src = 'data:image/jpeg;base64,' + c.jpg[j]; c.imgs[j] = im;
      var keys = Object.keys(c.imgs); if (keys.length > 80) delete c.imgs[keys[0]]; return null; }
    function W2S(p, W, H) { var q = sub(p, center), ca = Math.cos(st.yaw), sa = Math.sin(st.yaw), x1 = ca*q[0] - sa*q[1], y1 = sa*q[0] + ca*q[1], z1 = q[2];
      var cb = Math.cos(st.pitch), sb = Math.sin(st.pitch), s = Math.min(W, H) / (2.1 * extent) * st.zoom; return [W/2 + x1*s, H/2 - (cb*z1 + sb*y1)*s, cb*y1 - sb*z1]; }
    function tlabel(k) { return T ? T[k].toFixed(1) + ' s of ' + T[N - 1].toFixed(1) + ' s' : 'step ' + (k + 1) + ' of ' + N; }
    function lerp(a, b, f) { return 'rgb(' + a.map(function (x, i) { return Math.round(x + (b[i] - x) * f); }).join(',') + ')'; }
    // ---- camera cards
    function fitCanvas(card, c) {   // measure the free space with the images out of the way: a page's own canvas CSS must not size it
      var pair = card.el.querySelector('.mv-pair'); card.view.style.display = card.cov.style.display = 'none';
      var bx = pair.getBoundingClientRect(), d = window.devicePixelRatio || 1; card.view.style.display = ''; card.cov.style.display = compact ? 'none' : '';
      var bw = Math.max(40, compact ? bx.width : (bx.width - 6) / 2), bh = Math.max(30, bx.height || 120), k = Math.min(bw / c.size[0], bh / c.size[1]);
      var w = Math.max(10, Math.round(c.size[0] * k)), h = Math.max(10, Math.round(c.size[1] * k));
      [card.view, card.cov].forEach(function (x) { if (x.width !== w * d || x.height !== h * d) { x.width = w * d; x.height = h * d; x.style.width = w + 'px'; x.style.height = h + 'px'; } });
      return [w, h, d]; }
    function drawCard(i, seen) {
      var c = cams[i], card = cards[i], f = fitCanvas(card, c), w = f[0], h = f[1], d = f[2], s = w / c.size[0];
      var g = card.view.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); g.fillStyle = tok(root, 'panel'); g.fillRect(0, 0, w, h);
      var j = nearestJpg(c, cur), im = j >= 0 ? image(c, j) : null, uv = uvAt(c, cur);
      if (im) { g.drawImage(im, 0, 0, w, h); if (Math.abs(c.jpg_idx[j] - cur) > Math.max(2, N / 120)) { g.fillStyle = 'rgba(0,0,0,.45)'; g.fillRect(0, 0, w, h); } }
      else {
        var o = c.outline && c.outline[cur];
        if (o) { poly(g, o.map(function (q) { return [q[0] * s, q[1] * s]; })); g.fillStyle = 'rgba(255,255,255,.10)'; g.fill(); g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 1; g.stroke(); }
        if (uv) { g.fillStyle = '#4ade80'; for (var k = 0; k < uv.length; k += 2) { g.beginPath(); g.arc(uv[k] * s, uv[k + 1] * s, Math.max(1.2, w / 90), 0, 7); g.fill(); } }
      }
      if (c.ids && c.ids[cur]) c.ids[cur].forEach(function (id) { seen[id] = (seen[id] || 0) + 1; });
      // coverage: incremental, rebuilt when going back
      if (!card.covCv || card.covAt > cur || card.covCv.width !== Math.round(w * d)) { card.covCv = document.createElement('canvas'); card.covCv.width = Math.round(w * d); card.covCv.height = Math.round(h * d); card.covAt = -1; card.cells = {}; card.nAll = 0; }
      var cg = card.covCv.getContext('2d'); cg.setTransform(d, 0, 0, d, 0, 0); cg.fillStyle = 'rgba(96,165,250,.18)';
      for (var kk = card.covAt + 1; kk <= cur; kk++) { var u = uvAt(c, kk); if (!u) continue; card.nAll += u.length / 2;
        for (var q = 0; q < u.length; q += 2) { cg.fillRect(u[q] * s - 1, u[q + 1] * s - 1, 2, 2); card.cells[Math.min(7, Math.floor(u[q] / c.size[0] * 8)) + 8 * Math.min(5, Math.floor(u[q + 1] / c.size[1] * 6))] = 1; } }
      card.covAt = cur;
      g = card.cov.getContext('2d'); g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = tok(root, 'panel'); g.fillRect(0, 0, card.cov.width, card.cov.height);
      g.setTransform(d, 0, 0, d, 0, 0); g.fillStyle = 'rgba(96,165,250,.12)';
      Object.keys(card.cells).forEach(function (kc) { kc = +kc; g.fillRect((kc % 8) * w / 8, Math.floor(kc / 8) * h / 6, w / 8 - 1, h / 6 - 1); });
      g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(card.covCv, 0, 0);
      var cov = c.cov && c.cov[cur] != null ? c.cov[cur] : Object.keys(card.cells).length / 48 * 100, nn = c.n && c.n[cur] != null ? c.n[cur] : card.nAll;
      var nowN = c.ids && c.ids[cur] ? c.ids[cur].length : (uv ? uv.length / 2 : 0);   // the matched frame's corners; uv holds every frame since the last step
      card.note.textContent = compact ? (nowN ? nowN + ' now' : 'none now') + ' · ' + Math.round(cov) + ' % covered'
        : (nowN ? nowN + ' corners now' : 'no detection now') + ' · ' + Math.round(cov) + ' % of the image · ' + nn.toLocaleString() + ' so far';
      card.note.title = (nowN ? nowN + ' corners now' : 'no detection now') + ' · ' + Math.round(cov) + ' % of the image · ' + nn.toLocaleString() + ' so far';
    }
    // ---- 3D
    function draw3d() {
      var r = setup(cv), g = r[0], W = r[1], H = r[2]; if (!W) return;
      var S2 = function (p) { return W2S(p, W, H); }, muted = tok(root, 'muted'), isD = dark(root);
      if (showAxes) {
        var z0 = Math.min.apply(null, pts.map(function (p) { return p[2]; })) - 0.05 * extent, gs = Math.pow(10, Math.floor(Math.log10(extent / 3))), n = Math.ceil(extent / gs) + 1;
        var cx = Math.round(center[0] / gs) * gs, cy = Math.round(center[1] / gs) * gs, gc = tok(root, 'grid');
        for (var i = -n; i <= n; i++) { line(g, S2([cx + i*gs, cy - n*gs, z0]), S2([cx + i*gs, cy + n*gs, z0]), gc, 1); line(g, S2([cx - n*gs, cy + i*gs, z0]), S2([cx + n*gs, cy + i*gs, z0]), gc, 1); }
        var o0 = [cx - n*gs, cy - n*gs, z0]; [[1, 0, 0, 'x'], [0, 1, 0, 'y'], [0, 0, 1, 'z up']].forEach(function (a) { var e = add(o0, scl(a, 2 * gs)); line(g, S2(o0), S2(e), muted, 1.5);
          var se = S2(e); g.fillStyle = muted; g.font = '12px system-ui,sans-serif'; g.fillText(a[3] + ' (m)', se[0] + 4, se[1] - 4); });
        g.fillStyle = muted; g.fillText('grid ' + (gs >= 1 ? gs + ' m' : Math.round(gs * 100) + ' cm'), 12, H - 12);
      }
      if (B) { poly(g, B.outline.map(S2)); g.fillStyle = 'rgba(127,127,127,.14)'; g.fill(); g.strokeStyle = muted; g.lineWidth = 1; g.stroke();
        var bc = tok(root, 'board'); B.pts.forEach(function (p) { disc(g, S2(p), 2.1, bc); }); }
      else { g.fillStyle = alpha(muted, 0.5); L.forEach(function (p) { var q = S2(p); g.fillRect(q[0] - 1, q[1] - 1, 2, 2); }); }
      var lo = isD ? [40, 70, 140] : [191, 219, 254], hi = isD ? [191, 219, 254] : [30, 58, 138], last = T ? T[N - 1] || 1 : N - 1;
      root.querySelector('.mv-pathlab').textContent = isD ? 'Rig path, brighter is later' : 'Rig path, darker is later';
      for (var k = 1; k < N; k++) { var a = S2(P[k - 1]), b = S2(P[k]); if (k > cur) line(g, a, b, 'rgba(140,140,140,.22)', 1.5); else line(g, a, b, lerp(lo, hi, (T ? T[k] : k) / last), 3); }
      disc(g, S2(P[0]), 5, tok(root, 'start'));
      // selected camera: frustum with the image on its face (pinhole rectangle: an exact parallelogram here)
      var cp = camPose(sel, cur), c = cams[sel], K = c.k, cc = c.color;
      var face = function (u, v, dd) { return add(cp.t, mv(cp.R, [(u - K[2]) / K[0] * dd, (v - K[3]) / K[1] * dd, dd])); };
      var A0 = S2(face(0, 0, 1)), A1 = S2(face(c.size[0], 0, 1)), A3 = S2(face(0, c.size[1], 1));
      var longE = Math.max(Math.hypot(A1[0] - A0[0], A1[1] - A0[1]), Math.hypot(A3[0] - A0[0], A3[1] - A0[1]), 1e-6);
      var dep = Math.min(extent * 0.16, Math.max(extent * 0.06, 120 / longE));
      var cs = [[0, 0], [c.size[0], 0], [c.size[0], c.size[1]], [0, c.size[1]]].map(function (q) { return S2(face(q[0], q[1], dep)); }), o = S2(cp.t);
      cs.forEach(function (q) { line(g, o, q, cc, 1.6); });
      var src = cards[sel].view;
      if (showImg && src.width) { var p00 = cs[0], p10 = cs[1], p01 = cs[3], det = (p10[0] - p00[0]) * (p01[1] - p00[1]) - (p10[1] - p00[1]) * (p01[0] - p00[0]);
        g.save(); g.globalAlpha = det > 0 ? 0.92 : 0.4; g.transform((p10[0] - p00[0]) / src.width, (p10[1] - p00[1]) / src.width, (p01[0] - p00[0]) / src.height, (p01[1] - p00[1]) / src.height, p00[0], p00[1]);
        g.drawImage(src, 0, 0); g.restore(); }
      poly(g, cs); g.strokeStyle = cc; g.lineWidth = 2; g.stroke(); if (!showImg) { g.fillStyle = alpha(cc, 0.18); g.fill(); }
      var zc = col(cp.R, 2);
      if (B) { var den = dot(zc, bn), dist = Math.abs(dot(sub(cp.t, B.center), bn));
        if (Math.abs(den) > 1e-6) { var sr = dot(sub(B.center, cp.t), bn) / den; if (sr > 0) line(g, o, S2(add(cp.t, scl(zc, sr))), cc, 1.6, [6, 5]); }
        readout.textContent = c.name + ': ' + (dist * 100).toFixed(0) + ' cm from the board, ' + (Math.acos(Math.min(1, Math.abs(den))) * 180 / Math.PI).toFixed(0) + '° off its normal';
      } else readout.textContent = c.name + ' at ' + tlabel(cur);
      [['#e11d48', 0], ['#16a34a', 1], ['#2563eb', 2]].forEach(function (a) { line(g, S2(P[cur]), S2(add(P[cur], scl(col(RS[cur], a[1]), extent * 0.07))), a[0], 2.5); });
      disc(g, S2(P[cur]), 6, tok(root, 'cur'), '#111827');
    }
    function drawAll() { var seen = {}; cams.forEach(function (c, i) { drawCard(i, seen); });
      if (cams.some(function (c) { return c.ids; })) { var k2 = Object.keys(seen).filter(function (id) { return seen[id] >= 2; }).length;
        covis.textContent = cams.length > 1 ? 'Co-visible now: ' + k2 + ' of ' + Object.keys(seen).length + ' detected corners seen by two or more cameras.' : ''; }
      draw3d(); lab.textContent = tlabel(cur); rng.title = 'step ' + (cur + 1) + ' of ' + N; }
    function pick(i) { sel = i; camSel.value = i; cards.forEach(function (c, j) { c.el.setAttribute('aria-current', j === sel); }); drawAll(); }
    camSel.onchange = function () { pick(+camSel.value); };
    rng.oninput = function () { cur = +rng.value; drawAll(); };
    spB.onclick = function () { speed = speed >= 4 ? 1 : speed * 2; spB.textContent = speed + '×'; };
    var t0, i0;
    function step(ts) { if (!playing) return; if (t0 === undefined) t0 = ts; cur = Math.min(N - 1, i0 + Math.round((ts - t0) * speed / 12000 * (N - 1))); rng.value = cur; drawAll();
      if (cur >= N - 1) stop(); else requestAnimationFrame(step); }
    function stop() { playing = false; play.textContent = '▶︎ Play'; }
    play.onclick = function () { if (playing) return stop(); i0 = cur >= N - 1 ? 0 : cur; t0 = undefined; playing = true; play.textContent = '❚❚ Pause'; requestAnimationFrame(step); };
    Object.keys(vb).forEach(function (k) { vb[k].onclick = function () { st.yaw = PRE[k][0]; st.pitch = PRE[k][1]; st.zoom = 1; Object.keys(vb).forEach(function (x) { vb[x].setAttribute('aria-pressed', x === k); }); draw3d(); }; });
    axesB.onclick = function () { showAxes = !showAxes; axesB.setAttribute('aria-pressed', showAxes); draw3d(); };
    imgB.onclick = function () { showImg = !showImg; imgB.setAttribute('aria-pressed', showImg); draw3d(); };
    orbit(cv, st, draw3d); fullscreen(box, fsB, drawAll);
    return {draw: drawAll};
  }

  // =====================================================================================
  // Solved rig
  // =====================================================================================
  function Rig(root, R) {
    if (R.v !== VERSION) { root.textContent = 'This view needs a newer player (spec v' + R.v + ').'; return; }
    var cams = R.cams.map(function (c) { var o = Object.assign({}, c); o.Rm = R9(c.R); return o; });
    var imus = (R.imus || []).map(function (m) { var o = Object.assign({}, m); o.Rm = R9(m.R); return o; });
    // rig frame: forward = mean optical axis, right = baseline (cam0 -> cam1) off forward, down = forward x right
    var F = unit(cams.reduce(function (a, c) { return add(a, col(c.Rm, 2)); }, [0, 0, 0]));
    if (!(norm(F) > 0.5)) F = col(cams[0].Rm, 2);
    var Rt = col(cams[0].Rm, 0);
    if (cams.length > 1) { var b = sub(cams[1].p, cams[0].p); b = sub(b, scl(F, dot(b, F))); if (norm(b) > 1) Rt = dot(b, Rt) >= 0 ? b : scl(b, -1); }
    Rt = unit(sub(Rt, scl(F, dot(Rt, F))));
    var Q = [Rt, cross(F, Rt), F];
    function toRig(p) { return [dot(p, Q[0]), dot(p, Q[1]), dot(p, Q[2])]; }
    var all = [[0, 0, 0]].concat(cams.map(function (c) { return toRig(c.p); }), imus.map(function (m) { return toRig(m.p); }));
    var rc = mean(all), span = 10; all.forEach(function (p) { span = Math.max(span, norm(sub(p, rc))); });
    var nn = Infinity; cams.forEach(function (a, i) { cams.forEach(function (b2, j) { if (j > i) nn = Math.min(nn, norm(sub(a.p, b2.p))); }); });
    var FR = Math.max(8, Math.min(span * 0.42, isFinite(nn) ? nn * 0.85 : Infinity)), rext = span + FR * 0.45, floorY = Math.max.apply(null, all.map(function (p) { return p[1]; })) + FR * 0.85;
    var PRE = {iso: [0.65, -0.38], front: [Math.PI, 0], top: [Math.PI, -Math.PI / 2 + 1e-3], side: [Math.PI / 2, 0]};
    var st = {yaw: PRE.iso[0], pitch: PRE.iso[1], zoom: 1, moved: 0, touched: false}, rsel = 'imu', hits = [], showSig = false, hover = null, noImu = !!R.origin, ref = R.ref || null, showRef = !!(R.ref && R.ref.cams);
    var refCams = ref ? cams.map(function (c, i) { var q = ref.cams[i]; return q ? {p: q.p, Rm: q.R ? R9(q.R) : null} : null; }) : [];
    if (noImu) rsel = 0;   // camera-only rig: the reference camera is the origin
    var hasSig = cams.some(function (c) { return c.sig_p; });
    // ---- DOM: canvas, toolbar under it, pick panel beside
    var box = el('div', 'mv-box mv-rig'), main = el('div', 'mv-main'), stage = el('div', 'mv-stage'), cv = el('canvas'), bar = el('div', 'mv-bar'), key = el('div', 'mv-key');
    cv.setAttribute('aria-label', 'Solved rig in 3D'); cv.tabIndex = 0;
    var vb = {}; ['3D', 'Front', 'Top', 'Side'].forEach(function (n, i) { var k = ['iso', 'front', 'top', 'side'][i]; vb[k] = btn(n, i === 0); bar.appendChild(vb[k]); });
    var sigB = btn('σ × 10', false); sigB.title = 'Draw each camera\'s 1σ lever and rotation uncertainty, scaled 10 times'; if (hasSig) bar.appendChild(sigB);
    var refB = btn('Reference: ' + (ref ? esc(ref.label) : ''), showRef); refB.title = 'Overlay the reference rig: dashed bodies at its poses'; if (showRef) bar.appendChild(refB);
    var fsB = btn('Full screen'); fsB.className = 'mv-gap'; bar.appendChild(fsB);
    stage.appendChild(cv); stage.appendChild(key); main.appendChild(stage); main.appendChild(bar);
    var info = el('div', 'mv-info'), picks = el('div', 'mv-picks'), infoT = el('div'); info.appendChild(picks); info.appendChild(infoT);
    box.appendChild(main); box.appendChild(info); root.appendChild(box);
    var two = el('div', 'mv-two', '<div class="mv-fig"><b>From above</b><div></div></div><div class="mv-fig"><b>From the front</b><div></div></div>');
    var tbl = el('div', 'mv-scroll', '<table></table>'); root.appendChild(two); root.appendChild(tbl);
    var pickB = {};
    (noImu ? [] : [['imu', 'IMU']]).concat(cams.map(function (c, i) { return [i, c.name]; }), imus.map(function (m, i) { return ['a' + i, m.name]; })).forEach(function (x) {
      var bb = btn(x[1], false); bb.onclick = function () { select(x[0]); }; picks.appendChild(bb); pickB[x[0]] = bb; });
    // ---- projection (rig frame -> screen), and rig-frame vectors -> view space for shading
    function V(q) { var ca = Math.cos(st.yaw), sa = Math.sin(st.yaw), x1 = ca*q[0] + sa*q[2], z1 = -sa*q[0] + ca*q[2], cb = Math.cos(st.pitch), sb = Math.sin(st.pitch);
      return [x1, cb*q[1] - sb*z1, sb*q[1] + cb*z1]; }
    var W = 0, H = 0, S = 1;
    function P(p) { var v = V(sub(p, rc)); return [W/2 + v[0]*S, H/2 + v[1]*S, v[2]]; }
    function camR(c, v) { return toRig(mv(c.Rm, v)); }
    function shade(g, pts, face, light) { poly(g, pts); g.fillStyle = face; g.fill(); g.fillStyle = 'rgba(0,0,0,' + (0.5 * (1 - light)).toFixed(3) + ')'; g.fill(); g.strokeStyle = 'rgba(0,0,0,.28)'; g.lineWidth = 0.8; g.stroke(); }
    function box3(g, o, Rm, sz, face, lens) {   // a shaded box; faces turned away are skipped; lens on +z
      var hx = sz[0]/2, hy = sz[1]/2, hz = sz[2]/2, Lt = unit([0.35, -0.75, -0.55]);
      var F6 = [[[0,0,1],[[-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]]],[[0,0,-1],[[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz]]],[[1,0,0],[[hx,-hy,-hz],[hx,hy,-hz],[hx,hy,hz],[hx,-hy,hz]]],
                [[-1,0,0],[[-hx,-hy,-hz],[-hx,hy,-hz],[-hx,hy,hz],[-hx,-hy,hz]]],[[0,1,0],[[-hx,hy,-hz],[hx,hy,-hz],[hx,hy,hz],[-hx,hy,hz]]],[[0,-1,0],[[-hx,-hy,-hz],[hx,-hy,-hz],[hx,-hy,hz],[-hx,-hy,hz]]]];
      var Rr = function (v) { return toRig(mv(Rm, v)); };
      F6.forEach(function (f) { var n = V(Rr(f[0])); if (n[2] >= -0.02) return;
        shade(g, f[1].map(function (v) { return P(add(o, Rr(v))); }), face, 0.35 + 0.65 * Math.max(0, dot(n, Lt)));
        if (lens && f[0][2] === 1) { var ring = [], pup = []; for (var k = 0; k < 20; k++) { var t = k * Math.PI / 10;
            ring.push(P(add(o, Rr([Math.cos(t)*hy*0.78, Math.sin(t)*hy*0.78, hz])))); pup.push(P(add(o, Rr([Math.cos(t)*hy*0.34, Math.sin(t)*hy*0.34, hz])))); }
          poly(g, ring); g.fillStyle = '#10151d'; g.fill(); poly(g, pup); g.fillStyle = 'rgba(148,197,240,.75)'; g.fill(); } });
    }
    function wire(g, o, Rm, sz, color) {   // a dashed wireframe body: the reference camera
      var hx = sz[0] / 2, hy = sz[1] / 2, hz = sz[2] / 2, q = [];
      [-1, 1].forEach(function (a) { [-1, 1].forEach(function (b2) { [-1, 1].forEach(function (d) { q.push(P(add(o, toRig(mv(Rm, [a*hx, b2*hy, d*hz]))))); }); }); });
      [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function (e) { line(g, q[e[0]], q[e[1]], color, 1.4, [3, 3]); }); }
    function arrow(g, a, b2, c, w) { line(g, a, b2, c, w); var d = Math.atan2(b2[1] - a[1], b2[0] - a[0]), l = Math.max(5, w * 2.8); g.fillStyle = c; g.beginPath();
      g.moveTo(b2[0] + Math.cos(d)*l*0.6, b2[1] + Math.sin(d)*l*0.6); g.lineTo(b2[0] + Math.cos(d + 2.5)*l, b2[1] + Math.sin(d + 2.5)*l); g.lineTo(b2[0] + Math.cos(d - 2.5)*l, b2[1] + Math.sin(d - 2.5)*l); g.closePath(); g.fill(); }
    function hull(pts) { pts = pts.slice().sort(function (a, b2) { return a[0] - b2[0] || a[1] - b2[1]; }); var cr = function (o, a, b2) { return (a[0]-o[0])*(b2[1]-o[1]) - (a[1]-o[1])*(b2[0]-o[0]); }, lo = [], up = [];
      pts.forEach(function (p) { while (lo.length > 1 && cr(lo[lo.length-2], lo[lo.length-1], p) <= 0) lo.pop(); lo.push(p); });
      for (var k = pts.length - 1; k >= 0; k--) { while (up.length > 1 && cr(up[up.length-2], up[up.length-1], pts[k]) <= 0) up.pop(); up.push(pts[k]); } lo.pop(); up.pop(); return lo.concat(up); }
    function selKind() { return rsel === 'imu' ? 'imu' : String(rsel)[0] === 'a' ? 'aux' : 'cam'; }
    function draw() {
      var r = setup(cv), g = r[0]; W = r[1]; H = r[2]; if (!W) return; hits = []; S = Math.min(W, H) / (2.05 * rext) * st.zoom;
      var bg = tok(root, 'card'), soft = tok(root, 'soft'), ink = tok(root, 'ink'), mut = tok(root, 'muted'), ln = tok(root, 'line'), acc = tok(root, 'accent');
      var G = g.createRadialGradient(W/2, H*0.42, 0, W/2, H*0.42, Math.max(W, H)*0.75); G.addColorStop(0, bg); G.addColorStop(1, soft); g.fillStyle = G; g.fillRect(0, 0, W, H);
      // floor grid fading out from the rig
      var R0 = rext * 1.35, stp = Math.pow(10, Math.floor(Math.log10(rext / 2))); if (rext / stp > 9) stp *= 2;
      var n = Math.ceil(R0 / stp), cx = Math.round(rc[0] / stp) * stp, cz = Math.round(rc[2] / stp) * stp;
      for (var i = -n; i <= n; i++) for (var ax = 0; ax < 2; ax++) { var a0 = ax ? [cx + i*stp, floorY, cz - n*stp] : [cx - n*stp, floorY, cz + i*stp], a1 = ax ? [cx + i*stp, floorY, cz + n*stp] : [cx + n*stp, floorY, cz + i*stp];
        for (var s2 = 0; s2 < 8; s2++) { var u = a0.map(function (v, k) { return v + (a1[k] - v) * s2 / 8; }), w = a0.map(function (v, k) { return v + (a1[k] - v) * (s2 + 1) / 8; });
          var md = Math.hypot((u[0] + w[0]) / 2 - rc[0], (u[2] + w[2]) / 2 - rc[2]) / R0, al = Math.max(0, 0.75 * (1 - md * md)); if (al > 0.02) line(g, P(u), P(w), alpha(ln, al.toFixed(3)), 1); } }
      var shadow = function (p, c, rad) { var pts = []; for (var k = 0; k < 24; k++) { var t = k * Math.PI / 12; pts.push(P([p[0] + Math.cos(t)*rad, floorY, p[2] + Math.sin(t)*rad])); } poly(g, pts); g.fillStyle = alpha(c, 0.1); g.fill(); };
      cams.forEach(function (c) { shadow(toRig(c.p), c.color, FR * 0.3); }); imus.forEach(function (m) { shadow(toRig(m.p), ink, FR * 0.2); }); if (!noImu) shadow([0, 0, 0], ink, FR * 0.26);
      var O = P([0, 0, 0]), items = [], labs = [], kind = selKind();
      // lever lines: faint for all, strong for the selected sensor; the baseline solid
      cams.forEach(function (c, i) { var o = P(toRig(c.p)), on = kind === 'cam' && rsel === i; items.push({d: Math.max(O[2], o[2]) + 1e3, f: function () { line(g, O, o, alpha(on ? c.color : mut, on ? 0.95 : 0.35), on ? 1.6 : 1, [3, 4]); }}); });
      imus.forEach(function (m, i) { var o = P(toRig(m.p)), on = rsel === 'a' + i; items.push({d: Math.max(O[2], o[2]) + 1e3, f: function () { line(g, O, o, alpha(mut, on ? 0.95 : 0.35), on ? 1.6 : 1, [3, 4]); }}); });
      if (cams.length > 1) { var pa = P(toRig(cams[0].p)), pb = P(toRig(cams[1].p)); items.push({d: Math.max(pa[2], pb[2]) + 999, f: function () { line(g, pa, pb, alpha(ink, 0.7), 1.4); }}); }
      cams.forEach(function (c, i) {
        var p = toRig(c.p), o = P(p), on = kind === 'cam' && rsel === i, dim = kind === 'cam' && !on, body = FR * 0.26,
          reach = Math.min(c.reach != null ? c.reach : Infinity, c.domain != null ? c.domain : Infinity); if (!isFinite(reach)) reach = null;
        items.push({d: o[2] + FR * 0.5, f: function () {   // field-of-view rim: gradient fill, edge dashed past the fit
          if (!c.rim || c.rim.length < 4) return; g.globalAlpha = dim ? 0.4 : 1;
          var pts = c.rim.map(function (v) { return P(add(p, scl(camR(c, v), FR))); }), tip = P(add(p, scl(camR(c, [0, 0, 1]), FR)));
          var gr = g.createLinearGradient(o[0], o[1], tip[0], tip[1]); gr.addColorStop(0, alpha(c.color, on ? 0.5 : 0.38)); gr.addColorStop(1, alpha(c.color, on ? 0.16 : 0.08));
          poly(g, hull(pts.concat([o]))); g.fillStyle = gr; g.fill();
          for (var k = 0; k < pts.length; k++) { var v0 = c.rim[k], v1 = c.rim[(k + 1) % pts.length], past = reach != null && Math.max(Math.acos(Math.min(1, v0[2])), Math.acos(Math.min(1, v1[2]))) * 180 / Math.PI > reach + 0.5;
            line(g, pts[k], pts[(k + 1) % pts.length], alpha(c.color, on ? 0.95 : 0.6), on ? 1.8 : 1.1, past ? [4, 3] : null); }
          [0, pts.length >> 2, pts.length >> 1, (3 * pts.length) >> 2].forEach(function (k) { line(g, o, pts[k], alpha(c.color, 0.35), 1); });
          g.globalAlpha = 1; }});
        items.push({d: o[2], f: function () {
          g.globalAlpha = dim ? 0.5 : 1;
          if (on) { g.beginPath(); g.arc(o[0], o[1], Math.max(14, body * S * 1.1), 0, 7); g.fillStyle = alpha(c.color, 0.22); g.fill(); }
          box3(g, p, c.Rm, [body * 2.1, body, body * 0.9], c.color, true);
          if (showSig && c.sig_p) {   // 1-sigma lever ellipsoid (IMU axes) and rotation cone, x10
            var e = []; for (var a = 0; a < 12; a++) for (var bb = 0; bb < 6; bb++) { var th = a * Math.PI / 6, ph = (bb + 0.5) * Math.PI / 6;
              e.push(P(add(p, toRig([Math.sin(ph)*Math.cos(th)*c.sig_p[0]*10, Math.sin(ph)*Math.sin(th)*c.sig_p[1]*10, Math.cos(ph)*c.sig_p[2]*10])))); }
            poly(g, hull(e)); g.fillStyle = alpha(c.color, 0.35); g.fill(); g.strokeStyle = tok(root, 'ink'); g.lineWidth = 1.2; g.stroke();
            var sr = Math.hypot(c.sig_r[0], c.sig_r[1]) * 10, ring = []; for (var k2 = 0; k2 < 24; k2++) { var t2 = k2 * Math.PI / 12; ring.push(P(add(p, camR(c, [Math.tan(sr) * Math.cos(t2) * FR * 1.2, Math.tan(sr) * Math.sin(t2) * FR * 1.2, FR * 1.2])))); }
            poly(g, ring); g.strokeStyle = c.color; g.setLineDash([2, 2]); g.stroke(); g.setLineDash([]);
          }
          g.globalAlpha = 1; }});
        var rc0 = showRef ? refCams[i] : null;
        if (rc0) { var rp = toRig(rc0.p), ro = P(rp); items.push({d: ro[2] - 0.5, f: function () {
          g.globalAlpha = dim ? 0.45 : 1; if (norm(sub(rp, p)) > 0.3) line(g, o, ro, alpha(tok(root, 'ink'), 0.55), 1.1, [2, 3]);
          wire(g, rp, rc0.Rm || c.Rm, [body * 2.1, body, body * 0.9], alpha(c.color, 0.85)); g.globalAlpha = 1; }}); }
        labs.push({t: c.name, x: o[0], y: o[1], c: c.color, act: on, dim: dim}); hits.push({k: i, x: o[0], y: o[1]});
      });
      imus.forEach(function (m, i) { var p = toRig(m.p), o = P(p), on = rsel === 'a' + i;
        items.push({d: o[2], f: function () { box3(g, p, m.Rm, [FR * 0.16, FR * 0.16, FR * 0.16], ink, false); }});
        if (on || hover === 'a' + i) labs.push({t: m.name, x: o[0], y: o[1], c: ink, act: on}); hits.push({k: 'a' + i, x: o[0], y: o[1]}); });
      var I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      if (!noImu) items.push({d: O[2] - 1, f: function () {
        if (kind === 'imu') { g.beginPath(); g.arc(O[0], O[1], Math.max(13, FR * 0.24 * S * 1.1), 0, 7); g.fillStyle = alpha(ink, 0.14); g.fill(); }
        box3(g, [0, 0, 0], I3, [FR * 0.24, FR * 0.24, FR * 0.24], ink, false);
        [['x', 0, '#e11d48'], ['y', 1, '#16a34a'], ['z', 2, '#2563eb']].forEach(function (A) { var e = P(toRig(scl(I3[A[1]], FR * 0.62))); if (Math.hypot(e[0] - O[0], e[1] - O[1]) < 6) return;
          arrow(g, O, e, A[2], 2); labs.push({t: A[0], x: e[0] + (e[0] - O[0]) * 0.14, y: e[1] + (e[1] - O[1]) * 0.14, c: A[2], axis: true}); }); }});
      if (!noImu) { hits.push({k: 'imu', x: O[0], y: O[1]}); labs.push({t: 'IMU', x: O[0], y: O[1], c: ink, act: kind === 'imu'}); }
      items.sort(function (a, b2) { return b2.d - a.d; }).forEach(function (it) { it.f(); });
      // selection: the lever as dimension leaders along the rig axes (right, down, forward), in mm
      if (kind !== 'imu') { var sp = kind === 'cam' ? toRig(cams[rsel].p) : toRig(imus[+String(rsel).slice(1)].p), cc = kind === 'cam' ? cams[rsel].color : ink;
        var p1 = [sp[0], 0, 0], p2 = [sp[0], sp[1], 0], legs = [[[0, 0, 0], p1, 'right', sp[0]], [p1, p2, 'down', sp[1]], [p2, sp, 'forward', sp[2]]];
        legs.forEach(function (L) { if (Math.abs(L[3]) < 0.5) return; var a = P(L[0]), b2 = P(L[1]); line(g, a, b2, alpha(cc, 0.9), 1.3, [1, 3]);
          var mx = (a[0] + b2[0]) / 2, my = (a[1] + b2[1]) / 2; labs.push({t: L[2] + ' ' + sgn(L[3], 1), x: mx, y: my, c: cc, dimLabel: true}); }); }
      // labels: pills placed where nothing else is
      g.font = '600 12px system-ui,-apple-system,"Segoe UI",sans-serif'; g.textAlign = 'left'; var used = [];
      labs.sort(function (a, b2) { return (a.axis ? 0 : a.dimLabel ? 2 : 1) - (b2.axis ? 0 : b2.dimLabel ? 2 : 1); }).forEach(function (L) {
        g.font = (L.dimLabel ? '' : '600 ') + (L.dimLabel ? '11px' : '12px') + ' system-ui,-apple-system,"Segoe UI",sans-serif';
        var tw = g.measureText(L.t).width, bw = L.axis ? tw + 4 : tw + 12, bh = L.axis ? 14 : 20, pick = null;
        var c0 = L.axis ? [[0, 0]] : [[0, -24], [0, 24], [28 + tw / 2, 0], [-28 - tw / 2, 0], [24, -22], [-24, -22], [24, 24], [-24, 24], [0, -40], [0, 40], [44 + tw / 2, -30], [-44 - tw / 2, -30], [44 + tw / 2, 30], [-44 - tw / 2, 30], [0, -60], [0, 60]];
        if (L.dimLabel) c0 = [[0, 0], [0, -14], [0, 14], [tw / 2 + 10, 0], [-tw / 2 - 10, 0]];
        c0.some(function (o) { var x = L.x + o[0] - bw / 2, y = L.y + o[1] - bh / 2; if (x < 2 || y < 2 || x + bw > W - 2 || y + bh > H - 2) return false;
          if (used.some(function (q) { return x < q[2] && q[0] < x + bw && y < q[3] && q[1] < y + bh; })) return false; pick = [x, y]; return true; });
        if (!pick) { if (L.dimLabel) return; pick = [Math.max(2, Math.min(W - bw - 2, L.x + c0[0][0] - bw / 2)), Math.max(2, Math.min(H - bh - 2, L.y + c0[0][1] - bh / 2))]; }
        used.push([pick[0], pick[1], pick[0] + bw, pick[1] + bh]);
        if (!L.axis && !L.dimLabel) { var px = Math.max(pick[0], Math.min(L.x, pick[0] + bw)), py = Math.max(pick[1], Math.min(L.y, pick[1] + bh));
          if (Math.hypot(px - L.x, py - L.y) > 18) line(g, [L.x, L.y], [px, py], alpha(L.c, 0.6), 1); }
        if (L.axis) { g.fillStyle = L.c; g.fillText(L.t, pick[0] + 2, pick[1] + 11); return; }
        g.globalAlpha = L.dim ? 0.55 : 1; g.beginPath(); if (g.roundRect) g.roundRect(pick[0], pick[1], bw, bh, 6); else g.rect(pick[0], pick[1], bw, bh);
        g.fillStyle = alpha(bg, 0.88); g.fill(); g.strokeStyle = L.act || L.dimLabel ? L.c : alpha(ln, 0.9); g.lineWidth = L.act ? 1.4 : 1; g.stroke();
        g.fillStyle = L.dimLabel ? L.c : ink; g.fillText(L.t, pick[0] + 6, pick[1] + 14); g.globalAlpha = 1; });
      key.textContent = 'Grid ' + stp + ' mm · ' + (noImu ? 'origin: ' + R.origin : 'arrows: IMU x y z') + ' · rig frame: forward along the cameras, right along the baseline' + (cams.some(function (c) { return c.reach != null || c.domain != null; }) ? ' · dashed rim: past the corners the solve used, or past the model’s 90° limit' : '')
        + (showRef ? ' · dashed boxes: ' + ref.label + (ref.align === 'cam0_rotation' ? ' (aligned on cam0’s rotation)' : '') : '');
    }
    function showInfo() {
      Object.keys(pickB).forEach(function (k) { pickB[k].setAttribute('aria-pressed', String(k) === String(rsel)); });
      var h = '<table>', row = function (a, b2) { h += '<tr><th>' + a + '</th><td>' + b2 + '</td></tr>'; }, kind = selKind();
      if (kind === 'imu') { var u = R.imu || {};
        if (u.Dw) row('Gyro scale', u.Dw.map(function (x) { return x.toFixed(4); }).join(', ')); if (u.Da) row('Accel scale', u.Da.map(function (x) { return x.toFixed(4); }).join(', '));
        if (u.tga != null) row('t_ga', sgn(u.tga, 2) + ' ms'); if (!u.Dw && u.tga == null) row('IMU', 'base frame, origin of the rig'); }
      else if (kind === 'aux') { var m = imus[+String(rsel).slice(1)]; row('Lever', m.p.map(function (x) { return sgn(x, 2); }).join(', ') + ' mm'); row('Distance', norm(m.p).toFixed(1) + ' mm'); row('td', sgn(m.td, 3) + ' ms'); }
      else { var c = cams[rsel], I = c.intr, sp = c.sig_p; row('Model', esc(c.model)); row('fx, fy', I[0].toFixed(2) + ', ' + I[1].toFixed(2) + ' px'); row('cx, cy', I[2].toFixed(2) + ', ' + I[3].toFixed(2) + ' px');
        if (I.length > 4) row('Distortion', I.slice(4).map(function (x) { return x.toFixed(5); }).join(', '));
        row('Field of view', (c.fov.h == null ? 'past the fit' : c.fov.h.toFixed(1) + '°') + ' × ' + (c.fov.v == null ? 'past the fit' : c.fov.v.toFixed(1) + '°') + (c.reach != null ? ' (corners to ' + c.reach.toFixed(0) + '° off axis)' : ''));
        row('Lever', c.p.map(function (x) { return sgn(x, 2); }).join(', ') + ' mm' + (sp ? ' ± ' + sp.map(function (x) { return x.toFixed(2); }).join(', ') : ''));
        row('Distance', norm(c.p).toFixed(1) + ' mm'); row('td', sgn(c.td, 3) + ' ms');
        if (Math.abs(c.tr) > 0.01) row('Readout', sgn(c.tr, 2) + ' ms'); if (c.reproj_med != null) row('Reprojection median / P95', c.reproj_med.toFixed(3) + ' / ' + (c.reproj_p95 == null ? '—' : c.reproj_p95.toFixed(3)) + ' px');
        var rq = ref && refCams[rsel];
        if (rq) { var dp = sub(c.p, rq.p); row(esc(ref.label) + ' lever', rq.p.map(function (x) { return sgn(x, 2); }).join(', ') + ' mm');
          row('MVIS − ' + esc(ref.label), dp.map(function (x) { return sgn(x, 2); }).join(', ') + ' mm (|Δ| ' + norm(dp).toFixed(2) + ' mm)'
              + (rq.Rm ? (ref.align === 'cam0_rotation' && rsel === 0 ? ', rotation aligned on this camera' : ', rotation ' + rotDiff(c.Rm, rq.Rm).toFixed(5) + ' rad') : '')); } }
      infoT.innerHTML = h + '</table>';
      [].forEach.call(tbl.querySelectorAll('tbody tr'), function (tr, k) { tr.style.background = kind === 'cam' && k === rsel ? alpha(tok(root, 'accent'), 0.08) : ''; });
    }
    function rotDiff(A, B) {   // angle of A B^T in rad (|log(A B^T)|)
      var M = mm(A, [[B[0][0], B[1][0], B[2][0]], [B[0][1], B[1][1], B[2][1]], [B[0][2], B[1][2], B[2][2]]]);
      return Math.acos(Math.max(-1, Math.min(1, (M[0][0] + M[1][1] + M[2][2] - 1) / 2))); }
    function select(k) { rsel = k; st.touched = true; showInfo(); draw(); }
    function svgs() {
      var ink = tok(root, 'ink'), mut = tok(root, 'muted'), rp = cams.map(function (c) { return toRig(c.p); });
      function fan(c, ax) {   // angles measured from the camera's own axis, so fields of view near 180 degrees don't wrap
        var z = camR(c, [0, 0, 1]), a0 = Math.atan2(z[ax], z[2]), wrap = function (x) { while (x > Math.PI) x -= 2 * Math.PI; while (x <= -Math.PI) x += 2 * Math.PI; return x; };
        // the field of view in this drawing's plane: where the rim crosses the plane (the out-of-plane rig axis is 0),
        // whatever way the camera is turned about its own axis
        var rr = (c.rim || []).map(function (v) { return camR(c, v); }), out = ax === 0 ? 1 : 0, a = [];
        for (var k = 0; k < rr.length; k++) { var q0 = rr[k], q1 = rr[(k + 1) % rr.length];
          if ((q0[out] <= 0) !== (q1[out] <= 0)) { var t = q0[out] / (q0[out] - q1[out]), q = add(q0, scl(sub(q1, q0), t)); a.push(wrap(Math.atan2(q[ax], q[2]) - a0)); } }
        return a.length >= 2 ? [a0 + Math.min.apply(null, a), a0 + Math.max.apply(null, a)] : null; }
      function panel(holder, pts, ax, fans, flip) {
        var Wd = 420, Hd = 300, xs = pts.map(function (p) { return p[0]; }), ys = pts.map(function (p) { return p[1]; });
        var cx = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2, cy = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
        var s = Math.min(330 / Math.max(Math.max.apply(null, xs) - Math.min.apply(null, xs), 40), 150 / Math.max(Math.max.apply(null, ys) - Math.min.apply(null, ys), 20));
        var X = function (x) { return Wd / 2 + (x - cx) * s; }, Y = function (y) { return Hd / 2 + 30 + (y - cy) * s; }, g = '<svg viewBox="0 0 ' + Wd + ' ' + Hd + '" role="img" font-family="system-ui,sans-serif" font-size="11">';
        var fanLab = {}, fanAt = {};
        if (fans) cams.forEach(function (c, i) { var a = fan(c, ax); if (!a) return; var p = pts[i + 1], Lr = 38, sx = flip ? -1 : 1;
          var lim = Math.min(c.reach != null ? c.reach : Infinity, c.domain != null ? c.domain : Infinity), z0 = camR(c, [0, 0, 1]), ac = Math.atan2(z0[ax], z0[2]);
          var past = isFinite(lim) && Math.max(Math.abs(a[0] - ac), Math.abs(a[1] - ac)) * 180 / Math.PI > lim + 0.5;
          var pa = [X(p[0]) + Lr * Math.sin(a[0]) * sx, Y(p[1]) - Lr * Math.cos(a[0])], pb = [X(p[0]) + Lr * Math.sin(a[1]) * sx, Y(p[1]) - Lr * Math.cos(a[1])];
          g += '<path d="M' + X(p[0]) + ',' + Y(p[1]) + ' L' + pa[0] + ',' + pa[1] + ' A' + Lr + ',' + Lr + ' 0 ' + (a[1] - a[0] > Math.PI ? 1 : 0) + ',' + (flip ? 0 : 1) + ' ' + pb[0] + ',' + pb[1] + ' Z" fill="' + c.color + '" fill-opacity=".16" stroke="' + c.color + '"' + (past ? ' stroke-dasharray="4 3"' : '') + '/>';
          var lab = past ? 'past the fit' : ((a[1] - a[0]) * 180 / Math.PI).toFixed(0) + '°', key = Math.round(X(p[0]) / 10) + ',' + Math.round(Y(p[1]) / 10); fanLab[key] = fanLab[key] && fanLab[key].split(' / ').indexOf(lab) < 0 ? fanLab[key] + ' / ' + lab : (fanLab[key] || lab); fanAt[key] = [X(p[0]), Y(p[1]) - Lr - 4]; });
        Object.keys(fanLab).sort(function (a, b) { return fanAt[a][0] - fanAt[b][0]; }).forEach(function (k, n) { g += '<text x="' + fanAt[k][0] + '" y="' + (fanAt[k][1] - (n % 2) * 13) + '" text-anchor="middle" fill="' + mut + '">' + fanLab[k] + '</text>'; });
        // cameras at the same spot in this view (a stacked pair) share one label
        var groups = []; cams.forEach(function (c, i) { var p = pts[i + 1], gg = groups.find(function (q) { return Math.hypot(X(q.p[0]) - X(p[0]), Y(q.p[1]) - Y(p[1])) < 10; });
          if (gg) gg.idx.push(i); else groups.push({p: p, idx: [i]}); });
        groups.sort(function (a, b) { return X(a.p[0]) - X(b.p[0]); });
        groups.forEach(function (q, n) { var p = q.p; g += '<line x1="' + X(pts[0][0]) + '" y1="' + Y(pts[0][1]) + '" x2="' + X(p[0]) + '" y2="' + Y(p[1]) + '" stroke="' + mut + '" stroke-dasharray="3 3"/>';
          if (fans && !(noImu && Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) < 0.5)) g += '<text x="' + X(p[0]) + '" y="' + (Y(p[1]) + 33 + (n % 2) * 13) + '" text-anchor="middle" fill="' + mut + '">' + Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]).toFixed(1) + ' mm in plane</text>'; });
        if (cams.length > 1) { var b0 = pts[1], b1 = pts[2], yb = Math.max.apply(null, pts.map(function (q) { return Y(q[1]); })) + 52;
          g += '<line x1="' + X(b0[0]) + '" y1="' + yb + '" x2="' + X(b1[0]) + '" y2="' + yb + '" stroke="' + ink + '"/><text x="' + (X(b0[0]) + X(b1[0])) / 2 + '" y="' + (yb + 15) + '" text-anchor="middle" fill="' + ink + '" font-weight="600">' + norm(sub(cams[1].p, cams[0].p)).toFixed(2) + ' mm baseline (cam0–cam1)</text>'; }
        if (!noImu) g += '<rect x="' + (X(pts[0][0]) - 5) + '" y="' + (Y(pts[0][1]) - 5) + '" width="10" height="10" fill="' + ink + '"/><text x="' + X(pts[0][0]) + '" y="' + (Y(pts[0][1]) + (fans ? 20 : -10)) + '" text-anchor="middle" fill="' + ink + '">' + esc(R.origin || 'IMU') + '</text>';
        groups.forEach(function (q) { var p = q.p; q.idx.forEach(function (i, k) { g += '<circle cx="' + (X(p[0]) + (k - (q.idx.length - 1) / 2) * 9) + '" cy="' + Y(p[1]) + '" r="6" fill="' + cams[i].color + '" stroke="' + tok(root, 'card') + '"/>'; });
          g += '<text x="' + X(p[0]) + '" y="' + (Y(p[1]) + 20) + '" text-anchor="middle" fill="' + ink + '" font-weight="600">' + q.idx.map(function (i) { return esc(cams[i].name); }).join(' / ') + '</text>'; });
        if (showRef) cams.forEach(function (c, i) { var q = refCams[i]; if (!q) return; var rp2 = toRig(q.p), qq = fans ? [rp2[0], -rp2[2]] : [-rp2[0], rp2[1]], p = pts[i + 1];
          g += '<line x1="' + X(p[0]) + '" y1="' + Y(p[1]) + '" x2="' + X(qq[0]) + '" y2="' + Y(qq[1]) + '" stroke="' + mut + '" stroke-dasharray="2 2"/><circle cx="' + X(qq[0]) + '" cy="' + Y(qq[1]) + '" r="5" fill="none" stroke="' + c.color + '" stroke-width="1.5" stroke-dasharray="2 1.5"/>'; });
        var bar2 = Math.pow(10, Math.round(Math.log10(60 / s))); g += '<line x1="' + (Wd - 14 - bar2 * s) + '" y1="16" x2="' + (Wd - 14) + '" y2="16" stroke="' + ink + '" stroke-width="2"/><text x="' + (Wd - 14) + '" y="11" text-anchor="end" fill="' + mut + '">' + bar2 + ' mm</text></svg>';
        holder.innerHTML = g; }
      var hs = two.querySelectorAll('.mv-fig div');
      panel(hs[0], [[0, 0]].concat(rp.map(function (p) { return [p[0], -p[2]]; })), 0, true, false);
      panel(hs[1], [[0, 0]].concat(rp.map(function (p) { return [-p[0], p[1]]; })), 1, false, true);
      var t = '<thead><tr><th>Camera</th><th>Lever x, y, z (mm)</th><th>Distance (mm)</th><th>td (ms)</th><th>Readout (ms)</th><th>Field of view</th><th>Reprojection median / P95 (px)</th>' + (ref ? '<th>Δ ' + esc(ref.label) + '</th>' : '') + '</tr></thead><tbody>';
      cams.forEach(function (c) { t += '<tr><th>' + esc(c.name) + '</th><td>' + c.p.map(function (x) { return sgn(x, 1); }).join(', ') + '</td><td>' + norm(c.p).toFixed(1) + '</td><td>' + sgn(c.td, 2) + '</td><td>'
        + (Math.abs(c.tr) > 0.01 ? c.tr.toFixed(2) : '—') + '</td><td>' + (c.fov.h == null ? 'past the fit' : c.fov.h.toFixed(0) + '°') + ' × ' + (c.fov.v == null ? 'past the fit' : c.fov.v.toFixed(0) + '°') + '</td><td>'
        + (c.reproj_med == null ? '—' : c.reproj_med.toFixed(3) + ' / ' + (c.reproj_p95 == null ? '—' : c.reproj_p95.toFixed(3))) + '</td>' + (ref ? '<td>' + (refCams[cams.indexOf(c)] ? norm(sub(c.p, refCams[cams.indexOf(c)].p)).toFixed(2) + ' mm' + (refCams[cams.indexOf(c)].Rm && !(ref.align === 'cam0_rotation' && cams.indexOf(c) === 0) ? ', ' + rotDiff(c.Rm, refCams[cams.indexOf(c)].Rm).toFixed(4) + ' rad' : '') : '—') + '</td>' : '') + '</tr>'; });
      tbl.querySelector('table').innerHTML = t + '</tbody>';
    }
    function near(e) { var r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, best = null, bd = 26;
      hits.forEach(function (h) { var d = Math.hypot(h.x - x, h.y - y); if (d < bd) { bd = d; best = h; } }); return best; }
    cv.addEventListener('click', function (e) { if (st.moved > 3) return; var b2 = near(e); if (b2) select(b2.k); });
    cv.addEventListener('pointermove', function (e) { if (e.buttons) return; var b2 = near(e), k = b2 ? b2.k : null; if (k !== hover) { hover = k; cv.style.cursor = k != null ? 'pointer' : 'grab'; draw(); } });
    Object.keys(vb).forEach(function (k) { vb[k].onclick = function () { st.touched = true; st.yaw = PRE[k][0]; st.pitch = PRE[k][1]; st.zoom = 1; Object.keys(vb).forEach(function (x) { vb[x].setAttribute('aria-pressed', x === k); }); draw(); }; });
    sigB.onclick = function () { showSig = !showSig; sigB.setAttribute('aria-pressed', showSig); draw(); };
    refB.onclick = function () { showRef = !showRef; refB.setAttribute('aria-pressed', showRef); draw(); svgs(); };
    orbit(cv, st, function () { st.touched = true; draw(); }); fullscreen(box, fsB, draw); showInfo();
    // idle sway while on screen and untouched; off for reduced motion
    if (!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches)) (function () {
      var seen = false, y0 = st.yaw, t0 = null;
      if ('IntersectionObserver' in window) new IntersectionObserver(function (es) { seen = es[0].isIntersecting; }).observe(cv); else seen = true;
      function tick(ts) { if (st.touched) return; if (seen && !document.hidden) { if (t0 == null) t0 = ts; st.yaw = y0 + 0.32 * Math.sin((ts - t0) / 2600); draw(); } requestAnimationFrame(tick); }
      requestAnimationFrame(tick); })();
    return {draw: function () { draw(); svgs(); showInfo(); }};
  }

  // ---------- boot ----------
  var made = [];
  function render(root, spec) { var v = root.dataset.mvv === 'rig' ? Rig(root, spec) : Scene(root, spec, {mode: root.dataset.mode}); if (v) { made.push(v); v.draw(); } }
  function boot(root) {
    if (root.dataset.mvvReady) return; root.dataset.mvvReady = '1';
    var data = root.querySelector('script[type="application/json"]');
    if (data) return render(root, JSON.parse(data.textContent));
    if (!root.dataset.src) return;   // data-src: fetch the spec (the sensor pages load it on demand)
    var note = el('p', 'mv-lead', 'Loading\u2026'); root.appendChild(note);
    fetch(root.dataset.src).then(function (r) { return r.json(); }).then(function (spec) { note.remove(); render(root, spec); })
      .catch(function () { note.textContent = 'The figure could not load.'; });
  }
  function lazy(root) {   // boot when scrolled into view
    if (!root.dataset.src || !('IntersectionObserver' in window)) return boot(root);
    var io = new IntersectionObserver(function (es) { if (es[0].isIntersecting) { io.disconnect(); boot(root); } }, {rootMargin: '200px'}); io.observe(root);
  }
  function redrawAll() { made.forEach(function (v) { v.draw(); }); }
  window.mvisViews = {boot: boot, redraw: redrawAll, version: VERSION,
    init: function (scope) { [].forEach.call((scope || document).querySelectorAll('.mvv[data-mvv]'), lazy); }};
  var rt; addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(redrawAll, 80); });
  if (window.matchMedia) matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
  document.addEventListener('themechange', redrawAll);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { window.mvisViews.init(); }); else window.mvisViews.init();
})();
