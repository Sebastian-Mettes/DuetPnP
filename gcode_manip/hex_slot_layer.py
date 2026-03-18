#!/usr/bin/env python3
"""
hex_slot_layer.py — Slotted overlay layer for geodesic spherical cap.

Generates a thin spherical shell (LAYER_T thick) sitting on top of the
geodesic cap's outer surface (ID = R_OUTER).  Each hex/pentagon cell has
a through-slot cut, defined by:

  Slot centerline : a polygon whose vertices are each d mm inward from the
                    cell vertex, along the cell-centre → vertex line.
  Slot width      : t mm total (t/2 each side of the centreline, sharp
                    miter corners).
  Slot walls      : radial (normal to local sphere surface).
  Independence    : each cell's slot is a closed loop, isolated from
                    neighbouring cells.

Usage
-----
  python hex_slot_layer.py                    # d=0.75, t=0.70, layer=0.1 mm
  python hex_slot_layer.py --d 1.0 --t 0.8
  python hex_slot_layer.py --area 25.0 --t 0.7   # area mode

Outputs
-------
  /mnt/user-data/outputs/geodesic_slot_layer.stl
"""

import argparse, math, os, struct, sys, time
from collections import defaultdict

import numpy as np

# ── Import shared constants / helpers from the existing pipeline ─────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from transform_pipeline import (
    build_geodesic_mesh, _make_R_mat, _CAP_AXIS_APPROX,
    FREQ, R_OUTER, COS_CAP,
)

# ── Parameters ────────────────────────────────────────────────────────────────
LAYER_T   = 0.1            # mm — layer thickness
R_BOT     = R_OUTER        # inner surface of layer (rests on cap outer surface)
R_TOP_DEF = R_OUTER + LAYER_T  # outer surface of layer (updated from args)

D_DEFAULT = 0.75           # mm — inward offset from cell vertex to CL vertex
T_DEFAULT = 0.70           # mm — total slot width

OUT_DIR   = "/mnt/user-data/outputs"
OUT_STL   = os.path.join(OUT_DIR, "geodesic_slot_layer.stl")

# ── 2D geometry helpers ───────────────────────────────────────────────────────

def _norm2(v):
    l = math.hypot(float(v[0]), float(v[1]))
    return v / l if l > 1e-30 else v

def _signed_area_2d(pts):
    """Shoelace signed area — positive means CCW."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return a * 0.5

def _area_2d(pts):
    return abs(_signed_area_2d(pts))

def _ensure_ccw(pts):
    """Return pts as np.array in CCW order."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts if _signed_area_2d(pts) > 0 else pts[::-1]

def _offset_polygon_miter(pts_2d, d):
    """
    Offset a 2D CCW polygon by distance d (positive=outward, negative=inward).
    Uses miter joins (sharp corners).  Returns np.array of same length.
    """
    pts_2d = np.asarray(pts_2d, dtype=np.float64)
    n   = len(pts_2d)
    out = []
    for i in range(n):
        p_prev = pts_2d[(i - 1) % n]
        p_cur  = pts_2d[i]
        p_next = pts_2d[(i + 1) % n]

        # Outward edge normals for the two edges meeting at p_cur.
        # For a CCW polygon the outward normal of edge p→q is
        # rotate(normalize(q-p), -90°) = (dy, -dx).
        e1 = _norm2(p_cur  - p_prev);  n1 = np.array([ e1[1], -e1[0]])
        e2 = _norm2(p_next - p_cur);   n2 = np.array([ e2[1], -e2[0]])

        # Miter bisector
        bis     = n1 + n2
        bis_len = np.linalg.norm(bis)
        if bis_len < 1e-10:
            out.append(p_cur + d * n1)
        else:
            bis   = bis / bis_len
            cos_a = float(np.dot(bis, n1))
            if abs(cos_a) < 1e-10:
                out.append(p_cur + d * n1)
            else:
                out.append(p_cur + (d / cos_a) * bis)
    return np.array(out)

# ── Sphere projection ─────────────────────────────────────────────────────────

def _to_sphere(x, y, R):
    """
    Radially project 2D (x,y) from the rotated frame onto sphere of radius R.
    The cap sits on the +Z hemisphere, so z = sqrt(R²-x²-y²).
    """
    z2 = R * R - x * x - y * y
    z  = math.sqrt(max(z2, 0.0))
    return np.array([float(x), float(y), z], dtype=np.float64)

# ── Slot geometry ─────────────────────────────────────────────────────────────

def _centerline_2d(cell_xy, d):
    """
    Compute centerline polygon: each vertex moved d inward from the cell
    vertex along the cell-centre → vertex direction.
    """
    cell_xy = np.asarray(cell_xy, dtype=np.float64)
    C       = cell_xy.mean(axis=0)
    return np.array([V + d * _norm2(C - V) for V in cell_xy])

def _centerline_area(cell_xy, d):
    return _area_2d(_centerline_2d(cell_xy, d))

def _d_for_area(cell_xy, target_area):
    """Binary search for d such that centerline polygon area == target_area."""
    # Maximum d is the apothem (shortest vertex-to-centre distance).
    max_d = min(np.linalg.norm(cell_xy.mean(0) - v) for v in cell_xy) * 0.999
    # Sanity: if target >= full cell area, return d=0; if target<=0, return max_d
    if target_area >= _area_2d(cell_xy):
        return 0.0
    if target_area <= 0:
        return max_d
    # area decreases as d increases; binary search
    lo, hi = 0.0, max_d
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _centerline_area(cell_xy, mid) > target_area:
            lo = mid   # area too large → need more inward offset
        else:
            hi = mid
    return (lo + hi) * 0.5

# ── STL output ────────────────────────────────────────────────────────────────

def _tri_normal(a, b, c):
    n = np.cross(b - a, c - a)
    l = np.linalg.norm(n)
    return (n / l).astype(np.float32) if l > 1e-30 else np.zeros(3, np.float32)

def _write_stl(path, triangles):
    """Write binary STL. triangles = list of (v0, v1, v2) float array-likes."""
    label = b'Geodesic slot layer'
    hdr   = label + b' ' * (80 - len(label))
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(struct.pack('<I', len(triangles)))
        for a, b, c in triangles:
            a = np.asarray(a, np.float32)
            b = np.asarray(b, np.float32)
            c = np.asarray(c, np.float32)
            f.write(_tri_normal(a, b, c).tobytes())
            f.write(a.tobytes()); f.write(b.tobytes()); f.write(c.tobytes())
            f.write(b'\x00\x00')

# ── Main build ────────────────────────────────────────────────────────────────

def build_slot_layer(d=D_DEFAULT, t=T_DEFAULT, target_area=None,
                     layer_t=LAYER_T, out_stl=OUT_STL):

    R_TOP = R_BOT + layer_t

    print("─── Hex Slot Layer ─────────────────────────────────────────")
    print(f"  Mode        : {'area (target={:.4f} mm²)'.format(target_area) if target_area else 'd-mode'}")
    print(f"  d           : {d:.3f} mm" if target_area is None else "  d           : computed per cell")
    print(f"  t (width)   : {t:.3f} mm  (t/2={t/2:.3f} each side)")
    print(f"  layer_t     : {layer_t:.3f} mm")
    print(f"  R_BOT/R_TOP : {R_BOT:.3f} / {R_TOP:.3f} mm")
    t0 = time.time()

    print(f"\n  Building geodesic mesh (freq={FREQ}) …")
    mesh = build_geodesic_mesh(FREQ, R_OUTER)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Derive exact CAP_AXIS from hex centroids (same logic as transform_pipeline)
    hex_cents_n = []
    for face in mesh.faces:
        if len(face) != 6: continue
        c = np.mean([mesh.vertices[vi] for vi in face], axis=0)
        hex_cents_n.append(c / np.linalg.norm(c))
    best_hi  = int(np.argmax(np.array(hex_cents_n) @ _CAP_AXIS_APPROX))
    CAP_AXIS = hex_cents_n[best_hi]
    R_MAT    = _make_R_mat(CAP_AXIS)
    print(f"  CAP_AXIS (exact): [{CAP_AXIS[0]:.6f}, {CAP_AXIS[1]:.6f}, {CAP_AXIS[2]:.6f}]")

    # Select cap faces
    cap_faces = []
    for face in mesh.faces:
        if len(face) not in (5, 6): continue
        c = np.mean([mesh.vertices[vi] for vi in face], axis=0)
        if (c / np.linalg.norm(c)) @ CAP_AXIS >= COS_CAP:
            cap_faces.append(list(face))
    print(f"  Cap faces   : {len(cap_faces)}")

    # ── Boundary edge detection ───────────────────────────────────────────────
    # boundary edge = appears in exactly one cap face; direction as traversed
    edge_count = defaultdict(int)
    edge_dir   = {}          # key=(min,max) → (a,b) as in the single cap face
    for face in cap_faces:
        n = len(face)
        for i in range(n):
            a, b  = face[i], face[(i + 1) % n]
            key   = (min(a, b), max(a, b))
            edge_count[key] += 1
            edge_dir[key]   = (a, b)   # overwritten, but boundary edges appear once

    boundary_edges = [edge_dir[k] for k, cnt in edge_count.items() if cnt == 1]
    print(f"  Boundary edges: {len(boundary_edges)}")

    # ── Accumulate triangles ──────────────────────────────────────────────────
    tris = []

    # Statistics
    all_cl_areas = []
    all_cl_sides = []
    all_d_values = []

    # ── Per-cell processing ───────────────────────────────────────────────────
    for face in cap_faces:
        n = len(face)

        # Rotate world vertices into cap-aligned frame
        verts_rot = [R_MAT @ mesh.vertices[vi] for vi in face]

        # 2D XY in rotated frame — ensure CCW
        xy_raw = np.array([[v[0], v[1]] for v in verts_rot])
        if _signed_area_2d(xy_raw) < 0:
            xy_raw    = xy_raw[::-1]
            verts_rot = verts_rot[::-1]   # keep 3D and 2D in sync (for reference)
        xy = xy_raw   # CCW, shape (n, 2)

        # Per-cell d value
        d_cell = _d_for_area(xy, target_area) if target_area is not None else d
        all_d_values.append(d_cell)

        # Centerline polygon (CCW)
        cl_xy = _ensure_ccw(_centerline_2d(xy, d_cell))

        # Slot outer and inner loops (both CCW after offset of CCW polygon)
        outer_xy = _offset_polygon_miter(cl_xy, +t * 0.5)
        inner_xy = _offset_polygon_miter(cl_xy, -t * 0.5)

        # Guard: inner loop should not collapse (t/2 < d guaranteed by caller,
        # but check anyway)
        if _area_2d(inner_xy) < 1e-6 or _area_2d(outer_xy) < 1e-6:
            print(f"  Warning: slot degenerate for face {face}, skipping")
            continue

        # ── Statistics (on centerline in 3D on R_BOT) ────────────────────────
        cl_3d = [_to_sphere(p[0], p[1], R_BOT) for p in cl_xy]
        # 3D area via fan triangulation from cl_3d[0]
        cl_area_3d = sum(
            0.5 * float(np.linalg.norm(
                np.cross(cl_3d[i] - cl_3d[0], cl_3d[i + 1] - cl_3d[0])
            ))
            for i in range(1, n - 1)
        )
        all_cl_areas.append(cl_area_3d)
        all_cl_sides.extend(
            float(np.linalg.norm(cl_3d[(i + 1) % n] - cl_3d[i]))
            for i in range(n)
        )

        # ── Project all loops to 3D at R_TOP and R_BOT ───────────────────────
        cell_top  = [_to_sphere(p[0], p[1], R_TOP) for p in xy]
        cell_bot  = [_to_sphere(p[0], p[1], R_BOT) for p in xy]
        outer_top = [_to_sphere(p[0], p[1], R_TOP) for p in outer_xy]
        outer_bot = [_to_sphere(p[0], p[1], R_BOT) for p in outer_xy]
        inner_top = [_to_sphere(p[0], p[1], R_TOP) for p in inner_xy]
        inner_bot = [_to_sphere(p[0], p[1], R_BOT) for p in inner_xy]

        # Vertex alignment: cell[i] ↔ outer[i] ↔ inner[i] (all derived from
        # the same vertex ordering, so index correspondence is natural).

        # ── Top face (normal ≈ +Z, CCW when viewed from above) ───────────────

        # Outer frame: ring between cell boundary (outer) and outer_slot (inner hole).
        # For a CCW outer boundary and CCW inner boundary, the frame triangulation:
        #   (A[i], A[j], B[i])  and  (A[j], B[j], B[i])
        # produces CCW top-face triangles (verified geometrically).
        for i in range(n):
            j = (i + 1) % n
            tris.append((cell_top[i],  cell_top[j],  outer_top[i]))
            tris.append((cell_top[j],  outer_top[j], outer_top[i]))

        # Inner disk: fan from inner_top[0] — CCW for top face
        for i in range(1, n - 1):
            tris.append((inner_top[0], inner_top[i], inner_top[i + 1]))

        # ── Bottom face (normal ≈ −Z, winding reversed from top) ─────────────
        for i in range(n):
            j = (i + 1) % n
            tris.append((cell_bot[i],  outer_bot[i], cell_bot[j]))
            tris.append((cell_bot[j],  outer_bot[i], outer_bot[j]))

        for i in range(1, n - 1):
            tris.append((inner_bot[0], inner_bot[i + 1], inner_bot[i]))

        # ── Outer slot wall ───────────────────────────────────────────────────
        # The outer slot loop is CCW (viewed from above).  For a CCW polygon,
        # the outward normal of edge i→j faces to the right of travel, i.e.
        # away from the polygon interior (= away from hex centre = outward).
        # Wall winding for outward-facing normal (viewed from outside the slot):
        #   (top[i], bot[i], bot[j])  then  (top[i], bot[j], top[j])
        for i in range(n):
            j = (i + 1) % n
            tris.append((outer_top[i], outer_bot[i], outer_bot[j]))
            tris.append((outer_top[i], outer_bot[j], outer_top[j]))

        # ── Inner slot wall ───────────────────────────────────────────────────
        # The inner slot loop is CCW.  Its outward-facing wall faces inward
        # (toward hex centre), i.e. to the LEFT of travel (into the slot).
        # Reverse winding relative to outer wall:
        #   (top[i], bot[j], bot[i])  then  (top[i], top[j], bot[j])
        for i in range(n):
            j = (i + 1) % n
            tris.append((inner_top[i], inner_bot[j], inner_bot[i]))
            tris.append((inner_top[i], inner_top[j], inner_bot[j]))

    # ── Cap perimeter wall ────────────────────────────────────────────────────
    # boundary_edges stores (a, b) as traversed inside the cap face (CCW winding).
    # For a CCW face, the outward-facing wall is to the right of a→b.
    # Wall winding for outward-facing normal:
    #   (a_top, a_bot, b_bot)  then  (a_top, b_bot, b_top)
    for (a, b) in boundary_edges:
        va_r = R_MAT @ mesh.vertices[a]
        vb_r = R_MAT @ mesh.vertices[b]
        a_top = _to_sphere(va_r[0], va_r[1], R_TOP)
        a_bot = _to_sphere(va_r[0], va_r[1], R_BOT)
        b_top = _to_sphere(vb_r[0], vb_r[1], R_TOP)
        b_bot = _to_sphere(vb_r[0], vb_r[1], R_BOT)
        tris.append((a_top, a_bot, b_bot))
        tris.append((a_top, b_bot, b_top))

    # ── Write STL ─────────────────────────────────────────────────────────────
    _write_stl(out_stl, tris)
    sz = os.path.getsize(out_stl)

    # Verify header
    with open(out_stl, 'rb') as f:
        f.read(80)
        n_declared = struct.unpack('<I', f.read(4))[0]
    assert n_declared == len(tris), f"STL header mismatch: {n_declared} vs {len(tris)}"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Triangles  : {len(tris)}")
    print(f"  STL size   : {sz // 1024} kB  ({sz} bytes)")
    print(f"  Header OK  : declared={n_declared} ✓")
    print(f"  Output     : {out_stl}")

    print(f"\n  ── Slot Statistics ───────────────────────────────────────")
    if target_area is not None:
        print(f"  Target CL area  : {target_area:.4f} mm²")
        print(f"  d range         : {min(all_d_values):.4f} – {max(all_d_values):.4f} mm")
        print(f"  d mean ± std    : {np.mean(all_d_values):.4f} ± {np.std(all_d_values):.4f} mm")
    else:
        print(f"  d (fixed)       : {d:.4f} mm")
    print(f"  t (slot width)  : {t:.4f} mm")
    print(f"  Cells           : {len(all_cl_areas)}")
    print(f"  CL area mean    : {np.mean(all_cl_areas):.4f} mm²")
    print(f"  CL area range   : {min(all_cl_areas):.4f} – {max(all_cl_areas):.4f} mm²")
    print(f"  CL side mean    : {np.mean(all_cl_sides):.4f} mm")
    print(f"  CL side range   : {min(all_cl_sides):.4f} – {max(all_cl_sides):.4f} mm")
    print(f"  Elapsed         : {time.time() - t0:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build a slotted geodesic cap overlay layer STL.")
    ap.add_argument("--d",       type=float, default=D_DEFAULT,
                    help="Inward vertex offset (mm, default %(default)s)")
    ap.add_argument("--t",       type=float, default=T_DEFAULT,
                    help="Total slot width (mm, default %(default)s)")
    ap.add_argument("--area",    type=float, default=None,
                    help="Target centerline area per slot (mm²) — overrides --d")
    ap.add_argument("--layer-t", type=float, default=LAYER_T,
                    help="Layer thickness (mm, default %(default)s)")
    ap.add_argument("--out-stl", default=OUT_STL,
                    help="Output STL path")
    args = ap.parse_args()

    build_slot_layer(
        d           = args.d,
        t           = args.t,
        target_area = args.area,
        layer_t     = args.layer_t,
        out_stl     = args.out_stl,
    )
