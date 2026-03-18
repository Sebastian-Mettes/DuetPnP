#!/usr/bin/env python3
"""
transform_pipeline.py
=====================
Two-stage pipeline for printing a geodesic spherical cap on a flat-bed printer.

COORDINATE CONVENTION
---------------------
  WORLD frame   – original geodesic sphere (CAP_AXIS tilted at θ=20.9°, φ=180°)
  FLAT frame    – world rotated so CAP_AXIS → +Z, then orthographically flattened
                  (this is what Cura slices)
    R_mat       : world → rotated   (R_mat @ v_world = v_rotated)

STAGE 1 – forward_transform()
    1. Build the geodesic cap at R_outer and R_inner.
    2. Rotate every vertex so the cap centre aligns with +Z.
    3. Flatten orthographically:
         outer vertices → keep (x_rot, y_rot), set z = THICKNESS
         inner vertices → keep (x_rot, y_rot), set z = 0
    4. Write flat STL for Cura.
    5. Write vertex_table.json with one unified XY mesh.
       Each vertex stores:
         "xy"    : [x, y]      – flat XY position (shared by both surfaces)
         "outer" : [x, y, z]   – world position on outer sphere
         "inner" : [x, y, z]   – world position on inner sphere

STAGE 2 – reverse_transform_gcode()
    The G-code is processed layer by layer (each distinct Z value in the file).
    For each layer at flat z_layer:

        t = z_layer / THICKNESS          # 0 = inner surface, 1 = outer surface

    For every G0/G1 move at flat (fx, fy, z_layer):
        1. Barycentric lookup in the XY mesh → weights (w0, w1, w2) for triangle
        2. Interpolate curved positions on both surfaces:
               p_outer = Σ wi · outer_world[i]
               p_inner = Σ wi · inner_world[i]
        3. Layer surface point:  p_world = lerp(p_inner, p_outer, t)
        4. Rewrite move with (p_world.x, p_world.y, p_world.z)
        5. E-correction: E_new = E_cura × (curved_seg_len / flat_seg_len)

    Points outside the XY mesh domain (travels to/from skirt etc.) are passed
    through unchanged.
"""

import numpy as np
from collections import defaultdict
import struct, json, os, re, time

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
FREQ       = 33
R_OUTER    = 322.62
THICKNESS  = 0.5
R_INNER    = R_OUTER - THICKNESS

CAP_HALF_ANGLE = 15.0
COS_CAP        = np.cos(np.radians(CAP_HALF_ANGLE))

# Approximate axis used only to identify the best cap hex from the mesh.
# The exact axis (and R_MAT) are computed inside forward_transform() from
# the actual mesh centroid so the centre hex lands at exactly XY = (0, 0).
_THETA_APPROX = np.radians(20.9)
_PHI_APPROX   = np.radians(180.0)
_CAP_AXIS_APPROX = np.array([
    np.sin(_THETA_APPROX) * np.cos(_PHI_APPROX),
    np.sin(_THETA_APPROX) * np.sin(_PHI_APPROX),
    np.cos(_THETA_APPROX),
])


def _make_R_mat(cap_axis):
    """Build orthonormal rotation matrix that maps cap_axis → +Z."""
    ref = np.array([0., 0., 1.]) if abs(cap_axis[2]) < 0.9 else np.array([1., 0., 0.])
    U   = np.cross(cap_axis, ref);  U /= np.linalg.norm(U)
    V   = np.cross(cap_axis, U);    V /= np.linalg.norm(V)
    R   = np.array([U, V, cap_axis])
    assert np.allclose(R @ cap_axis, [0, 0, 1], atol=1e-10), "R_mat axis check failed"
    assert np.allclose(R @ R.T, np.eye(3),       atol=1e-10), "R_mat orthonormal check failed"
    return R


# ─────────────────────────────────────────────────────────────
# GEODESIC MESH
# ─────────────────────────────────────────────────────────────

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = [np.array(v, dtype=float) for v in vertices]
        self.faces    = [list(f) for f in faces]

def _fix_winding(mesh):
    n = len(mesh.faces)
    if n == 0: return
    ue = defaultdict(set)
    for fi, face in enumerate(mesh.faces):
        nn = len(face)
        for i in range(nn):
            a, b = face[i], face[(i+1)%nn]
            ue[(min(a,b), max(a,b))].add(fi)
    vis = [False]*n; vis[0] = True; q = [0]
    while q:
        fi = q.pop(0); face = mesh.faces[fi]; nn = len(face)
        for i in range(nn):
            a, b = face[i], face[(i+1)%nn]
            for fj in ue[(min(a,b), max(a,b))]:
                if fj == fi or vis[fj]: continue
                vis[fj] = True
                fj_face = mesh.faces[fj]; nj = len(fj_face)
                if any(fj_face[k]==a and fj_face[(k+1)%nj]==b for k in range(nj)):
                    mesh.faces[fj] = fj_face[::-1]
                q.append(fj)

def _make_icosahedron():
    phi = (1+np.sqrt(5))/2
    v = [[-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
         [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
         [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1]]
    v = [np.array(x)/np.linalg.norm(x) for x in v]
    f = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
         [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
         [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
         [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    m = Mesh(v, f); _fix_winding(m); return m

def _order_faces_around_vertex(mesh, vi, fis):
    if len(fis) <= 1: return fis
    fn = {}; fp = {}
    for fi in fis:
        face = mesh.faces[fi]; idx = face.index(vi); n = len(face)
        fn[fi] = face[(idx+1)%n]; fp[fi] = face[(idx-1)%n]
    ordered = [fis[0]]; vis = {fis[0]}
    for _ in range(len(fis)-1):
        cur = ordered[-1]; tgt = fn[cur]
        for fj in fis:
            if fj not in vis and fp[fj] == tgt:
                ordered.append(fj); vis.add(fj); break
    return ordered

def _dual(mesh):
    nv = [np.mean([mesh.vertices[vi] for vi in face], axis=0) for face in mesh.faces]
    vfm = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for v in face: vfm[v].append(fi)
    nf = [_order_faces_around_vertex(mesh, vi, vfm[vi])
          for vi in range(len(mesh.vertices))]
    r = Mesh(nv, nf); _fix_winding(r); return r

def _project_sphere(mesh, r):
    for i, v in enumerate(mesh.vertices):
        n = np.linalg.norm(v)
        if n > 0: mesh.vertices[i] = r * v / n

def _subdivide(freq, radius):
    ico = _make_icosahedron()
    vm = {}; verts = []; faces = []
    def gv(pos):
        p = pos/np.linalg.norm(pos)*radius
        k = tuple(np.round(p, 10))
        if k not in vm: vm[k] = len(verts); verts.append(p)
        return vm[k]
    n = freq
    for face in ico.faces:
        v0,v1,v2 = [ico.vertices[face[i]] for i in range(3)]
        g = {}
        for i in range(n+1):
            for j in range(n-i+1):
                g[(i,j)] = gv((i/n)*v0+(j/n)*v1+((n-i-j)/n)*v2)
        for i in range(n):
            for j in range(n-i):
                faces.append([g[(i,j)], g[(i,j+1)], g[(i+1,j)]])
                if j+1 <= n-i-1:
                    faces.append([g[(i,j+1)], g[(i+1,j+1)], g[(i+1,j)]])
    m = Mesh(verts, faces); _fix_winding(m); return m

def build_geodesic_mesh(freq, radius):
    tri = _subdivide(freq, radius)
    hm  = _dual(tri)
    _project_sphere(hm, radius)
    return hm

def polygon_area_3d(verts):
    total = np.zeros(3)
    for i in range(len(verts)):
        total += np.cross(verts[i], verts[(i+1)%len(verts)])
    return 0.5 * np.linalg.norm(total)


# ─────────────────────────────────────────────────────────────
# STAGE 1 – FORWARD TRANSFORM
# ─────────────────────────────────────────────────────────────

def forward_transform(
    out_stl   = "/mnt/user-data/outputs/geodesic_cap_FLAT.stl",
    out_table = "vertex_table.json",
):
    """
    Vertex table schema
    -------------------
    {
      "R_outer"   : float,
      "R_inner"   : float,
      "thickness" : float,
      "vertices"  : [
          {
            "xy"    : [x, y],       // flat XY (same footprint for both surfaces)
            "outer" : [x, y, z],    // ROTATED-frame position on outer sphere (cap centre → XY=0,0)
            "inner" : [x, y, z]     // ROTATED-frame position on inner sphere
          }, ...
      ],
      "triangles" : [[i,j,k], ...]  // XY-space triangulation (fan of outer cap faces)
    }

    Reverse lookup for any flat (fx, fy, z_layer):
        t       = z_layer / thickness          # 0=inner, 1=outer
        p_outer = bary_interp(outer positions)
        p_inner = bary_interp(inner positions)
        p_world = lerp(p_inner, p_outer, t)
    """
    print("─── Stage 1: Forward Transform ─────────────────────────")
    t0 = time.time()

    print(f"  Building geodesic mesh (freq={FREQ}) …")
    mesh_o = build_geodesic_mesh(FREQ, R_OUTER)
    mesh_i = build_geodesic_mesh(FREQ, R_INNER)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ── Derive exact CAP_AXIS from mesh before selecting cap faces ──
    # Use the hex centroid closest to the approximate axis as the exact centre.
    # This guarantees the centre hex lands at XY=(0,0) after rotation.
    hex_cents_n = []
    for face in mesh_o.faces:
        if len(face) != 6: continue
        c = np.mean([mesh_o.vertices[vi] for vi in face], axis=0)
        hex_cents_n.append(c / np.linalg.norm(c))
    hex_cents_n = np.array(hex_cents_n)
    best_hi  = int(np.argmax(hex_cents_n @ _CAP_AXIS_APPROX))
    CAP_AXIS = hex_cents_n[best_hi]
    R_MAT    = _make_R_mat(CAP_AXIS)

    _centre_rot = R_MAT @ (CAP_AXIS * R_OUTER)
    print(f"  Exact CAP_AXIS: [{CAP_AXIS[0]:.8f}, {CAP_AXIS[1]:.8f}, {CAP_AXIS[2]:.8f}]")
    print(f"  Centre hex XY after rotation: [{_centre_rot[0]:.2e}, {_centre_rot[1]:.2e}]  (target: 0, 0)")

    # Identify cap faces using exact axis
    def cap_faces_of(mesh):
        out = []
        for face in mesh.faces:
            if len(face) not in (5, 6): continue
            c = np.mean([mesh.vertices[vi] for vi in face], axis=0)
            if (c / np.linalg.norm(c)) @ CAP_AXIS >= COS_CAP:
                out.append(face)
        return out

    cap_o = cap_faces_of(mesh_o)
    cap_i = cap_faces_of(mesh_i)
    print(f"  Cap faces: outer={len(cap_o)}, inner={len(cap_i)}")

    # ── Build unified XY vertex list ──────────────────────────
    # Coords stored are in the ROTATED frame (R_MAT applied), not world frame.
    # This means:
    #   • centre hex centroid → (0, 0, R_outer) on outer surface
    #   • surface is parallel to the XY plane
    #   • after Z-offset: inner surface min Z = 0, all Z values are ≥ 0
    # The LookupTable therefore returns print-bed frame coordinates directly.

    xy_list    = []      # [x_rot, y_rot] – flat XY (same as rotated XY)
    outer_list = []      # ROTATED-frame point on outer sphere
    inner_list = []      # ROTATED-frame point on inner sphere
    vert_map   = {}      # rounded(xy) → index

    def add_vertex_outer(v_world_o, v_world_i):
        """Register a matched outer/inner pair. Store rotated coords, key on XY."""
        v_rot_o = R_MAT @ v_world_o
        v_rot_i = R_MAT @ v_world_i
        key = (round(v_rot_o[0], 4), round(v_rot_o[1], 4))
        if key in vert_map:
            return vert_map[key]
        idx = len(xy_list)
        xy_list.append([v_rot_o[0], v_rot_o[1]])
        outer_list.append(v_rot_o.tolist())   # rotated frame
        inner_list.append(v_rot_i.tolist())   # rotated frame
        vert_map[key] = idx
        return idx

    # Match inner vertices to outer by unit direction
    inner_used = sorted({vi for face in cap_i for vi in face})
    inner_by_dir = {}
    for vi in inner_used:
        u = mesh_i.vertices[vi] / np.linalg.norm(mesh_i.vertices[vi])
        inner_by_dir[tuple(np.round(u, 5))] = mesh_i.vertices[vi]

    def get_inner_world(v_outer):
        u = v_outer / np.linalg.norm(v_outer)
        return inner_by_dir.get(tuple(np.round(u, 5)))

    # Build outer face index list (for STL + triangulation)
    outer_face_idx = []
    for face in cap_o:
        local = []
        for vi in face:
            v_o = mesh_o.vertices[vi]
            v_i = get_inner_world(v_o)
            if v_i is None:
                # shouldn't happen; fallback to radial projection
                v_i = v_o * (R_INNER / R_OUTER)
            local.append(add_vertex_outer(v_o, v_i))
        outer_face_idx.append(local)

    # Inner face index: same XY vertices, just need them for the STL (not the table)
    inner_face_idx = []
    for face in cap_i:
        local = []
        for vi in face:
            v_i   = mesh_i.vertices[vi]
            u     = v_i / np.linalg.norm(v_i)
            key_i = tuple(np.round(u, 5))
            # find corresponding outer vertex to get the shared table index
            # We can look up by the same rounded-XY key
            v_rot = R_MAT @ v_i
            key   = (round(v_rot[0], 4), round(v_rot[1], 4))
            idx   = vert_map.get(key)
            if idx is None:
                # inner-only vertex (boundary rounding mismatch) — add with outer=radial
                v_o_approx = v_i * (R_OUTER / R_INNER)
                idx = add_vertex_outer(v_o_approx, v_i)
            local.append(idx)
        inner_face_idx.append(local)

    N = len(xy_list)
    print(f"  Unified XY vertices: {N}")

    # Side walls: same as before, match outer↔inner by unit direction
    edge_cnt = defaultdict(int)
    dir_set  = set()
    for face in cap_o:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i+1)%n]
            edge_cnt[(min(a,b), max(a,b))] += 1
            dir_set.add((a, b))

    # For the flat STL we need separate outer/inner vertex positions
    # outer at z=THICKNESS, inner at z=0 — reconstruct from xy_list
    def flat_pos_outer(idx): return np.array([xy_list[idx][0], xy_list[idx][1], THICKNESS])
    def flat_pos_inner(idx): return np.array([xy_list[idx][0], xy_list[idx][1], 0.0])

    # Build a map from outer mesh vertex index → table index (for wall lookup)
    outer_vi_to_table = {}
    for fi, face in enumerate(cap_o):
        for k, vi in enumerate(face):
            outer_vi_to_table[vi] = outer_face_idx[fi][k]

    wall_tris = []
    for (u_e, v_e), cnt in edge_cnt.items():
        if cnt == 1:
            a, b   = (u_e, v_e) if (u_e, v_e) in dir_set else (v_e, u_e)
            ao     = outer_vi_to_table.get(a)
            bo     = outer_vi_to_table.get(b)
            if ao is None or bo is None: continue
            # ai/bi share same table index as ao/bo (same XY), wall uses both z levels
            wall_tris += [(ao, bo, bo), (ao, bo, ao)]  # placeholder — handled separately below

    print(f"  Wall edges: {len(wall_tris)//2}")

    # ── Triangulate and build flat STL ───────────────────────
    def fan(idx):
        return [(idx[0], idx[i], idx[i+1]) for i in range(1, len(idx)-1)]

    # Build a flat vertex array: first N entries = outer (z=H), next N = inner (z=0)
    flat_verts = (
        [np.array([xy_list[i][0], xy_list[i][1], THICKNESS], dtype=np.float32) for i in range(N)]
      + [np.array([xy_list[i][0], xy_list[i][1], 0.0],       dtype=np.float32) for i in range(N)]
    )
    # Outer faces use indices 0..N-1, inner faces use N..2N-1
    all_tris = []
    outer_tris_table = []
    for face in outer_face_idx:
        for t in fan(face):
            all_tris.append(t)
            outer_tris_table.append(list(t))
    for face in inner_face_idx:
        for t in fan(face):
            # shift to inner range and flip winding for inward normal
            all_tris.append((t[0]+N, t[2]+N, t[1]+N))

    # Real wall quads: outer edge a→b top, then down to inner
    wall_tris_stl = []
    for (u_e, v_e), cnt in edge_cnt.items():
        if cnt == 1:
            a, b = (u_e, v_e) if (u_e, v_e) in dir_set else (v_e, u_e)
            ao   = outer_vi_to_table.get(a)
            bo   = outer_vi_to_table.get(b)
            if ao is None or bo is None: continue
            # quad: ao_top, bo_top, bo_bot, ao_bot
            all_tris.append((ao,    bo,    bo+N))
            all_tris.append((ao,    bo+N,  ao+N))

    print(f"  Total STL triangles: {len(all_tris)}")

    # Write binary STL
    va = flat_verts

    def nrm(v0, v1, v2):
        n = np.cross(v1-v0, v2-v0).astype(np.float32)
        l = np.linalg.norm(n)
        return n/l if l > 1e-12 else n

    header = (b"Geodesic cap FLAT - rotation + orthographic flatten" + b'\0'*80)[:80]
    with open(out_stl, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', len(all_tris)))
        for tri in all_tris:
            v0,v1,v2 = va[tri[0]], va[tri[1]], va[tri[2]]
            f.write(nrm(v0,v1,v2).astype('<f4').tobytes())
            f.write(v0.astype('<f4').tobytes())
            f.write(v1.astype('<f4').tobytes())
            f.write(v2.astype('<f4').tobytes())
            f.write(b'\x00\x00')

    exp = 80 + 4 + len(all_tris)*50
    act = os.path.getsize(out_stl)
    assert exp == act, f"STL size mismatch {exp} vs {act}"
    print(f"  Flat STL → {out_stl}  ({act//1024} kB)  ✓")

    # Repair winding
    try:
        import trimesh
        m = trimesh.load_mesh(out_stl)
        trimesh.repair.fix_normals(m); trimesh.repair.fix_winding(m)
        if m.volume < 0: m.invert()
        m.export(out_stl)
        m2 = trimesh.load_mesh(out_stl)
        print(f"  Trimesh repair → watertight={m2.is_watertight}, "
              f"winding={m2.is_winding_consistent}, vol={m2.volume:.0f} mm³")
    except Exception as e:
        print(f"  (trimesh repair skipped: {e})")

    # ── Z-translation: shift world coords so min-Z = 0 ──────────
    # Find minimum Z across all outer and inner world positions
    all_z    = [outer_list[i][2] for i in range(N)] + [inner_list[i][2] for i in range(N)]
    z_offset = min(all_z)
    outer_list_t = [[v[0], v[1], v[2] - z_offset] for v in outer_list]
    inner_list_t = [[v[0], v[1], v[2] - z_offset] for v in inner_list]
    print(f"  Z-offset: {z_offset:.4f} mm subtracted  "
          f"(world Z range after translation: 0 … {max(all_z)-z_offset:.4f} mm)")

    # ── Write vertex table ────────────────────────────────────
    table = {
        "R_outer"   : R_OUTER,
        "R_inner"   : R_INNER,
        "thickness" : THICKNESS,
        "z_offset"  : z_offset,    # world coords = stored coords + z_offset
        "vertices"  : [
            {
                "xy"    : xy_list[i],
                "outer" : outer_list_t[i],
                "inner" : inner_list_t[i],
            }
            for i in range(N)
        ],
        "triangles" : outer_tris_table,
    }
    with open(out_table, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"  Vertex table → {out_table}  ({os.path.getsize(out_table)//1024} kB)  ✓")

    # ── Sanity check via LookupTable ─────────────────────────
    # Write a temp table, load it, verify outer/inner/mid lookups
    import json as _json, tempfile as _tmp
    _tmp_path = _tmp.mktemp(suffix=".json")
    _json.dump(table, open(_tmp_path, "w"))
    lut_check = LookupTable(_tmp_path)
    os.remove(_tmp_path)

    print(f"\n  LookupTable round-trip (centroid of first triangle):")
    i0, i1, i2 = outer_tris_table[0]
    cx_ = (xy_list[i0][0] + xy_list[i1][0] + xy_list[i2][0]) / 3
    cy_ = (xy_list[i0][1] + xy_list[i1][1] + xy_list[i2][1]) / 3
    ref_outer = np.array([(outer_list_t[i0][k]+outer_list_t[i1][k]+outer_list_t[i2][k])/3 for k in range(3)])
    ref_inner = np.array([(inner_list_t[i0][k]+inner_list_t[i1][k]+inner_list_t[i2][k])/3 for k in range(3)])
    got_outer = np.array(lut_check.lookup(cx_, cy_, THICKNESS))
    got_inner = np.array(lut_check.lookup(cx_, cy_, 0.0))
    got_mid   = np.array(lut_check.lookup(cx_, cy_, THICKNESS/2))
    ref_mid   = (ref_outer + ref_inner) / 2
    print(f"    outer error : {np.linalg.norm(got_outer-ref_outer):.2e} mm  "
          f"({'✓' if np.linalg.norm(got_outer-ref_outer)<1e-6 else '✗'})")
    print(f"    inner error : {np.linalg.norm(got_inner-ref_inner):.2e} mm  "
          f"({'✓' if np.linalg.norm(got_inner-ref_inner)<1e-6 else '✗'})")
    print(f"    mid error   : {np.linalg.norm(got_mid  -ref_mid  ):.2e} mm  "
          f"({'✓' if np.linalg.norm(got_mid  -ref_mid  )<1e-6 else '✗'})")
    print(f"    outer Z     : {got_outer[2]:.4f} mm  (outer > inner)")
    print(f"    inner Z     : {got_inner[2]:.4f} mm  (inner < outer; global min Z = 0)")
    outside = lut_check.lookup(99999., 99999., 5.)
    print(f"    outside→None: {'✓' if outside is None else '✗'}")

    # Area distortion
    flat_areas, orig_areas = [], []
    for face_idx, face in zip(outer_face_idx, cap_o):
        vs = [np.array(xy_list[i]) for i in face_idx]
        n = len(vs); a = 0.0
        for k in range(n):
            a += vs[k][0]*vs[(k+1)%n][1] - vs[(k+1)%n][0]*vs[k][1]
        flat_areas.append(abs(a)/2)
        orig_areas.append(polygon_area_3d([mesh_o.vertices[vi] for vi in face]))
    fa, oa = np.array(flat_areas), np.array(orig_areas)
    print(f"\n  Orthographic area distortion (outer surface):")
    print(f"    Orig avg: {oa.mean():.4f} mm²   Flat avg: {fa.mean():.4f} mm²")
    print(f"    Max err : {np.abs(fa/oa-1).max()*100:.3f}%   "
          f"Mean err: {np.abs(fa/oa-1).mean()*100:.3f}%")

    print(f"\n  Total time: {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────
# STAGE 2 – REVERSE TRANSFORM
# ─────────────────────────────────────────────────────────────

class LookupTable:
    """
    Loads the vertex table and provides layer-surface interpolation.

    For any flat point (fx, fy) and layer fraction t ∈ [0,1]:
        p_world = lerp( inner_surface(fx,fy), outer_surface(fx,fy), t )
        t = z_layer / thickness    (0 = inner/bottom, 1 = outer/top)

    Points outside the XY mesh return None (caller passes through unchanged).
    """
    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        self.thickness = data["thickness"]

        verts = data["vertices"]
        self.flat_xy    = np.array([v["xy"]    for v in verts])   # (N,2)
        self.outer_world = np.array([v["outer"] for v in verts])  # (N,3)
        self.inner_world = np.array([v["inner"] for v in verts])  # (N,3)

        self.tris = np.array(data["triangles"], dtype=int)         # (T,3)

        # Pre-compute edge vectors for all triangles (used in bary lookup)
        self.tv0  = self.flat_xy[self.tris[:,0]]   # (T,2)
        self.tv1  = self.flat_xy[self.tris[:,1]]
        self.tv2  = self.flat_xy[self.tris[:,2]]
        self.d00  = self.tv1 - self.tv0
        self.d01  = self.tv2 - self.tv0
        a = (self.d00 * self.d00).sum(1)
        b = (self.d00 * self.d01).sum(1)
        c = (self.d01 * self.d01).sum(1)
        self.denom = a*c - b*b                     # (T,)  precomputed denominator

        print(f"  LookupTable: {len(verts)} verts, {len(self.tris)} tris loaded")

    def _bary_weights(self, fx, fy):
        """
        Return (ti, w0, w1, w2) for the first triangle containing (fx, fy),
        or None if the point is outside all triangles.
        """
        p  = np.array([fx, fy])
        dp = p - self.tv0                          # (T,2)
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(np.abs(self.denom) > 1e-20,
                         ((self.d01*self.d01).sum(1)*(dp*self.d00).sum(1) -
                          (self.d00*self.d01).sum(1)*(dp*self.d01).sum(1)) / self.denom,
                         -1.)
            v = np.where(np.abs(self.denom) > 1e-20,
                         ((self.d00*self.d00).sum(1)*(dp*self.d01).sum(1) -
                          (self.d00*self.d01).sum(1)*(dp*self.d00).sum(1)) / self.denom,
                         -1.)
        w = 1. - u - v
        hits = np.where((u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9))[0]
        if len(hits) == 0:
            return None
        ti = hits[0]
        return ti, w[ti], u[ti], v[ti]

    def lookup(self, fx, fy, z_layer):
        """
        Map flat (fx, fy, z_layer) → world (x, y, z).
        Returns None if (fx, fy) is outside the mesh domain.

        z_layer is the flat printer Z (0 … THICKNESS).
        t = z_layer / thickness linearly interpolates between inner and outer
        curved surfaces, giving the correct interpolated surface for each layer.
        """
        result = self._bary_weights(fx, fy)
        if result is None:
            return None   # outside domain — caller will pass through unchanged

        ti, w0, w1, w2 = result
        i0, i1, i2 = self.tris[ti]

        # Interpolate world positions on outer and inner surfaces
        p_outer = (w0 * self.outer_world[i0]
                 + w1 * self.outer_world[i1]
                 + w2 * self.outer_world[i2])
        p_inner = (w0 * self.inner_world[i0]
                 + w1 * self.inner_world[i1]
                 + w2 * self.inner_world[i2])

        # Layer fraction: 0 = inner (bottom), 1 = outer (top)
        t = np.clip(z_layer / self.thickness, 0.0, 1.0)
        p_world = p_inner + t * (p_outer - p_inner)
        return tuple(p_world.tolist())


_COORD_RE = re.compile(r'([XYZEF])\s*(-?\d*\.?\d+(?:[eE][+\-]?\d+)?)', re.IGNORECASE)

def _parse(line):
    return {m.group(1).upper(): float(m.group(2)) for m in _COORD_RE.finditer(line)}

def _rebuild(line, upd):
    return _COORD_RE.sub(
        lambda m: f"{m.group(1).upper()}{upd[m.group(1).upper()]:.4f}"
                  if m.group(1).upper() in upd else m.group(0),
        line)


def reverse_transform_gcode(
    in_gcode   = "flat_cap.gcode",
    out_gcode  = "curved_cap.gcode",
    table_path = "vertex_table.json",
    seg_len    = 1.0,    # mm — max sub-segment length
    bed_size   = 225.0,  # mm — output XY shifted by +bed_size/2
    relative_e = True,   # True = Cura relative extrusion (M83); False = absolute
):
    """
    Process Cura G-code with 1 mm segmentation + layer-surface transform.

    For every G0/G1 move:
      1. Subdivide into steps of <= seg_len mm in flat space.
      2. Each sub-step independently queries the LookupTable at its
         (x, y, z_layer) → world position on the interpolated layer surface.
      3. Emit one G-code line per sub-step with transformed coords.
      4. For extrusion moves, distribute E proportionally to curved length
         of each sub-step so total extrusion matches the curved path.
      5. Moves entirely outside the mesh domain pass through unchanged.
    """
    print("\n─── Stage 2: Reverse Transform ─────────────────────────")
    if not os.path.exists(in_gcode):
        print(f"  ✗ Not found: {in_gcode}")
        print("  Slice geodesic_cap_FLAT.stl in Cura, save as that path, then re-run.")
        return

    lut = LookupTable(table_path)
    xy_offset = bed_size / 2.0
    print(f"  Bed size: {bed_size} mm  →  XY offset: +{xy_offset:.1f} mm")
    print(f"  Extrusion mode: {'relative (M83)' if relative_e else 'absolute (M82)'}")

    # Machine state (flat coords)
    cx = cy = cz = ce = 0.0
    current_layer_z = 0.0
    n_layers = 0

    # Statistics
    n_raw_moves  = 0
    n_sub_moves  = 0
    n_outside    = 0
    flat_total   = 0.0
    curved_total = 0.0
    escale_list  = []

    out_lines = []

    with open(in_gcode) as f:
        lines_in = f.readlines()

    try:
        from tqdm import tqdm
        _iter = tqdm(lines_in,
                     desc="  Transforming",
                     unit="lines",
                     bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
                     ncols=75)
    except ImportError:
        print(f"  Processing {len(lines_in)} G-code lines (seg_len={seg_len} mm)  [install tqdm for a progress bar] ...")
        _iter = lines_in

    for raw in _iter:
        line = raw.rstrip("\n")
        s    = line.strip()

        if not s or s.startswith(";"):
            out_lines.append(line + "\n")
            continue

        cmd = s.split()[0].upper()
        if cmd not in ("G0", "G1"):
            out_lines.append(line + "\n")
            continue

        coords = _parse(line)
        n_raw_moves += 1

        nx = coords.get("X", cx)
        ny = coords.get("Y", cy)
        nz = coords.get("Z", cz)
        # In relative mode E on the line is the delta for this move;
        # in absolute mode it's the target position and ce tracks current.
        ne = coords.get("E", 0.0 if relative_e else ce)

        if nz != current_layer_z:
            current_layer_z = nz
            n_layers += 1

        is_extrude = ("E" in coords) and (cmd == "G1")

        dx = nx - cx; dy = ny - cy; dz = nz - cz
        flat_len = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Zero-length move (Z hop only or redundant): single transformed point
        if flat_len < 1e-9:
            we = lut.lookup(nx, ny, nz)
            if we is not None:
                upd = {"X": we[0]+xy_offset, "Y": we[1]+xy_offset, "Z": we[2]}
                if "E" in coords:
                    upd["E"] = ne
                out_lines.append(_rebuild(line, upd) + "\n")
            else:
                # Outside domain but still need offset on any X/Y present
                upd_out = {}
                if "X" in coords: upd_out["X"] = nx + xy_offset
                if "Y" in coords: upd_out["Y"] = ny + xy_offset
                if "E" in coords: upd_out["E"] = ne
                out_lines.append(_rebuild(line, upd_out) + "\n" if upd_out else line + "\n")
            cx, cy, cz, ce = nx, ny, nz, ne
            n_sub_moves += 1
            continue

        # Subdivide into n_steps of <= seg_len each
        n_steps = max(1, int(np.ceil(flat_len / seg_len)))

        flat_pts  = [np.array([cx + (k/n_steps)*dx,
                                cy + (k/n_steps)*dy,
                                cz + (k/n_steps)*dz])
                     for k in range(n_steps + 1)]
        world_pts = [lut.lookup(p[0], p[1], p[2]) for p in flat_pts]

        # If entire move is outside domain, pass through with offset still applied
        if all(w is None for w in world_pts[1:]):
            n_outside += 1
            upd_out = {}
            if "X" in coords: upd_out["X"] = nx + xy_offset
            if "Y" in coords: upd_out["Y"] = ny + xy_offset
            if "E" in coords: upd_out["E"] = ne
            out_lines.append(_rebuild(line, upd_out) + "\n" if upd_out else line + "\n")
            cx, cy, cz, ce = nx, ny, nz, ne
            n_sub_moves += 1
            continue

        # Curved length of each sub-step (fall back to flat if either end outside)
        curved_lens = []
        for k in range(n_steps):
            ws_k = world_pts[k]
            we_k = world_pts[k+1]
            if ws_k is not None and we_k is not None:
                cl = float(np.sqrt(sum((a-b)**2 for a, b in zip(ws_k, we_k))))
            else:
                fp = flat_pts[k+1] - flat_pts[k]
                cl = float(np.linalg.norm(fp))
            curved_lens.append(cl)

        total_curved = sum(curved_lens)
        if is_extrude and flat_len > 1e-9:
            escale_list.append(total_curved / flat_len)
            flat_total   += flat_len
            curved_total += total_curved

        # Emit one G-code line per sub-step
        running_e = ce
        for k in range(n_steps):
            we_k = world_pts[k+1]
            fp1  = flat_pts[k+1]

            if we_k is not None:
                out_x, out_y, out_z = we_k[0], we_k[1], we_k[2]
            else:
                out_x, out_y, out_z = fp1[0], fp1[1], fp1[2]

            new_line = f"{cmd} X{out_x+xy_offset:.4f} Y{out_y+xy_offset:.4f} Z{out_z:.4f}"

            if is_extrude:
                # Fraction of original E delta for this sub-step, scaled by curved/flat ratio.
                # relative mode: ne is already the delta for the whole move.
                # absolute mode: (ne - ce) is the delta; running_e accumulates from ce.
                e_frac = curved_lens[k] / flat_len if flat_len > 1e-9 else 1.0 / n_steps
                if relative_e:
                    sub_e     = ne * e_frac
                    new_line += f" E{sub_e:.5f}"
                else:
                    running_e += (ne - ce) * e_frac
                    new_line  += f" E{running_e:.5f}"

            # Preserve feedrate F on first sub-step only
            if k == 0:
                fc = coords.get("F")
                if fc is not None:
                    new_line += f" F{fc:.0f}"

            out_lines.append(new_line + "\n")
            n_sub_moves += 1

        cx, cy, cz = nx, ny, nz
        # relative: ce unused (each line's E is self-contained)
        # absolute: track last emitted E so next move's delta is correct
        ce = 0.0 if relative_e else (running_e if is_extrude else ne)

    with open(out_gcode, "w") as f:
        f.writelines(out_lines)

    expansion = n_sub_moves / max(n_raw_moves, 1)
    print(f"  Output              : {out_gcode}")
    print(f"  Layers              : {n_layers}")
    print(f"  Raw G0/G1 moves     : {n_raw_moves}")
    print(f"  Emitted sub-moves   : {n_sub_moves}  (x{expansion:.1f} expansion)")
    print(f"  Outside domain      : {n_outside} moves passed through")
    if escale_list:
        ea = np.array(escale_list)
        print(f"  E-scale (per move)  : min={ea.min():.5f}  max={ea.max():.5f}  mean={ea.mean():.5f}")
    if flat_total > 0:
        print(f"  Length total        : flat={flat_total:.1f} mm  "
              f"curved={curved_total:.1f} mm  ratio={curved_total/flat_total:.5f}")



# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    p1 = sub.add_parser("flatten")
    p1.add_argument("--out-stl",   default="/mnt/user-data/outputs/geodesic_cap_FLAT.stl")
    p1.add_argument("--out-table", default="/mnt/user-data/outputs/vertex_table.json")
    p2 = sub.add_parser("reverse")
    p2.add_argument("in_gcode")
    p2.add_argument("--out-gcode",  default="curved_cap.gcode")
    p2.add_argument("--table",      default="/mnt/user-data/outputs/vertex_table.json")
    p2.add_argument("--bed-size",    default=220.0, type=float, help="Printer bed size in mm (default 225)")
    p2.add_argument("--seg-len",     default=1.0,   type=float, help="Max segment length in mm (default 1.0)")
    p2.add_argument("--absolute-e",  action="store_true", help="Use absolute extrusion (default: relative)")
    args = p.parse_args()

    if args.cmd == "flatten":
        forward_transform(out_stl=args.out_stl, out_table=args.out_table)
    elif args.cmd == "reverse":
        reverse_transform_gcode(args.in_gcode, args.out_gcode, args.table,
                                seg_len=args.seg_len, bed_size=args.bed_size,
                                relative_e=not args.absolute_e)
    else:
        print("Running Stage 1 by default. For Stage 2:")
        print("  python transform_pipeline.py reverse <flat.gcode>\n")
        forward_transform()
