"""
Tier 0 synthetic GT shapes for the Mask2PolyMin benchmark.

Step 0 (done): hand-authored unit-scale family definitions, reviewed one at a time via
`render_preview`. Step 1 (done): placement (scale/rotate/center), subpixel
rasterization, and the committed gt_shapes/ output — 0°-rotation PNG+JSON pairs per
(family, size) plus per-size gallery contact sheets. Regenerating the GT files is an
explicit action (`synth_shapes.py --write-gt`), never a side effect of a benchmark run.
Step 2 (this file): the distortion pipeline (`distort`), contour extraction, and the
`dataset()` generator — the single enumeration point for `run_benchmark.py`.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

PREVIEW_DIR = Path(__file__).parent / "shape_review"
GT_DIR = Path(__file__).parent / "gt_shapes"

MARGIN_PX = 24            # per side; distortion (step 2) must never touch the canvas border
SIZES = (48, 128, 320)    # circumscribed diameter axis; the small slot has per-family fallbacks
SMALL_SIZE_OVERRIDE = {"car": 64, "plane": 64}  # review outcomes (corner-spacing constraint): car per step 0, plane per step 1 via its coarse d64 variant
ROTATIONS_DEG = (0.0, 10.0, 22.5, 37.0, 45.0)
MIN_CORNER_SPACING_PX = 4.5  # corner_metrics matches nearest-neighbour with tau = 2 px, so GT corners must stay >= 2*tau apart
RASTER_SHIFT = 8          # fractional bits for fixed-point cv2.fillPoly

# family-specific design constants echoed into the JSON "params" field
FAMILY_PARAMS = {"rect": {"rect_aspect": 1.8}, "star": {"star_inner_ratio": 0.45}}

# segmentation-noise levels (order of application: elastic jitter -> blur + re-threshold
# -> boundary speckle). Levels 2 and 4 are the noise-review-gate anchors (formerly levels
# 1 and 2); levels 1 and 3 interpolate between the reviewed settings.
NOISE_LEVELS = {
    0: {"blur_sigma": 0.0, "jitter_amp": 0.0, "speckle_p": 0.0},    # clean rasterization
    1: {"blur_sigma": 0.5, "jitter_amp": 0.5, "speckle_p": 0.0},    # good segmentation net
    2: {"blur_sigma": 1.0, "jitter_amp": 1.0, "speckle_p": 0.0},    # decent segmentation net
    3: {"blur_sigma": 1.5, "jitter_amp": 1.75, "speckle_p": 0.03},  # mediocre segmentation net
    4: {"blur_sigma": 2.0, "jitter_amp": 2.5, "speckle_p": 0.06},   # sloppy segmentation net
}
JITTER_CORR_LEN_PX = 8.0  # Gaussian smoothing sigma of the displacement field: wobble, not salt-and-pepper
SPECKLE_BAND_PX = 2       # speckle flips are confined to this band around the boundary


def _finalize(raw) -> np.ndarray:
    """Center a hand-authored polygon by its bounding box and scale so the
    circumscribed radius (max vertex distance from center) is exactly 1.
    """
    verts = np.asarray(raw, dtype=np.float64)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts = verts - center
    radius = np.linalg.norm(verts, axis=1).max()
    return verts / radius


def _min_spacing(verts: np.ndarray) -> float:
    """Smallest pairwise distance between any two vertices."""
    dmat = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    return float(dmat.min())


def _reflex_mask(verts: np.ndarray) -> np.ndarray:
    """True at each vertex whose interior turn is reflex (interior angle > 180 deg)."""
    nxt = np.roll(verts, -1, axis=0)
    prev = np.roll(verts, 1, axis=0)
    signed_area = 0.5 * np.sum(verts[:, 0] * nxt[:, 1] - nxt[:, 0] * verts[:, 1])
    orientation = np.sign(signed_area)
    edge_in = verts - prev
    edge_out = nxt - verts
    cross = edge_in[:, 0] * edge_out[:, 1] - edge_in[:, 1] * edge_out[:, 0]
    return np.sign(cross) != orientation


def _rect() -> np.ndarray:
    hw, hh = 1.8, 1.0  # elongation 1.8:1, not a square
    return _finalize([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])


def _lshape() -> np.ndarray:
    w, h = 0.66, 1.0    # narrower than tall: letter-like proportions, not a square
    t = 0.32            # uniform stroke thickness (was 0.4 on the old unit square; 4/5 as thick)
    raw = [
        (0, 0),
        (t, 0),
        (t, h - t),
        (w, h - t),
        (w, h),
        (0, h),
    ]
    return _finalize(raw)


def _hexagon() -> np.ndarray:
    angles = np.deg2rad([-90, -30, 30, 90, 150, 210])  # pointy top/bottom, flat left/right sides
    raw = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return _finalize(raw)


def _star() -> np.ndarray:
    outer_r, inner_r = 1.0, 0.45  # inner/outer ratio 0.45 per the catalog
    k = np.arange(5)
    outer_ang = np.deg2rad(-90 + 72 * k)   # 5 outer tips, first pointing straight up
    inner_ang = outer_ang + np.deg2rad(36)  # 5 inner (reflex) notches, interleaved
    raw = np.empty((10, 2))
    raw[0::2, 0] = outer_r * np.cos(outer_ang)
    raw[0::2, 1] = outer_r * np.sin(outer_ang)
    raw[1::2, 0] = inner_r * np.cos(inner_ang)
    raw[1::2, 1] = inner_r * np.sin(inner_ang)
    return _finalize(raw)


def _tab() -> np.ndarray:
    # union of two rectangles, same proportions as examples/simple_bitmap_example.py:
    # body 40x60, a 10x20 bump on the right edge, offset 10 from the top.
    raw = [
        (0, 0),
        (40, 0),
        (40, 10),
        (50, 10),
        (50, 30),
        (40, 30),
        (40, 60),
        (0, 60),
    ]
    return _finalize(raw)


def _plane() -> np.ndarray:
    # top-view airliner silhouette, nose up. Traced from an icon (shape_review/plane.jpg):
    # thresholded, contour via cv2.findContours, fitted with Mask2PolyMin at tolerance 1.0,
    # rotated nose-up, mirror-symmetrized, unit-normalized; then simplified by review
    # decision: engine pods removed (straight wing leading edges) and the nose flat
    # collapsed to a single sharp apex. Min vertex spacing ~0.10 unit (tailplane chord),
    # so this family's smallest legal size is still d128, not d48.
    raw = [
        (-0.1344, -0.2943),  # wing leading root, left
        (-0.9529, 0.1678),   # wingtip leading
        (-0.9550, 0.2965),   # wingtip trailing
        (-0.1302, 0.0754),   # wing trailing root (reflex)
        (-0.0909, 0.6571),   # aft fuselage, left
        (-0.3647, 0.8140),   # tailplane leading tip
        (-0.3669, 0.9135),   # tailplane trailing tip
        (0.0000, 0.8162),    # tail notch (reflex)
        (0.3669, 0.9135),
        (0.3647, 0.8140),
        (0.0909, 0.6571),
        (0.1302, 0.0754),
        (0.9550, 0.2965),
        (0.9529, 0.1678),
        (0.1344, -0.2943),   # wing leading root, right
        (0.1359, -0.7526),   # nose shoulder, right
        (0.0000, -0.9337),   # nose apex — sharp
        (-0.1359, -0.7526),  # nose shoulder, left
    ]
    return _finalize(raw)


def _plane64() -> np.ndarray:
    # coarsened variant of _plane for the d64 slot only (step-1 review decision): the traced
    # geometry's fine detail cannot meet the 4.5 px corner floor below d128. Same silhouette,
    # but the tailplane and wingtip chords are widened so every vertex pair is >= 0.145 unit
    # apart (4.64 px at d64).
    raw = [
        (-0.134, -0.294),  # wing leading root, left
        (-0.953, 0.155),   # wingtip leading
        (-0.955, 0.305),   # wingtip trailing
        (-0.130, 0.075),   # wing trailing root (reflex)
        (-0.091, 0.657),   # aft fuselage, left
        (-0.360, 0.790),   # tailplane leading tip
        (-0.380, 0.935),   # tailplane trailing tip
        (0.000, 0.816),    # tail notch (reflex)
        (0.380, 0.935),
        (0.360, 0.790),
        (0.091, 0.657),
        (0.130, 0.075),
        (0.955, 0.305),
        (0.953, 0.155),
        (0.134, -0.294),   # wing leading root, right
        (0.136, -0.753),   # nose shoulder, right
        (0.000, -0.935),   # nose apex — sharp
        (-0.136, -0.753),  # nose shoulder, left
    ]
    return _finalize(raw)


def _house() -> np.ndarray:
    # pentagon body (gable roof, no overhang) with a centered door notch in the bottom edge
    half_w = 0.5          # wall half-width
    apex_y, eave_y, bottom_y = -0.55, 0.0, 0.7
    door_hw, door_top = 0.12, 0.4  # door half-width, door lintel height
    raw = [
        (0.0, apex_y),
        (half_w, eave_y),
        (half_w, bottom_y),
        (door_hw, bottom_y),
        (door_hw, door_top),   # reflex: door lintel, right
        (-door_hw, door_top),  # reflex: door lintel, left
        (-door_hw, bottom_y),
        (-half_w, bottom_y),
        (-half_w, eave_y),
    ]
    return _finalize(raw)


def _ship() -> np.ndarray:
    # side-view ship, bow to the right. Nothing axis-aligned except the cabin roofs:
    # raked stern, deck sheer rising toward the bow, trapezoid (leaning) cabin walls.
    raw = [
        (-0.90, -0.03),  # stern deck corner (raked stern)
        (-0.62, -0.05),  # lower cabin base, left (reflex)
        (-0.55, -0.32),  # lower cabin roof, left (wall leans inward)
        (-0.37, -0.32),  # step to upper cabin (reflex)
        (-0.32, -0.56),  # upper cabin roof, left
        (-0.10, -0.56),  # upper cabin roof, right
        (-0.05, -0.32),  # step down (reflex)
        (0.13, -0.32),   # lower cabin roof, right
        (0.20, -0.07),   # lower cabin base, right (reflex)
        (0.90, -0.14),   # bow tip
        (0.58, 0.35),    # hull bottom, bow side (raked)
        (-0.74, 0.35),   # hull bottom, stern side
    ]
    return _finalize(raw)


def _arrow() -> np.ndarray:
    # map direction-arrow pointing up: triangular head (~57 deg tip) on a straight
    # shaft, two reflex barbs where the head base meets the shaft.
    raw = [
        (0.00, -1.00),   # tip
        (0.38, -0.30),   # head base, right
        (0.16, -0.30),   # barb, right (reflex)
        (0.16, 0.95),    # shaft end, right
        (-0.16, 0.95),   # shaft end, left
        (-0.16, -0.30),  # barb, left (reflex)
        (-0.38, -0.30),  # head base, left
    ]
    return _finalize(raw)


def _car() -> np.ndarray:
    # low sedan profile facing right (review decision: was a rear-view silhouette with
    # mirrors). A silhouette cannot express a true 3/4 view, so the profile with a
    # slanted nose face is the closest realization. Shallow deck/roof bends, obtuse
    # roofline corners, two wheel bumps protruding below the rocker line — small
    # sub-features that tempt the fitter to drop them. Min spacing ~0.175 unit
    # (wheel-arch chords), so this family keeps the d64 fallback slot.
    raw = [
        (-0.88, -0.08),  # trunk deck, rear
        (-0.52, -0.14),  # rear-window base (shallow bend)
        (-0.25, -0.42),  # roof, rear
        (0.15, -0.42),   # roof, front
        (0.48, -0.10),   # windshield base / hood rear
        (0.88, -0.04),   # hood front
        (1.00, 0.20),    # nose, bumper top (slanted front face)
        (0.95, 0.50),    # front bumper, bottom
        (0.73, 0.50),    # front wheel arch, front root (reflex)
        (0.65, 0.66),    # front wheel, front bottom
        (0.45, 0.66),    # front wheel, rear bottom
        (0.37, 0.50),    # front wheel arch, rear root (reflex)
        (-0.37, 0.50),   # rear wheel arch, front root (reflex)
        (-0.45, 0.66),   # rear wheel, front bottom
        (-0.65, 0.66),   # rear wheel, rear bottom
        (-0.73, 0.50),   # rear wheel arch, rear root (reflex)
        (-0.95, 0.50),   # rear bumper, bottom
        (-1.00, 0.15),   # rear bumper, top
    ]
    return _finalize(raw)


FAMILIES = {
    "rect": _rect,
    "lshape": _lshape,
    "hexagon": _hexagon,
    "star": _star,
    "tab": _tab,
    "plane": _plane,
    "house": _house,
    "ship": _ship,
    "arrow": _arrow,
    "car": _car,
}


# size-specific design variants, looked up by gt_polygon before the family default:
# a variant fills a size slot the family's main geometry cannot legally occupy
SIZE_VARIANTS = {("plane", 64): _plane64}


def family_sizes(family: str) -> tuple[int, ...]:
    """Circumscribed diameters this family is generated at; the small slot honours the
    per-family fallbacks (car and plane: 64 instead of 48)."""
    small = SMALL_SIZE_OVERRIDE.get(family, SIZES[0])
    return tuple(sorted({small, *SIZES[1:]}))


def canvas_center(canvas_px: int) -> float:
    """Canvas centre in pixel-center coordinates: the image domain is [-0.5, N - 0.5],
    so its midpoint is (N - 1) / 2 — a half-integer on our even canvases, which keeps
    mirror-symmetric shapes symmetric on the pixel grid."""
    return (canvas_px - 1) / 2.0


def gt_polygon(family: str, size_px: int) -> tuple[np.ndarray, int]:
    """Place a unit-scale family at 0° rotation: scale to circumscribed diameter size_px
    and center on the square canvas of size_px + 2 * MARGIN_PX. Returns the closed
    polygon (first == last) in float canvas pixel coords, and the canvas side length.
    Asserts the corner-spacing constraint that corner_metrics relies on."""
    verts = SIZE_VARIANTS.get((family, size_px), FAMILIES[family])()
    canvas = size_px + 2 * MARGIN_PX
    corners = verts * (size_px / 2.0) + canvas_center(canvas)
    spacing = _min_spacing(corners)
    assert spacing >= MIN_CORNER_SPACING_PX, (
        f"{family} at d{size_px}: corner spacing {spacing:.2f} px < {MIN_CORNER_SPACING_PX} px"
    )
    return np.vstack([corners, corners[:1]]), canvas


def rotate_polygon(poly_xy: np.ndarray, angle_deg: float, center_xy) -> np.ndarray:
    """Rotate float polygon coordinates about a point. Rotation is always applied to the
    polygon before rasterization — never to a rendered mask, which would corrupt the GT
    with resampling artifacts."""
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(poly_xy) - center_xy) @ rot.T + center_xy


def rasterize(poly_xy: np.ndarray, canvas_hw: tuple[int, int]) -> np.ndarray:
    """Subpixel rasterization of a closed polygon: fixed-point cv2.fillPoly, {0, 1} uint8."""
    mask = np.zeros(canvas_hw, dtype=np.uint8)
    pts = np.round(np.asarray(poly_xy) * (1 << RASTER_SHIFT)).astype(np.int32)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1, cv2.LINE_8, RASTER_SHIFT)
    return mask


def _json_pairs(pts: np.ndarray) -> str:
    """Coordinate list serialized one [x, y] pair per line, floats at full precision
    (shortest round-trip repr), so regenerating unchanged shapes is byte-identical."""
    lines = ",\n".join(f"    [{json.dumps(float(x))}, {json.dumps(float(y))}]" for x, y in pts)
    return "[\n" + lines + "\n  ]"


def _write_pair(folder: Path, family: str, size_px: int) -> None:
    """Write one canonical GT pair: the JSON description (the pipeline's source of truth)
    and the 0/255 PNG mask derived from it for in-repo review."""
    polygon, canvas = gt_polygon(family, size_px)
    shape_id = f"{family}_d{size_px:03d}_a0"
    cv2.imwrite(str(folder / f"{shape_id}.png"), rasterize(polygon, (canvas, canvas)) * 255)
    params = {"margin_px": MARGIN_PX, **FAMILY_PARAMS.get(family, {})}
    if (family, size_px) in SIZE_VARIANTS:
        params["variant"] = "coarse"
    text = (
        "{\n"
        f'  "shape_id": {json.dumps(shape_id)},\n'
        f'  "family": {json.dumps(family)},\n'
        f'  "size_px": {size_px},\n'
        '  "angle_deg": 0.0,\n'
        f'  "canvas_hw": [{canvas}, {canvas}],\n'
        f'  "polygon_xy": {_json_pairs(polygon)},\n'
        f'  "corners_xy": {_json_pairs(polygon[:-1])},\n'
        f'  "params": {json.dumps(params)}\n'
        "}\n"
    )
    (folder / f"{shape_id}.json").write_text(text)
    print(f"  {shape_id}: canvas {canvas}x{canvas}, "
          f"min corner spacing {_min_spacing(polygon[:-1]):.2f} px")


def _gallery(path: Path, rows: list[tuple[str, str, int]], title: str) -> None:
    """Contact sheet: one row per (label, family, size_px), one column per rotation
    angle; each cell shows the rasterized mask with GT polygon and corner dots overlaid."""
    n_r, n_c = len(rows), len(ROTATIONS_DEG)
    fig, axes = plt.subplots(n_r, n_c, figsize=(1.9 * n_c, 1.9 * n_r), squeeze=False)
    for i, (label, family, size_px) in enumerate(rows):
        base, canvas = gt_polygon(family, size_px)
        c = canvas_center(canvas)
        for j, angle in enumerate(ROTATIONS_DEG):
            ax = axes[i][j]
            poly = rotate_polygon(base, angle, (c, c))
            ax.imshow(rasterize(poly, (canvas, canvas)), cmap="gray", vmin=0, vmax=1)
            ax.plot(poly[:, 0], poly[:, 1], "r-", linewidth=0.7)
            ax.scatter(poly[:-1, 0], poly[:-1, 1], c="tab:blue", s=6, zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{angle:g}°", fontsize=9)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  {path.name}: {n_r} families x {n_c} angles")


def write_gt(out_dir: Path = GT_DIR) -> None:
    """Write the committed canonical GT files — one PNG+JSON pair per (family, size) at 0°
    rotation in per-size folders — plus, into shape_review/, one gallery contact sheet per
    size slot on which every (family, size, angle) instance appears exactly once."""
    n_pairs = 0
    for family in FAMILIES:
        for size_px in family_sizes(family):
            folder = out_dir / f"d{size_px:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            _write_pair(folder, family, size_px)
            n_pairs += 1

    # one contact sheet per size folder, listing exactly the families stored in it,
    # so every (family, size, angle) instance appears on exactly one sheet
    titles = {
        48: "families passing at d048 (car and plane fall back to d064)",
        64: "the d064 fallback families (plane: coarse variant)",
    }
    all_sizes = sorted({s for f in FAMILIES for s in family_sizes(f)})
    PREVIEW_DIR.mkdir(exist_ok=True)
    n_sheets = n_cells = 0
    for size_px in all_sizes:
        rows = [(f"{f} d{size_px:03d}", f, size_px)
                for f in FAMILIES if size_px in family_sizes(f)]
        _gallery(PREVIEW_DIR / f"gallery_d{size_px:03d}.png", rows,
                 titles.get(size_px, f"all families at d{size_px}"))
        n_sheets += 1
        n_cells += len(rows) * len(ROTATIONS_DEG)
    print(f"{n_pairs} PNG+JSON pairs -> {out_dir}\n"
          f"{n_sheets} galleries ({n_cells} instances) -> {PREVIEW_DIR}")


def load_gt(gt_dir: Path = GT_DIR) -> list[dict]:
    """Load the canonical GT JSONs in stable enumeration order (FAMILIES order, sizes
    ascending) — the order the per-record seeds are derived from."""
    records = []
    for family in FAMILIES:
        for size_px in family_sizes(family):
            path = gt_dir / f"d{size_px:03d}" / f"{family}_d{size_px:03d}_a0.json"
            rec = json.loads(path.read_text())
            rec["polygon_xy"] = np.asarray(rec["polygon_xy"])
            rec["corners_xy"] = np.asarray(rec["corners_xy"])
            records.append(rec)
    return records


def distort(mask: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate segmentation-net output from a clean {0, 1} mask: elastic jitter ->
    Gaussian blur + re-threshold at 0.5 -> boundary speckle, per NOISE_LEVELS[level].
    Deterministic given (mask, level, rng state)."""
    p = NOISE_LEVELS[level]
    if level == 0:
        return mask.copy()
    h, w = mask.shape
    field = mask.astype(np.float32)
    if p["jitter_amp"] > 0:
        # two independent smooth displacement components: Gaussian-smoothed white noise,
        # rescaled so each component's per-pixel std is exactly jitter_amp
        disp = []
        for _ in range(2):
            white = rng.standard_normal((h, w)).astype(np.float32)
            smooth = cv2.GaussianBlur(white, (0, 0), JITTER_CORR_LEN_PX)
            disp.append(smooth * (p["jitter_amp"] / max(float(smooth.std()), 1e-9)))
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        field = cv2.remap(field, gx + disp[0], gy + disp[1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if p["blur_sigma"] > 0:
        field = cv2.GaussianBlur(field, (0, 0), p["blur_sigma"])
    out = (field >= 0.5).astype(np.uint8)
    if p["speckle_p"] > 0:
        r = SPECKLE_BAND_PX
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        band = cv2.dilate(out, kernel) - cv2.erode(out, kernel)
        flips = (band > 0) & (rng.random((h, w)) < p["speckle_p"])
        out ^= flips.astype(np.uint8)
        k3 = np.ones((3, 3), np.uint8)  # close/open makes the flipped pixels blobby, not salt-and-pepper
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k3)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)
    return out


def extract_contour(mask: np.ndarray) -> np.ndarray | None:
    """Benchmark input contour of a distorted mask: largest external contour by area
    (cv2.findContours, CHAIN_APPROX_NONE — integer pixel coordinates, exactly what a real
    user of the library feeds the fitter), closed float (N, 2). None if the mask is empty."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return np.vstack([pts, pts[:1]])


def _record_rng(shape_idx: int, angle_idx: int, level: int, rep: int) -> np.random.Generator:
    """Per-record RNG seeded from the record's position on the dataset axes,
    so any benchmark row can be regenerated in isolation."""
    return np.random.default_rng(np.random.SeedSequence([shape_idx, angle_idx, level, rep]))


def dataset(reps: int = 3, gt_dir: Path = GT_DIR):
    """Yield one benchmark record per (shape, angle, noise level, rep) — the single
    enumeration point for run_benchmark.py. Level 0 is deterministic and gets one rep;
    default volume is 30 x 5 x (1 + 4 x reps) = 1950 records.

    Record fields:
      contour_id, family, size_px, angle_deg, noise_level, rep — identity/aggregation axes
      gt_polygon_xy — rotated GT polygon, closed float (M, 2): reference for fidelity metrics
      gt_corners_xy — rotated GT corners (K, 2): reference for corner_metrics
      gt_mask       — clean {0, 1} rasterization: reference for iou_rasterized
      contour_xy    — extracted contour of the distorted mask, closed float (N, 2):
                      the input both algorithms are run on
    """
    for shape_idx, gt in enumerate(load_gt(gt_dir)):
        canvas_hw = tuple(gt["canvas_hw"])
        c = canvas_center(canvas_hw[0])
        for angle_idx, angle_deg in enumerate(ROTATIONS_DEG):
            poly = rotate_polygon(gt["polygon_xy"], angle_deg, (c, c))
            gt_mask = rasterize(poly, canvas_hw)
            for level in sorted(NOISE_LEVELS):
                for rep in range(1 if level == 0 else reps):
                    rng = _record_rng(shape_idx, angle_idx, level, rep)
                    contour = extract_contour(distort(gt_mask, level, rng))
                    if contour is None:  # distortion annihilated the mask; benchmark logs the gap
                        print(f"WARNING: empty distorted mask, skipping "
                              f"{gt['shape_id']} a{angle_deg:g} n{level} r{rep}")
                        continue
                    yield {
                        "contour_id": f"{gt['family']}_d{gt['size_px']:03d}"
                                      f"_a{angle_deg:g}_n{level}_r{rep}",
                        "family": gt["family"],
                        "size_px": gt["size_px"],
                        "angle_deg": angle_deg,
                        "noise_level": level,
                        "rep": rep,
                        "canvas_hw": canvas_hw,
                        "gt_polygon_xy": poly,
                        "gt_corners_xy": poly[:-1],
                        "gt_mask": gt_mask,
                        "contour_xy": contour,
                    }


def render_noise_preview(size_px: int = 128, angle_deg: float = 22.5) -> Path:
    """One row per family at (size_px, angle_deg) x one column per noise level: the
    distorted mask with the GT polygon (red) and extracted contour (cyan) overlaid.
    Cells use the true dataset seeds (rep 0), so they show exactly what the benchmark
    consumes for those records."""
    angle_idx = ROTATIONS_DEG.index(angle_deg)
    rows = [(i, gt) for i, gt in enumerate(load_gt()) if gt["size_px"] == size_px]
    levels = sorted(NOISE_LEVELS)
    fig, axes = plt.subplots(len(rows), len(levels),
                             figsize=(2.2 * len(levels), 2.2 * len(rows)), squeeze=False)
    for r, (shape_idx, gt) in enumerate(rows):
        canvas_hw = tuple(gt["canvas_hw"])
        c = canvas_center(canvas_hw[0])
        poly = rotate_polygon(gt["polygon_xy"], angle_deg, (c, c))
        gt_mask = rasterize(poly, canvas_hw)
        for j, level in enumerate(levels):
            ax = axes[r][j]
            noisy = distort(gt_mask, level, _record_rng(shape_idx, angle_idx, level, 0))
            contour = extract_contour(noisy)
            ax.imshow(noisy, cmap="gray", vmin=0, vmax=1)
            ax.plot(poly[:, 0], poly[:, 1], "r-", linewidth=0.8)
            if contour is not None:
                ax.plot(contour[:, 0], contour[:, 1], "-", color="deepskyblue", linewidth=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"level {level}", fontsize=10)
            if j == 0:
                ax.set_ylabel(gt["family"], fontsize=9)
    fig.suptitle(f"noise levels at d{size_px:03d}, {angle_deg:g}° "
                 f"(GT red, extracted contour cyan; rep 0, true dataset seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    PREVIEW_DIR.mkdir(exist_ok=True)
    out = PREVIEW_DIR / "preview_noise.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def render_preview(family: str) -> Path:
    """Rasterize a unit-scale family at a fixed review canvas: filled mask,
    GT outline, and vertex dots (reflex corners marked separately).
    """
    verts = FAMILIES[family]()
    reflex = _reflex_mask(verts)

    canvas_r, margin = 250, 50
    size = 2 * (canvas_r + margin)
    c = size / 2.0
    px = verts * canvas_r + c
    closed_px = np.vstack([px, px[:1]])

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], 1)

    min_spacing = _min_spacing(verts)
    bbox_unit = verts.max(axis=0) - verts.min(axis=0)
    bbox_px = bbox_unit * canvas_r

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, cmap="gray", alpha=0.35)
    ax.plot(closed_px[:, 0], closed_px[:, 1], "r-", linewidth=1.5)
    ax.scatter(px[~reflex, 0], px[~reflex, 1], c="blue", s=40, zorder=3, label="corner")
    if reflex.any():
        ax.scatter(px[reflex, 0], px[reflex, 1], c="orange", s=70, marker="s", zorder=3, label="reflex")
    ax.set_title(f"{family}  ({len(verts)} corners, {int(reflex.sum())} reflex)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    PREVIEW_DIR.mkdir(exist_ok=True)
    out_path = PREVIEW_DIR / f"{family}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"{family}: {len(verts)} corners, {int(reflex.sum())} reflex, "
          f"min unit spacing {min_spacing:.3f} (need >= 0.19 @48px)")
    print(f"  bounding box: {bbox_unit[0]:.3f} x {bbox_unit[1]:.3f} unit "
          f"= {bbox_px[0]:.0f} x {bbox_px[1]:.0f} px on this {size}x{size} review canvas")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier 0 synthetic GT shape generator")
    parser.add_argument("--write-gt", action="store_true",
                        help="write the canonical gt_shapes/ files (PNG+JSON pairs + galleries)")
    parser.add_argument("--preview", nargs="*", metavar="FAMILY",
                        help="render shape_review/ previews (all families if none given)")
    parser.add_argument("--preview-noise", action="store_true",
                        help="render shape_review/preview_noise.png (families x noise levels)")
    args = parser.parse_args()
    if args.write_gt:
        write_gt()
    if args.preview is not None:
        for name in args.preview or list(FAMILIES):
            render_preview(name)
    if args.preview_noise:
        render_noise_preview()
    if not args.write_gt and args.preview is None and not args.preview_noise:
        parser.print_help()
