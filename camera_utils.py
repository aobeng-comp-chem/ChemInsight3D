"""Camera-state helpers shared by the viewer, its image exporters and session files.

Camera-state contract
---------------------
A camera state is a plain dict:

    position, focal, viewup : 3 floats each, in world units
    parallel_scale          : float > 0, half the visible height under parallel
                              projection, i.e. the apparent zoom. None only in
                              a state read from an older session file.
    parallel_projection     : bool

Distance is deliberately absent: it is |position - focal|, so storing it
separately could only contradict those two. Under parallel projection the
distance does not affect apparent size at all; only parallel_scale does.

apply_camera_state() restores in a fixed order: projection mode, focal point,
position, view-up, then parallel scale. The caller then resets the clipping
range (vtkRenderer.ResetCameraClippingRange), which only moves the near/far
planes; it never refits the view or changes the scale.

Session files store the same keys, plus 'distance' so that v1.0.0, which
requires that key, can still open sessions written by later versions.
"""

from __future__ import annotations

import math

import numpy as np

_VECTOR_KEYS = ("position", "focal", "viewup")


def capture_camera_state(cam):
    """Snapshot a vtkCamera as a camera-state dict."""
    return {
        "position": [float(v) for v in cam.GetPosition()],
        "focal": [float(v) for v in cam.GetFocalPoint()],
        "viewup": [float(v) for v in cam.GetViewUp()],
        "parallel_scale": float(cam.GetParallelScale()),
        "parallel_projection": bool(cam.GetParallelProjection()),
    }


def apply_camera_state(cam, state):
    """
    Restore a camera-state dict onto a vtkCamera, in the contract's order.

    A state whose parallel_scale is None (older session files) leaves the
    camera's current scale untouched; the caller decides how to set it.
    """
    cam.SetParallelProjection(bool(state.get("parallel_projection", True)))
    cam.SetFocalPoint(*state["focal"])
    cam.SetPosition(*state["position"])
    cam.SetViewUp(*state["viewup"])
    if state.get("parallel_scale") is not None:
        cam.SetParallelScale(state["parallel_scale"])


def camera_state_to_session(state):
    """Camera-state dict -> JSON-ready session block (adds the derived distance)."""
    block = {key: list(state[key]) for key in _VECTOR_KEYS}
    block["distance"] = float(np.linalg.norm(
        np.subtract(state["position"], state["focal"])))
    block["parallel_scale"] = state["parallel_scale"]
    block["parallel_projection"] = state["parallel_projection"]
    return block


def _finite_vector(value):
    try:
        vec = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        return None
    return vec


def camera_state_from_session(block):
    """
    Validate a session file's camera block and return a camera-state dict,
    or None when the block is missing or unusable.

    Older sessions (v1.0.0) have no parallel_scale or parallel_projection.
    Those come back with parallel_scale=None and parallel projection on (the
    only mode the viewer has used); the stored 'distance' is ignored because
    position and focal already define it.
    """
    if not isinstance(block, dict):
        return None
    vectors = {key: _finite_vector(block.get(key)) for key in _VECTOR_KEYS}
    if any(v is None for v in vectors.values()):
        return None
    direction = vectors["position"] - vectors["focal"]
    if np.linalg.norm(direction) < 1e-9:
        return None
    if np.linalg.norm(np.cross(direction, vectors["viewup"])) < 1e-9:
        return None   # view-up is zero or parallel to the view direction

    scale = block.get("parallel_scale")
    if not (isinstance(scale, (int, float)) and not isinstance(scale, bool)
            and math.isfinite(scale) and scale > 0):
        scale = None
    projection = block.get("parallel_projection", True)
    if not isinstance(projection, bool):
        projection = True

    return {
        **{key: [float(v) for v in vec] for key, vec in vectors.items()},
        "parallel_scale": float(scale) if scale is not None else None,
        "parallel_projection": projection,
    }


def overflow_zoom(points, projection_matrix, margin=0.97):
    """
    Zoom factor (<= 1.0) that brings every point inside the frame.

    points            : (n, 3) world coordinates
    projection_matrix : (4, 4) composite world-to-normalized-device matrix
    margin            : the largest allowed |x| or |y| in normalized device
                        coordinates, leaving a small border at the image edge

    Returns 1.0 when nothing overflows, or when there are no usable points.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return 1.0
    clip = np.c_[pts, np.ones(len(pts))] @ np.asarray(projection_matrix, dtype=float).T
    w = clip[:, 3]
    ok = np.abs(w) > 1e-12
    if not np.any(ok):
        return 1.0
    ndc = np.abs(clip[ok, :2] / w[ok, None])
    ndc = ndc[np.all(np.isfinite(ndc), axis=1)]
    if len(ndc) == 0:
        return 1.0
    max_ndc = float(ndc.max())
    return 1.0 if max_ndc <= margin else margin / max_ndc
