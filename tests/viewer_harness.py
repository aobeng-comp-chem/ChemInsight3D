"""
Shared off-screen harness for tests that drive the real MultiCubeVisualizer.

SKIP_REASON is set when Qt, VTK or off-screen OpenGL is unavailable, so the
tests that import this module can skip instead of erroring.
"""
import os
import platform
import re
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if "CI3D_CUBE_CACHE_DIR" not in os.environ:      # keep test grids out of the user cache
    import atexit
    import shutil
    import tempfile
    os.environ["CI3D_CUBE_CACHE_DIR"] = tempfile.mkdtemp(prefix="ci3d-test-cache-")
    atexit.register(shutil.rmtree, os.environ["CI3D_CUBE_CACHE_DIR"], True)

SKIP_REASON = None
try:
    import pyvista as pv
    import vtk
    from PyQt5.QtWidgets import QApplication

    import ChemInsight3D as ci

    pv.global_theme.allow_empty_mesh = True
    _APP = QApplication.instance() or QApplication(sys.argv[:1])
    _probe = pv.Plotter(off_screen=True, window_size=(64, 64))
    _probe.screenshot(return_img=True)
    _probe.close()
except Exception as exc:   # no Qt / VTK / off-screen OpenGL in this environment
    SKIP_REASON = f"off-screen rendering unavailable: {exc}"

BOHR = 0.529177
WATER = [(8, 0.0, 0.0, 0.0), (1, 1.43, 1.1, 0.0), (1, -1.43, 1.1, 0.0)]   # bohr
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FECO5_FCHK = os.path.join(REPO, "test_files", "closed-shell", "gaussian", "feco5-hf.fchk")


def make_cube(atoms=WATER, width=2.0, axis=0, centre=(0.0, 0.0, 0.0), n=40, half=14.0):
    """
    A p-like orbital, coordinate[axis] * exp(-r^2 / width), on a cube grid.
    width=2 stays inside the atom frame; width=60 reaches the grid edges.
    """
    ax = np.linspace(-half, half, n)
    grid = np.meshgrid(ax, ax, ax, indexing="ij")
    rel = [g - c for g, c in zip(grid, centre)]
    data = rel[axis] * np.exp(-(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) / width)
    data /= np.abs(data).max()
    result = dict(
        atom_info=[(z, x * BOHR, y * BOHR, zz * BOHR) for z, x, y, zz in atoms],
        bohr_const=BOHR, origin=np.array([-half] * 3),
        spacing=np.array([ax[1] - ax[0]] * 3), nx=n, ny=n, nz=n, grid=data)
    return ci.MultiCubeVisualizer()._result_to_cube(result)


_FECO5_CACHE = {}


def feco5_homo_cube(grid_quality=60):
    """Fe(CO)5 HOMO computed from the tracked fchk fixture, or None if it is missing."""
    if not os.path.isfile(FECO5_FCHK):
        return None
    if grid_quality not in _FECO5_CACHE:
        import fchk_read
        _ene_a, occ_a, _ene_b, _occ_b = fchk_read.get_orbital_energies_and_occupations_fchk(FECO5_FCHK)
        homo = int(np.count_nonzero(np.asarray(occ_a) > 0.5))
        result = fchk_read.compute_cube_data_fchk(
            FECO5_FCHK, [homo], "alpha", grid_quality, 4.0, BOHR)[0]
        _FECO5_CACHE[grid_quality] = ci.MultiCubeVisualizer()._result_to_cube(result)
    return _FECO5_CACHE[grid_quality]


def make_viewer(cubes=(), size=(800, 600)):
    """
    Mirror visualize(): plotter, renderer configuration, scene, background,
    axes, startup camera.
    """
    v = ci.MultiCubeVisualizer()
    v.plotter = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    v.plotter.enable_parallel_projection()
    v._configure_renderer()
    for i, cube in enumerate(cubes):
        v._append_cube_to_list(f"cube{i}.cube", cube)
    v.plotter.set_background(v.background_color)
    if v.cubes:
        v.update_visualization()
    v.plotter.show_axes()
    v._init_camera()
    return v


def set_fixed_camera(v, direction=(1.0, -1.2, 0.8), up=(0.0, 0.0, 1.0), scale_factor=0.62):
    """
    A camera computed only from the cube's atoms, independent of the app's
    framing code, so lighting references don't move when framing changes.
    """
    coords = np.asarray(v.cubes[v.current_cube_index]["coordinates"], dtype=float)
    centre = (coords.min(axis=0) + coords.max(axis=0)) / 2
    extent = float(np.ptp(coords, axis=0).max()) + 6.0
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    cam = v.plotter.renderer.GetActiveCamera()
    cam.SetFocalPoint(*centre)
    cam.SetPosition(*(centre + d * extent * 10))
    cam.SetViewUp(*up)
    cam.OrthogonalizeViewUp()
    cam.SetParallelScale(extent * scale_factor)
    v.plotter.renderer.ResetCameraClippingRange()


def render(v):
    """Render and capture without overlays; screenshot() alone can return a stale frame."""
    with v._export_overlays_hidden():
        v.plotter.render()
        return v.plotter.screenshot(return_img=True)


def environment_fingerprint():
    """
    Platform, VTK version and loaded GL driver libraries. Reference images are
    only comparable between runs with the same fingerprint, since MSAA, depth
    peeling and SSAO output vary between GPU drivers.
    """
    # Drivers load with the GL context and unload when it closes, so look
    # while a probe render is alive.
    probe = pv.Plotter(off_screen=True, window_size=(32, 32))
    probe.add_mesh(pv.Sphere())
    probe.render()
    drivers = []
    try:
        with open("/proc/self/maps") as maps:
            found = re.findall(r"/((?:lib(?:d3d12|nvidia-glcore|nvidia-eglcore|gallium|OSMesa|"
                               r"GLX_mesa|GLX_nvidia|EGL_mesa|EGL_nvidia)|\w+_dri)[^/\s]*?)\.so",
                               maps.read())
            drivers = sorted(set(found))
    except OSError:
        pass
    finally:
        probe.close()
    return f"{platform.system()}|vtk {vtk.vtkVersion.GetVTKVersion()}|{','.join(drivers) or 'unknown'}"
