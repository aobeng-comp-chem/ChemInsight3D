"""Render an export image at its final size in a separate off-screen plotter.

The alternative, vtkWindowToImageFilter magnification (screenshot(scale=N)),
renders the image as N x N tiles at window size. Screen-space effects are then
computed per tile and can show seams at tile borders, and pixel-sized things
(line widths, label fonts) stay window-sized, so they shrink relative to the
image.

render_full_size() instead builds a throwaway off-screen plotter at the final
resolution with copies of the scene's 3D props, the same camera and the same
RenderStyle, with pixel sizes multiplied by the scale. The copies get their
own mappers and properties, so no GPU resources are shared between the two
render windows. It returns None when the size is beyond MAX_DIMENSION or the
render fails, and the caller falls back to tiling.
"""

from __future__ import annotations

import camera_utils
import rendering_style

# Largest image side rendered in one piece. Every current desktop GPU allows
# at least 16384 x 16384 renderbuffers; above that, tiling is the only option.
MAX_DIMENSION = 16384


def _copy_actor(actor, scale):
    import vtk
    old_mapper = actor.GetMapper()
    if old_mapper is None or old_mapper.GetInput() is None:
        return None
    mapper = old_mapper.NewInstance()
    mapper.ShallowCopy(old_mapper)
    mapper.SetInputData(old_mapper.GetInput())
    factor, units = vtk.reference(0.0), vtk.reference(0.0)
    old_mapper.GetRelativeCoincidentTopologyLineOffsetParameters(factor, units)
    mapper.SetRelativeCoincidentTopologyLineOffsetParameters(float(factor), float(units))

    copy = vtk.vtkActor()
    copy.SetMapper(mapper)
    prop = vtk.vtkProperty()
    prop.DeepCopy(actor.GetProperty())
    prop.SetLineWidth(actor.GetProperty().GetLineWidth() * scale)
    prop.SetPointSize(actor.GetProperty().GetPointSize() * scale)
    copy.SetProperty(prop)
    if actor.GetBackfaceProperty() is not None:
        back = vtk.vtkProperty()
        back.DeepCopy(actor.GetBackfaceProperty())
        copy.SetBackfaceProperty(back)
    copy.SetPosition(actor.GetPosition())
    copy.SetOrientation(actor.GetOrientation())
    copy.SetScale(actor.GetScale())
    copy.SetOrigin(actor.GetOrigin())
    if actor.GetUserMatrix() is not None:
        copy.SetUserMatrix(actor.GetUserMatrix())
    return copy


def _copy_billboard_text(actor, scale):
    import vtk
    copy = vtk.vtkBillboardTextActor3D()
    copy.SetInput(actor.GetInput())
    copy.SetPosition(actor.GetPosition())
    dx, dy = actor.GetDisplayOffset()
    copy.SetDisplayOffset(int(round(dx * scale)), int(round(dy * scale)))
    text = vtk.vtkTextProperty()
    text.ShallowCopy(actor.GetTextProperty())
    text.SetFontSize(int(round(actor.GetTextProperty().GetFontSize() * scale)))
    sx, sy = actor.GetTextProperty().GetShadowOffset()
    text.SetShadowOffset(int(round(sx * scale)), int(round(sy * scale)))
    copy.SetTextProperty(text)
    return copy


def copy_scene_props(renderer, scale):
    """Copies of the renderer's visible 3D props, pixel sizes multiplied by scale."""
    import vtk
    copies = []
    props = renderer.GetViewProps()
    props.InitTraversal()
    prop = props.GetNextProp()
    while prop is not None:
        if prop.GetVisibility():
            if isinstance(prop, vtk.vtkBillboardTextActor3D):
                copies.append(_copy_billboard_text(prop, scale))
            elif isinstance(prop, vtk.vtkActor):
                copy = _copy_actor(prop, scale)
                if copy is not None:
                    copies.append(copy)
            # 2D overlays (title, population text) are never part of an export.
        prop = props.GetNextProp()
    return copies


def render_full_size(source_plotter, style, scale, transparent):
    """
    The source plotter's 3D scene rendered at scale x its window size, or
    None when that is not possible here (the caller then falls back to tiled
    magnification).
    """
    import pyvista as pv

    width, height = source_plotter.ren_win.GetSize()
    target = (int(width * scale), int(height * scale))
    if min(target) < 1 or max(target) > MAX_DIMENSION:
        return None
    plotter = None
    try:
        plotter = pv.Plotter(off_screen=True, window_size=target, lighting="none")
        renderer = plotter.renderer
        renderer.SetBackground(source_plotter.renderer.GetBackground())
        for prop in copy_scene_props(source_plotter.renderer, scale):
            renderer.AddViewProp(prop)
        camera_utils.apply_camera_state(
            renderer.GetActiveCamera(),
            camera_utils.capture_camera_state(source_plotter.renderer.GetActiveCamera()))
        plotter.camera_set = True   # keep PyVista from resetting the camera on first render
        rendering_style.apply_render_style(plotter, style)
        rendering_style.install_auto_render_passes(plotter, lambda: style)
        renderer.ResetCameraClippingRange()
        image = plotter.screenshot(return_img=True, transparent_background=transparent)
        if image is None or tuple(image.shape[:2]) != (target[1], target[0]):
            return None
        return image
    except Exception as exc:   # no off-screen context, out of GPU memory, ...
        print(f"Full-size export render failed ({exc}); using tiled export instead.")
        return None
    finally:
        if plotter is not None:
            plotter.close()
