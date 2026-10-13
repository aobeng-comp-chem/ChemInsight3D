"""Lighting, materials and render-quality settings for the 3D viewer, in one place.

RenderStyle holds every visual setting the viewer owns: the light rig
(LightingSettings), the material preset and whether depth peeling may switch
on. It starts from one of two presets (STYLE_PRESETS); editing a light control
makes it "Custom". The GUI, session files and image export all read the same
RenderStyle:

    apply_render_style(plotter, style)   lights and depth peeling
    install_auto_render_passes(...)      re-decide depth peeling every frame
    style.material(actor_class)          Material for add_mesh / apply_material

The plotter must be created with lighting='none', so the lights here are the
only ones. "Default" reproduces PyVista's default light kit (vtkLightKit
defaults) and the v1.0.0 materials exactly, so figures render as they did in
v1.0.0. "Glossy" follows the Mathematica cube-plot notebook
(Lighting -> "Neutral", Specularity[White, 50]).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace

STYLE_VERSION = 3

WHITE = (1.0, 1.0, 1.0)
KIT_KEY_COLOR = (1.0, 248 / 255, 230 / 255)    # vtkLightKit key light, warmth 0.6
KIT_FILL_COLOR = (232 / 255, 238 / 255, 1.0)   # vtkLightKit fill light, warmth 0.4

# vtkLight.SetLightType values
_HEADLIGHT, _CAMERA_LIGHT, _SCENE_LIGHT = 1, 2, 3


# ── Lights ────────────────────────────────────────────────────────────────────

LAYOUTS = ("kit", "quad")


@dataclass(frozen=True)
class LightingSettings:
    """
    A five-light rig. Angles are degrees in the camera frame (or in the
    frozen scene_frame when lock_to_camera is off): azimuth 0 = from the
    viewer, positive = from the right; elevation positive = from above. The
    headlight shines along the view direction in both layouts.

    layout "kit" (key/fill/rim): the fill light mirrors the key: azimuth
    -key_azimuth, elevation key_elevation * fill_elevation_factor, intensity
    key_intensity * fill_ratio. Rim lights sit behind at ±rim_azimuth,
    elevation 0, intensity key_intensity * rim_ratio.

    layout "quad" (four diagonal lights, like Mathematica's "Neutral"
    lighting): lights at (±key_azimuth, ±key_elevation); the upper pair at
    key_intensity, the lower pair at key_intensity * fill_ratio. The
    fill-elevation and rim settings are unused.
    """
    layout: str = "kit"
    key_azimuth: float = 10.0
    key_elevation: float = 50.0
    key_intensity: float = 0.75
    fill_ratio: float = 1 / 3
    fill_elevation_factor: float = -1.5
    headlight: float = 0.25
    rim_lights: bool = True
    rim_ratio: float = 1 / 3.5
    rim_azimuth: float = 110.0
    key_color: tuple = KIT_KEY_COLOR
    fill_color: tuple = KIT_FILL_COLOR
    lock_to_camera: bool = True
    # World-space [right, up, back] axes of the camera at the moment the lock
    # was released; lights stay fixed in the scene relative to this frame.
    scene_frame: list | None = None


def light_direction(azimuth_deg, elevation_deg):
    """Unit vector toward the light in the camera frame (+x right, +y up, +z to viewer)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))


def light_specs(settings):
    """
    The rig as (name, camera-frame direction, intensity, colour, on) tuples,
    in a fixed order: headlight, key, fill, right rim, left rim ("kit"), or
    headlight, upper right, upper left, lower right, lower left ("quad").
    """
    s = settings
    key = s.key_intensity
    if s.layout == "quad":
        az, el, low = s.key_azimuth, s.key_elevation, key * s.fill_ratio
        return [
            ("head", (0.0, 0.0, 1.0), s.headlight, WHITE, True),
            ("upper_right", light_direction(az, el), key, s.key_color, True),
            ("upper_left", light_direction(-az, el), key, s.key_color, True),
            ("lower_right", light_direction(az, -el), low, s.fill_color, True),
            ("lower_left", light_direction(-az, -el), low, s.fill_color, True),
        ]
    return [
        ("head", (0.0, 0.0, 1.0), s.headlight, WHITE, True),
        ("key", light_direction(s.key_azimuth, s.key_elevation), key, s.key_color, True),
        ("fill", light_direction(-s.key_azimuth, s.key_elevation * s.fill_elevation_factor),
         key * s.fill_ratio, s.fill_color, True),
        ("rim_right", light_direction(s.rim_azimuth, 0.0), key * s.rim_ratio, WHITE, s.rim_lights),
        ("rim_left", light_direction(-s.rim_azimuth, 0.0), key * s.rim_ratio, WHITE, s.rim_lights),
    ]


def camera_frame(camera):
    """World-space [right, up, back] unit axes of a vtkCamera."""
    pos, focal = camera.GetPosition(), camera.GetFocalPoint()
    back = _normalize([p - f for p, f in zip(pos, focal)])
    up = list(camera.GetViewUp())
    dot = sum(u * b for u, b in zip(up, back))
    up = _normalize([u - dot * b for u, b in zip(up, back)])
    right = _cross(up, back)
    return [right, up, back]


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _to_world(direction, frame):
    right, up, back = frame
    return [direction[0] * r + direction[1] * u + direction[2] * b
            for r, u, b in zip(right, up, back)]


def _light_rig(renderer):
    """The renderer's five app-owned lights, created once and then mutated in place."""
    rig = getattr(renderer, "_ci3d_lights", None)
    if rig is None:
        import vtk
        rig = []
        for _ in range(5):
            light = vtk.vtkLight()
            light.SetPositional(False)
            light.SetFocalPoint(0.0, 0.0, 0.0)
            light.SetAmbientColor(0.0, 0.0, 0.0)
            renderer.AddLight(light)
            rig.append(light)
        renderer._ci3d_lights = rig
    return rig


def apply_lighting(renderer, settings):
    """Point the rig's lights at the settings. Lights are never removed or re-added."""
    frame = None if settings.lock_to_camera else settings.scene_frame
    for light, (name, direction, intensity, color, on) in zip(
            _light_rig(renderer), light_specs(settings)):
        if frame is None:
            light.SetLightType(_HEADLIGHT if name == "head" else _CAMERA_LIGHT)
            light.SetPosition(*direction)
        else:
            light.SetLightType(_SCENE_LIGHT)
            light.SetPosition(*_to_world(direction, frame))
        # A directional light shines from position toward focal point. VTK
        # moves a headlight's focal point to the camera's on every render, so
        # reset it here or a former headlight would point the wrong way.
        light.SetFocalPoint(0.0, 0.0, 0.0)
        light.SetIntensity(float(intensity))
        light.SetDiffuseColor(*color)
        light.SetSpecularColor(*color)
        light.SetSwitch(bool(on))


# ── Materials ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Material:
    ambient: float
    diffuse: float
    specular: float
    specular_power: float

    def kwargs(self):
        """Keyword arguments for plotter.add_mesh."""
        return dict(ambient=self.ambient, diffuse=self.diffuse,
                    specular=self.specular, specular_power=self.specular_power)


ACTOR_CLASSES = ("lobe", "atom", "bond", "highlight")

MATERIAL_PRESETS = {
    # v1.0.0 values; "highlight" is what PyVista gave the selection spheres by default.
    "Legacy": {
        "lobe": Material(0.15, 0.85, 0.40, 20),
        "atom": Material(0.08, 0.85, 0.50, 40),
        "bond": Material(0.08, 0.90, 0.30, 20),
        "highlight": Material(0.0, 1.0, 0.0, 100),
    },
    # The notebook's Specularity[White, 50]: strong, sharp white highlights,
    # the same on every surface. Chosen from matched renders (2026-10-11).
    "Glossy": {
        "lobe": Material(0.30, 0.75, 0.80, 50),
        "atom": Material(0.30, 0.75, 0.80, 50),
        "bond": Material(0.30, 0.75, 0.80, 50),
        "highlight": Material(0.35, 0.65, 0.0, 10),
    },
}


def apply_material(actor, material):
    """Restyle an existing actor's surface lighting coefficients."""
    prop = actor.GetProperty()
    prop.SetAmbient(material.ambient)
    prop.SetDiffuse(material.diffuse)
    prop.SetSpecular(material.specular)
    prop.SetSpecularPower(material.specular_power)


# ── Presets and the style itself ──────────────────────────────────────────────

# Mathematica's Lighting -> "Neutral": four white lights in front of the
# viewer on the image diagonals (ImageScaled {0|2, 0|2, 2}, i.e. about 45°
# across and 35° up or down). The ambient part lives in the Glossy material.
_NEUTRAL_LIGHTS = LightingSettings(
    layout="quad", key_azimuth=45.0, key_elevation=35.0, key_intensity=0.45,
    fill_ratio=1.0, headlight=0.0, rim_lights=False, key_color=WHITE, fill_color=WHITE)

DEFAULT_PRESET = "Default"
GLOSSY_PRESET = "Glossy"
CUSTOM = "Custom"
# name -> (lighting, material preset)
STYLE_PRESETS = {
    DEFAULT_PRESET: (LightingSettings(), "Legacy"),
    GLOSSY_PRESET: (_NEUTRAL_LIGHTS, "Glossy"),
}
# Names saved by development builds since v1.0.0. Their other presets
# (Publication, Print, Screen) load as the default.
RENAMED_PRESETS = {"Legacy light kit": DEFAULT_PRESET, "Default (v1.0.0)": DEFAULT_PRESET,
                   "Neutral (glossy)": GLOSSY_PRESET}


@dataclass(frozen=True)
class RenderStyle:
    """Everything the viewer owns about how the scene is lit and shaded."""
    preset: str = DEFAULT_PRESET        # a STYLE_PRESETS name, or CUSTOM after an edit
    lighting: LightingSettings = field(default_factory=LightingSettings)
    materials: str = "Legacy"
    depth_peeling: bool = True          # peel automatically whenever something is translucent
    version: int = STYLE_VERSION

    @classmethod
    def from_preset(cls, name, **overrides):
        lighting, materials = STYLE_PRESETS[name]
        return cls(preset=name, lighting=lighting, materials=materials, **overrides)

    def with_preset(self, name):
        """This style switched to a preset, keeping the peeling setting."""
        lighting, materials = STYLE_PRESETS[name]
        return replace(self, preset=name, lighting=lighting, materials=materials)

    def material(self, actor_class):
        return MATERIAL_PRESETS[self.materials][actor_class]

    def to_dict(self):
        d = asdict(self)
        d["lighting"]["key_color"] = list(self.lighting.key_color)
        d["lighting"]["fill_color"] = list(self.lighting.fill_color)
        return d

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a style from a session block. Unknown keys are ignored and
        missing or malformed values fall back to defaults, so a damaged block
        can't stop a session from loading. A named preset's missing light
        values come from that preset; a Custom block's from the default rig.
        Presets that no longer exist load as the default.
        """
        if not isinstance(data, dict):
            return cls()
        preset = data.get("preset")
        preset = RENAMED_PRESETS.get(preset, preset)
        peeling = data.get("depth_peeling")
        peeling = peeling if isinstance(peeling, bool) else cls.depth_peeling
        if preset == CUSTOM:
            materials = data.get("materials")
            return cls(preset=CUSTOM,
                       lighting=_settings_from_dict(LightingSettings(), data.get("lighting")),
                       materials=materials if materials in MATERIAL_PRESETS else "Legacy",
                       depth_peeling=peeling)
        if preset not in STYLE_PRESETS:
            return cls(depth_peeling=peeling)
        lighting, materials = STYLE_PRESETS[preset]
        return cls(preset=preset, lighting=_settings_from_dict(lighting, data.get("lighting")),
                   materials=materials, depth_peeling=peeling)


def _as_bool(value, default):
    return value if isinstance(value, bool) else default


def _as_number(value, default, lo=-math.inf, hi=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return min(max(float(value), lo), hi)


def _as_color(value, default):
    if (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                    and math.isfinite(c) for c in value)):
        return tuple(min(max(float(c), 0.0), 1.0) for c in value)
    return default


def _as_frame(value):
    try:
        frame = [[float(c) for c in axis] for axis in value]
    except (TypeError, ValueError):
        return None
    if len(frame) != 3 or any(len(axis) != 3 or not all(map(math.isfinite, axis))
                              for axis in frame):
        return None
    return frame


_LIGHTING_LIMITS = {
    "key_azimuth": (-180.0, 180.0), "key_elevation": (-90.0, 90.0),
    "key_intensity": (0.0, 1.5), "fill_ratio": (0.0, 1.0),
    "fill_elevation_factor": (-2.0, 2.0), "headlight": (0.0, 0.5),
    "rim_ratio": (0.0, 1.0), "rim_azimuth": (0.0, 180.0),
}


def _settings_from_dict(defaults, data):
    if not isinstance(data, dict):
        return defaults
    values = {}
    for f in fields(LightingSettings):
        default = getattr(defaults, f.name)
        raw = data.get(f.name)
        if f.name in _LIGHTING_LIMITS:
            values[f.name] = _as_number(raw, default, *_LIGHTING_LIMITS[f.name])
        elif f.name in ("rim_lights", "lock_to_camera"):
            values[f.name] = _as_bool(raw, default)
        elif f.name in ("key_color", "fill_color"):
            values[f.name] = _as_color(raw, default)
        elif f.name == "scene_frame":
            values[f.name] = _as_frame(raw) if raw is not None else None
        elif f.name == "layout":
            values[f.name] = raw if raw in LAYOUTS else default
    settings = LightingSettings(**values)
    if not settings.lock_to_camera and settings.scene_frame is None:
        settings = replace(settings, lock_to_camera=True)   # unusable frame: fall back to camera lights
    return settings


# ── Applying a style to a plotter ─────────────────────────────────────────────

def apply_render_style(plotter, style):
    """Push the lights to a plotter and settle depth peeling."""
    apply_lighting(plotter.renderer, style.lighting)
    enable_coincident_line_offset()
    update_render_passes(plotter, style)


def enable_coincident_line_offset():
    """
    Turn on VTK's polygon-offset resolution of coincident topology (off by
    default in VTK 9.6). It is a global mapper setting: polygons are pushed
    back by the same small depth offset, so nothing moves visibly, and lines
    lying exactly on a surface (the lobe grid overlay) win the depth test
    instead of flickering in and out of it.
    """
    import vtk
    vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()


def install_auto_render_passes(plotter, current_style):
    """
    Re-decide depth peeling at the start of every frame, so any change that
    makes something translucent (opacity slider, highlights, grid lines) is
    covered without each code path having to remember it. current_style is a
    callable returning the RenderStyle in force. Installed once per plotter.
    """
    if getattr(plotter, "_ci3d_pass_observer", None) is None:
        plotter._ci3d_pass_observer = plotter.ren_win.AddObserver(
            "StartEvent", lambda *_: update_render_passes(plotter, current_style()))


def scene_has_translucency(renderer):
    """True when any visible actor is drawn with opacity below 1."""
    actors = renderer.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextActor()
    while actor is not None:
        if actor.GetVisibility() and actor.GetProperty().GetOpacity() < 1.0:
            return True
        actor = actors.GetNextActor()
    return False


def update_render_passes(plotter, style):
    """
    Depth peeling on when the style allows it and something is translucent;
    returns whether it is on. Without it VTK falls back to weighted
    order-independent transparency, which weights overlapping layers almost
    equally: a red 50 % plane in front of a blue one rendered (96, 0, 96)
    instead of the correct (127, 0, 64) (VTK 9.6). Peeling composites
    front-to-back exactly.
    """
    renderer = plotter.renderer
    peel = bool(style.depth_peeling) and scene_has_translucency(renderer)
    renderer.SetUseDepthPeeling(peel)
    if peel:
        renderer.SetMaximumNumberOfPeels(8)
        renderer.SetOcclusionRatio(0.0)
    return peel
