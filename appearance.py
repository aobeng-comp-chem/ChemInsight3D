"""Appearance settings: lobe phase palettes, a typed settings snapshot, validation,
and reading/writing the session block.

The viewer's own attributes (current_mol_theme, atom_scale, lobe_pos_color, ...)
stay the single source of truth. AppearanceSettings is only a snapshot of them,
used to save, load, reset and compare, so there is no second state to drift.

Sessions store a versioned "appearance" block and keep writing the older flat
keys ('mol_theme', 'color_scheme', ...) so v1.0.0 can still open them. On load
the block wins; sessions without it are read from the flat keys.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

APPEARANCE_SCHEMA_VERSION = 2


# ── Phase palettes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Palette:
    id: str
    label: str
    positive: tuple       # colour of the positive-phase lobe, RGB 0-1
    negative: tuple       # colour of the negative-phase lobe
    colorblind_safe: bool = True


# colorblind_safe: the two colours stay clearly apart (CIELAB distance > 40)
# under simulated deuteranopia and protanopia (Machado et al. 2009). Red/Green
# measured 15.8 under deuteranopia, so it is kept for old sessions but marked.
PALETTES = (
    Palette("crimson_steel_blue", "Crimson / Steel-blue (default)",
            (0.60, 0.15, 0.18), (0.55, 0.72, 0.82)),
    Palette("red_blue", "Red / Blue", (0.85, 0.10, 0.10), (0.10, 0.30, 0.85)),
    Palette("orange_blue", "Orange / Blue", (0.90, 0.45, 0.05), (0.15, 0.40, 0.85)),
    Palette("purple_teal", "Purple / Teal", (0.45, 0.20, 0.70), (0.35, 0.82, 0.72)),
    Palette("orange_teal", "Orange / Teal", (0.92, 0.50, 0.05), (0.05, 0.60, 0.65)),
    Palette("purple_gold", "Purple / Gold", (0.55, 0.10, 0.75), (0.85, 0.72, 0.05)),
    Palette("white_grey", "White / Grey", (0.95, 0.95, 0.95), (0.45, 0.45, 0.45)),
    Palette("red_green", "Red / Green (not colour-blind safe)",
            (0.85, 0.12, 0.12), (0.12, 0.70, 0.25), colorblind_safe=False),
)
PALETTES_BY_ID = {p.id: p for p in PALETTES}
DEFAULT_PALETTE = "crimson_steel_blue"
CUSTOM_PALETTE = "custom"
CUSTOM_LABEL = "Custom"

# Names older sessions stored in 'color_scheme'.
LEGACY_PALETTE_NAMES = {
    "Crimson / Steel-blue (default)": "crimson_steel_blue",
    "Mathematica (default)": "crimson_steel_blue",
    "Red / Blue": "red_blue",
    "Red / Green": "red_green",
    "Orange / Teal": "orange_teal",
    "Purple / Gold": "purple_gold",
    "White / Grey": "white_grey",
}

# Display name <-> stable id for the molecular representation.
REPRESENTATIONS = {
    "ball_and_stick": "Ball and Stick",
    "sticks": "Sticks",
    "tubes": "Tubes",
    "space_filling": "Space Filling",
}
REPRESENTATION_IDS = {name: rid for rid, name in REPRESENTATIONS.items()}
# v1.0.0 fails on a representation it doesn't know, so its flat 'mol_theme'
# key gets the closest one it has.
LEGACY_REPRESENTATIONS = {"Tubes": "Sticks"}

SCALE_RANGE = (0.25, 3.0)       # atom and bond size multipliers (slider 25-300 %)
OPACITY_RANGE = (0.10, 1.0)     # lobe opacity (slider 10-100 %)
BACKGROUNDS = ("black", "white")


def palette_for_colors(positive, negative):
    """The palette whose colours these are, or CUSTOM_PALETTE."""
    for p in PALETTES:
        if _same_rgb(p.positive, positive) and _same_rgb(p.negative, negative):
            return p.id
    return CUSTOM_PALETTE


def _same_rgb(a, b, tol=1e-6):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# ── The settings snapshot ─────────────────────────────────────────────────────

@dataclass
class AppearanceSettings:
    representation: str = "Ball and Stick"    # display name, as the viewer uses it
    atom_scale: float = 1.0
    bond_scale: float = 1.0
    palette: str = DEFAULT_PALETTE            # palette id or CUSTOM_PALETTE
    positive_rgb: tuple = PALETTES_BY_ID[DEFAULT_PALETTE].positive
    negative_rgb: tuple = PALETTES_BY_ID[DEFAULT_PALETTE].negative
    surface_opacity: float = 1.0
    background: str = "black"
    close_open_lobes: bool = False
    show_mesh: bool = True                    # the mesh overlay on the lobes
    show_atom_labels: bool = False


def appearance_to_session(settings):
    """The versioned 'appearance' session block."""
    d = asdict(settings)
    return {
        "schema_version": APPEARANCE_SCHEMA_VERSION,
        "representation": REPRESENTATION_IDS.get(d["representation"], "ball_and_stick"),
        "atom_scale": d["atom_scale"],
        "bond_scale": d["bond_scale"],
        "scheme_id": d["palette"],
        "positive_rgb": [float(c) for c in d["positive_rgb"]],
        "negative_rgb": [float(c) for c in d["negative_rgb"]],
        "surface_opacity": d["surface_opacity"],
        "background": d["background"],
        "close_open_lobes": d["close_open_lobes"],
        "show_mesh_overlay": d["show_mesh"],
        "show_atom_labels": d["show_atom_labels"],
    }


def legacy_session_keys(settings):
    """The flat keys v1.0.0 reads, kept in every session for older versions."""
    palette = PALETTES_BY_ID.get(settings.palette)
    legacy_name = next((name for name, pid in LEGACY_PALETTE_NAMES.items()
                        if palette is not None and pid == palette.id
                        and name != "Mathematica (default)"), CUSTOM_LABEL)
    return {
        "opacity": settings.surface_opacity,
        "mol_theme": LEGACY_REPRESENTATIONS.get(settings.representation, settings.representation),
        "atom_scale": settings.atom_scale,
        "bond_scale": settings.bond_scale,
        "background": settings.background,
        "show_wireframe": settings.show_mesh,
        "close_lobes": settings.close_open_lobes,
        "show_labels": settings.show_atom_labels,
        "color_scheme": legacy_name,
    }


def appearance_from_session(state):
    """
    AppearanceSettings from a session dict. Reads the 'appearance' block when
    present and usable, otherwise the older flat keys. Missing or malformed
    values fall back to defaults; explicit RGB values always win over a
    palette name that disagrees with them.
    """
    base = AppearanceSettings()
    if not isinstance(state, dict):
        return base
    block = state.get("appearance")
    if isinstance(block, dict):
        return _from_block(block, base)
    return _from_legacy(state, base)


def _from_block(block, base):
    rep = REPRESENTATIONS.get(block.get("representation"), base.representation)
    positive = _rgb(block.get("positive_rgb"))
    negative = _rgb(block.get("negative_rgb"))
    scheme = block.get("scheme_id")
    if positive is None or negative is None:
        palette = PALETTES_BY_ID.get(scheme, PALETTES_BY_ID[DEFAULT_PALETTE])
        positive, negative, scheme = palette.positive, palette.negative, palette.id
    else:
        scheme = palette_for_colors(positive, negative)     # RGB is authoritative
    return AppearanceSettings(
        representation=rep,
        atom_scale=_number(block.get("atom_scale"), base.atom_scale, *SCALE_RANGE),
        bond_scale=_number(block.get("bond_scale"), base.bond_scale, *SCALE_RANGE),
        palette=scheme, positive_rgb=positive, negative_rgb=negative,
        surface_opacity=_number(block.get("surface_opacity"), base.surface_opacity, *OPACITY_RANGE),
        background=block.get("background") if block.get("background") in BACKGROUNDS else base.background,
        close_open_lobes=_flag(block.get("close_open_lobes"), base.close_open_lobes),
        show_mesh=_show_mesh(block.get("show_mesh_overlay"), block.get("mesh_style"), base.show_mesh),
        show_atom_labels=_flag(block.get("show_atom_labels"), base.show_atom_labels),
    )


def _from_legacy(state, base):
    rep = state.get("mol_theme")
    palette = PALETTES_BY_ID[LEGACY_PALETTE_NAMES.get(state.get("color_scheme"), DEFAULT_PALETTE)]
    return AppearanceSettings(
        representation=rep if rep in REPRESENTATION_IDS else base.representation,
        atom_scale=_number(state.get("atom_scale"), base.atom_scale, *SCALE_RANGE),
        bond_scale=_number(state.get("bond_scale"), base.bond_scale, *SCALE_RANGE),
        palette=palette.id, positive_rgb=palette.positive, negative_rgb=palette.negative,
        surface_opacity=_number(state.get("opacity"), base.surface_opacity, *OPACITY_RANGE),
        background=state.get("background") if state.get("background") in BACKGROUNDS else base.background,
        close_open_lobes=_flag(state.get("close_lobes"), base.close_open_lobes),
        show_mesh=_flag(state.get("show_wireframe"), base.show_mesh),
        show_atom_labels=_flag(state.get("show_labels"), base.show_atom_labels),
    )


def _show_mesh(shown, style, default):
    """
    Whether the mesh is on. Development builds between v1.0.0 and the
    on/off mesh saved a 'mesh_style' name instead; any style but 'off'
    means on.
    """
    if isinstance(shown, bool):
        return shown
    if isinstance(style, str):
        return style != "off"
    return default


def _number(value, default, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return min(max(float(value), lo), hi)


def _flag(value, default):
    return value if isinstance(value, bool) else default


def _rgb(value):
    if (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(c, (int, float)) and not isinstance(c, bool)
                    and math.isfinite(c) for c in value)):
        return tuple(min(max(float(c), 0.0), 1.0) for c in value)
    return None
