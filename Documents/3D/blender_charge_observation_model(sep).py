"""
Blender 4.x scientific visualization generator

Creates a fully editable 3D charge distribution and z=0 observation-plane model.
All visible annotations are converted to meshes for GLB/PowerPoint compatibility.

Usage:
1. Open Blender 4.x -> Scripting.
2. Create a new text block and paste this entire script.
3. Press Run Script.
4. Optionally set AUTO_EXPORT = True before running.
"""

import bpy
import os
from mathutils import Vector


# =============================================================================
# USER SETTINGS
# =============================================================================

# Coordinate system
SPACE_SCALE = 1.5                 # 1.0 = original space, 1.5 = 50% wider space
AXIS_LENGTH = 2.0 * SPACE_SCALE   # Coordinate range is now -3 ... +3
AXIS_RADIUS = 0.018
AXIS_VERTICES = 24
ARROW_LENGTH = 0.22
ARROW_RADIUS = 0.075
ORIGIN_RADIUS = 0.055

# Tick marks and text
TICK_VALUES = (-3, -2, -1, 0, 1, 2, 3)
TICK_LENGTH = 0.11
TICK_RADIUS = 0.008
TICK_LABEL_SIZE = 0.125
TICK_LABEL_OFFSET = 0.18
AXIS_LABEL_SIZE = 0.24
AXIS_LABEL_OFFSET = 0.16
TEXT_EXTRUDE = 0.006
TEXT_BEVEL_DEPTH = 0.001
TEXT_RESOLUTION = 3

# Labels are designed to face this default presentation viewpoint.
# They remain ordinary editable mesh objects after creation.
LABEL_VIEWPOINT = (9.75, -11.25, 8.4)

# Observation plane and visual grid
PLANE_HALF_SIZE = 2.0 * SPACE_SCALE
PLANE_ALPHA = 0.20
GRID_DIVISIONS = 8               # Change to 32 for a 32 x 32 visual grid
GRID_THICKNESS = 0.010
GRID_HEIGHT = 0.008
GRID_Z_OFFSET = 0.010            # Small offset avoids coplanar z-fighting

# Point charges: edit only this list to change positions/values
CHARGES = [
    {"pos": (-1.1, -0.7, 0.8), "q": +1.0},
    {"pos": ( 0.9, -1.0, 1.3), "q": -0.8},
    {"pos": (-0.4,  1.1, 1.6), "q": +0.7},
    {"pos": ( 1.2,  0.7, 0.6), "q": -1.0},
    {"pos": ( 0.2,  0.1, 1.9), "q": +0.5},
]

CHARGE_RADIUS = 0.19
CHARGE_RADIUS_MIN_FACTOR = 0.92
CHARGE_RADIUS_MAX_FACTOR = 1.10
CHARGE_SEGMENTS = 32
CHARGE_RINGS = 16
CHARGE_SYMBOL_SIZE = 0.22
CHARGE_SYMBOL_GAP = 0.014

# Projection guides
SHOW_PROJECTION_LINES = True
PROJECTION_DASHED = True
PROJECTION_RADIUS = 0.007
PROJECTION_DASH_LENGTH = 0.10
PROJECTION_GAP_LENGTH = 0.065
PROJECTION_START_Z = 0.025
PROJECTION_MARKER_RADIUS = 0.040
PROJECTION_MARKER_HEIGHT = 0.012

# Automatic GLB export
AUTO_EXPORT = False
EXPORT_PATH = "//charge_observation_model.glb"

# Preview camera/light (excluded from GLB by the export settings below)
CAMERA_LOCATION = (9.75, -11.25, 8.4)
CAMERA_TARGET = (0.0, 0.0, 0.65)
LIGHT_LOCATION = (6.0, -6.0, 10.5)
LIGHT_ENERGY = 900.0
LIGHT_SIZE = 7.5

# RGBA material colors
COLOR_AXIS_X = (0.88, 0.12, 0.10, 1.0)
COLOR_AXIS_Y = (0.12, 0.62, 0.22, 1.0)
COLOR_AXIS_Z = (0.10, 0.30, 0.90, 1.0)
COLOR_ORIGIN = (0.12, 0.13, 0.16, 1.0)
COLOR_TEXT = (0.08, 0.09, 0.12, 1.0)
COLOR_PLANE = (0.34, 0.72, 0.94, PLANE_ALPHA)
COLOR_GRID = (0.25, 0.43, 0.55, 1.0)
COLOR_POSITIVE = (0.95, 0.16, 0.08, 1.0)
COLOR_NEGATIVE = (0.08, 0.32, 0.92, 1.0)
COLOR_SYMBOL = (0.98, 0.98, 0.98, 1.0)
COLOR_PROJECTION = (0.28, 0.30, 0.36, 1.0)


# =============================================================================
# SCENE / COLLECTION UTILITIES
# =============================================================================

def clear_scene():
    """Remove the existing scene and its local data blocks."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    # Removing unused data keeps repeated script runs clean.
    for data_group in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for data_block in list(data_group):
            if data_block.users == 0:
                data_group.remove(data_block)


def create_collection(name, parent=None):
    """Create and link a collection to the scene or to another collection."""
    collection = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(collection)
    else:
        parent.children.link(collection)
    return collection


def move_to_collection(obj, target_collection):
    """Move an object into exactly one target collection."""
    # Do not use ``target_collection in obj.users_collection`` here.
    # Some Blender 4.x bpy_prop_collection implementations accept only a
    # string key in __contains__, which raises TypeError for a Collection.
    is_already_linked = any(
        collection == target_collection
        for collection in obj.users_collection
    )
    if not is_already_linked:
        target_collection.objects.link(obj)
    for old_collection in list(obj.users_collection):
        if old_collection != target_collection:
            old_collection.objects.unlink(obj)


def assign_material(obj, material):
    if obj.data is not None and hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(material)


def set_mesh_smooth(obj, smooth=True):
    if obj.type != 'MESH':
        return
    for polygon in obj.data.polygons:
        polygon.use_smooth = smooth


def set_cylinder_side_smooth(obj):
    """Smooth cylindrical sides while keeping large end caps flat."""
    if obj.type != 'MESH':
        return
    for polygon in obj.data.polygons:
        polygon.use_smooth = len(polygon.vertices) <= 4


# =============================================================================
# MATERIALS
# =============================================================================

def create_material(name, rgba, metallic=0.0, roughness=0.48):
    """
    Create a simple glTF-friendly Principled BSDF material.

    Blender 4.2+ uses surface_render_method. Blender 4.0/4.1 used
    blend_method, so the latter is touched only as a compatibility fallback.
    The glTF exporter also reads the Principled Alpha socket directly.
    """
    rgba = tuple(float(value) for value in rgba)
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = rgba
    material.use_backface_culling = False

    # Do not depend on Blender automatically adding or naming default nodes.
    # Custom startup files, localized builds, and some Blender versions can
    # leave a new material's node tree empty. Build the minimal glTF-friendly
    # graph explicitly so it is deterministic in every environment.
    node_tree = material.node_tree
    if node_tree is None:
        raise RuntimeError(f"{name}: Material node tree could not be enabled")

    nodes = node_tree.nodes
    nodes.clear()
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.name = "Material Output"
    output.location = (300.0, 0.0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.name = "Principled BSDF"
    bsdf.location = (0.0, 0.0)
    node_tree.links.new(bsdf.outputs[0], output.inputs[0])

    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = rgba[3]

    if rgba[3] < 1.0:
        # Current Blender 4.x API where available.
        if hasattr(material, "surface_render_method"):
            try:
                material.surface_render_method = 'BLENDED'
            except (TypeError, ValueError):
                # Some intermediate builds expose only DITHERED.
                material.surface_render_method = 'DITHERED'
        # Blender 4.0/4.1 compatibility path only.
        elif hasattr(material, "blend_method"):
            material.blend_method = 'BLEND'

    return material


def create_all_materials():
    return {
        "AxisX": create_material("AxisX", COLOR_AXIS_X, roughness=0.42),
        "AxisY": create_material("AxisY", COLOR_AXIS_Y, roughness=0.42),
        "AxisZ": create_material("AxisZ", COLOR_AXIS_Z, roughness=0.42),
        "Origin": create_material("OriginMaterial", COLOR_ORIGIN, roughness=0.48),
        "Text": create_material("TextMaterial", COLOR_TEXT, roughness=0.55),
        "Plane": create_material("ObservationPlane", COLOR_PLANE, roughness=0.65),
        "Grid": create_material("GridMaterial", COLOR_GRID, roughness=0.58),
        "Positive": create_material("PositiveCharge", COLOR_POSITIVE, roughness=0.34),
        "Negative": create_material("NegativeCharge", COLOR_NEGATIVE, roughness=0.34),
        "Symbol": create_material("ChargeSymbol", COLOR_SYMBOL, roughness=0.50),
        "Projection": create_material("ProjectionMaterial", COLOR_PROJECTION, roughness=0.62),
    }


# =============================================================================
# BASIC GEOMETRY
# =============================================================================

def create_cylinder_between(
    name,
    point_a,
    point_b,
    radius,
    material,
    collection,
    vertices=24,
):
    """Create a mesh cylinder whose local Z axis joins two arbitrary points."""
    point_a = Vector(point_a)
    point_b = Vector(point_b)
    direction = point_b - point_a
    length = direction.length

    if length <= 1e-8:
        raise ValueError(f"Cannot create zero-length cylinder: {name}")

    midpoint = (point_a + point_b) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=max(8, int(vertices)),
        radius=radius,
        depth=length,
        end_fill_type='NGON',
        location=midpoint,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_Mesh"
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')
    assign_material(obj, material)
    set_cylinder_side_smooth(obj)
    move_to_collection(obj, collection)
    return obj


def create_axis(axis_letter, direction, material, collection):
    """Create a full negative-to-positive coordinate-axis shaft."""
    direction = Vector(direction).normalized()
    start = -direction * AXIS_LENGTH
    end = direction * AXIS_LENGTH
    return create_cylinder_between(
        name=f"Axis_{axis_letter}",
        point_a=start,
        point_b=end,
        radius=AXIS_RADIUS,
        material=material,
        collection=collection,
        vertices=AXIS_VERTICES,
    )


def create_arrow(axis_letter, direction, material, collection):
    """Create a cone arrowhead at the positive end of an axis."""
    direction = Vector(direction).normalized()
    base = direction * AXIS_LENGTH
    center = base + direction * (ARROW_LENGTH * 0.5)

    bpy.ops.mesh.primitive_cone_add(
        vertices=max(8, int(AXIS_VERTICES)),
        radius1=ARROW_RADIUS,
        radius2=0.0,
        depth=ARROW_LENGTH,
        end_fill_type='NGON',
        location=center,
    )
    obj = bpy.context.object
    obj.name = f"Arrow_{axis_letter}"
    obj.data.name = f"Arrow_{axis_letter}_Mesh"
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')
    assign_material(obj, material)
    set_cylinder_side_smooth(obj)
    move_to_collection(obj, collection)
    return obj


def create_origin(material, collection):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        radius=ORIGIN_RADIUS,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = "Origin"
    obj.data.name = "Origin_Mesh"
    assign_material(obj, material)
    set_mesh_smooth(obj, True)
    move_to_collection(obj, collection)
    return obj


# =============================================================================
# TEXT -> MESH
# =============================================================================

def orient_text_toward(obj, target):
    """Point the text object's local +Z (front normal) toward a target."""
    direction = Vector(target) - obj.location
    if direction.length <= 1e-8:
        return
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')


def create_text_mesh(
    body,
    name,
    location,
    size,
    material,
    collection,
    face_target=LABEL_VIEWPOINT,
):
    """Create a Text object, orient it, then immediately convert it to Mesh."""
    font_curve = bpy.data.curves.new(type='FONT', name=f"{name}_Font")
    font_curve.body = str(body)
    font_curve.align_x = 'CENTER'
    font_curve.align_y = 'CENTER'
    font_curve.size = size
    font_curve.extrude = TEXT_EXTRUDE
    font_curve.bevel_depth = TEXT_BEVEL_DEPTH
    font_curve.bevel_resolution = 0
    font_curve.resolution_u = max(1, int(TEXT_RESOLUTION))
    font_curve.fill_mode = 'BOTH'

    obj = bpy.data.objects.new(name, font_curve)
    collection.objects.link(obj)
    obj.location = Vector(location)
    orient_text_toward(obj, face_target)
    assign_material(obj, material)

    # Object conversion is reliable from Blender's Scripting workspace as long
    # as the object is active and Blender is in Object Mode.
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target='MESH')

    mesh_obj = bpy.context.object
    mesh_obj.name = name
    mesh_obj.data.name = f"{name}_Mesh"
    return mesh_obj


def tick_name(value):
    if value < 0:
        return f"m{abs(int(value))}"
    return str(int(value))


def create_ticks_and_labels(
    axis_letter,
    direction,
    axis_material,
    text_material,
    tick_collection,
    label_collection,
):
    direction = Vector(direction).normalized()

    if axis_letter == 'X':
        tick_direction = Vector((0.0, 0.0, 1.0))
        label_offset = Vector((0.0, -TICK_LABEL_OFFSET, 0.060))
    elif axis_letter == 'Y':
        tick_direction = Vector((0.0, 0.0, 1.0))
        label_offset = Vector((TICK_LABEL_OFFSET, 0.0, 0.060))
    else:
        tick_direction = Vector((1.0, 0.0, 0.0))
        label_offset = Vector((-TICK_LABEL_OFFSET, 0.10, 0.0))

    for value in TICK_VALUES:
        if abs(float(value)) > AXIS_LENGTH + 1e-6:
            continue

        center = direction * float(value)
        half_tick = tick_direction * (TICK_LENGTH * 0.5)
        suffix = tick_name(value)

        create_cylinder_between(
            name=f"TickMark_{axis_letter}_{suffix}",
            point_a=center - half_tick,
            point_b=center + half_tick,
            radius=TICK_RADIUS,
            material=axis_material,
            collection=tick_collection,
            vertices=12,
        )

        create_text_mesh(
            body=str(int(value)),
            name=f"TickLabel_{axis_letter}_{suffix}",
            location=center + label_offset,
            size=TICK_LABEL_SIZE,
            material=text_material,
            collection=label_collection,
        )


def create_axis_label(axis_letter, direction, material, label_collection):
    direction = Vector(direction).normalized()
    location = direction * (AXIS_LENGTH + ARROW_LENGTH + AXIS_LABEL_OFFSET)
    if axis_letter in {'X', 'Y'}:
        location.z += 0.10

    return create_text_mesh(
        body=axis_letter,
        name=f"Label_{axis_letter}",
        location=location,
        size=AXIS_LABEL_SIZE,
        material=material,
        collection=label_collection,
    )


# =============================================================================
# OBSERVATION PLANE AND GRID
# =============================================================================

def create_observation_plane(material, collection):
    h = float(PLANE_HALF_SIZE)
    vertices = [
        (-h, -h, 0.0),
        ( h, -h, 0.0),
        ( h,  h, 0.0),
        (-h,  h, 0.0),
    ]
    faces = [(0, 1, 2, 3)]

    mesh = bpy.data.meshes.new("Observation_Plane_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Observation_Plane", mesh)
    collection.objects.link(obj)
    assign_material(obj, material)
    return obj


def append_box(vertices, faces, center, dimensions):
    """Append a rectangular prism to shared vertex/face arrays."""
    cx, cy, cz = center
    hx = dimensions[0] * 0.5
    hy = dimensions[1] * 0.5
    hz = dimensions[2] * 0.5
    start = len(vertices)

    vertices.extend([
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ])

    local_faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    faces.extend(tuple(start + index for index in face) for face in local_faces)


def create_grid(material, collection):
    """
    Create one low-poly Mesh containing all grid lines as thin boxes.
    Even at GRID_DIVISIONS=32 this remains lightweight.
    """
    divisions = max(1, int(GRID_DIVISIONS))
    h = float(PLANE_HALF_SIZE)
    z = float(GRID_Z_OFFSET)
    thickness = max(0.001, float(GRID_THICKNESS))
    height = max(0.001, float(GRID_HEIGHT))
    length = 2.0 * h + thickness

    vertices = []
    faces = []
    for index in range(divisions + 1):
        coordinate = -h + (2.0 * h * index / divisions)
        append_box(
            vertices,
            faces,
            center=(0.0, coordinate, z),
            dimensions=(length, thickness, height),
        )
        append_box(
            vertices,
            faces,
            center=(coordinate, 0.0, z),
            dimensions=(thickness, length, height),
        )

    mesh = bpy.data.meshes.new("Grid_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Grid", mesh)
    collection.objects.link(obj)
    assign_material(obj, material)
    obj["grid_divisions"] = divisions
    return obj


# =============================================================================
# CHARGES AND PROJECTION GUIDES
# =============================================================================

def charge_radius_from_q(q):
    factor = 0.90 + 0.20 * abs(float(q))
    factor = max(CHARGE_RADIUS_MIN_FACTOR, min(CHARGE_RADIUS_MAX_FACTOR, factor))
    return CHARGE_RADIUS * factor


def create_charge(
    index,
    charge_data,
    materials,
    charge_collection,
    charge_label_collection,
):
    position = Vector(charge_data["pos"])
    q = float(charge_data["q"])
    is_positive = q >= 0.0
    sign_name = "Positive" if is_positive else "Negative"
    material = materials["Positive"] if is_positive else materials["Negative"]
    radius = charge_radius_from_q(q)
    name = f"Charge_{index:02d}_{sign_name}"

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=max(12, int(CHARGE_SEGMENTS)),
        ring_count=max(8, int(CHARGE_RINGS)),
        radius=radius,
        location=position,
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_Mesh"
    assign_material(obj, material)
    set_mesh_smooth(obj, True)
    move_to_collection(obj, charge_collection)
    obj["charge_q"] = q
    obj["position_xyz"] = tuple(float(value) for value in position)

    # Put the sign slightly above the visible sphere surface, facing the same
    # default presentation viewpoint as the axis labels.
    toward_view = Vector(LABEL_VIEWPOINT) - position
    if toward_view.length <= 1e-8:
        toward_view = Vector((0.0, -1.0, 1.0))
    toward_view.normalize()
    symbol_location = position + toward_view * (radius + CHARGE_SYMBOL_GAP)
    symbol = create_text_mesh(
        body="+" if is_positive else "-",
        name=f"ChargeSymbol_{index:02d}_{sign_name}",
        location=symbol_location,
        size=CHARGE_SYMBOL_SIZE,
        material=materials["Symbol"],
        collection=charge_label_collection,
    )

    # Parenting keeps the sign attached if the user moves/rotates the charge.
    symbol.parent = obj
    symbol.matrix_parent_inverse = obj.matrix_world.inverted()
    return obj, radius


def create_projection_line(index, charge_obj, material, collection):
    """Create an optional solid or dashed vertical projection plus plane marker."""
    x, y, z = charge_obj.location
    start_z = min(float(PROJECTION_START_Z), z)
    end_z = z

    if end_z - start_z > 1e-6:
        if PROJECTION_DASHED:
            cursor = start_z
            dash_index = 1
            dash_length = max(0.01, float(PROJECTION_DASH_LENGTH))
            gap_length = max(0.0, float(PROJECTION_GAP_LENGTH))
            while cursor < end_z - 1e-6:
                dash_end = min(cursor + dash_length, end_z)
                create_cylinder_between(
                    name=f"Projection_{index:02d}_Dash_{dash_index:02d}",
                    point_a=(x, y, cursor),
                    point_b=(x, y, dash_end),
                    radius=PROJECTION_RADIUS,
                    material=material,
                    collection=collection,
                    vertices=10,
                )
                cursor = dash_end + gap_length
                dash_index += 1
        else:
            create_cylinder_between(
                name=f"Projection_{index:02d}",
                point_a=(x, y, start_z),
                point_b=(x, y, end_z),
                radius=PROJECTION_RADIUS,
                material=material,
                collection=collection,
                vertices=10,
            )

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20,
        radius=PROJECTION_MARKER_RADIUS,
        depth=PROJECTION_MARKER_HEIGHT,
        end_fill_type='NGON',
        location=(x, y, PROJECTION_START_Z),
    )
    marker = bpy.context.object
    marker.name = f"Projection_Marker_{index:02d}"
    marker.data.name = f"Projection_Marker_{index:02d}_Mesh"
    assign_material(marker, material)
    set_cylinder_side_smooth(marker)
    move_to_collection(marker, collection)
    return marker


def create_all_charges(materials, collections):
    created = []
    for index, charge_data in enumerate(CHARGES, start=1):
        if "pos" not in charge_data or "q" not in charge_data:
            raise ValueError(f"CHARGES[{index - 1}] must contain 'pos' and 'q'")
        if len(charge_data["pos"]) != 3:
            raise ValueError(f"CHARGES[{index - 1}]['pos'] must have three values")

        charge_obj, radius = create_charge(
            index=index,
            charge_data=charge_data,
            materials=materials,
            charge_collection=collections["Charges"],
            charge_label_collection=collections["ChargeLabels"],
        )
        created.append((charge_obj, radius))

        if SHOW_PROJECTION_LINES:
            create_projection_line(
                index=index,
                charge_obj=charge_obj,
                material=materials["Projection"],
                collection=collections["ProjectionLines"],
            )
    return created


# =============================================================================
# CAMERA, LIGHT, WORLD, EXPORT
# =============================================================================

def point_object_at(obj, target, track_axis='-Z', up_axis='Y'):
    direction = Vector(target) - obj.location
    if direction.length > 1e-8:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = direction.to_track_quat(track_axis, up_axis)


def create_camera_and_light(collection):
    scene = bpy.context.scene

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    collection.objects.link(camera)
    camera.location = Vector(CAMERA_LOCATION)
    camera_data.lens = 52.0
    camera_data.sensor_width = 36.0
    point_object_at(camera, CAMERA_TARGET)
    scene.camera = camera

    light_data = bpy.data.lights.new("Area_Light", type='AREA')
    light = bpy.data.objects.new("Area_Light", light_data)
    collection.objects.link(light)
    light.location = Vector(LIGHT_LOCATION)
    light_data.energy = LIGHT_ENERGY
    light_data.shape = 'DISK'
    light_data.size = LIGHT_SIZE
    point_object_at(light, (0.0, 0.0, 0.7))

    return camera, light


def configure_scene():
    scene = bpy.context.scene
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        # Compatibility fallback for early Blender 4.x builds.
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.resolution_x = 1200
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.92, 0.93, 0.95, 1.0)
        background.inputs["Strength"].default_value = 0.55

    scene.unit_settings.system = 'NONE'


def export_glb():
    export_path = bpy.path.abspath(EXPORT_PATH)
    if not export_path.lower().endswith(".glb"):
        export_path += ".glb"

    export_directory = os.path.dirname(export_path)
    if export_directory:
        os.makedirs(export_directory, exist_ok=True)

    if not hasattr(bpy.ops.export_scene, "gltf"):
        raise RuntimeError("glTF 2.0 exporter is unavailable in this Blender installation")

    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=False,
        export_materials='EXPORT',
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_yup=True,
        export_apply=True,
    )

    if not os.path.isfile(export_path) or os.path.getsize(export_path) == 0:
        raise RuntimeError(f"GLB export did not produce a valid file: {export_path}")
    print(f"[Charge Model] GLB exported: {export_path}")


# =============================================================================
# VALIDATION AND MAIN
# =============================================================================

def validate_result():
    remaining_text = [obj.name for obj in bpy.context.scene.objects if obj.type == 'FONT']
    if remaining_text:
        raise RuntimeError(f"Unconverted Text objects remain: {remaining_text}")

    required_meshes = {
        "Axis_X", "Axis_Y", "Axis_Z",
        "Arrow_X", "Arrow_Y", "Arrow_Z",
        "Origin", "Observation_Plane", "Grid",
    }
    missing = [
        name for name in sorted(required_meshes)
        if bpy.data.objects.get(name) is None or bpy.data.objects[name].type != 'MESH'
    ]
    if missing:
        raise RuntimeError(f"Required mesh objects are missing: {missing}")

    mesh_count = sum(1 for obj in bpy.context.scene.objects if obj.type == 'MESH')
    print(f"[Charge Model] Validation passed: {mesh_count} mesh objects, "
          f"{len(bpy.data.materials)} materials")


def main():
    clear_scene()
    configure_scene()

    # Outliner organization
    coordinate_system = create_collection("Coordinate_System")
    labels = create_collection("Labels", parent=coordinate_system)
    tick_marks = create_collection("Tick_Marks", parent=coordinate_system)

    observation = create_collection("Observation")
    grid_collection = create_collection("Grid", parent=observation)

    charges = create_collection("Charges")
    charge_labels = create_collection("Charge_Labels", parent=charges)
    projection_lines = create_collection("Projection_Lines", parent=charges)

    scene_setup = create_collection("Scene_Setup")

    collections = {
        "CoordinateSystem": coordinate_system,
        "Labels": labels,
        "TickMarks": tick_marks,
        "Observation": observation,
        "Grid": grid_collection,
        "Charges": charges,
        "ChargeLabels": charge_labels,
        "ProjectionLines": projection_lines,
        "SceneSetup": scene_setup,
    }

    materials = create_all_materials()

    axes = {
        'X': (Vector((1.0, 0.0, 0.0)), materials["AxisX"]),
        'Y': (Vector((0.0, 1.0, 0.0)), materials["AxisY"]),
        'Z': (Vector((0.0, 0.0, 1.0)), materials["AxisZ"]),
    }

    for axis_letter, (direction, material) in axes.items():
        create_axis(axis_letter, direction, material, coordinate_system)
        create_arrow(axis_letter, direction, material, coordinate_system)
        create_ticks_and_labels(
            axis_letter=axis_letter,
            direction=direction,
            axis_material=material,
            text_material=materials["Text"],
            tick_collection=tick_marks,
            label_collection=labels,
        )
        create_axis_label(axis_letter, direction, material, labels)

    create_origin(materials["Origin"], coordinate_system)
    create_observation_plane(materials["Plane"], observation)
    create_grid(materials["Grid"], grid_collection)
    create_all_charges(materials, collections)
    create_camera_and_light(scene_setup)

    validate_result()

    if AUTO_EXPORT:
        export_glb()

    # Leave the scene in a clean, predictable state.
    bpy.ops.object.select_all(action='DESELECT')
    print("[Charge Model] Scene generation complete.")


if __name__ == "__main__":
    main()
