# BlendCraft bpy compatibility exercise script.
# Runs against the embedded RustPython bpy/mathutils subset.

import bpy
from mathutils import Vector, Euler, Quaternion

# Prefer an existing mesh if the scene already has one; otherwise create a cube.
obj = None
if bpy.data.objects:
    for o in bpy.data.objects:
        obj = o
        break
if obj is None:
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    obj = bpy.context.object

obj.name = "BC Compatibility Cube"
obj.location = Vector((1.0, 2.0, 3.0))

# Rotation via Euler -> Quaternion (matches Blender-style flow)
rotation = Euler((0.0, 0.0, 0.25)).to_quaternion()
obj.rotation_quaternion = Quaternion(tuple(rotation))

# Material create / edit / assign
mat = bpy.data.materials.new(name="BC Compatibility Material")
mat.diffuse_color = (0.2, 0.55, 0.9, 1.0)
mat.metallic = 0.3
mat.roughness = 0.4
obj.data.materials.append(mat)

assert bpy.data.materials["BC Compatibility Material"].name == "BC Compatibility Material"
assert abs(obj.rotation_euler.z - 0.25) < 0.0001
assert obj.location.x == 1.0

print("BC Compatibility Material")
print("BlendCraft scripting compatibility test passed.")
