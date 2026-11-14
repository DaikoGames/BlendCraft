# BlendCraft pure-Python bpy / mathutils compatibility layer for RustPython.
# Mutates global _BC dict; Rust loads the snapshot into _BC and reads it back.
# Avoid `import io` — RustPython's pure-Python stdlib (io, etc.) needs the
# frozen pylib; a tiny StringIO stand-in keeps the shim self-contained.
import math, sys, types
# No json/io imports — avoids LookupError: unknown encoding: latin1 on
# embedded RustPython when codec tables are incomplete.

class _StringIO(object):
    def __init__(self):
        self._buf = []
    def write(self, s):
        self._buf.append(str(s))
        return len(str(s))
    def getvalue(self):
        return "".join(self._buf)
    def flush(self):
        pass

_stdout = _StringIO()
_stderr = _StringIO()
sys.stdout = _stdout
sys.stderr = _stderr

def _unique_name(base):
    names = {o["name"] for o in _BC["objects"]}
    if base not in names:
        return base
    i = 1
    while True:
        n = "%s.%03d" % (base, i)
        if n not in names:
            return n
        i += 1

def _deselect_all():
    for o in _BC["objects"]:
        o["selected"] = False

class Vector(list):
    def __init__(self, seq=(0.0, 0.0, 0.0)):
        seq = list(seq)[:3]
        while len(seq) < 3:
            seq.append(0.0)
        list.__init__(self, [float(x) for x in seq])
    @property
    def x(self): return self[0]
    @x.setter
    def x(self, v): self[0] = float(v)
    @property
    def y(self): return self[1]
    @y.setter
    def y(self, v): self[1] = float(v)
    @property
    def z(self): return self[2]
    @z.setter
    def z(self, v): self[2] = float(v)
    @property
    def length(self):
        return math.sqrt(self[0]**2 + self[1]**2 + self[2]**2)
    def cross(self, other):
        a, b = self, Vector(other)
        return Vector((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]))
    def __repr__(self):
        return "Vector((%s, %s, %s))" % (self[0], self[1], self[2])
    def copy(self):
        return Vector(self)

class Euler(list):
    def __init__(self, seq=(0.0, 0.0, 0.0), order="XYZ"):
        seq = list(seq)[:3]
        while len(seq) < 3:
            seq.append(0.0)
        list.__init__(self, [float(x) for x in seq])
        self.order = order
    @property
    def x(self): return self[0]
    @x.setter
    def x(self, v): self[0] = float(v)
    @property
    def y(self): return self[1]
    @y.setter
    def y(self, v): self[1] = float(v)
    @property
    def z(self): return self[2]
    @z.setter
    def z(self, v): self[2] = float(v)
    def to_quaternion(self):
        x, y, z = self[0]*0.5, self[1]*0.5, self[2]*0.5
        cx, sx = math.cos(x), math.sin(x)
        cy, sy = math.cos(y), math.sin(y)
        cz, sz = math.cos(z), math.sin(z)
        w = cx*cy*cz + sx*sy*sz
        qx = sx*cy*cz - cx*sy*sz
        qy = cx*sy*cz + sx*cy*sz
        qz = cx*cy*sz - sx*sy*cz
        return Quaternion((w, qx, qy, qz))
    def __repr__(self):
        return "Euler((%s, %s, %s))" % (self[0], self[1], self[2])

class Quaternion(list):
    def __init__(self, seq=(1.0, 0.0, 0.0, 0.0)):
        seq = list(seq)
        if len(seq) == 3:
            seq = [1.0] + seq
        while len(seq) < 4:
            seq.append(0.0)
        list.__init__(self, [float(x) for x in seq[:4]])
    @property
    def w(self): return self[0]
    @property
    def x(self): return self[1]
    @property
    def y(self): return self[2]
    @property
    def z(self): return self[3]
    def to_euler(self, order="XYZ"):
        w, x, y, z = self[0], self[1], self[2], self[3]
        sinr_cosp = 2*(w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2*(w*y - z*x)
        pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return Euler((roll, pitch, yaw), order)
    def __repr__(self):
        return "Quaternion((%s, %s, %s, %s))" % (self[0], self[1], self[2], self[3])

mathutils = types.ModuleType("mathutils")
mathutils.Vector = Vector
mathutils.Euler = Euler
mathutils.Quaternion = Quaternion
sys.modules["mathutils"] = mathutils

class _MatProxy(object):
    def __init__(self, idx):
        self._idx = idx
    @property
    def name(self):
        return _BC["materials"][self._idx]["name"]
    @name.setter
    def name(self, v):
        _BC["materials"][self._idx]["name"] = str(v)
    @property
    def diffuse_color(self):
        return tuple(_BC["materials"][self._idx]["diffuse_color"])
    @diffuse_color.setter
    def diffuse_color(self, v):
        t = list(v)[:4]
        while len(t) < 4: t.append(1.0)
        _BC["materials"][self._idx]["diffuse_color"] = [float(x) for x in t]
    @property
    def metallic(self):
        return _BC["materials"][self._idx]["metallic"]
    @metallic.setter
    def metallic(self, v):
        _BC["materials"][self._idx]["metallic"] = float(v)
    @property
    def roughness(self):
        return _BC["materials"][self._idx]["roughness"]
    @roughness.setter
    def roughness(self, v):
        _BC["materials"][self._idx]["roughness"] = float(v)

class _MatSlots(object):
    def __init__(self, obj_idx):
        self._obj_idx = obj_idx
    def append(self, mat):
        if isinstance(mat, _MatProxy):
            _BC["objects"][self._obj_idx]["material"] = mat._idx
        else:
            name = mat if isinstance(mat, str) else getattr(mat, "name", str(mat))
            for i, m in enumerate(_BC["materials"]):
                if m["name"] == name:
                    _BC["objects"][self._obj_idx]["material"] = i
                    return
            raise KeyError(name)

class _ObjData(object):
    def __init__(self, obj_idx):
        self.materials = _MatSlots(obj_idx)

class _LocProxy(object):
    def __init__(self, obj_idx):
        self._i = obj_idx
    def _arr(self):
        return _BC["objects"][self._i]["location"]
    def __getitem__(self, i):
        return self._arr()[i]
    def __setitem__(self, i, v):
        self._arr()[i] = float(v)
    @property
    def x(self): return self._arr()[0]
    @x.setter
    def x(self, v): self._arr()[0] = float(v)
    @property
    def y(self): return self._arr()[1]
    @y.setter
    def y(self, v): self._arr()[1] = float(v)
    @property
    def z(self): return self._arr()[2]
    @z.setter
    def z(self, v): self._arr()[2] = float(v)
    def __iter__(self):
        return iter(self._arr())
    def __repr__(self):
        a = self._arr()
        return "Vector((%s, %s, %s))" % (a[0], a[1], a[2])

class _EulerProxy(object):
    def __init__(self, obj_idx):
        self._i = obj_idx
    def _arr(self):
        return _BC["objects"][self._i]["rotation_euler"]
    def __getitem__(self, i):
        return self._arr()[i]
    def __setitem__(self, i, v):
        self._arr()[i] = float(v)
    @property
    def x(self): return self._arr()[0]
    @x.setter
    def x(self, v): self._arr()[0] = float(v)
    @property
    def y(self): return self._arr()[1]
    @y.setter
    def y(self, v): self._arr()[1] = float(v)
    @property
    def z(self): return self._arr()[2]
    @z.setter
    def z(self, v): self._arr()[2] = float(v)
    def __iter__(self):
        return iter(self._arr())

class _ObjProxy(object):
    def __init__(self, idx):
        self._idx = idx
    @property
    def name(self):
        return _BC["objects"][self._idx]["name"]
    @name.setter
    def name(self, v):
        _BC["objects"][self._idx]["name"] = str(v)
    @property
    def location(self):
        return _LocProxy(self._idx)
    @location.setter
    def location(self, value):
        if hasattr(value, "__iter__"):
            t = list(value)[:3]
        else:
            t = [value.x, value.y, value.z]
        while len(t) < 3: t.append(0.0)
        _BC["objects"][self._idx]["location"] = [float(x) for x in t]
    @property
    def rotation_euler(self):
        return _EulerProxy(self._idx)
    @rotation_euler.setter
    def rotation_euler(self, value):
        if hasattr(value, "__iter__"):
            t = list(value)[:3]
        else:
            t = [value.x, value.y, value.z]
        while len(t) < 3: t.append(0.0)
        _BC["objects"][self._idx]["rotation_euler"] = [float(x) for x in t]
    @property
    def rotation_quaternion(self):
        e = _BC["objects"][self._idx]["rotation_euler"]
        return Euler(e).to_quaternion()
    @rotation_quaternion.setter
    def rotation_quaternion(self, value):
        if isinstance(value, Quaternion):
            q = value
        elif hasattr(value, "__iter__"):
            q = Quaternion(value)
        else:
            q = Quaternion((value.w, value.x, value.y, value.z))
        e = q.to_euler()
        _BC["objects"][self._idx]["rotation_euler"] = [e[0], e[1], e[2]]
    @property
    def scale(self):
        return Vector(_BC["objects"][self._idx]["scale"])
    @scale.setter
    def scale(self, value):
        t = list(value)[:3]
        while len(t) < 3: t.append(1.0)
        _BC["objects"][self._idx]["scale"] = [float(x) for x in t]
    @property
    def data(self):
        return _ObjData(self._idx)
    @property
    def active_material(self):
        m = _BC["objects"][self._idx].get("material")
        if m is None:
            return None
        return _MatProxy(m)
    def select_set(self, state):
        _BC["objects"][self._idx]["selected"] = bool(state)

class _Objects(object):
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0: key += len(_BC["objects"])
            if key < 0 or key >= len(_BC["objects"]):
                raise IndexError(key)
            return _ObjProxy(key)
        for i, o in enumerate(_BC["objects"]):
            if o["name"] == key:
                return _ObjProxy(i)
        raise KeyError(key)
    def __iter__(self):
        for i in range(len(_BC["objects"])):
            yield _ObjProxy(i)
    def __len__(self):
        return len(_BC["objects"])
    def __contains__(self, name):
        return any(o["name"] == name for o in _BC["objects"])
    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

class _Materials(object):
    def new(self, name="Material"):
        existing = {m["name"] for m in _BC["materials"]}
        base = name
        i = 1
        while name in existing:
            name = "%s.%03d" % (base, i)
            i += 1
        _BC["materials"].append({
            "name": name,
            "diffuse_color": [0.8, 0.8, 0.8, 1.0],
            "metallic": 0.0,
            "roughness": 0.5,
            "emissive": [0.0, 0.0, 0.0],
        })
        return _MatProxy(len(_BC["materials"]) - 1)
    def __getitem__(self, key):
        if isinstance(key, int):
            return _MatProxy(key)
        for i, m in enumerate(_BC["materials"]):
            if m["name"] == key:
                return _MatProxy(i)
        raise KeyError(key)
    def __len__(self):
        return len(_BC["materials"])
    def __iter__(self):
        for i in range(len(_BC["materials"])):
            yield _MatProxy(i)

class _Data(object):
    objects = _Objects()
    materials = _Materials()

class _Scene(object):
    @property
    def objects(self):
        return _Objects()

class _ViewLayerObjects(object):
    def __iter__(self):
        for i, o in enumerate(_BC["objects"]):
            if o.get("selected"):
                yield _ObjProxy(i)

class _ViewLayer(object):
    @property
    def objects(self):
        return _ViewLayerObjects()

class _Context(object):
    @property
    def object(self):
        a = _BC.get("active")
        if a is None or a < 0 or a >= len(_BC["objects"]):
            return None
        return _ObjProxy(a)
    @property
    def active_object(self):
        return self.object
    @property
    def scene(self):
        return _Scene()
    @property
    def view_layer(self):
        return _ViewLayer()
    @property
    def selected_objects(self):
        return [_ObjProxy(i) for i, o in enumerate(_BC["objects"]) if o.get("selected")]

def _add_mesh(name, prim, params, location=None, rotation=None):
    _deselect_all()
    loc = [0.0, 0.0, 0.0]
    rot = [0.0, 0.0, 0.0]
    if location is not None:
        loc = [float(x) for x in list(location)[:3]]
        while len(loc) < 3: loc.append(0.0)
    if rotation is not None:
        rot = [float(x) for x in list(rotation)[:3]]
        while len(rot) < 3: rot.append(0.0)
    name = _unique_name(name)
    _BC["objects"].append({
        "id": None,
        "name": name,
        "type": "MESH",
        "primitive": prim,
        "primitive_params": [float(x) for x in params],
        "material": None,
        "location": loc,
        "rotation_euler": rot,
        "scale": [1.0, 1.0, 1.0],
        "selected": True,
    })
    _BC["active"] = len(_BC["objects"]) - 1
    return {"FINISHED"}

class _MeshOps(object):
    def primitive_cube_add(self, size=2.0, location=None, rotation=None, **kw):
        return _add_mesh("Cube", "Cube", [size], location, rotation)
    def primitive_plane_add(self, size=2.0, location=None, rotation=None, **kw):
        return _add_mesh("Plane", "Plane", [size], location, rotation)
    def primitive_uv_sphere_add(self, radius=1.0, segments=32, ring_count=16, location=None, rotation=None, **kw):
        return _add_mesh("Sphere", "Sphere", [radius, segments, ring_count], location, rotation)
    def primitive_ico_sphere_add(self, radius=1.0, subdivisions=2, location=None, rotation=None, **kw):
        return _add_mesh("IcoSphere", "IcoSphere", [radius, subdivisions], location, rotation)
    def primitive_cylinder_add(self, radius=1.0, depth=2.0, vertices=32, location=None, rotation=None, **kw):
        return _add_mesh("Cylinder", "Cylinder", [radius, depth, vertices], location, rotation)
    def primitive_cone_add(self, radius=1.0, depth=2.0, vertices=32, location=None, rotation=None, **kw):
        return _add_mesh("Cone", "Cone", [radius, depth, vertices], location, rotation)
    def primitive_torus_add(self, major_radius=1.0, minor_radius=0.25, major_segments=48, minor_segments=12, location=None, rotation=None, **kw):
        return _add_mesh("Torus", "Torus", [major_radius, minor_radius, major_segments, minor_segments], location, rotation)
    def primitive_monkey_add(self, size=2.0, location=None, rotation=None, **kw):
        return _add_mesh("Monkey", "Monkey", [size], location, rotation)

class _ObjectOps(object):
    def select_all(self, action="TOGGLE"):
        if action == "SELECT":
            for o in _BC["objects"]:
                o["selected"] = True
        elif action == "DESELECT":
            for o in _BC["objects"]:
                o["selected"] = False
        elif action == "INVERT":
            for o in _BC["objects"]:
                o["selected"] = not o["selected"]
        return {"FINISHED"}
    def delete(self, **kw):
        keep = [o for o in _BC["objects"] if not o.get("selected")]
        old_name = None
        a = _BC.get("active")
        if a is not None and 0 <= a < len(_BC["objects"]) and not _BC["objects"][a].get("selected"):
            old_name = _BC["objects"][a]["name"]
        _BC["objects"] = keep
        _BC["active"] = None
        if old_name:
            for i, o in enumerate(_BC["objects"]):
                if o["name"] == old_name:
                    _BC["active"] = i
                    break
        return {"FINISHED"}

class _Ops(object):
    mesh = _MeshOps()
    object = _ObjectOps()

class _BpyModule(object):
    data = _Data()
    context = _Context()
    ops = _Ops()

bpy = _BpyModule()
sys.modules["bpy"] = bpy
