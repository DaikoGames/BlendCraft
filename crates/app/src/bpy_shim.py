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
        list.__init__(self, [float(x) for x in list(seq)[:3]])
        while len(self) < 3:
            self.append(0.0)
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
        return math.sqrt(self[0]*self[0] + self[1]*self[1] + self[2]*self[2])
    def length_squared(self):
        return self[0]*self[0] + self[1]*self[1] + self[2]*self[2]
    def normalized(self):
        L = self.length
        if L < 1e-12:
            return Vector((0.0, 0.0, 0.0))
        return Vector((self[0]/L, self[1]/L, self[2]/L))
    def normalize(self):
        n = self.normalized()
        self[0], self[1], self[2] = n[0], n[1], n[2]
        return self
    def dot(self, other):
        o = list(other)
        return self[0]*float(o[0]) + self[1]*float(o[1]) + self[2]*float(o[2])
    def cross(self, other):
        o = list(other)
        return Vector((self[1]*o[2]-self[2]*o[1], self[2]*o[0]-self[0]*o[2], self[0]*o[1]-self[1]*o[0]))
    def lerp(self, other, t):
        o = list(other)
        t = float(t)
        return Vector((self[0]+(float(o[0])-self[0])*t, self[1]+(float(o[1])-self[1])*t, self[2]+(float(o[2])-self[2])*t))
    def __add__(self, other):
        o = list(other)
        return Vector((self[0]+float(o[0]), self[1]+float(o[1]), self[2]+float(o[2])))
    def __sub__(self, other):
        o = list(other)
        return Vector((self[0]-float(o[0]), self[1]-float(o[1]), self[2]-float(o[2])))
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            s = float(other)
            return Vector((self[0]*s, self[1]*s, self[2]*s))
        o = list(other)
        return Vector((self[0]*float(o[0]), self[1]*float(o[1]), self[2]*float(o[2])))
    def __rmul__(self, other):
        return self.__mul__(other)
    def __neg__(self):
        return Vector((-self[0], -self[1], -self[2]))
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
    def slerp(self, other, factor):
        # Simple normalized lerp fallback (good enough for stubs)
        a = list(self); b = list(other)
        t = float(factor)
        # flip if needed for shortest path
        dot = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
        if dot < 0:
            b = [-x for x in b]
            dot = -dot
        if dot > 0.9995:
            out = [a[i] + t*(b[i]-a[i]) for i in range(4)]
        else:
            th0 = math.acos(max(-1.0, min(1.0, dot)))
            th = th0 * t
            s0 = math.sin(th0)
            s1 = math.sin(th0 - th) / s0
            s2 = math.sin(th) / s0
            out = [s1*a[i] + s2*b[i] for i in range(4)]
        n = math.sqrt(sum(x*x for x in out)) or 1.0
        return Quaternion([x/n for x in out])
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


class Matrix(list):
    """Minimal 4x4 row-major matrix (mathutils.Matrix subset)."""
    def __init__(self, rows=None):
        if rows is None:
            list.__init__(self, [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        else:
            list.__init__(self, [[float(c) for c in row[:4]] for row in list(rows)[:4]])
            while len(self) < 4:
                self.append([0.0, 0.0, 0.0, 0.0])
            for row in self:
                while len(row) < 4:
                    row.append(0.0)
    @staticmethod
    def Identity(size=4):
        return Matrix()
    @staticmethod
    def Translation(v):
        m = Matrix()
        vv = list(v)
        m[0][3] = float(vv[0]); m[1][3] = float(vv[1]); m[2][3] = float(vv[2])
        return m
    def to_translation(self):
        return Vector((self[0][3], self[1][3], self[2][3]))
    def copy(self):
        return Matrix(self)
    def __mul__(self, other):
        if isinstance(other, Matrix):
            out = Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            for i in range(4):
                for j in range(4):
                    s = 0.0
                    for k in range(4):
                        s += self[i][k] * other[k][j]
                    out[i][j] = s
            return out
        # Matrix * Vector (as column, w=1)
        v = list(other)[:3]
        while len(v) < 3:
            v.append(0.0)
        x = self[0][0]*v[0] + self[0][1]*v[1] + self[0][2]*v[2] + self[0][3]
        y = self[1][0]*v[0] + self[1][1]*v[1] + self[1][2]*v[2] + self[1][3]
        z = self[2][0]*v[0] + self[2][1]*v[1] + self[2][2]*v[2] + self[2][3]
        return Vector((x, y, z))
    def inverted(self):
        # Gauss-Jordan invert 4x4
        m = [row[:] + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(self)]
        for col in range(4):
            piv = col
            for r in range(col + 1, 4):
                if abs(m[r][col]) > abs(m[piv][col]):
                    piv = r
            if abs(m[piv][col]) < 1e-12:
                return Matrix()  # singular → identity fallback
            m[col], m[piv] = m[piv], m[col]
            div = m[col][col]
            m[col] = [x / div for x in m[col]]
            for r in range(4):
                if r == col:
                    continue
                fac = m[r][col]
                m[r] = [a - fac * b for a, b in zip(m[r], m[col])]
        return Matrix([row[4:] for row in m])
    def __repr__(self):
        return "Matrix(%r)" % (list(self),)

class Color(list):
    def __init__(self, seq=(0.0, 0.0, 0.0)):
        list.__init__(self, [float(x) for x in list(seq)[:3]])
        while len(self) < 3:
            self.append(0.0)
    @property
    def r(self): return self[0]
    @r.setter
    def r(self, v): self[0] = float(v)
    @property
    def g(self): return self[1]
    @g.setter
    def g(self, v): self[1] = float(v)
    @property
    def b(self): return self[2]
    @b.setter
    def b(self, v): self[2] = float(v)
    def copy(self):
        return Color(self)
    def from_hsv(self, h, s, v):
        h = float(h) % 1.0
        s = float(s); v = float(v)
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0: self[0], self[1], self[2] = v, t, p
        elif i == 1: self[0], self[1], self[2] = q, v, p
        elif i == 2: self[0], self[1], self[2] = p, v, t
        elif i == 3: self[0], self[1], self[2] = p, q, v
        elif i == 4: self[0], self[1], self[2] = t, p, v
        else: self[0], self[1], self[2] = v, p, q
        return self
    @property
    def hsv(self):
        r, g, b = self[0], self[1], self[2]
        mx = max(r, g, b); mn = min(r, g, b)
        df = mx - mn
        if mx == 0:
            s = 0.0
        else:
            s = df / mx
        if df == 0:
            h = 0.0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        else:
            h = (60 * ((r - g) / df) + 240) % 360
        return (h / 360.0, s, mx)
    def __repr__(self):
        return "Color((%s, %s, %s))" % (self[0], self[1], self[2])

mathutils = types.ModuleType("mathutils")
mathutils.Vector = Vector
mathutils.Euler = Euler
mathutils.Quaternion = Quaternion
mathutils.Matrix = Matrix
mathutils.Color = Color
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

class _VertColl(object):
    def __init__(self, obj_idx):
        self._idx = obj_idx
    def __len__(self):
        return len(_BC["objects"][self._idx].get("vertices") or [])
    def __getitem__(self, i):
        verts = _BC["objects"][self._idx].get("vertices") or []
        if isinstance(i, slice):
            return [Vector(v) for v in verts[i]]
        return Vector(verts[i])
    def __iter__(self):
        for v in (_BC["objects"][self._idx].get("vertices") or []):
            yield Vector(v)

class _ObjData(object):
    def __init__(self, obj_idx):
        self._idx = obj_idx
        self.materials = _MatSlots(obj_idx)
    @property
    def vertices(self):
        return _VertColl(self._idx)

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
    def active_material(self):
        m = _BC["objects"][self._idx].get("material")
        if m is None:
            return None
        return _MatProxy(m)
    def select_set(self, state):
        _BC["objects"][self._idx]["selected"] = bool(state)
    @property
    def type(self):
        return _BC["objects"][self._idx].get("type", "MESH")
    @property
    def hide_viewport(self):
        return not _BC["objects"][self._idx].get("visible", True)
    @hide_viewport.setter
    def hide_viewport(self, v):
        _BC["objects"][self._idx]["visible"] = not bool(v)
    @property
    def parent(self):
        p = _BC["objects"][self._idx].get("parent")
        if p is None:
            return None
        if 0 <= p < len(_BC["objects"]):
            return _ObjProxy(p)
        return None
    @parent.setter
    def parent(self, value):
        if value is None:
            _BC["objects"][self._idx]["parent"] = None
        elif isinstance(value, _ObjProxy):
            if value._idx == self._idx:
                raise ValueError("cannot parent object to itself")
            _BC["objects"][self._idx]["parent"] = value._idx
        else:
            raise TypeError("parent must be an object or None")
    @property
    def active_material_index(self):
        return int(_BC["objects"][self._idx].get("active_material_index") or 0)
    @active_material_index.setter
    def active_material_index(self, v):
        _BC["objects"][self._idx]["active_material_index"] = int(v)
        slots = _BC["objects"][self._idx].get("material_slots") or []
        if 0 <= int(v) < len(slots) and slots[int(v)] is not None:
            _BC["objects"][self._idx]["material"] = slots[int(v)]
    def _local_matrix(self):
        o = _BC["objects"][self._idx]
        loc = o["location"]; e = o["rotation_euler"]; s = o["scale"]
        # Rz * Ry * Rx * Scale + Translation (matches XYZ euler)
        cx, sx = math.cos(e[0]), math.sin(e[0])
        cy, sy = math.cos(e[1]), math.sin(e[1])
        cz, sz = math.cos(e[2]), math.sin(e[2])
        # R = Rz Ry Rx
        r00 = cz*cy; r01 = cz*sy*sx - sz*cx; r02 = cz*sy*cx + sz*sx
        r10 = sz*cy; r11 = sz*sy*sx + cz*cx; r12 = sz*sy*cx - cz*sx
        r20 = -sy;   r21 = cy*sx;            r22 = cy*cx
        return Matrix([
            [r00*s[0], r01*s[1], r02*s[2], loc[0]],
            [r10*s[0], r11*s[1], r12*s[2], loc[1]],
            [r20*s[0], r21*s[1], r22*s[2], loc[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
    @property
    def matrix_local(self):
        return self._local_matrix()
    @property
    def matrix_world(self):
        m = self._local_matrix()
        p = _BC["objects"][self._idx].get("parent")
        guard = 0
        while p is not None and 0 <= p < len(_BC["objects"]) and guard < 64:
            pm = _ObjProxy(p)._local_matrix()
            m = pm * m
            p = _BC["objects"][p].get("parent")
            guard += 1
        return m
    @property
    def dimensions(self):
        verts = _BC["objects"][self._idx].get("vertices") or []
        if not verts:
            s = _BC["objects"][self._idx]["scale"]
            return Vector((abs(s[0]), abs(s[1]), abs(s[2])))
        xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
        s = _BC["objects"][self._idx]["scale"]
        return Vector((
            (max(xs)-min(xs))*abs(s[0]),
            (max(ys)-min(ys))*abs(s[1]),
            (max(zs)-min(zs))*abs(s[2]),
        ))
    def keyframe_insert(self, data_path, frame=None, **kw):
        if frame is None:
            frame = int(_BC.get("frame_current") or 1)
        o = _BC["objects"][self._idx]
        if data_path == "location":
            val = list(o["location"])
        elif data_path in ("rotation_euler", "rotation"):
            data_path = "rotation_euler"
            val = list(o["rotation_euler"])
        elif data_path == "scale":
            val = list(o["scale"])
        else:
            raise ValueError("unsupported data_path %r" % data_path)
        if "keyframes" not in _BC:
            _BC["keyframes"] = []
        # replace existing key at same object/frame/path
        kept = []
        for k in _BC["keyframes"]:
            if not (k.get("object") == self._idx and k.get("frame") == int(frame) and k.get("data_path") == data_path):
                kept.append(k)
        kept.append({"object": self._idx, "frame": int(frame), "data_path": data_path, "value": [float(x) for x in val]})
        _BC["keyframes"] = kept
        return True
    @property
    def data(self):
        # For lights/cameras expose energy/color/fov on a tiny data proxy
        o = _BC["objects"][self._idx]
        if o.get("type") == "LIGHT":
            return _LightData(self._idx)
        if o.get("type") == "CAMERA":
            return _CameraData(self._idx)
        return _ObjData(self._idx)

class _LightData(object):
    def __init__(self, obj_idx):
        self._idx = obj_idx
    @property
    def energy(self):
        e = _BC["objects"][self._idx].get("light_energy")
        return 10.0 if e is None else float(e)
    @energy.setter
    def energy(self, v):
        _BC["objects"][self._idx]["light_energy"] = float(v)
    @property
    def color(self):
        c = _BC["objects"][self._idx].get("light_color")
        if c is None:
            return Color((1.0, 1.0, 1.0))
        return Color(c)
    @color.setter
    def color(self, v):
        t = list(v)[:3]
        while len(t) < 3: t.append(1.0)
        _BC["objects"][self._idx]["light_color"] = [float(x) for x in t]
    @property
    def type(self):
        return _BC["objects"][self._idx].get("light_type") or "POINT"
    @property
    def distance(self):
        r = _BC["objects"][self._idx].get("light_range")
        return 25.0 if r is None else float(r)
    @distance.setter
    def distance(self, v):
        _BC["objects"][self._idx]["light_range"] = float(v)
    @property
    def shadow_soft_size(self):
        return float(_BC["objects"][self._idx].get("light_spot_blend") or 0.15)
    @property
    def spot_size(self):
        import math as _m
        deg = _BC["objects"][self._idx].get("light_spot_size")
        if deg is None: deg = 45.0
        return float(deg) * _m.pi / 180.0
    @spot_size.setter
    def spot_size(self, v):
        import math as _m
        _BC["objects"][self._idx]["light_spot_size"] = float(v) * 180.0 / _m.pi
    @property
    def spot_blend(self):
        return float(_BC["objects"][self._idx].get("light_spot_blend") or 0.15)
    @spot_blend.setter
    def spot_blend(self, v):
        _BC["objects"][self._idx]["light_spot_blend"] = float(v)

class _CameraData(object):
    def __init__(self, obj_idx):
        self._idx = obj_idx
    @property
    def lens(self):
        return _BC["objects"][self._idx].get("camera_fov") or 40.0
    @lens.setter
    def lens(self, v):
        _BC["objects"][self._idx]["camera_fov"] = float(v)
    @property
    def angle(self):
        import math as _m
        fov = _BC["objects"][self._idx].get("camera_fov") or 40.0
        return float(fov) * _m.pi / 180.0
    @property
    def type(self):
        return "ORTHO" if _BC["objects"][self._idx].get("camera_ortho") else "PERSP"
    @type.setter
    def type(self, v):
        _BC["objects"][self._idx]["camera_ortho"] = (str(v).upper() == "ORTHO")
    @property
    def ortho_scale(self):
        return float(_BC["objects"][self._idx].get("camera_ortho_size") or 6.0)
    @ortho_scale.setter
    def ortho_scale(self, v):
        _BC["objects"][self._idx]["camera_ortho_size"] = float(v)
    @property
    def clip_start(self):
        return float(_BC["objects"][self._idx].get("camera_near") or 0.1)
    @clip_start.setter
    def clip_start(self, v):
        _BC["objects"][self._idx]["camera_near"] = float(v)
    @property
    def clip_end(self):
        return float(_BC["objects"][self._idx].get("camera_far") or 200.0)
    @clip_end.setter
    def clip_end(self, v):
        _BC["objects"][self._idx]["camera_far"] = float(v)

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

class _CollectionProxy(object):
    def __init__(self, idx):
        self._idx = idx
    @property
    def name(self):
        return _BC["collections"][self._idx]["name"]
    @name.setter
    def name(self, v):
        _BC["collections"][self._idx]["name"] = str(v)
    @property
    def objects(self):
        return _CollectionObjects(self._idx)
    def link(self, obj):
        if not isinstance(obj, _ObjProxy):
            raise TypeError("expected object")
        cols = _BC.setdefault("collections", [])
        c = cols[self._idx]
        objs = c.setdefault("objects", [])
        if obj._idx not in objs:
            objs.append(obj._idx)
        _BC["objects"][obj._idx]["collection"] = self._idx
    def unlink(self, obj):
        if not isinstance(obj, _ObjProxy):
            raise TypeError("expected object")
        c = _BC["collections"][self._idx]
        objs = c.setdefault("objects", [])
        if obj._idx in objs:
            objs.remove(obj._idx)
        if _BC["objects"][obj._idx].get("collection") == self._idx:
            _BC["objects"][obj._idx]["collection"] = None

class _CollectionObjects(object):
    def __init__(self, col_idx):
        self._col = col_idx
    def __iter__(self):
        for i in _BC["collections"][self._col].get("objects") or []:
            if 0 <= i < len(_BC["objects"]):
                yield _ObjProxy(i)
    def __len__(self):
        return len(_BC["collections"][self._col].get("objects") or [])

class _Collections(object):
    def new(self, name="Collection"):
        if "collections" not in _BC:
            _BC["collections"] = []
        existing = {c["name"] for c in _BC["collections"]}
        base = name; i = 1
        while name in existing:
            name = "%s.%03d" % (base, i); i += 1
        _BC["collections"].append({"id": None, "name": name, "visible": True, "objects": []})
        return _CollectionProxy(len(_BC["collections"]) - 1)
    def __getitem__(self, key):
        cols = _BC.get("collections") or []
        if isinstance(key, int):
            return _CollectionProxy(key)
        for i, c in enumerate(cols):
            if c["name"] == key:
                return _CollectionProxy(i)
        raise KeyError(key)
    def __iter__(self):
        for i in range(len(_BC.get("collections") or [])):
            yield _CollectionProxy(i)
    def __len__(self):
        return len(_BC.get("collections") or [])

class _Data(object):
    objects = _Objects()
    materials = _Materials()
    collections = _Collections()

class _Scene(object):
    @property
    def objects(self):
        return _Objects()
    @property
    def frame_current(self):
        return int(_BC.get("frame_current") or 1)
    @frame_current.setter
    def frame_current(self, v):
        _BC["frame_current"] = int(v)

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
        "visible": True,
        "parent": None,
        "light_type": None,
        "light_energy": None,
        "light_color": None,
        "camera_fov": None,
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
    def primitive_circle_add(self, radius=1.0, vertices=32, location=None, rotation=None, **kw):
        return _add_mesh("Circle", "Circle", [radius, vertices], location, rotation)
    def primitive_grid_add(self, size=2.0, x_subdivisions=10, y_subdivisions=10, location=None, rotation=None, **kw):
        return _add_mesh("Grid", "Grid", [size, x_subdivisions, y_subdivisions], location, rotation)
    def primitive_capsule_add(self, radius=0.5, depth=2.0, vertices=16, location=None, rotation=None, **kw):
        return _add_mesh("Capsule", "Capsule", [radius, depth, vertices, 8], location, rotation)
    def primitive_pyramid_add(self, size=2.0, depth=2.0, location=None, rotation=None, **kw):
        return _add_mesh("Pyramid", "Pyramid", [size, depth], location, rotation)

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
        else:
            # TOGGLE
            if any(o.get("selected") for o in _BC["objects"]):
                for o in _BC["objects"]:
                    o["selected"] = False
            else:
                for o in _BC["objects"]:
                    o["selected"] = True
        return {"FINISHED"}
    def delete(self, **kw):
        keep = [o for o in _BC["objects"] if not o.get("selected")]
        old_name = None
        a = _BC.get("active")
        if a is not None and 0 <= a < len(_BC["objects"]) and not _BC["objects"][a].get("selected"):
            old_name = _BC["objects"][a]["name"]
        # Fix parent indices after deletion
        old_to_new = {}
        ni = 0
        for oi, o in enumerate(_BC["objects"]):
            if not o.get("selected"):
                old_to_new[oi] = ni
                ni += 1
        for o in keep:
            p = o.get("parent")
            if p is not None:
                o["parent"] = old_to_new.get(p)
        _BC["objects"] = keep
        _BC["active"] = None
        if old_name:
            for i, o in enumerate(_BC["objects"]):
                if o["name"] == old_name:
                    _BC["active"] = i
                    break
        return {"FINISHED"}
    def duplicate(self, **kw):
        src = [dict(o) for o in _BC["objects"] if o.get("selected")]
        if not src:
            return {"CANCELLED"}
        _deselect_all()
        for o in src:
            d = dict(o)
            d["id"] = None
            d["name"] = _unique_name(o["name"])
            d["selected"] = True
            # parent index still points at original parent (unchanged)
            _BC["objects"].append(d)
            _BC["active"] = len(_BC["objects"]) - 1
        return {"FINISHED"}
    def empty_add(self, type="PLAIN_AXES", location=None, rotation=None, **kw):
        _deselect_all()
        loc = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0]
        if location is not None:
            loc = [float(x) for x in list(location)[:3]]
            while len(loc) < 3: loc.append(0.0)
        if rotation is not None:
            rot = [float(x) for x in list(rotation)[:3]]
            while len(rot) < 3: rot.append(0.0)
        shape_map = {
            "PLAIN_AXES": "Plain Axes",
            "ARROWS": "Arrows",
            "SINGLE_ARROW": "Single Arrow",
            "CIRCLE": "Circle",
            "CUBE": "Cube",
            "SPHERE": "Sphere",
            "CONE": "Cone",
        }
        shape = shape_map.get(str(type).upper(), "Plain Axes")
        name = _unique_name("Empty")
        _BC["objects"].append({
            "id": None,
            "name": name,
            "type": "EMPTY",
            "primitive": "",
            "primitive_params": [],
            "material": None,
            "material_slots": [],
            "active_material_index": 0,
            "location": loc,
            "rotation_euler": rot,
            "scale": [1.0, 1.0, 1.0],
            "selected": True,
            "visible": True,
            "parent": None,
            "collection": None,
            "light_type": None,
            "light_energy": None,
            "light_color": None,
            "light_range": None,
            "light_spot_size": None,
            "light_spot_blend": None,
            "camera_fov": None,
            "camera_ortho": None,
            "camera_ortho_size": None,
            "camera_near": None,
            "camera_far": None,
            "empty_shape": shape,
            "empty_size": 1.0,
            "vertices": [],
        })
        _BC["active"] = len(_BC["objects"]) - 1
        return {"FINISHED"}
    def light_add(self, type="POINT", location=None, rotation=None, **kw):
        _deselect_all()
        loc = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0]
        if location is not None:
            loc = [float(x) for x in list(location)[:3]]
            while len(loc) < 3: loc.append(0.0)
        if rotation is not None:
            rot = [float(x) for x in list(rotation)[:3]]
            while len(rot) < 3: rot.append(0.0)
        lt = str(type).upper()
        if lt not in ("POINT", "SUN", "SPOT", "AREA"):
            lt = "POINT"
        name = _unique_name(lt.title() if lt != "POINT" else "Point")
        _BC["objects"].append({
            "id": None,
            "name": name,
            "type": "LIGHT",
            "primitive": "",
            "primitive_params": [],
            "material": None,
            "location": loc,
            "rotation_euler": rot,
            "scale": [1.0, 1.0, 1.0],
            "selected": True,
            "visible": True,
            "parent": None,
            "light_type": lt,
            "light_energy": None,
            "light_color": None,
            "camera_fov": None,
        })
        _BC["active"] = len(_BC["objects"]) - 1
        return {"FINISHED"}
    def camera_add(self, location=None, rotation=None, **kw):
        _deselect_all()
        loc = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0]
        if location is not None:
            loc = [float(x) for x in list(location)[:3]]
            while len(loc) < 3: loc.append(0.0)
        if rotation is not None:
            rot = [float(x) for x in list(rotation)[:3]]
            while len(rot) < 3: rot.append(0.0)
        name = _unique_name("Camera")
        _BC["objects"].append({
            "id": None,
            "name": name,
            "type": "CAMERA",
            "primitive": "",
            "primitive_params": [],
            "material": None,
            "location": loc,
            "rotation_euler": rot,
            "scale": [1.0, 1.0, 1.0],
            "selected": True,
            "visible": True,
            "parent": None,
            "light_type": None,
            "light_energy": None,
            "light_color": None,
            "camera_fov": 40.0,
        })
        _BC["active"] = len(_BC["objects"]) - 1
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
