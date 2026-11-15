//! Alembic (.abc) import and export, written from scratch in pure Rust (Ogawa container).
//!
//! Supported: PolyMesh / SubD objects (positions, face counts, face indices; topology may
//! change per sample, which is what liquid surface caches need), Xform hierarchies
//! (matrix and TRS operators, animated), time sampling. Everything else (cameras, curves,
//! points, normals, UVs, arbitrary geometry parameters) is skipped on import.
//!
//! Property headers use the official AbcCoreOgawa bit layout (size hint, POD, extent,
//! metadata index) so files written here open in Blender and other Alembic 1.7/1.8 tools.
//! The Ogawa version bytes are the official `{0x00, 0x01}`. The reader remains tolerant of
//! minor layout variations from other writers.
use anyhow::{anyhow, bail, Result};
use forge_core::{scene_at_frame, MeshData, MeshSequence, Object, Scene, TransformKey, Vertex};
use glam::{Mat4, Vec3};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::Arc;

const TOP: u64 = 0x8000_0000_0000_0000;

/// Writer-side bit layout of the 32-bit property info word.
/// Matches official Alembic AbcCoreOgawa / Blender:
///   bits 0-1   property kind (0=compound, 1=scalar, 2=array)
///   bits 2-3   size hint for following integers (0=u8, 1=u16, 2=u32) — smallest that fits
///   bits 4-7   POD type
///   bit  8     has time-sampling index (0x100)
///   bit  9     has first/last changed indices (0x200)
///   bits 12-19 extent
///   bits 20-27 metadata index (0xFF = inline metadata string follows)
mod layout {
    pub const POD_SHIFT: u32 = 4;
    pub const HAS_TS: u32 = 0x100;
    pub const HAS_FIRST_LAST: u32 = 0x200;
    pub const EXTENT_SHIFT: u32 = 12;
    pub const META_SHIFT: u32 = 20;
    pub const META_INLINE: u32 = 0xFF;
}

// Plain-old-data type codes of the format.
const POD_BOOL: u8 = 0;
const POD_U8: u8 = 1;
const POD_U32: u8 = 5;
const POD_I32: u8 = 6;
const POD_F32: u8 = 10;
const POD_F64: u8 = 11;

// ---------------------------------------------------------------------------
// Ogawa reader
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Child {
    Group(u64),
    Data(u64),
}

struct Ogawa<'a> {
    buf: &'a [u8],
    root: u64,
}

impl<'a> Ogawa<'a> {
    fn open(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < 16 || &buf[0..5] != b"Ogawa" {
            bail!("not an Ogawa Alembic file (HDF5-based .abc files from old Alembic versions are not supported)");
        }
        if buf[5] != 0xFF {
            bail!("the Alembic archive was not closed properly (it is incomplete)");
        }
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&buf[8..16]);
        Ok(Self { buf, root: u64::from_le_bytes(raw) })
    }

    fn u64_at(&self, pos: u64) -> Result<u64> {
        let p = pos as usize;
        let end = p.checked_add(8).ok_or_else(|| anyhow!("corrupt Alembic offset"))?;
        if end > self.buf.len() {
            bail!("Alembic offset points outside the file");
        }
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&self.buf[p..end]);
        Ok(u64::from_le_bytes(raw))
    }

    fn num_children(&self, group: u64) -> Result<usize> {
        if group == 0 {
            return Ok(0);
        }
        let n = self.u64_at(group)? as usize;
        if n.checked_mul(8).map_or(true, |b| b > self.buf.len()) {
            bail!("corrupt Alembic group size");
        }
        Ok(n)
    }

    fn child(&self, group: u64, index: usize) -> Result<Child> {
        let v = self.u64_at(group + 8 + 8 * index as u64)?;
        Ok(if v & TOP != 0 { Child::Data(v & !TOP) } else { Child::Group(v) })
    }

    fn data(&self, pos: u64) -> Result<&'a [u8]> {
        if pos == 0 {
            return Ok(&[]);
        }
        let size = self.u64_at(pos)? as usize;
        let start = pos as usize + 8;
        let end = start.checked_add(size).ok_or_else(|| anyhow!("corrupt Alembic data size"))?;
        if end > self.buf.len() {
            bail!("Alembic data block points outside the file");
        }
        Ok(&self.buf[start..end])
    }

    fn child_data(&self, group: u64, index: usize) -> Result<&'a [u8]> {
        match self.child(group, index)? {
            Child::Data(p) => self.data(p),
            Child::Group(_) => bail!("expected a data block in the Alembic file"),
        }
    }

    fn child_group(&self, group: u64, index: usize) -> Result<u64> {
        match self.child(group, index)? {
            Child::Group(p) => Ok(p),
            Child::Data(_) => bail!("expected a group in the Alembic file"),
        }
    }
}

fn rd_u32(b: &[u8], pos: usize) -> Option<u32> {
    let end = pos.checked_add(4)?;
    if end > b.len() {
        return None;
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&b[pos..end]);
    Some(u32::from_le_bytes(raw))
}

fn rd_f64(b: &[u8], pos: usize) -> Option<f64> {
    let end = pos.checked_add(8)?;
    if end > b.len() {
        return None;
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&b[pos..end]);
    Some(f64::from_le_bytes(raw))
}

fn printable(bytes: &[u8]) -> Option<String> {
    if bytes.iter().all(|c| (0x20..0x7F).contains(c)) {
        Some(String::from_utf8_lossy(bytes).to_string())
    } else {
        None
    }
}

/// Read a 1/2/4-byte unsigned integer according to the Alembic size hint
/// (bits 2-3 of the property info word). Blender and the official library
/// use the compact widths; assuming 4 bytes always was the main reason
/// Blender .abc files failed to import.
fn hinted(buf: &[u8], hint: u32, at: &mut usize) -> Option<u32> {
    let (value, advance) = match hint {
        0 => (*buf.get(*at)? as u32, 1usize),
        1 => {
            let end = at.checked_add(2)?;
            if end > buf.len() {
                return None;
            }
            let mut raw = [0u8; 2];
            raw.copy_from_slice(&buf[*at..end]);
            (u16::from_le_bytes(raw) as u32, 2)
        }
        _ => {
            let end = at.checked_add(4)?;
            if end > buf.len() {
                return None;
            }
            let mut raw = [0u8; 4];
            raw.copy_from_slice(&buf[*at..end]);
            (u32::from_le_bytes(raw), 4)
        }
    };
    *at += advance;
    Some(value)
}

// ---------------------------------------------------------------------------
// Time sampling
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TimeSampling {
    per_cycle: f64,
    times: Vec<f64>,
}

impl TimeSampling {
    fn identity() -> Self {
        Self { per_cycle: 1.0, times: vec![0.0] }
    }

    fn time_of(&self, i: usize) -> f64 {
        let k = self.times.len().max(1);
        self.times.get(i % k).copied().unwrap_or(0.0) + (i / k) as f64 * self.per_cycle
    }

    /// Largest sample index whose time is not after `t` (clamped to the sample count).
    fn index_for_time(&self, t: f64, count: usize) -> usize {
        if count <= 1 {
            return 0;
        }
        let k = self.times.len().max(1);
        let first = self.times.first().copied().unwrap_or(0.0);
        let cycle = if self.per_cycle > 0.0 { ((t - first) / self.per_cycle + 1e-6).floor().max(0.0) as usize } else { 0 };
        let mut best = cycle * k;
        for j in 0..k {
            if self.time_of(cycle * k + j) <= t + 1e-6 {
                best = cycle * k + j;
            }
        }
        best.min(count - 1)
    }
}

fn parse_time_samplings(buf: &[u8]) -> Vec<TimeSampling> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 16 <= buf.len() {
        let Some(per_cycle) = rd_f64(buf, pos + 4) else { break };
        let Some(n) = rd_u32(buf, pos + 12) else { break };
        let n = n as usize;
        pos += 16;
        if n == 0 || pos + n * 8 > buf.len() {
            break;
        }
        let times: Vec<f64> = (0..n).filter_map(|i| rd_f64(buf, pos + i * 8)).collect();
        pos += n * 8;
        out.push(TimeSampling { per_cycle, times });
    }
    if out.is_empty() {
        out.push(TimeSampling::identity());
    }
    out
}

// ---------------------------------------------------------------------------
// Headers (objects and properties)
// ---------------------------------------------------------------------------

fn parse_object_headers_at(b: &[u8], indexed: &[String]) -> Option<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= b.len() {
        let Some(name_len) = rd_u32(b, pos).map(|v| v as usize) else { break };
        pos += 4;
        // Zero-length name usually means padding / end of real headers.
        if name_len == 0 {
            break;
        }
        // Absurd lengths are corruption or trailing hash data; stop rather than panic.
        if name_len > 4096 || pos + name_len > b.len() {
            break;
        }
        let name_bytes = &b[pos..pos + name_len];
        pos += name_len;
        // Prefer strict printable names, but fall back to lossy UTF-8 so Blender
        // objects with unusual characters still appear in the hierarchy.
        let name = match printable(name_bytes) {
            Some(n) => n,
            None => String::from_utf8_lossy(name_bytes).into_owned(),
        };
        let Some(&index_byte) = b.get(pos) else { break };
        pos += 1;
        let index = index_byte as usize;
        let md = if index == 0xFF {
            let Some(ml) = rd_u32(b, pos).map(|v| v as usize) else { break };
            pos += 4;
            if ml > 65_536 || pos + ml > b.len() {
                break;
            }
            let meta_bytes = &b[pos..pos + ml];
            pos += ml;
            match printable(meta_bytes) {
                Some(s) => s,
                None => String::from_utf8_lossy(meta_bytes).into_owned(),
            }
        } else if index == 0 {
            String::new()
        } else {
            // Indexed metadata: 1-based index into the archive's string table.
            // Out-of-range is not fatal — keep an empty schema string.
            indexed.get(index.wrapping_sub(1)).cloned().unwrap_or_default()
        };
        out.push((name, md));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn parse_object_headers(buf: &[u8], indexed: &[String]) -> Option<Vec<(String, String)>> {
    // Object header blocks end with 32 bytes of hashes (official Alembic).
    // Also try 16 and 0 in case a writer omits or shortens them.
    for trailer in [32usize, 16, 8, 0] {
        if buf.len() > trailer {
            if let Some(v) = parse_object_headers_at(&buf[..buf.len() - trailer], indexed) {
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
    }
    // Last resort: parse the whole buffer; partial results are better than failure.
    parse_object_headers_at(buf, indexed)
}

#[derive(Clone, Debug)]
struct PropHeader {
    name: String,
    ptype: u8, // 0 compound, 1 scalar, 2 array
    ts: u32,
}

/// Parse a packed property-header list using the official Alembic bit layout
/// and variable-width integers (the size hint in bits 2-3).
///
/// Layout of the 32-bit info word (see Alembic AbcCoreOgawa and threers):
///   bits 0-1   property kind (0=compound, 1=scalar, 2=array)
///   bits 2-3   size hint for following integers (0=u8, 1=u16, 2=u32)
///   bits 4-7   POD type
///   bit  8     isHomogenous / related flags
///   bit  9     has first/last changed indices
///   bits 12-19 extent
///   bits 20-27 metadata index (0xFF = inline metadata string follows)
///
/// After the info word come (for non-compound): sample_count, optional
/// first/last changed, optional time-sampling index, then the name length
/// and the name bytes, then optional inline metadata.
fn parse_prop_headers(buf: &[u8], nts: usize) -> Option<Vec<PropHeader>> {
    // Object / property header blocks often end with 16 or 32 bytes of hashes.
    for trailer in [0usize, 16, 32] {
        if buf.len() <= trailer {
            continue;
        }
        let data = &buf[..buf.len() - trailer];
        if let Some(headers) = parse_prop_headers_body(data, nts) {
            if !headers.is_empty() {
                return Some(headers);
            }
        }
    }
    None
}

fn parse_prop_headers_body(buf: &[u8], nts: usize) -> Option<Vec<PropHeader>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 4 <= buf.len() {
        let info = rd_u32(buf, at)?;
        at += 4;
        let ptype = (info & 3) as u8;
        if ptype > 2 {
            break;
        }
        let hint = (info & 0x0c) >> 2;
        let mut ts: u32 = 0;

        if ptype != 0 {
            // sample count (always present for scalar/array)
            let _sample_count = hinted(buf, hint, &mut at)?;
            // first / last changed indices
            if info & 0x200 != 0 {
                let _ = hinted(buf, hint, &mut at)?;
                let _ = hinted(buf, hint, &mut at)?;
            }
            // time sampling index (present when the corresponding flag is set)
            if info & 0x100 != 0 {
                if let Some(idx) = hinted(buf, hint, &mut at) {
                    if (idx as usize) < nts {
                        ts = idx;
                    }
                } else {
                    break;
                }
            }
        }

        // name length + name
        let name_len = hinted(buf, hint, &mut at)? as usize;
        if name_len == 0 || name_len > 512 || at + name_len > buf.len() {
            break;
        }
        let name_bytes = &buf[at..at + name_len];
        at += name_len;
        let name = match printable(name_bytes) {
            Some(n) => n,
            None => String::from_utf8_lossy(name_bytes).into_owned(),
        };

        // inline metadata when the metadata index field is 0xFF
        if (info & 0xff0_0000) >> 20 == 0xff {
            let meta_len = hinted(buf, hint, &mut at)? as usize;
            if at + meta_len > buf.len() {
                break;
            }
            at += meta_len;
        }

        out.push(PropHeader { name, ptype, ts });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

struct Compound {
    group: u64,
    headers: Vec<PropHeader>,
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

struct Archive<'a> {
    og: Ogawa<'a>,
    times: Vec<TimeSampling>,
    indexed: Vec<String>,
}

#[derive(Clone)]
struct ObjRef {
    name: String,
    md: String,
    group: u64,
}

impl<'a> Archive<'a> {
    fn open(buf: &'a [u8]) -> Result<Self> {
        let og = Ogawa::open(buf)?;
        let n = og.num_children(og.root)?;
        if n < 4 {
            bail!("the Alembic archive has an unexpected structure ({n} root entries)");
        }
        let times = if n > 4 { parse_time_samplings(og.child_data(og.root, 4)?) } else { vec![TimeSampling::identity()] };
        let mut indexed = Vec::new();
        if n > 5 {
            let d = og.child_data(og.root, 5)?;
            let mut pos = 0usize;
            while pos < d.len() {
                let len = d[pos] as usize;
                pos += 1;
                if pos + len > d.len() {
                    break;
                }
                indexed.push(String::from_utf8_lossy(&d[pos..pos + len]).to_string());
                pos += len;
            }
        }
        Ok(Self { og, times, indexed })
    }

    fn ts(&self, index: u32) -> &TimeSampling {
        self.times.get(index as usize).unwrap_or(&self.times[0])
    }

    fn top_object(&self) -> Result<ObjRef> {
        Ok(ObjRef { name: "ABC".to_string(), md: String::new(), group: self.og.child_group(self.og.root, 2)? })
    }

    fn children(&self, obj: &ObjRef) -> Result<Vec<ObjRef>> {
        let n = self.og.num_children(obj.group)?;
        if n < 2 {
            return Ok(Vec::new());
        }
        let Child::Data(pos) = self.og.child(obj.group, n - 1)? else { return Ok(Vec::new()) };
        // Tolerate objects whose child-header block we cannot fully decode
        // (common with some Blender fluid / Mantaflow exports). An empty list
        // lets the walk continue under other branches instead of aborting the
        // whole import with "could not read the child object list of 'Liquid'".
        let headers = match parse_object_headers(self.og.data(pos)?, &self.indexed) {
            Some(h) => h,
            None => return Ok(Vec::new()),
        };
        let mut out = Vec::new();
        for (i, (name, md)) in headers.into_iter().enumerate() {
            if i + 1 >= n - 1 {
                break;
            }
            // Skip corrupt child group references rather than failing the import.
            if let Ok(group) = self.og.child_group(obj.group, i + 1) {
                out.push(ObjRef { name, md, group });
            }
        }
        Ok(out)
    }

    fn compound(&self, group: u64) -> Result<Compound> {
        let n = self.og.num_children(group)?;
        if n == 0 {
            return Ok(Compound { group, headers: Vec::new() });
        }
        let buf = self.og.child_data(group, n - 1)?;
        let mut headers = parse_prop_headers(buf, self.times.len())
            .ok_or_else(|| anyhow!("could not decode the Alembic property headers (unsupported file layout)"))?;
        // Some writers leave trailing padding or extra hash data; keep only as
        // many headers as there are property children so name→index stays valid.
        let expected = n.saturating_sub(1);
        if headers.len() > expected {
            headers.truncate(expected);
        }
        // If we got fewer headers than children, still proceed — the missing
        // ones simply cannot be looked up by name.
        Ok(Compound { group, headers })
    }

    fn object_props(&self, obj: &ObjRef) -> Result<Compound> {
        if self.og.num_children(obj.group)? == 0 {
            return Ok(Compound { group: 0, headers: Vec::new() });
        }
        let g = self.og.child_group(obj.group, 0)?;
        self.compound(g)
    }

    fn prop(&self, c: &Compound, name: &str) -> Option<(u64, PropHeader)> {
        let i = c.headers.iter().position(|h| h.name == name)?;
        let g = self.og.child_group(c.group, i).ok()?;
        Some((g, c.headers[i].clone()))
    }

    fn sub_compound(&self, c: &Compound, name: &str) -> Option<Compound> {
        let (g, h) = self.prop(c, name)?;
        if h.ptype != 0 {
            return None;
        }
        self.compound(g).ok()
    }

    fn sample_count(&self, group: u64, h: &PropHeader) -> usize {
        let n = self.og.num_children(group).unwrap_or(0);
        if h.ptype == 2 {
            n / 2
        } else {
            n
        }
    }

    fn sample_bytes(&self, group: u64, h: &PropHeader, s: usize) -> Result<&'a [u8]> {
        let index = if h.ptype == 2 { 2 * s } else { s };
        let d = self.og.child_data(group, index)?;
        if d.len() < 16 {
            bail!("Alembic sample is too short");
        }
        Ok(&d[16..])
    }

    fn array_dim(&self, group: u64, s: usize) -> Option<u64> {
        let d = self.og.child_data(group, 2 * s + 1).ok()?;
        if d.len() >= 8 {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&d[..8]);
            Some(u64::from_le_bytes(raw))
        } else {
            None
        }
    }

    fn read_vec3_array(&self, group: u64, h: &PropHeader, s: usize) -> Result<Vec<Vec3>> {
        let bytes = self.sample_bytes(group, h, s)?;
        let count = self.array_dim(group, s);
        let wide = matches!(count, Some(n) if n as usize * 24 == bytes.len() && n as usize * 12 != bytes.len());
        let mut out = Vec::new();
        if wide {
            for c in bytes.chunks_exact(24) {
                let f = |o: usize| {
                    let mut raw = [0u8; 8];
                    raw.copy_from_slice(&c[o..o + 8]);
                    f64::from_le_bytes(raw) as f32
                };
                out.push(Vec3::new(f(0), f(8), f(16)));
            }
        } else {
            for c in bytes.chunks_exact(12) {
                let f = |o: usize| {
                    let mut raw = [0u8; 4];
                    raw.copy_from_slice(&c[o..o + 4]);
                    f32::from_le_bytes(raw)
                };
                out.push(Vec3::new(f(0), f(4), f(8)));
            }
        }
        Ok(out)
    }

    fn read_i32_array(&self, group: u64, h: &PropHeader, s: usize) -> Result<Vec<i32>> {
        let bytes = self.sample_bytes(group, h, s)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| {
                let mut raw = [0u8; 4];
                raw.copy_from_slice(c);
                i32::from_le_bytes(raw)
            })
            .collect())
    }

    fn read_f64s(&self, group: u64, h: &PropHeader, s: usize) -> Result<Vec<f64>> {
        let bytes = self.sample_bytes(group, h, s)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|c| {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(c);
                f64::from_le_bytes(raw)
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

struct XformReader {
    ops: Vec<(u8, usize)>,
    vals_group: u64,
    vals_header: PropHeader,
    count: usize,
    inherits: bool,
}

impl XformReader {
    fn load(ar: &Archive, obj: &ObjRef) -> Option<XformReader> {
        let props = ar.object_props(obj).ok()?;
        let xf = ar.sub_compound(&props, ".xform")?;
        let (ops_g, ops_h) = ar.prop(&xf, ".ops")?;
        let ops_bytes = ar.sample_bytes(ops_g, &ops_h, 0).ok()?;
        let ops: Vec<(u8, usize)> = ops_bytes
            .iter()
            .map(|b| {
                let ty = b >> 4;
                let n = match ty {
                    0 | 1 => 3,
                    2 => 4,
                    3 => 16,
                    4 | 5 | 6 => 1,
                    _ => 0,
                };
                (ty, n)
            })
            .collect();
        let (vals_group, vals_header) = ar.prop(&xf, ".vals")?;
        let count = ar.sample_count(vals_group, &vals_header);
        let inherits = match ar.prop(&xf, ".inherits") {
            Some((g, h)) => ar.sample_bytes(g, &h, 0).ok().and_then(|b| b.first().copied()).map_or(true, |v| v != 0),
            None => true,
        };
        Some(XformReader { ops, vals_group, vals_header, count, inherits })
    }

    fn matrix_at(&self, ar: &Archive, t: f64) -> Mat4 {
        let s = if self.count > 1 { ar.ts(self.vals_header.ts).index_for_time(t, self.count) } else { 0 };
        let Ok(vals) = ar.read_f64s(self.vals_group, &self.vals_header, s) else { return Mat4::IDENTITY };
        let needed: usize = self.ops.iter().map(|o| o.1).sum();
        if vals.len() < needed {
            return Mat4::IDENTITY;
        }
        let mut m = Mat4::IDENTITY;
        let mut at = 0usize;
        for (ty, n) in &self.ops {
            let v = &vals[at..at + n];
            at += n;
            let op = match *ty {
                0 => Mat4::from_scale(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)),
                1 => Mat4::from_translation(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)),
                2 => {
                    let axis = Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
                    if axis.length_squared() > 1e-12 {
                        Mat4::from_axis_angle(axis.normalize(), (v[3] as f32).to_radians())
                    } else {
                        Mat4::IDENTITY
                    }
                }
                3 => {
                    // Alembic stores matrix ops row-major; glam is column-major.
                    let mut a = [0.0f32; 16];
                    for (i, x) in v.iter().take(16).enumerate() {
                        a[i] = *x as f32;
                    }
                    Mat4::from_cols_array(&a).transpose()
                }
                4 => Mat4::from_rotation_x((v[0] as f32).to_radians()),
                5 => Mat4::from_rotation_y((v[0] as f32).to_radians()),
                6 => Mat4::from_rotation_z((v[0] as f32).to_radians()),
                _ => Mat4::IDENTITY,
            };
            m = m * op;
        }
        m
    }
}

fn world_at(ar: &Archive, chain: &[XformReader], t: f64) -> Mat4 {
    let mut m = Mat4::IDENTITY;
    for x in chain.iter().rev() {
        m = x.matrix_at(ar, t) * m;
        if !x.inherits {
            break;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

pub struct ImportedObject {
    pub object: Object,
    /// Per-frame meshes when the geometry is animated.
    pub sequence: Option<MeshSequence>,
    /// World-transform keys when the transform is animated.
    pub keys: Vec<TransformKey>,
}

pub struct AlembicImport {
    pub objects: Vec<ImportedObject>,
    pub start_frame: i32,
    pub end_frame: i32,
    pub notes: Vec<String>,
}

struct Ctx {
    fps: f32,
    scale: f32,
    objects: Vec<ImportedObject>,
    notes: Vec<String>,
    min_frame: i32,
    max_frame: i32,
    skipped: usize,
}

fn triangulate(counts: &[i32], indices: &[i32], vertex_count: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut at = 0usize;
    for &c in counts {
        let c = c.max(0) as usize;
        if at + c > indices.len() {
            break;
        }
        let poly = &indices[at..at + c];
        at += c;
        if c < 3 || poly.iter().any(|&i| i < 0 || i as usize >= vertex_count) {
            continue;
        }
        // Alembic polygons are stored clockwise: reverse to the renderer's winding.
        for k in 1..c - 1 {
            out.push(poly[0] as u32);
            out.push(poly[k + 1] as u32);
            out.push(poly[k] as u32);
        }
    }
    out
}

fn walk(ar: &Archive, obj: &ObjRef, chain: &mut Vec<XformReader>, parent_name: Option<&str>, ctx: &mut Ctx) -> Result<()> {
    // Schema strings from Blender / Alembic 1.7–1.8 usually contain these tokens.
    // Also accept a few common variants seen in fluid / Mantaflow caches.
    let md_lower = obj.md.to_ascii_lowercase();
    let is_mesh = obj.md.contains("PolyMesh")
        || obj.md.contains("SubD")
        || md_lower.contains("polymesh")
        || md_lower.contains("abcgeom_polymesh")
        || md_lower.contains("abcgeom_subd");
    let is_xform = obj.md.contains("Xform") || md_lower.contains("abcgeom_xform");

    // Prefer treating as Xform when both flags could match (rare).
    if is_xform {
        if let Some(x) = XformReader::load(ar, obj) {
            chain.push(x);
            for child in ar.children(obj)? {
                walk(ar, &child, chain, Some(obj.name.as_str()), ctx)?;
            }
            chain.pop();
            // Some writers put geometry properties on the same object as the xform.
            // If this node also looks like a mesh, still try to import it.
            if is_mesh {
                let _ = import_mesh(ar, obj, chain, parent_name, ctx);
            }
            return Ok(());
        }
    }
    if is_mesh {
        // import_mesh can fail on a single object; don't abort the whole archive.
        if let Err(e) = import_mesh(ar, obj, chain, parent_name, ctx) {
            ctx.notes.push(format!("skipped mesh '{}': {e}", obj.name));
        }
        return Ok(());
    }
    if !obj.md.is_empty() && !is_xform {
        ctx.skipped += 1;
    }
    for child in ar.children(obj)? {
        walk(ar, &child, chain, parent_name, ctx)?;
    }
    Ok(())
}

fn import_mesh(ar: &Archive, obj: &ObjRef, chain: &[XformReader], parent_name: Option<&str>, ctx: &mut Ctx) -> Result<()> {
    let props = ar.object_props(obj)?;
    let geom = ar
        .sub_compound(&props, ".geom")
        .ok_or_else(|| anyhow!("mesh '{}' has no geometry data", obj.name))?;
    let (gp, hp) = ar.prop(&geom, "P").ok_or_else(|| anyhow!("mesh '{}' has no positions", obj.name))?;
    let (gc, hc) = ar.prop(&geom, ".faceCounts").ok_or_else(|| anyhow!("mesh '{}' has no face counts", obj.name))?;
    let (gi, hi) = ar.prop(&geom, ".faceIndices").ok_or_else(|| anyhow!("mesh '{}' has no face indices", obj.name))?;
    let np = ar.sample_count(gp, &hp);
    let nc = ar.sample_count(gc, &hc);
    let ni = ar.sample_count(gi, &hi);
    if np == 0 || nc == 0 || ni == 0 {
        bail!("mesh '{}' has no samples", obj.name);
    }
    let scale = ctx.scale;
    let ts_mesh = ar.ts(hp.ts);

    let mut frames: Vec<Arc<MeshData>> = Vec::with_capacity(np);
    let mut shared_indices: Option<Vec<u32>> = None;
    for s in 0..np {
        let positions = ar.read_vec3_array(gp, &hp, s)?;
        let indices = if nc == 1 && ni == 1 {
            if shared_indices.is_none() {
                let counts = ar.read_i32_array(gc, &hc, 0)?;
                let idx = ar.read_i32_array(gi, &hi, 0)?;
                shared_indices = Some(triangulate(&counts, &idx, positions.len()));
            }
            shared_indices.clone().unwrap_or_default()
        } else {
            let counts = ar.read_i32_array(gc, &hc, s.min(nc - 1))?;
            let idx = ar.read_i32_array(gi, &hi, s.min(ni - 1))?;
            triangulate(&counts, &idx, positions.len())
        };
        let mut mesh = MeshData { name: obj.name.clone(), ..Default::default() };
        mesh.vertices = positions
            .iter()
            .map(|p| Vertex { position: (*p * scale).to_array(), normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0] })
            .collect();
        mesh.indices = indices;
        mesh.compute_normals();
        frames.push(Arc::new(mesh));
    }

    // Sample times used for the transform keys.
    let mut times: Vec<f64> = (0..np).map(|s| ts_mesh.time_of(s)).collect();
    if np == 1 {
        let mut best: Option<Vec<f64>> = None;
        for x in chain.iter() {
            if x.count > 1 {
                let ts = ar.ts(x.vals_header.ts);
                let t: Vec<f64> = (0..x.count).map(|i| ts.time_of(i)).collect();
                if best.as_ref().map_or(true, |b| t.len() > b.len()) {
                    best = Some(t);
                }
            }
        }
        if let Some(b) = best {
            times = b;
        }
    }
    let matrices: Vec<Mat4> = times.iter().map(|t| world_at(ar, chain, *t)).collect();
    let first = matrices[0];
    let animated = matrices.iter().any(|m| {
        m.to_cols_array().iter().zip(first.to_cols_array().iter()).any(|(a, b)| (a - b).abs() > 1e-5)
    });

    let fps = ctx.fps as f64;
    let name = match parent_name {
        Some(p) => p.to_string(),
        None => obj.name.clone(),
    };
    let mut object = Object::new(name, MeshData::default());
    let (s0, r0, t0) = first.to_scale_rotation_translation();
    object.scale = s0;
    object.rotation = r0;
    object.translation = t0 * scale;
    object.mesh = frames[0].clone();

    let mut keys = Vec::new();
    if animated {
        for (t, m) in times.iter().zip(matrices.iter()) {
            let (s, r, tr) = m.to_scale_rotation_translation();
            let frame = (*t * fps).round() as i32;
            keys.push(TransformKey { frame, location: Some(tr * scale), rotation: Some(r), scale: Some(s) });
            ctx.min_frame = ctx.min_frame.min(frame);
            ctx.max_frame = ctx.max_frame.max(frame);
        }
    }

    let sequence = if np > 1 {
        let t_first = ts_mesh.time_of(0);
        let t_second = ts_mesh.time_of(1);
        let cache_fps = if t_second > t_first { (1.0 / (t_second - t_first)) as f32 } else { ctx.fps };
        let start = (t_first * fps) as f32;
        let end = (ts_mesh.time_of(np - 1) * fps).round() as i32;
        ctx.min_frame = ctx.min_frame.min(start.round() as i32);
        ctx.max_frame = ctx.max_frame.max(end);
        Some(MeshSequence::new(format!("Alembic: {}", obj.name), frames, start, cache_fps))
    } else {
        None
    };
    ctx.objects.push(ImportedObject { object, sequence, keys });
    Ok(())
}

pub fn parse_alembic_bytes(bytes: &[u8], scene_fps: f32, scale: f32) -> Result<AlembicImport> {
    let ar = Archive::open(bytes)?;
    let top = ar.top_object()?;
    let mut ctx = Ctx {
        fps: if scene_fps > 0.0 { scene_fps } else { 24.0 },
        scale: if scale > 0.0 { scale } else { 1.0 },
        objects: Vec::new(),
        notes: Vec::new(),
        min_frame: i32::MAX,
        max_frame: i32::MIN,
        skipped: 0,
    };
    let mut chain: Vec<XformReader> = Vec::new();
    for child in ar.children(&top)? {
        walk(&ar, &child, &mut chain, None, &mut ctx)?;
    }
    if ctx.objects.is_empty() {
        bail!("the Alembic file contains no PolyMesh/SubD objects that BlendCraft can import (check that the file is Ogawa, not HDF5, and that meshes use the standard .geom/P/.faceCounts/.faceIndices properties)");
    }
    if ctx.skipped > 0 {
        ctx.notes.push(format!("{} unsupported object(s) skipped (cameras, curves, points, ...)", ctx.skipped));
    }
    let (start, end) = if ctx.min_frame <= ctx.max_frame { (ctx.min_frame, ctx.max_frame) } else { (1, 1) };
    Ok(AlembicImport { objects: ctx.objects, start_frame: start, end_frame: end, notes: ctx.notes })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_alembic(path: &Path, scene_fps: f32, scale: f32) -> Result<AlembicImport> {
    let bytes = std::fs::read(path).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
    parse_alembic_bytes(&bytes, scene_fps, scale)
}

#[cfg(target_arch = "wasm32")]
pub fn load_alembic(_path: &Path, _scene_fps: f32, _scale: f32) -> Result<AlembicImport> {
    bail!("Alembic import from a path is not available on WebAssembly")
}

// ---------------------------------------------------------------------------
// Ogawa writer
// ---------------------------------------------------------------------------

enum Node {
    Group(Vec<Node>),
    Data(Vec<u8>),
}

fn write_node(buf: &mut Vec<u8>, node: &Node) -> u64 {
    match node {
        Node::Data(d) => {
            if d.is_empty() {
                return TOP;
            }
            let pos = buf.len() as u64;
            buf.extend_from_slice(&(d.len() as u64).to_le_bytes());
            buf.extend_from_slice(d);
            pos | TOP
        }
        Node::Group(children) => {
            if children.is_empty() {
                return 0;
            }
            let refs: Vec<u64> = children.iter().map(|c| write_node(buf, c)).collect();
            let pos = buf.len() as u64;
            buf.extend_from_slice(&(refs.len() as u64).to_le_bytes());
            for r in refs {
                buf.extend_from_slice(&r.to_le_bytes());
            }
            pos
        }
    }
}

fn sample_key(_bytes: &[u8], _dims: &[u8], _pod: u8, _extent: u8) -> [u8; 16] {
    // Official Alembic uses SpookyHash here for sample identity / dedup. Readers do
    // not verify the digest against the payload when opening, so zeros are safe and
    // avoid any risk of a bad custom hash implementation confusing tools.
    [0u8; 16]
}

enum PropKind {
    Compound(Vec<Prop>),
    Scalar { pod: u8, extent: u8, ts: u32, samples: Vec<Vec<u8>> },
    Array { pod: u8, extent: u8, ts: u32, samples: Vec<(Vec<u8>, u64)> },
}

struct Prop {
    name: String,
    md: String,
    kind: PropKind,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Table of indexed metadata strings (archive child 5). Index 0 is the empty metadata, the
/// strings are stored 1-based, exactly like Blender/libAlembic files do.
struct MdTable {
    entries: Vec<String>,
}

impl MdTable {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }
    fn index(&mut self, md: &str) -> u32 {
        if md.is_empty() {
            return 0;
        }
        if let Some(i) = self.entries.iter().position(|e| e == md) {
            return i as u32 + 1;
        }
        self.entries.push(md.to_string());
        self.entries.len() as u32
    }
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for e in &self.entries {
            out.push(e.len().min(255) as u8);
            out.extend_from_slice(&e.as_bytes()[..e.len().min(255)]);
        }
        out
    }
}

/// Integer width used for the numbers that follow the info word (size hint, bits 2-3):
/// 0 = one byte, 1 = two bytes, 2 = four bytes. Like the real writer, the smallest width
/// that fits every number of the header is used.
fn hint_for(max_value: usize) -> u32 {
    if max_value <= 0xFF {
        0
    } else if max_value <= 0xFFFF {
        1
    } else {
        2
    }
}

fn put_hinted(out: &mut Vec<u8>, hint: u32, v: u32) {
    match hint {
        0 => out.push(v as u8),
        1 => out.extend_from_slice(&(v as u16).to_le_bytes()),
        _ => out.extend_from_slice(&v.to_le_bytes()),
    }
}

fn write_prop_header(out: &mut Vec<u8>, p: &Prop, _md: &mut MdTable) {
    // Always use size-hint 2 (u32) so every following integer is 4 bytes. This matches
    // the common path in official writers and avoids width mismatches that crash Blender.
    let hint: u32 = 2;
    let name = p.name.as_bytes();
    // Always embed metadata inline (index 0xFF) when present. Indexed metadata is
    // optional and easy to get wrong; inline is what the official writer falls back to.
    let inline = !p.md.is_empty();
    let meta_idx: u32 = if inline { 0xFF } else { 0 };

    match &p.kind {
        PropKind::Compound(_) => {
            let info = (hint << 2) | (meta_idx << layout::META_SHIFT);
            out.extend_from_slice(&info.to_le_bytes());
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name);
            if inline {
                out.extend_from_slice(&(p.md.len() as u32).to_le_bytes());
                out.extend_from_slice(p.md.as_bytes());
            }
        }
        PropKind::Scalar { pod, extent, ts, samples } => {
            // Header sample count MUST equal the number of children written.
            write_value_header(out, 1, *pod, *extent, *ts, samples.len(), name, meta_idx, false);
            if inline {
                out.extend_from_slice(&(p.md.len() as u32).to_le_bytes());
                out.extend_from_slice(p.md.as_bytes());
            }
        }
        PropKind::Array { pod, extent, ts, samples } => {
            let homo = samples.len() <= 1 || samples.windows(2).all(|w| w[0].1 == w[1].1);
            write_value_header(out, 2, *pod, *extent, *ts, samples.len(), name, meta_idx, homo);
            if inline {
                out.extend_from_slice(&(p.md.len() as u32).to_le_bytes());
                out.extend_from_slice(p.md.as_bytes());
            }
        }
    }
}

/// Header of a scalar (kind 1) or array (kind 2) property.
/// Official rules (AbcCoreOgawa WritePropertyInfo):
/// - 0x800 only when the property is constant (we use it for a single sample)
/// - 0x200 only for a non-default first/last range (we never need it)
/// - 0x400 when array sample counts are homogenous
/// Never set 0x800 and 0x200 together.
fn write_value_header(
    out: &mut Vec<u8>,
    kind: u32,
    pod: u8,
    extent: u8,
    ts: u32,
    count: usize,
    name: &[u8],
    meta_idx: u32,
    homogenous: bool,
) {
    let hint: u32 = 2;
    let mut info = kind
        | (hint << 2)
        | ((pod as u32) << layout::POD_SHIFT)
        | ((extent as u32) << layout::EXTENT_SHIFT)
        | (meta_idx << layout::META_SHIFT);
    if ts != 0 {
        info |= layout::HAS_TS;
    }
    if homogenous {
        info |= 0x400;
    }
    if count <= 1 {
        info |= 0x800;
    }
    out.extend_from_slice(&info.to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    if ts != 0 {
        out.extend_from_slice(&ts.to_le_bytes());
    }
    out.extend_from_slice(&(name.len() as u32).to_le_bytes());
    out.extend_from_slice(name);
}

fn compound_node(children: &[Prop], md: &mut MdTable) -> Node {
    if children.is_empty() {
        return Node::Group(Vec::new());
    }
    let mut nodes: Vec<Node> = children.iter().map(|c| prop_node(c, md)).collect();
    let mut headers = Vec::new();
    for c in children {
        write_prop_header(&mut headers, c, md);
    }
    // Verified against a Blender-written file: property header blocks have no trailing hash.
    nodes.push(Node::Data(headers));
    Node::Group(nodes)
}

fn prop_node(p: &Prop, md: &mut MdTable) -> Node {
    match &p.kind {
        PropKind::Compound(children) => compound_node(children, md),
        PropKind::Scalar { pod, extent, samples, .. } => Node::Group(
            samples
                .iter()
                .map(|s| {
                    let mut d = sample_key(s, &[], *pod, *extent).to_vec();
                    d.extend_from_slice(s);
                    Node::Data(d)
                })
                .collect(),
        ),
        PropKind::Array { pod, extent, samples, .. } => {
            let mut nodes = Vec::new();
            for (bytes, _count) in samples {
                let mut d = sample_key(bytes, &[], *pod, *extent).to_vec();
                d.extend_from_slice(bytes);
                nodes.push(Node::Data(d));
                nodes.push(Node::Data(Vec::new()));
            }
            Node::Group(nodes)
        }
    }
}

struct ObjBuild {
    name: String,
    md: String,
    props: Vec<Prop>,
    children: Vec<ObjBuild>,
}

fn obj_node(o: &ObjBuild, md: &mut MdTable) -> Node {
    let mut nodes = vec![compound_node(&o.props, md)];
    for c in &o.children {
        nodes.push(obj_node(c, md));
    }
    // Real files always end an object group with the child-header block followed by a
    // 32-byte hash, also for objects without children (then it is the hash alone).
    // Object-header metadata: use inline form (0xFF + u32 size + bytes) whenever the
    // schema string is non-empty so we never depend on the archive metadata index table.
    let mut h = Vec::new();
    for c in &o.children {
        h.extend_from_slice(&(c.name.len() as u32).to_le_bytes());
        h.extend_from_slice(c.name.as_bytes());
        if c.md.is_empty() {
            h.push(0);
        } else {
            h.push(0xFF);
            h.extend_from_slice(&(c.md.len() as u32).to_le_bytes());
            h.extend_from_slice(c.md.as_bytes());
        }
    }
    let _ = md; // indexed table kept for compatibility; object headers use inline
    h.extend_from_slice(&[0u8; 32]);
    nodes.push(Node::Data(h));
    Node::Group(nodes)
}

struct TimeSamplingOut {
    max_samples: u32,
    per_cycle: f64,
    times: Vec<f64>,
}

fn build_archive(top: &ObjBuild, samplings: &[TimeSamplingOut]) -> Vec<u8> {
    let mut ts = Vec::new();
    for s in samplings {
        // u32 maxSamples, f64 timePerCycle, u32 numStoredTimes, f64[] storedTimes
        ts.extend_from_slice(&s.max_samples.to_le_bytes());
        ts.extend_from_slice(&s.per_cycle.to_le_bytes());
        ts.extend_from_slice(&(s.times.len() as u32).to_le_bytes());
        for t in &s.times {
            ts.extend_from_slice(&t.to_le_bytes());
        }
    }
    let alembic_version: i32 = 10803; // 1.8.3, the version found in a real Blender file
    let meta = b"FramesPerTimeUnit=24.000000;_ai_AlembicVersion=Alembic 1.8.3 (BlendCraft writer);_ai_Application=BlendCraft".to_vec();
    let mut md = MdTable::new();
    let top_node = obj_node(top, &mut md);
    let root = Node::Group(vec![
        Node::Data(0i32.to_le_bytes().to_vec()),
        Node::Data(alembic_version.to_le_bytes().to_vec()),
        top_node,
        Node::Data(meta),
        Node::Data(ts),
        Node::Data(md.to_bytes()),
    ]);
    let mut buf = vec![0u8; 16];
    let root_pos = write_node(&mut buf, &root);
    // Header: "Ogawa", frozen flag 0xFF, version bytes 00 01, root group position.
    buf[0..5].copy_from_slice(b"Ogawa");
    buf[5] = 0xFF;
    buf[6] = 0x00;
    buf[7] = 0x01;
    buf[8..16].copy_from_slice(&root_pos.to_le_bytes());
    buf
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Returns (positions_bytes, n_points, indices_bytes, n_indices, counts_bytes, n_faces, self_bounds_as_6xf64).
fn mesh_samples(mesh: &MeshData) -> (Vec<u8>, u64, Vec<u8>, u64, Vec<u8>, u64, [f64; 6]) {
    let nvert = mesh.vertices.len();
    let mut p = Vec::with_capacity(nvert * 3);
    let mut bmin = [f32::MAX; 3];
    let mut bmax = [f32::MIN; 3];
    for v in &mesh.vertices {
        // Sanitize non-finite positions so Blender does not crash on NaNs.
        let pos = [
            if v.position[0].is_finite() { v.position[0] } else { 0.0 },
            if v.position[1].is_finite() { v.position[1] } else { 0.0 },
            if v.position[2].is_finite() { v.position[2] } else { 0.0 },
        ];
        p.extend_from_slice(&pos);
        for k in 0..3 {
            bmin[k] = bmin[k].min(pos[k]);
            bmax[k] = bmax[k].max(pos[k]);
        }
    }
    if nvert == 0 {
        bmin = [0.0; 3];
        bmax = [0.0; 3];
    }
    // Drop triangles that reference out-of-range indices (can crash Blender).
    let mut idx: Vec<i32> = Vec::new();
    let mut face_count = 0i32;
    for t in mesh.indices.chunks_exact(3) {
        if (t[0] as usize) < nvert && (t[1] as usize) < nvert && (t[2] as usize) < nvert {
            // Alembic polygons are stored clockwise relative to outward normals.
            idx.extend_from_slice(&[t[0] as i32, t[2] as i32, t[1] as i32]);
            face_count += 1;
        }
    }
    let counts = vec![3i32; face_count as usize];
    let bounds = [
        bmin[0] as f64, bmin[1] as f64, bmin[2] as f64,
        bmax[0] as f64, bmax[1] as f64, bmax[2] as f64,
    ];
    (
        f32_bytes(&p),
        nvert as u64,
        i32_bytes(&idx),
        idx.len() as u64,
        i32_bytes(&counts),
        counts.len() as u64,
        bounds,
    )
}

fn safe_name(raw: &str, used: &mut HashSet<String>) -> String {
    let base: String = raw.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' }).collect();
    let base = if base.is_empty() { "Object".to_string() } else { base };
    let mut name = base.clone();
    let mut n = 1;
    while !used.insert(name.clone()) {
        name = format!("{base}_{n}");
        n += 1;
    }
    name
}

/// Serialize the scene (or the selected objects) to Alembic bytes. Animated transforms,
/// rigid-body bakes and mesh sequences (fluid bakes, imported caches) are sampled over the
/// timeline range; a static scene is written as a single sample.
pub fn build_alembic_bytes(scene: &Scene, only: Option<&BTreeSet<u64>>) -> Result<Vec<u8>> {
    let fps = scene.animation.fps.max(1) as f64;
    let start = scene.animation.frame_start;
    let end = scene.animation.frame_end.max(start);

    let exported: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| {
            let o = &scene.objects[i];
            scene.is_visible(i)
                && o.kind.is_mesh()
                && only.map_or(true, |s| s.contains(&o.id))
                && (!o.mesh.indices.is_empty() || scene.mesh_caches.contains_key(&o.id))
        })
        .collect();
    if exported.is_empty() {
        bail!("nothing to export: no visible mesh objects");
    }

    let any_cache = exported.iter().any(|&i| scene.mesh_caches.contains_key(&scene.objects[i].id));
    let any_motion = !scene.animation.objects.is_empty() || scene.physics_cache.is_some();
    let animated_range = any_cache || any_motion;
    let frames: Vec<i32> = if animated_range { (start..=end).collect() } else { vec![scene.animation.frame_current] };
    let evaluated: Vec<Scene> = frames.iter().map(|f| scene_at_frame(scene, *f)).collect();
    let worlds: Vec<Vec<Mat4>> = evaluated.iter().map(|s| s.world_matrices()).collect();
    let n = frames.len();
    // Sampling 0 = identity, sampling 1 = uniform (one sample every 1/fps seconds starting at
    // the first frame), exactly how Blender writes it. Frame f is at time f / fps.
    let max_samples = n.max(1) as u32;
    let samplings = vec![
        TimeSamplingOut { max_samples, per_cycle: 1.0, times: vec![0.0] },
        TimeSamplingOut { max_samples, per_cycle: 1.0 / fps, times: vec![frames[0] as f64 / fps] },
    ];

    let mut used: HashSet<String> = HashSet::new();
    let mut top = ObjBuild { name: "ABC".into(), md: String::new(), props: Vec::new(), children: Vec::new() };
    for &i in &exported {
        let o = &scene.objects[i];
        let name = safe_name(&o.name, &mut used);
        let mats: Vec<Mat4> = worlds.iter().map(|w| w[i]).collect();
        let first = mats[0].to_cols_array();
        let xform_animated = n > 1 && mats.iter().any(|m| m.to_cols_array().iter().zip(first.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
        // Alembic/Imath matrix ops store a 4×4 in row-major order. glam is column-major,
        // so transpose before serializing.
        let mat16 = |m: Mat4| -> Vec<f64> {
            m.transpose()
                .to_cols_array()
                .iter()
                .map(|v| *v as f64)
                .collect()
        };
        let (xts, xsamples): (u32, Vec<Vec<u8>>) = if xform_animated {
            (1, mats.iter().map(|m| f64_bytes(&mat16(*m))).collect())
        } else {
            (0, vec![f64_bytes(&mat16(mats[0]))])
        };
        // .ops: single matrix op (type 3 << 4 = 0x30). .inherits = true (Blender default).
        let mut xf_children = vec![
            Prop {
                name: ".ops".into(),
                md: String::new(),
                // Stored as a scalar, exactly like a real Blender file does.
                kind: PropKind::Scalar {
                    pod: POD_U8,
                    extent: 1,
                    ts: 0,
                    samples: vec![vec![0x30u8]],
                },
            },
            Prop {
                name: ".vals".into(),
                md: String::new(),
                kind: PropKind::Scalar {
                    pod: POD_F64,
                    extent: 16,
                    ts: xts,
                    samples: xsamples,
                },
            },
            Prop {
                name: ".inherits".into(),
                md: String::new(),
                kind: PropKind::Scalar {
                    pod: POD_BOOL,
                    extent: 1,
                    ts: 0,
                    samples: vec![vec![1u8]],
                },
            },
        ];
        if xform_animated {
            let chans: Vec<u8> = (0u32..16).flat_map(|c| c.to_le_bytes()).collect();
            xf_children.push(Prop {
                name: ".animChans".into(),
                md: String::new(),
                kind: PropKind::Array {
                    pod: POD_U32,
                    extent: 1,
                    ts: 0,
                    samples: vec![(chans, 16)],
                },
            });
        }

        // Same property order as a real Blender file: .inherits, .ops, .vals, (.animChans).
        xf_children.sort_by_key(|p| match p.name.as_str() {
            ".inherits" => 0,
            ".ops" => 1,
            ".vals" => 2,
            _ => 3,
        });

        // Geometry samples.
        let cache = scene.mesh_caches.get(&o.id);
        let geo_animated = cache.map_or(false, |c| c.len() > 1) && n > 1;
        let mesh_for = |k: usize| -> Arc<MeshData> {
            match cache {
                Some(c) => c
                    .mesh_at(frames[k] as f32, scene.animation.fps as f32)
                    .unwrap_or_else(|| o.mesh.clone()),
                None => o.mesh.clone(),
            }
        };
        let count = if geo_animated { n } else { 1 };
        let gts = if geo_animated { 1 } else { 0 };
        let mut p_samples = Vec::new();
        let mut i_samples = Vec::new();
        let mut c_samples = Vec::new();
        let mut bounds_samples: Vec<Vec<u8>> = Vec::new();
        for k in 0..count {
            let m = mesh_for(k);
            let (pb, pn, ib, inn, cb, cn, bounds) = mesh_samples(&m);
            p_samples.push((pb, pn));
            i_samples.push((ib, inn));
            c_samples.push((cb, cn));
            bounds_samples.push(f64_bytes(&bounds));
        }
        // GeomBase requires `.selfBnds` (note spelling — not "Bounds").
        // Box3d = 6 × f64. Blender's IGeomBaseSchema looks up this exact name.
        let geom = Prop {
            name: ".geom".into(),
            md: "schema=AbcGeom_PolyMesh_v1;schemaBaseType=AbcGeom_GeomBase_v1".into(),
            kind: PropKind::Compound(vec![
                Prop {
                    name: ".selfBnds".into(),
                    md: String::new(),
                    kind: PropKind::Scalar {
                        pod: POD_F64,
                        extent: 6,
                        ts: gts,
                        samples: bounds_samples,
                    },
                },
                Prop {
                    name: "P".into(),
                    md: "interpretation=point".into(),
                    kind: PropKind::Array {
                        pod: POD_F32,
                        extent: 3,
                        ts: gts,
                        samples: p_samples,
                    },
                },
                Prop {
                    name: ".faceIndices".into(),
                    md: String::new(),
                    kind: PropKind::Array {
                        pod: POD_I32,
                        extent: 1,
                        ts: gts,
                        samples: i_samples,
                    },
                },
                Prop {
                    name: ".faceCounts".into(),
                    md: String::new(),
                    kind: PropKind::Array {
                        pod: POD_I32,
                        extent: 1,
                        ts: gts,
                        samples: c_samples,
                    },
                },
            ]),
        };
        let shape_name = format!("{name}Shape");
        let shape = ObjBuild {
            name: shape_name,
            // Full schema metadata so Blender recognizes the PolyMesh schema.
            md: "schema=AbcGeom_PolyMesh_v1;schemaBaseType=AbcGeom_GeomBase_v1;schemaObjTitle=AbcGeom_PolyMesh_v1:.geom".into(),
            props: vec![geom],
            children: Vec::new(),
        };
        let xform = ObjBuild {
            name,
            md: "schema=AbcGeom_Xform_v3;schemaObjTitle=AbcGeom_Xform_v3:.xform".into(),
            props: vec![Prop {
                name: ".xform".into(),
                md: "schema=AbcGeom_Xform_v3".into(),
                kind: PropKind::Compound(xf_children),
            }],
            children: vec![shape],
        };
        top.children.push(xform);
    }
    Ok(build_archive(&top, &samplings))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn export_alembic(scene: &Scene, path: &Path, only: Option<&BTreeSet<u64>>) -> Result<()> {
    let bytes = build_alembic_bytes(scene, only)?;
    std::fs::write(path, bytes).map_err(|e| anyhow!("writing {}: {e}", path.display()))
}

#[cfg(target_arch = "wasm32")]
pub fn export_alembic(_scene: &Scene, _path: &Path, _only: Option<&BTreeSet<u64>>) -> Result<()> {
    bail!("Alembic export is not available on WebAssembly")
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::primitives::cube;

    fn cube_scene() -> Scene {
        let mut scene = Scene::default();
        let mut o = Object::new("Water Cube", cube(1.0));
        o.id = scene.alloc_id();
        o.translation = Vec3::new(1.0, 2.0, 3.0);
        scene.objects.push(o);
        scene
    }

    #[test]
    fn static_mesh_roundtrips() {
        let scene = cube_scene();
        let tris = scene.objects[0].mesh.indices.len();
        let verts = scene.objects[0].mesh.vertices.len();
        let bytes = build_alembic_bytes(&scene, None).expect("export");
        let imp = parse_alembic_bytes(&bytes, 24.0, 1.0).expect("import");
        assert_eq!(imp.objects.len(), 1);
        let obj = &imp.objects[0].object;
        assert_eq!(obj.mesh.vertices.len(), verts);
        assert_eq!(obj.mesh.indices.len(), tris);
        assert!((obj.translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
        assert!(imp.objects[0].sequence.is_none());
    }

    #[test]
    fn mesh_sequence_with_changing_topology_roundtrips() {
        let mut scene = cube_scene();
        scene.animation.frame_start = 1;
        scene.animation.frame_end = 3;
        let id = scene.objects[0].id;
        let small = Arc::new(cube(0.5));
        let big = Arc::new(cube(2.0));
        let mut tri = MeshData::default();
        tri.vertices = cube(1.0).vertices[..3].to_vec();
        tri.indices = vec![0, 1, 2];
        let frames = vec![small.clone(), Arc::new(tri), big.clone()];
        scene.mesh_caches.insert(id, MeshSequence::new("test", frames, 1.0, 24.0));
        let bytes = build_alembic_bytes(&scene, None).expect("export");
        let imp = parse_alembic_bytes(&bytes, 24.0, 1.0).expect("import");
        let seq = imp.objects[0].sequence.as_ref().expect("sequence");
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.frames[0].vertices.len(), small.vertices.len());
        assert_eq!(seq.frames[1].indices.len(), 3);
        assert_eq!(seq.frames[2].vertices.len(), big.vertices.len());
        assert_eq!(imp.start_frame, 1);
        assert_eq!(imp.end_frame, 3);
    }

    #[test]
    fn garbage_is_rejected_cleanly() {
        assert!(parse_alembic_bytes(b"not alembic", 24.0, 1.0).is_err());
        let mut truncated = build_alembic_bytes(&cube_scene(), None).unwrap();
        truncated.truncate(40);
        assert!(parse_alembic_bytes(&truncated, 24.0, 1.0).is_err());
    }

    #[test]
    fn property_header_blocks_have_no_trailer_and_leaf_objects_end_with_a_hash() {
        let bytes = build_alembic_bytes(&cube_scene(), None).unwrap();
        let ar = Archive::open(&bytes).unwrap();
        let top = ar.top_object().unwrap();
        let xf = ar.children(&top).unwrap().remove(0);
        let props = ar.object_props(&xf).unwrap();
        let n = ar.og.num_children(props.group).unwrap();
        let hdr = ar.og.child_data(props.group, n - 1).unwrap();
        let parsed = parse_prop_headers_body(hdr, ar.times.len()).unwrap();
        assert_eq!(parsed.len(), n - 1);
        // The block ends with the last property name (no zero padding / hash).
        assert!(hdr.last().map_or(false, |b| (0x20..0x7f).contains(b)));
        // Leaf (shape) object: [properties group, 32-byte hash block].
        let shape = ar.children(&xf).unwrap().remove(0);
        let cn = ar.og.num_children(shape.group).unwrap();
        assert_eq!(cn, 2);
        assert_eq!(ar.og.child_data(shape.group, 1).unwrap().len(), 32);
    }
}
