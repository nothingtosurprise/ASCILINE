/**
 * encoder.js - client-side ASCILINE .ascf encoder (mirror of codec.py).
 *
 * Produces bytes the shipped codec.js decodes exactly. It emits ONLY the tags
 * codec.js understands: 0 RAW, 1 ZLIB, 2 DELTA. It deliberately does NOT emit
 * tag 3 (RLE_FULL): the shipped codec.js throws on it, so any .ascf using tag 3
 * cannot play in the current player. Lossless (tolerance 0). A forced keyframe
 * every 48 frames bounds delta chains and lets late joiners resync.
 *
 * Deflate is pluggable so the same file runs in the browser (pako) and Node
 * (zlib) - both emit RFC1950 zlib, which codec.js inflates via
 * DecompressionStream('deflate').
 *
 *   makeEncoder(cellBytes, opts?) -> { encode(frame) -> {msg, shown}, reset() }
 *   ascfHeader(fps, mode, pixel, cols, rows) -> Uint8Array(14)
 *   encodeFramesToAscf(frames, meta, opts?) -> Uint8Array         (convenience)
 *
 * frame / shown are Uint8Array of length rows*cols*cellBytes. Pixel cells are
 * [B,G,R] (cellBytes 3); ASCII cells are [char,R,G,B] (cellBytes 4).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AscilineEncoder = api;
})(typeof self !== 'undefined' ? self : this, function () {
  const TAG_RAW = 0, TAG_ZLIB = 1, TAG_DELTA = 2;
  const KEYFRAME_INTERVAL = 48;
  const DELTA_MAX_FRAC = 0.60; // above this, delta loses; skip building it
  const ZLIB_MIN_FRAC = 0.10;  // below this, full-frame zlib loses; skip it

  function resolveDeflate(opts) {
    if (opts && opts.deflate) return opts.deflate;
    const level = (opts && opts.level != null) ? opts.level : 3;
    if (typeof pako !== 'undefined') return (u8) => pako.deflate(u8, { level });
    if (typeof window !== 'undefined' && window.pako) return (u8) => window.pako.deflate(u8, { level });
    if (typeof require !== 'undefined') {
      const zlib = require('zlib');
      return (u8) => new Uint8Array(zlib.deflateSync(Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength), { level }));
    }
    throw new Error('encoder: no deflate available (load pako before encoder.js)');
  }

  function concat(list) {
    let n = 0; for (const a of list) n += a.length;
    const out = new Uint8Array(n);
    let p = 0; for (const a of list) { out.set(a, p); p += a.length; }
    return out;
  }

  function msgHeader(frameIndex, tag) {
    const h = new Uint8Array(5);
    new DataView(h.buffer).setUint32(0, frameIndex >>> 0, false); // big-endian
    h[4] = tag;
    return h;
  }

  function ascfHeader(fps, mode, pixel, cols, rows) {
    const h = new Uint8Array(14);
    const dv = new DataView(h.buffer);
    h[0] = 0x41; h[1] = 0x53; h[2] = 0x43; h[3] = 0x46; // 'ASCF'
    dv.setFloat32(4, fps, false);
    h[8] = mode & 0xff;
    h[9] = pixel ? 1 : 0;
    dv.setUint16(10, cols, false);
    dv.setUint16(12, rows, false);
    return h;
  }

  function makeEncoder(cellBytes, opts) {
    const C = cellBytes;
    const deflate = resolveDeflate(opts);
    let prev = null;   // Uint8Array of the last shown frame
    let index = 0;

    function fullFrame(raw, frameIndex) {
      // Race ZLIB vs RAW only (no RLE_FULL: shipped codec.js can't decode it).
      const z = deflate(raw);
      if (z.length < raw.length) return concat([msgHeader(frameIndex, TAG_ZLIB), z]);
      return concat([msgHeader(frameIndex, TAG_RAW), raw]);
    }

    function encode(frame) {
      const raw = frame instanceof Uint8Array ? frame : new Uint8Array(frame);
      const frameIndex = index++;
      const keyframe = prev === null || (frameIndex % KEYFRAME_INTERVAL === 0) || prev.length !== raw.length;

      if (keyframe) {
        const msg = fullFrame(raw, frameIndex);
        prev = raw.slice();
        return { msg, shown: prev };
      }

      // Lossless changed-cell scan (tolerance 0).
      const nCells = (raw.length / C) | 0;
      const changed = [];
      for (let cell = 0; cell < nCells; cell++) {
        const o = cell * C;
        let diff = false;
        for (let c = 0; c < C; c++) { if (raw[o + c] !== prev[o + c]) { diff = true; break; } }
        if (diff) changed.push(cell);
      }
      const frac = nCells ? changed.length / nCells : 1;

      const candidates = [];
      if (frac < DELTA_MAX_FRAC) {
        const k = changed.length;
        const body = new Uint8Array(k * 4 + k * C);
        const dv = new DataView(body.buffer);
        for (let j = 0; j < k; j++) dv.setUint32(j * 4, changed[j], true); // LE indices
        let off = k * 4;
        for (let j = 0; j < k; j++) { const o = changed[j] * C; for (let c = 0; c < C; c++) body[off++] = raw[o + c]; }
        candidates.push({ tag: TAG_DELTA, payload: deflate(body) });
      }
      if (frac >= ZLIB_MIN_FRAC || candidates.length === 0) {
        const z = deflate(raw);
        if (z.length < raw.length) candidates.push({ tag: TAG_ZLIB, payload: z });
        else candidates.push({ tag: TAG_RAW, payload: raw });
      }

      let best = candidates[0];
      for (const cnd of candidates) if (cnd.payload.length < best.payload.length) best = cnd;
      let tag = best.tag, payload = best.payload;
      if (raw.length < payload.length) { tag = TAG_RAW; payload = raw; } // never exceed raw

      const msg = concat([msgHeader(frameIndex, tag), payload]);
      prev = raw.slice(); // lossless: what the client shows equals the true frame
      return { msg, shown: prev };
    }

    return { encode, reset() { prev = null; index = 0; } };
  }

  function encodeFramesToAscf(frames, meta, opts) {
    const cellBytes = meta.cellBytes != null ? meta.cellBytes : (meta.pixel ? 3 : 4);
    const mode = meta.mode != null ? meta.mode : 5;
    const enc = makeEncoder(cellBytes, opts);
    const parts = [ascfHeader(meta.fps, mode, cellBytes === 3, meta.cols, meta.rows)];
    for (const f of frames) {
      const { msg } = enc.encode(f);
      const lp = new Uint8Array(4);
      new DataView(lp.buffer).setUint32(0, msg.length, false); // big-endian length
      parts.push(lp, msg);
    }
    return concat(parts);
  }

  return { makeEncoder, ascfHeader, encodeFramesToAscf, concat,
           TAG_RAW, TAG_ZLIB, TAG_DELTA, KEYFRAME_INTERVAL };
});
