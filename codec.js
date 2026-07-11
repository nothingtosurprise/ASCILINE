/**
 * codec.js — Adaptive frame decoder for ASCILINE.
 *
 * Mirrors codec.py. Runs in the browser (attaches window.AscilineCodec) and in
 * Node (module.exports) so the end-to-end test exercises the exact shipped path.
 *
 * Wire format per binary frame:
 *   [4B frame_index big-endian][1B tag][payload]
 *   tag 0 RAW   : payload is the framebuffer bytes
 *   tag 1 ZLIB  : payload is zlib(framebuffer bytes)        -> 'deflate'
 *   tag 2 DELTA : payload is zlib(indices[uint32 LE] ++ changed values)
 *   tag 3 RLE   : payload is zlib(runs: [uint16 count][cell bytes]...)
 *
 * Decoding MUST stay in arrival order (deltas patch the previous frame), so
 * callers feed messages through a sequential queue (see makeDecoder).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AscilineCodec = api;
})(typeof self !== 'undefined' ? self : this, function () {
  const TAG_RAW = 0, TAG_ZLIB = 1, TAG_DELTA = 2, TAG_RLE_FULL = 3;

  async function inflate(bytes) {
    // Python zlib.compress -> RFC1950 zlib wrapper -> 'deflate' here.
    const ds = new DecompressionStream('deflate');
    const stream = new Blob([bytes]).stream().pipeThrough(ds);
    const buf = await new Response(stream).arrayBuffer();
    return new Uint8Array(buf);
  }

  /**
   * Create a stateful decoder. `cellBytes` = channels per cell (4 ASCII color,
   * 3 pixel). Returns { decode(message) -> {frameIndex, frame}, reset() }.
   * `frame` is a Uint8Array of the full framebuffer for that frame.
   */
  function makeDecoder(cellBytes) {
    let prev = null; // Uint8Array of last full frame

    async function decode(message) {
      const bytes = new Uint8Array(message);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const frameIndex = view.getUint32(0, false); // big-endian
      const tag = bytes[4];
      const payload = bytes.subarray(5);

      let frame;
      if (tag === TAG_RAW) {
        frame = payload.slice(); // own copy; becomes next prev
      } else if (tag === TAG_ZLIB) {
        frame = await inflate(payload);
      } else if (tag === TAG_DELTA) {
        const body = await inflate(payload);
        const k = body.length / (4 + cellBytes);
        const idx = new DataView(body.buffer, body.byteOffset, body.byteLength);
        frame = prev.slice(); // patch onto a copy of previous frame
        const valuesOffset = k * 4;
        for (let j = 0; j < k; j++) {
          const cell = idx.getUint32(j * 4, true); // little-endian indices
          const dst = cell * cellBytes;
          const src = valuesOffset + j * cellBytes;
          for (let c = 0; c < cellBytes; c++) frame[dst + c] = body[src + c];
        }
      } else if (tag === TAG_RLE_FULL) {
        const body = await inflate(payload);
        const bodyView = new DataView(body.buffer, body.byteOffset, body.byteLength);
        let totalCells = 0;
        let offset = 0;
        while (offset < body.length) {
          totalCells += bodyView.getUint16(offset, true);
          offset += 2 + cellBytes;
        }
        frame = new Uint8Array(totalCells * cellBytes);
        offset = 0;
        let dst = 0;
        while (offset < body.length) {
          const count = bodyView.getUint16(offset, true);
          const valOffset = offset + 2;
          if (cellBytes === 4) {
            const v0 = body[valOffset], v1 = body[valOffset+1], v2 = body[valOffset+2], v3 = body[valOffset+3];
            for (let i = 0; i < count; i++) {
              frame[dst++] = v0; frame[dst++] = v1; frame[dst++] = v2; frame[dst++] = v3;
            }
          } else if (cellBytes === 3) {
            const v0 = body[valOffset], v1 = body[valOffset+1], v2 = body[valOffset+2];
            for (let i = 0; i < count; i++) {
              frame[dst++] = v0; frame[dst++] = v1; frame[dst++] = v2;
            }
          } else {
            for (let i = 0; i < count; i++) {
              for (let c = 0; c < cellBytes; c++) frame[dst++] = body[valOffset + c];
            }
          }
          offset += 2 + cellBytes;
        }
      } else {
        throw new Error('Unknown ASCILINE codec tag: ' + tag);
      }
      prev = frame;
      return { frameIndex, frame };
    }

    return { decode, reset() { prev = null; } };
  }

  return { makeDecoder, inflate, TAG_RAW, TAG_ZLIB, TAG_DELTA, TAG_RLE_FULL };
});
