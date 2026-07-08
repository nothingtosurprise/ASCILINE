"""
codec.py — Adaptive per-frame codec for ASCILINE's binary WebSocket stream.

Wire format (one message per frame):
    [4 bytes: frame_index, big-endian uint32]
    [1 byte : codec tag]
    [payload ...]

Tags:
    0 RAW    payload = framebuffer bytes, as the legacy protocol sent them
    1 ZLIB   payload = zlib(framebuffer bytes)
    2 DELTA  payload = zlib( changed-cell indices [uint32 LE] ++ changed values )

The encoder picks the smallest applicable encoding per frame. The decoder lives
in codec.js (browser + Node) so the shipped path is the tested path; it never
needs to change for any of the encoder optimizations below.

Optimizations:
  - zlib level 3 (near level-6 ratio at roughly half the CPU)
  - smart candidate selection: only try DELTA when few cells changed and ZLIB
    when many did, skipping the obvious loser at the extremes (saves CPU, no
    size cost in the common middle range)
  - lossy temporal delta (conditional replenishment): a colour cell is only
    re-sent once it drifts past `tolerance` from what the viewer already sees.
    The CHARACTER plane is always exact. tolerance=0 is lossless and keeps the
    stream bit-exact. State is the previously-SHOWN frame, so error is bounded
    by `tolerance` and never drifts.
"""
import struct
import zlib
import numpy as np

TAG_RAW = 0
TAG_ZLIB = 1
TAG_DELTA = 2
TAG_RLE_FULL = 3

DEFAULT_LEVEL = 3        # zlib level: best size/CPU trade-off (see experiments/optimize.py)
KEYFRAME_INTERVAL = 48   # force a full frame this often for resync / late joiners

# Smart-selection thresholds (fraction of cells changed).
_DELTA_MAX_FRAC = 0.60   # above this, delta loses — don't bother building it
_ZLIB_MIN_FRAC = 0.10    # below this, full-frame zlib loses — don't bother

def _rle_encode(frame: np.ndarray) -> bytes:
    """Run-Length Encoding of a full frame. Returns bytes."""
    C = frame.shape[2]
    flat = frame.reshape(-1, C)
    diffs = np.any(flat[1:] != flat[:-1], axis=1)
    change_indices = np.where(diffs)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [len(flat)]))
    run_lengths = np.diff(change_indices)
    values = flat[change_indices[:-1]]
    
    out = bytearray()
    for count, val in zip(run_lengths, values):
        val_bytes = val.tobytes()
        while count > 65535:
            out.extend(struct.pack("<H", 65535))
            out.extend(val_bytes)
            count -= 65535
        if count > 0:
            out.extend(struct.pack("<H", count))
            out.extend(val_bytes)
    return bytes(out)

def _full_frame(raw: bytes, frame: np.ndarray, frame_index: int, level: int) -> bytes:
    # Race ZLIB vs RLE_FULL
    z_raw = zlib.compress(raw, level)
    
    rle_bytes = _rle_encode(frame)
    z_rle = zlib.compress(rle_bytes, level)
    
    if len(z_rle) < len(z_raw) and len(z_rle) < len(raw):
        return struct.pack(">IB", frame_index, TAG_RLE_FULL) + z_rle
    elif len(z_raw) < len(raw):
        return struct.pack(">IB", frame_index, TAG_ZLIB) + z_raw
    return struct.pack(">IB", frame_index, TAG_RAW) + raw

def encode_frame(frame: np.ndarray, prev: np.ndarray | None, frame_index: int,
                 level: int = DEFAULT_LEVEL, tolerance: int = 0):
    """
    Encode one framebuffer.

    :param frame: C-contiguous uint8 array, shape (rows, cols, C). C is 4 for
                  ASCII colour ([char,R,G,B]) or 3 for pixel mode ([B,G,R]).
    :param prev:  the previously-SHOWN frame (what the client currently displays)
                  or None for a keyframe.
    :param tolerance: max per-channel colour drift tolerated before re-sending a
                  cell (lossy). 0 = lossless. The character plane is always exact.
    :returns: (message_bytes, shown_frame) — shown_frame is what the client will
              now display and must be passed back as `prev` next call.
    """
    raw = frame.tobytes()
    keyframe = prev is None or (frame_index % KEYFRAME_INTERVAL == 0)
    if keyframe or prev.shape != frame.shape:
        return _full_frame(raw, frame, frame_index, level), frame.copy()

    C = frame.shape[2]
    diff = np.abs(frame.astype(np.int16) - prev.astype(np.int16))
    if C == 4:
        # channel 0 is the character (structure) -> exact; tolerance on colour
        char_changed = frame[:, :, 0] != prev[:, :, 0]
        if tolerance <= 0:
            color_changed = np.any(diff[:, :, 1:] != 0, axis=2)
        else:
            color_changed = np.any(diff[:, :, 1:] > tolerance, axis=2)
        changed = char_changed | color_changed
    else:
        changed = (np.any(diff != 0, axis=2) if tolerance <= 0
                   else np.any(diff > tolerance, axis=2))

    frac = float(changed.mean())
    ci = np.nonzero(changed.reshape(-1))[0].astype("<u4")

    # Lossy reconstruction the client will hold if we send a DELTA.
    delta_shown = prev.copy()
    delta_shown.reshape(-1, C)[ci] = frame.reshape(-1, C)[ci]

    candidates = []  # (tag, payload, shown_after_decode)
    if frac < _DELTA_MAX_FRAC:
        vals = frame.reshape(-1, C)[ci]
        delta = zlib.compress(ci.tobytes() + vals.tobytes(), level)
        candidates.append((TAG_DELTA, delta, delta_shown))
    
    # We still race Full ZLIB and Full RLE if they might win
    if frac >= _ZLIB_MIN_FRAC or not candidates:
        z_raw = zlib.compress(raw, level)
        rle_bytes = _rle_encode(frame)
        z_rle = zlib.compress(rle_bytes, level)
        
        if len(z_rle) < len(z_raw):
            candidates.append((TAG_RLE_FULL, z_rle, frame))
        else:
            candidates.append((TAG_ZLIB, z_raw, frame))

    tag, payload, shown = min(candidates, key=lambda c: len(c[1]))
    # Never exceed the raw frame (zlib can inflate incompressible data slightly).
    if len(raw) < len(payload):
        tag, payload, shown = TAG_RAW, raw, frame

    msg = struct.pack(">IB", frame_index, tag) + payload
    # If we sent a full frame, the client shows the TRUE frame, not the lossy one.
    return msg, (shown.copy() if shown is frame else shown)
