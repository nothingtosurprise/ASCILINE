# ASCILINE lossy DCT profile (tag 4)

An opt-in, lossy compression profile for pixel mode. It is a separate profile, not
a competitor in the per-frame tag race: you enable it with a flag. The default
RAW/ZLIB/DELTA behaviour and all existing `.ascf` files are byte-for-byte unchanged.

Everything stays in userland: no `<video>` element and no native decoder, just a
hand-written decoder painting a canvas. On typical footage the profile is about
4 to 5x smaller than the lossless path at matched quality.

## Pipeline
YUV 4:2:0 chroma subsampling, an 8x8 integer DCT with a JPEG-style perceptual
quantization matrix, per-block skip, luma block motion compensation, zigzag plus
run-length entropy, DC prediction, deadzone quantization, and a rate-distortion
skip.

## Wire format
Per-frame message stays `[4B index BE][1B tag][zlib payload]`. The profile uses
**tag 4**. Keyframes self-describe, so the 14-byte `.ascf` header does not change:
- I-frame payload (after inflate): `[ftype=0][QF u8][cols u16 BE][rows u16 BE]` then the Y, Cb, Cr planes.
- P-frame payload: `[ftype=1]` then the planes.

The 8x8 integer DCT basis and the zigzag order are fixed constants shared by
`codec.py` and `codec.js`; the luma and chroma quantization tables are derived
from `QF` on both sides. The integer IDCT and an integer YUV420->BGR keep the two
implementations bit-exact. The decoder outputs a standard BGR pixel framebuffer,
so the renderer is unchanged.

## Constraints (v1)
Pixel mode only (the ASCII character plane stays exact and is not transformed).
`cols` and `rows` must be multiples of 16; the compiler pads to the next multiple.
An old `codec.js` without tag-4 support will not play a profile `.ascf`; this
`codec.js` adds the tag-4 branch and a graceful fallback that repeats the last
frame on an unknown tag instead of throwing.

## Use
    python static_player/compiler.py input.mp4 --pixel --profile --qf 70 --cols 480

Higher `--qf` means better quality and a larger file.

## Test (bit-exact, cross-language)
    python experiments/profile_vectors.py
    node experiments/check_profile.js
