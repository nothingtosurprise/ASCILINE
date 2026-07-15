# Encode lossy-DCT-profile (tag 4) test vectors with codec.py, for cross-language
# bit-exact verification by check_profile.js. Mirrors gen_vectors.py -> check_vectors.js.
import os, sys, struct, hashlib, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codec import ProfileEncoder
W, H, N, QF = 160, 96, 40, 70
def frame(i):
    yy, xx = np.mgrid[0:H, 0:W]
    B=(40+180*xx/W).astype(np.uint8); G=(30+150*yy/H).astype(np.uint8); R=np.full((H,W),60,np.uint8)
    bx=int((0.5+0.4*math.sin(i/N*2*math.pi))*(W-24)); by=int((0.5+0.4*math.cos(i/N*2*math.pi))*(H-24))
    R[by:by+24,bx:bx+24]=230; G[by:by+24,bx:bx+24]=230; B[by:by+24,bx:bx+24]=80
    return np.ascontiguousarray(np.stack([B,G,R],2).astype(np.uint8))
d = os.path.join(os.path.dirname(__file__), "vectors"); os.makedirs(d, exist_ok=True)
enc = ProfileEncoder(W, H, QF); msgs = bytearray(); shas = []
for i in range(N):
    m, shown = enc.encode(frame(i)); msgs += struct.pack(">I", len(m)) + m
    shas.append(hashlib.sha256(shown).hexdigest())
open(os.path.join(d, "profile.bin"), "wb").write(bytes(msgs))
json.dump({"W":W,"H":H,"QF":QF,"nframes":N,"frame_shas":shas}, open(os.path.join(d,"profile_meta.json"),"w"))
print("wrote profile vectors: %d frames %dx%d QF%d, %.1f KB" % (N, W, H, QF, len(msgs)/1024))
