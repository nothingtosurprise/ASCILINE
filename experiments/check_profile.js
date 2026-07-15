// Decode the lossy-DCT-profile (tag 4) vectors with the shipped codec.js and assert
// bit-exact against the codec.py encoder. Mirrors check_vectors.js.
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const Codec = require(path.join(__dirname, '..', 'codec.js'));
const sha = u => crypto.createHash('sha256').update(Buffer.from(u)).digest('hex');
(async () => {
  const d = path.join(__dirname, 'vectors');
  const meta = JSON.parse(fs.readFileSync(path.join(d, 'profile_meta.json')));
  const buf = new Uint8Array(fs.readFileSync(path.join(d, 'profile.bin')));
  const dv = new DataView(buf.buffer); const msgs = []; let off = 0;
  while (off + 4 <= buf.length) { const len = dv.getUint32(off, false); off += 4; msgs.push(buf.subarray(off, off + len)); off += len; }

  const dec = Codec.makeDecoder(3); const shas = [];
  for (const m of msgs) shas.push(sha((await dec.decode(m)).frame));
  const ok = shas.length === meta.frame_shas.length && shas.every((s, i) => s === meta.frame_shas[i]);
  console.log('profile frames ' + shas.length + '/' + meta.nframes + ' : ' + (ok ? 'BIT-EXACT OK' : 'FAIL'));

  // The profile decoder reuses module-level scratch across blocks to keep the hot path
  // allocation-free. That is only sound because its block loop never awaits, so two
  // decoders can never interleave inside it. Guard the invariant rather than trust it.
  const a = Codec.makeDecoder(3), b = Codec.makeDecoder(3); const sa = [], sb = [];
  for (const m of msgs) { const [ra, rb] = await Promise.all([a.decode(m), b.decode(m)]); sa.push(sha(ra.frame)); sb.push(sha(rb.frame)); }
  const conc = sa.every((s, i) => s === meta.frame_shas[i]) && sb.every((s, i) => s === meta.frame_shas[i]);
  console.log('concurrent decoders, shared scratch : ' + (conc ? 'OK' : 'FAIL'));
  process.exit(ok && conc ? 0 : 1);
})();
