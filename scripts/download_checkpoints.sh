#!/bin/bash
# Download the released checkpoints (40 MB) into checkpoints/ and check their SHA-256.
set -e
cd "$(dirname "$0")/.."
URL=https://github.com/ShafayethJamil/lgn-km/releases/download/v1.0/lgnkm_checkpoints_v1.tar.gz
curl -L --fail -o lgnkm_checkpoints_v1.tar.gz "$URL"
tar -xzf lgnkm_checkpoints_v1.tar.gz
rm lgnkm_checkpoints_v1.tar.gz
python - <<'PY'
import hashlib, json
m = json.load(open('checkpoints/MANIFEST.json'))
bad = [k for k, v in m.items()
       if hashlib.sha256(open('checkpoints/' + k, 'rb').read()).hexdigest() != v['sha256']]
print(f"{len(m) - len(bad)} of {len(m)} checkpoints verified" + (f"; MISMATCH: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
PY
