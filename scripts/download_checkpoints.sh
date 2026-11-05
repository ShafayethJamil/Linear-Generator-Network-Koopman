#!/bin/bash
# Download the checkpoints into checkpoints/ and check their SHA-256 against checkpoints/MANIFEST*.json.
# Usage: bash scripts/download_checkpoints.sh [TAG]   (default v1.1)
set -e
cd "$(dirname "$0")/.."
TAG=${1:-v1.1}
URL=https://github.com/ShafayethJamil/Linear-Generator-Network-Koopman/releases/download/$TAG/lgnkm_checkpoints_$TAG.tar.gz
curl -L --fail -o lgnkm_checkpoints.tar.gz "$URL"
tar -xzf lgnkm_checkpoints.tar.gz
rm lgnkm_checkpoints.tar.gz
python - <<'PY'
import hashlib, json, glob
m = {}
for f in sorted(glob.glob('checkpoints/MANIFEST*.json')):
    m.update(json.load(open(f)))
bad = [k for k, v in m.items()
       if hashlib.sha256(open('checkpoints/' + k, 'rb').read()).hexdigest() != v['sha256']]
print(f"{len(m) - len(bad)} of {len(m)} checkpoints verified" + (f"; MISMATCH: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
PY
