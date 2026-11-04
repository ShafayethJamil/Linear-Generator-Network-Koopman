#!/bin/bash
# Maintainer: build the release asset from checkpoints/ (upload to the v1.0 release).
set -e
cd "$(dirname "$0")/.."
tar -czf lgnkm_checkpoints_v1.tar.gz checkpoints/MANIFEST.json checkpoints/ns checkpoints/nlse
sha256sum lgnkm_checkpoints_v1.tar.gz
