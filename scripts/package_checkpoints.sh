#!/bin/bash
# Maintainer: build the release asset from checkpoints/ (upload it to the release named by TAG).
set -e
cd "$(dirname "$0")/.."
TAG=${1:-v1.1}
tar -czf lgnkm_checkpoints_$TAG.tar.gz checkpoints/MANIFEST*.json $(ls -d checkpoints/*/)
sha256sum lgnkm_checkpoints_$TAG.tar.gz
