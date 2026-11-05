#!/bin/bash
# Build the checkpoint release asset from checkpoints/ (maintainers).
# Usage: bash scripts/package_checkpoints.sh [TAG]   (default v1.1)
set -e
cd "$(dirname "$0")/.."
TAG=${1:-v1.1}
tar -czf lgnkm_checkpoints_$TAG.tar.gz checkpoints/MANIFEST*.json $(ls -d checkpoints/*/)
sha256sum lgnkm_checkpoints_$TAG.tar.gz
