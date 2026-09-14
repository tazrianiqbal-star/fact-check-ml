#!/usr/bin/env bash
# Packages extension/ into a store-ready zip (manifest.json at the archive
# root, as Chrome Web Store / AMO / Edge Add-ons all require) at
# dist/fact-check-ml-extension-<version>.zip.
set -euo pipefail

cd "$(dirname "$0")"
VERSION=$(python3 -c "import json; print(json.load(open('extension/manifest.json'))['version'])")
OUT="dist/fact-check-ml-extension-${VERSION}.zip"

mkdir -p dist
rm -f "$OUT"
(cd extension && zip -r -X "../$OUT" . -x ".*")

echo "Packaged extension/ (v$VERSION) -> $OUT"
