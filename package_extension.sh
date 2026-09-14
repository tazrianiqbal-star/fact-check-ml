#!/usr/bin/env bash
# Packages extension/ into store-ready zips (manifest.json at the archive
# root, as every store requires) under dist/.
#
# Two variants, because Chrome/Edge's Manifest V3 validator hard-rejects
# background.scripts alongside background.service_worker, while Firefox
# needs background.scripts to run any background code at all (it doesn't
# support service_worker the way Chrome does -- see README's "Publishing
# the extension" section):
#   - dist/fact-check-ml-extension-<version>.zip            (Firefox/AMO)
#   - dist/fact-check-ml-extension-<version>-chromium.zip   (Chrome/Edge)
set -euo pipefail

cd "$(dirname "$0")"
VERSION=$(python3 -c "import json; print(json.load(open('extension/manifest.json'))['version'])")
mkdir -p dist

FIREFOX_OUT="dist/fact-check-ml-extension-${VERSION}.zip"
rm -f "$FIREFOX_OUT"
(cd extension && zip -r -X "../$FIREFOX_OUT" . -x ".*")
echo "Packaged extension/ (v$VERSION, Firefox/AMO) -> $FIREFOX_OUT"

CHROMIUM_OUT="dist/fact-check-ml-extension-${VERSION}-chromium.zip"
CHROMIUM_STAGE=$(mktemp -d)
trap 'rm -rf "$CHROMIUM_STAGE"' EXIT

cp -R extension/. "$CHROMIUM_STAGE/"
python3 -c "
import json
path = '$CHROMIUM_STAGE/manifest.json'
manifest = json.load(open(path))
manifest['background'].pop('scripts', None)
json.dump(manifest, open(path, 'w'), indent=2)
"
rm -f "$CHROMIUM_OUT"
(cd "$CHROMIUM_STAGE" && zip -r -X "$OLDPWD/$CHROMIUM_OUT" . -x ".*")
echo "Packaged extension/ (v$VERSION, Chrome/Edge -- background.scripts stripped) -> $CHROMIUM_OUT"
