#!/bin/bash
# Regenerate SVGs, then rasterize every assets/*.svg at its native artboard size.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python3 "$ROOT/scripts/brand/glyphs_ext.py"
C="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [ ! -x "$C" ]; then
  C="$HOME/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
fi
if [ ! -x "$C" ]; then
  echo 'Chrome unavailable. Set CHROME_BIN to a headless Chrome executable.' >&2
  exit 1
fi
WORK="$(mktemp -d /private/tmp/reshoot-brand.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$ROOT/assets/png"
python3 - "$ROOT" "$WORK" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
root, work = map(Path, sys.argv[1:])
rows = []
for source in sorted((root/'assets').glob('*.svg')):
    svg = ET.parse(source).getroot()
    w, h = svg.attrib['width'], svg.attrib['height']
    wrapper = work/(source.stem+'.html')
    wrapper.write_text(f'<html><head><style>html,body{{margin:0;width:{w}px;height:{h}px;background:#000;overflow:hidden}}img{{display:block;width:{w}px;height:{h}px}}</style></head><body><img src="{source.as_uri()}"></body></html>')
    rows.append(f'{source.stem}\t{w}\t{h}\t{wrapper.as_uri()}')
(work/'manifest').write_text('\n'.join(rows)+'\n')
PY
while IFS=$'\t' read -r name width height url; do
  "$C" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
    --user-data-dir="$WORK/profile" --no-first-run --no-default-browser-check \
    --force-device-scale-factor=1 --window-size="$width,$height" \
    --virtual-time-budget=2000 --screenshot="$ROOT/assets/png/$name.png" "$url"
done < "$WORK/manifest"
echo 'Rendered assets/png/*.png'
