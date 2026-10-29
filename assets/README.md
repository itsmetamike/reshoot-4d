# Reshoot 4D brand assets

Pure black `#000000` and lime `#C8FF00` are the brand colours. Grey is reserved for secondary copy. The banner is 1600x500; the social card is 1280x640.

The wordmark uses custom geometric lettering based on the brutus labs mark. The compact R4D mark is intended for avatars and favicons.

An SVG filter joins and softens the strokes: `feGaussianBlur` at 3.6 units for lime lettering or 4.2 for carved lettering, then `feColorMatrix` with alpha threshold `0 0 0 36 -16`. The carved treatment puts black letters into a lime box with a 19-unit corner radius. SVGs retain live geometry and filters.

- `reshoot4d_wordmark.svg`: horizontal lime lettering on black.
- `reshoot4d_carved.svg`: black lettering in a rounded lime box.
- `reshoot4d_mark.svg` and `favicon.svg`: compact carved R4D.
- `banner.svg` and `social_card.svg`: standalone repo imagery, including the original carved brutus mark. Unbounded is embedded so the supporting copy travels with the SVG.

UI headings, badges and buttons use Unbounded, vendored from `web/lib/fonts/Unbounded-latin.woff2`. The site copy and its SIL Open Font License are in `site/media/brand/Unbounded-latin.woff2` and `site/media/brand/OFL.txt`. The site uses local Space Grotesk for body copy. Wordmark lettering does not use Unbounded.

From the repository root:

```sh
python3 scripts/brand/glyphs_ext.py
bash scripts/brand/render.sh
```

The generator writes SVG assets and site copies. The renderer uses headless Chrome to write PNGs under `assets/png/`; override its executable with `CHROME_BIN=/path/to/chrome`. Inspect generated assets before publishing.
