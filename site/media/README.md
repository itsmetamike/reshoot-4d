# Site media

This folder contains the project page videos, matching JPEG posters, workflow stills and local brand files.

Videos are silent H.264 MP4s with matching JPEG posters. The page autoplays visible previews; its shared Pause button and reduced-motion preference control playback.

## Editor and workflow slots

| Video | Poster | Content |
| --- | --- | --- |
| `explainer.mp4` | `explainer.jpg` | 35-second, 1920x1080 explainer, made with fframes. |
| `step_capture.mp4` | `step_capture.jpg` | Four-second portrait capture preview. |
| `step_camera.mp4` | `step_camera.jpg` | 18.77-second editor recording; 4D camera editor recording. |
| `step_retake.mp4` | `step_retake.jpg` | Four-second retake preview; Example 1 retake. |
| `step_photoreal.mp4` | `step_photoreal.jpg` | Four-second final preview; Example 1 final. |
| `walkthrough.mp4` | `walkthrough.jpg` | 71-second, 1920x1080 walkthrough: workflow, 4D camera editor, Run, final shot. |
| `workflow_graph.jpg` | none | Current Reshoot 4D workflow graph (1920x1080), used by the repo README. |

Use readable UI recordings and matching posters. Check that captions describe the actual footage and review portrait media on desktop and mobile.

The README uses `workflow_graph.jpg`; the page shows the graph in `walkthrough.mp4`.

## Results

Six examples, each the source video next to three reshoots (M01 slow arc right, M02 arc left and rise, M03 gentle push) with a top-view camera path and live readouts. Every reshoot uses the default Photoreal Pass (CrossView + detailer). Built with `hero/minimal_review/compare_multi.py`.

| Video | Poster | Content |
| --- | --- | --- |
| `example_1.mp4` | `example_1.jpg` | Example 1, 1920x1080, 4 seconds, loops. |
| `example_2.mp4` | `example_2.jpg` | Example 2, 1920x1080, 4 seconds, loops. |
| `example_3.mp4` | `example_3.jpg` | Example 3, 1920x1080, 4 seconds, loops. |
| `example_4.mp4` | `example_4.jpg` | Example 4, 1920x1080, 4 seconds, loops. |
| `example_5.mp4` | `example_5.jpg` | Example 5, 1920x1080, 4 seconds, loops. |
| `example_6.mp4` | `example_6.jpg` | Example 6, 1920x1080, 4 seconds, loops. |

## Limitations

| File | Content |
| --- | --- |
| `limits_world.jpg` | Example 4: source and two reshoots; the view through the window differs. |
| `limits_fast_motion.jpg` | Example 6: source and reshoot mid-spin; her face smears in the reshoot. |

See [Credits](../../docs/CREDITS.md) for footage attribution.

## Asset inventory

The hero, four step videos, editor tour, workflow stills, six example videos and the limitation media are present. `hero.mp4` and `hero.jpg` are unused by the current page.

The `#github-link` button href in `site/index.html` defines the repository URL once. All GitHub documentation links derive from that href. Without JavaScript, documentation links use relative `../docs/` paths for a repository checkout.

The page has no build step or external scripts. Preview desktop and mobile, keyboard navigation, posters, playback and reduced motion before publishing.

## Brand files

`brand/` contains brutus artwork, Reshoot 4D wordmarks, compact marks and local fonts. The header uses `reshoot4d_wordmark_t.svg`; the footer uses `brutus_carved.png`.

The page preloads local Unbounded and Space Grotesk fonts. Keep their OFL notices with the font files. Brand colours are lime `#C8FF00` and black `#000000`.

See [Brand assets](../../assets/README.md) for regeneration commands and source inventory.
