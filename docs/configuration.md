# Configuration

Every runtime setting is an environment variable prefixed `TUZKAOCR_`. The same values back
the CLI, so a variable sets the default and a command-line flag overrides it for that
invocation.

## Where settings come from

Precedence, highest first:

1. Command-line flags (CLI only).
2. Real environment variables.
3. `.env` in the current working directory.
4. `tuzkaocr.env` next to the package directory.

Both files are loaded without overriding what is already set, so an exported variable always
beats a file.

!!! warning "`tuzkaocr.env` only applies to a checkout"
    It is looked up one level above the `tuzkaocr/` package directory. In a clone that is
    the repository root, so the file works. After `pip install .` the package lives in
    `site-packages/`, where no such file exists — an installed `tuzkaocr` command reads
    only real environment variables and `.env` in the working directory. The Docker images
    copy `tuzkaocr.env` into `/app`, so it applies there.

## Validation

Configuration is validated before models load, and a bad value aborts startup with a message
naming every problem at once:

```
Invalid configuration:
  - TUZKAOCR_PAGE_WORKERS must be >0, got 0
  - TUZKAOCR_DEVICE must be one of ('cpu', 'cuda', 'auto'), got 'gpu'
```

Rules: the thread, worker, size, and age counters must be greater than zero;
`TUZKAOCR_DEVICE` must be `cpu`, `cuda`, or `auto`; `TUZKAOCR_READING_ORDER` must be
`columns` or `basic`; `TUZKAOCR_HEIGHT_SCALE` must be between 0.1 and 10.0; and `TUZKAOCR_SPOOL_DIR`, when set, must already exist and be writable.

Booleans accept `1`, `true`, or `yes`, case-insensitively — anything else is false, so
`on`, `y`, and `enabled` all read as false. A non-numeric value for a numeric setting raises
a `ValueError` naming the variable.

## Compute and concurrency

| Variable | Default | Effect |
|---|---|---|
| `TUZKAOCR_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `TUZKAOCR_OCR_THREADS` | `4` | ONNX Runtime intra-op threads |
| `TUZKAOCR_LINE_WORKERS` | `4` | Parallel line-recognition threads per page |
| `TUZKAOCR_PAGE_WORKERS` | `2` | Concurrent pages — API worker threads, or CLI batch processes |
| `TUZKAOCR_CPU_MEM_ARENA` | `true` | ONNX Runtime CPU memory arena; see [memory](#memory) |

`auto` resolves to `cuda` only when `onnxruntime` reports a CUDA execution provider, and to
`cpu` otherwise. The plain `onnxruntime` wheel is CPU-only.

`TUZKAOCR_PAGE_WORKERS` is the main lever for both throughput and memory. The three
concurrency settings multiply: total threads land near
`PAGE_WORKERS × LINE_WORKERS × OCR_THREADS`, so oversubscribing a small machine slows it
down.

## Recognition quality

| Variable | Default | Effect |
|---|---|---|
| `TUZKAOCR_HEIGHT_SCALE` | `1.0` | Multiplies predicted line heights (valid range 0.1–10.0) |
| `TUZKAOCR_MAX_WIDTH` | `3400` | Maximum line-crop width fed to the recognizer, in pixels |
| `TUZKAOCR_ADAPTIVE_DOWNSAMPLE` | `true` | Size the printed layout pass to each page's text size (never for `handwritten`, `kurrent` or `kramarky`) |
| `TUZKAOCR_LAYOUT_LINE_HEIGHT` | `12` | Target text-line height, in pixels, at the layout model input when adaptive downsampling is on (valid range 4–64); `0` falls back to the fixed downsample |
| `TUZKAOCR_READING_ORDER` | `columns` | Order of text blocks on printed pages: `columns` (column by column, aware of printed rules) or `basic` (horizontal bands, the earlier order); see [reading order](#reading-order) |
| `TUZKAOCR_LARGE_TEXT` | `true` | Extra layout pass for text far larger than the body text, such as newspaper mastheads; see [large text](#large-text) |
| `TUZKAOCR_CROP_ENDPOINT_EXT` | `0.0` | Extend line crops at both ends, as a fraction of line height |
| `TUZKAOCR_COLUMN_SPLIT` | `false` | Order lines column-by-column within a region |

!!! note "Two of these are set per domain"
    `CROP_ENDPOINT_EXT` and `COLUMN_SPLIT` are chosen by the selected domain — `0.3` and
    `true` for `handwritten` and `kurrent`, `0.0` and `false` otherwise — which overrides
    the environment value. Your setting survives only for the printed domain on the API
    path. Treat them as tuning knobs for the printed models, not as global switches.

### Adaptive downsampling

The layout model works best when text lines reach it about 12 pixels tall. A fixed
downsample cannot guarantee that: a newspaper broadsheet with small type ends up with lines
of 6–7 pixels, so lines get missed, geometry becomes imprecise, and recognition degrades.

With adaptive downsampling on (the default for the printed domain), each page is measured
before the layout model runs: a fast image analysis (a few tens of milliseconds) estimates
its typical text-line height. The page is then scaled so its lines land at
`TUZKAOCR_LAYOUT_LINE_HEIGHT` pixels, and the layout model runs once at that scale. Large
book type is scaled down further than before, small newspaper type less, so pages with
ordinary type cost about the same as a fixed pass and broadsheets cost about 2×. The rare
page whose text size cannot be estimated reliably gets one extra coarse layout pass to
measure it. If the layout model then finds lines far taller than the target (more than 5/3
of it), the estimate was misled — typically by faint typewriter whose letters break into
dots — and the layout runs once more at the scale the model's own measurement implies.

Double-page spreads, which are split down the gutter, always use the fixed downsample.

It cannot be switched off per request; set `TUZKAOCR_ADAPTIVE_DOWNSAMPLE=false`
server-side, or pass `--no-adaptive` on the CLI, to use the fixed downsample for every page.

The `handwritten`, `kurrent` and `kramarky` domains always use the fixed downsample,
whatever this setting says: the text-size estimate is not reliable on handwriting, and it
has not been evaluated on Kramarky broadsheets.

### Reading order

The order of text blocks decides the order of the plain-text output and of the
`TextBlock` elements in ALTO and PAGE XML.

With `TUZKAOCR_READING_ORDER=columns` (the default), the page is divided recursively:

- Printed horizontal rules that span most of a section split it into parts read one after
  another. On newspapers this keeps a feuilleton "under the line" after the columns above
  it, including when the rule is broken at every column gutter.
- Rows of blocks that share the same columns stay together, so each column is read top to
  bottom before the next one starts.
- Full-width headings and banners split the page into sections.
- On a front page, the header row ("Ročník | date | Číslo") is read first, then the
  masthead band, then the columns.

Rule detection checks the continuity, slope and thickness of the original printed ink.
Slightly slanted or gently wavy dividers can still separate sections, and aligned fragments
can be joined across narrow column gutters. Occasional vertical strokes crossing a divider
do not by themselves make it too thick. This is bounded to near-horizontal rules (slope up
to 0.04, about 2.3 degrees); very faint, heavily broken or more steeply slanted dividers may
still be missed. The detector uses the existing layout-input image and adds no OCR pass:
each line is recognised once. The `basic` order and non-print domains do not use this
rule-aware ordering.

Before ordering, a one- or two-line block sitting directly under a paragraph, aligned with
it and set in the same type size, is joined to that paragraph. This is typically the last
line of the paragraph that the layout model split off. Headings, indented first lines of the
next paragraph, and lines below a printed rule are left as separate blocks.

`TUZKAOCR_READING_ORDER=basic` restores the earlier order: blocks are grouped into
horizontal bands and read left to right within each band, and orphan lines are not
re-attached. A single-column page reads top to bottom either way.

### Large text

At the scale chosen for body text, a newspaper masthead can be ten times taller than what the
layout model expects. The model then sees only fragments of letters, and the title comes out
as junk or not at all. With `TUZKAOCR_LARGE_TEXT=true` (the default), a page that contains
text at least four times the height of its body text gets one extra, coarser layout pass.
Large lines from that pass replace the fragments under them and are recognised from a
slightly widened crop. Pages without large text skip the extra pass. On pages that need it,
it costs about 0.2–0.7 s.

Both settings apply to the printed domain only. The `handwritten`, `kurrent` and
`kramarky` domains always use the `basic` order without the large-text pass. Double-page
spreads split down the gutter use the selected reading order on each half, but never the
large-text pass.

## Line roles

| Variable | Default | Effect |
|---|---|---|
| `TUZKAOCR_ROLE_CLASSIFIER` | `false` | Tag each ALTO line with a structural role |
| `TUZKAOCR_ROLE_MODEL` | `role-H5.onnx` | Bundled role classifier |

This is the server-wide default; the API's `role_classifier` form field and the CLI's
`--role-classifier` flag override it per request. See
[Output formats](output-formats.md#line-roles).

## ALTO output

| Variable | Default | Effect |
|---|---|---|
| `TUZKAOCR_ALTO_PROFILE` | `ndk` | `ndk` (Czech NDK ingest profile) or `basic` (pre-1.7.2 shape) |
| `TUZKAOCR_ALTO_AGENCY` | *empty* | Institution running the OCR, written as `processingAgency` |
| `TUZKAOCR_ALTO_LANG` | *empty* | Language code for `LANG`; empty means detect it from the text |
| `TUZKAOCR_ALTO_FONTFAMILY` | *empty* | Typeface written as `TextStyle/@FONTFAMILY` |
| `TUZKAOCR_ALTO_DPI` | `0` | Scan resolution used for `FONTSIZE`; `0` means read it from the image |
| `TUZKAOCR_ALTO_PAGE_WIDTH_MM` | `0` | Physical page width; the resolution is derived from it when the image records none |

`TUZKAOCR_ALTO_PROFILE` is the server-wide default; the API's `alto_profile` field and the
CLI's `--alto-profile` override it per request. The other variables affect only the `ndk`
profile.

Values TuzkaOCR cannot determine are left out of the ALTO rather than guessed. For NDK ingest:

- **Set `TUZKAOCR_ALTO_AGENCY`.** The profile requires `processingAgency`.
- **Make sure a resolution is available** for `FONTSIZE`. It is read from the image (JPEG, PNG
  or TIFF metadata), but values below 100 dpi are ignored as software defaults, and derivative
  JPEGs (such as IIIF images) often record none. Then set either `TUZKAOCR_ALTO_DPI` or
  `TUZKAOCR_ALTO_PAGE_WIDTH_MM` — e.g. a 4363 px scan of a 450 mm page is 246 dpi.
  Precedence: `TUZKAOCR_ALTO_DPI` → image metadata → page width.
- **Set `TUZKAOCR_ALTO_FONTFAMILY`** (`Fraktur`, `Antiqua`, …) if you know it; typefaces are not
  detected.
- **Set `TUZKAOCR_ALTO_LANG`** for material the language detector does not cover or pages with
  very little text — see [Language](output-formats.md#language).

The CLI flag `--ndk-warn` reports missing agency and page numbers during a run.

## Service limits and storage

| Variable | Default | Effect |
|---|---|---|
| `TUZKAOCR_RESULTS_DIR` | `results` | Where finished output is written |
| `TUZKAOCR_SPOOL_DIR` | *system temp* | Where uploads are streamed |
| `TUZKAOCR_MAX_JOB_AGE_HOURS` | `24` | Retention for jobs and result files |
| `TUZKAOCR_MAX_QUEUE` | `16` | Maximum queued + running jobs before 503 |
| `TUZKAOCR_MAX_UPLOAD_MB` | `256` | Request body limit; over this returns 413 |
| `TUZKAOCR_MAX_IMAGE_PIXELS` | `300000000` | Decoded pixel limit; over this fails the job with 422 |

The defaults are generous to accommodate large archival scans; tune down for stricter
deployments. The pixel limit applies to API jobs only — command-line processing has no
pixel cap.

Every accepted upload is streamed to a uniquely named file under the spool directory, decoded
once by its worker, and deleted after processing or rejection. Queued jobs hold the encoded
file on disk, never a decoded page in memory.

An explicitly configured spool directory is treated as private to one service instance and is
cleared of leftovers at startup — do not share one between instances. The shared system temp
directory is never swept. Keep the spool on real disk: on a RAM-backed `tmpfs`, large uploads
consume memory instead.

## Authentication

| Variable | Default | Effect |
|---|---|---|
| `TUZKAOCR_API_KEY` | *empty* | Single shared secret, sent as `X-API-Key` |
| `TUZKAOCR_API_KEYS_FILE` | *empty* | YAML file of `name: key` pairs; takes precedence |

Both empty means authentication is disabled, which is only safe on a trusted network. Details
in [HTTP API](api.md#authentication).

## Models

| Variable | Default |
|---|---|
| `TUZKAOCR_LAYOUT_MODEL` | `dec-B-v2.onnx` |
| `TUZKAOCR_OCR_MODEL` | `rec-E-v5.onnx` |
| `TUZKAOCR_KRAMARKY_LAYOUT_MODEL` | `dec-B-v1k.onnx` |
| `TUZKAOCR_KRAMARKY_OCR_MODEL` | `rec-E-v4k7.onnx` |
| `TUZKAOCR_HANDWRITTEN_LAYOUT_MODEL` | `dec-B-v2h.onnx` |
| `TUZKAOCR_HANDWRITTEN_OCR_MODEL` | `rec-I-v2.onnx` |
| `TUZKAOCR_KURRENT_LAYOUT_MODEL` | `dec-B-v2h.onnx` |
| `TUZKAOCR_KURRENT_OCR_MODEL` | `rec-I-v2.onnx` |
| `TUZKAOCR_VOCAB` | `vocab.json` |
| `TUZKAOCR_LANG_MODEL` | `lang-A-v1.npz` |

The defaults are the bundled files and normally need no change. See
[Models and domains](models.md) for how a name is resolved and when pinning makes sense.

## Memory

Peak memory is dominated by the layout detector — a fully convolutional segmentation network
run at up to about 1536 px. Its fp32 activations are a **transient of roughly 1.3 GiB per
page in flight**. The model files themselves (0.3–13 MB) and decoded pages (~10 MB) are
negligible next to that.

```
peak ≈ 0.1 GiB + PAGE_WORKERS × 1.3 GiB + scratch
```

`TUZKAOCR_CPU_MEM_ARENA` controls ONNX Runtime's CPU memory arena:

| Value | Behaviour | Cost |
|---|---|---|
| `true` (default) | A reusable allocation pool is kept. Lowest per-page latency, but the pool grows to the peak working set and is **never returned to the OS** — idle RSS stays high, around 2.4 GiB with two workers. | A tight container limit can OOM on a later page. |
| `false` | Memory is released between pages. Peak RSS drops about 45%, idle RSS falls to roughly 0.26 GiB. | About +35% single-page latency, but only about −6% batch throughput — the work is compute-bound. |

`false` is the recommendation for memory-constrained bulk work, where avoiding an OOM matters
more than single-page latency. With it, two page workers fit in about 2 GiB and one fits in
about 1.2 GiB.

For tight limits, prefer **one page worker per service instance, scaled out by instance
count**, over packing more workers into one process. And keep scratch — the spool and results
directories — on real disk, since `tmpfs` counts against a container's memory limit.
