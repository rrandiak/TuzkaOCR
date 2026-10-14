# Installation

!!! tip "Not comfortable with a terminal?"
    [Step by step, no experience needed](beginners.md) covers the same ground for
    readers who have never used a command line — no git, per-platform instructions,
    and a plain-language version of what each line does.

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| Disk | ~1.5 GB for dependencies, plus ~80 MB of bundled models |
| Hardware | Any x86-64 or ARM64 CPU; no GPU needed |
| Network | None at runtime — models ship inside the package |

Dependencies are version-pinned, so install into a virtual environment.

### Platform support

Linux, Windows, and Apple Silicon macOS are supported.

!!! warning "Intel Macs are not supported"
    Upstream `onnxruntime` no longer publishes x86-64 macOS wheels — 1.18 and later are
    arm64-only — so `pip install` fails on Intel hardware. Use the
    [Docker CPU image](docker.md) instead.

## Install from a checkout

There is no PyPI release; you install from a clone of the repository.

```bash
git clone https://github.com/moravianlibrary/TuzkaOCR.git
cd TuzkaOCR
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .
```

This puts `tuzkaocr` on your `PATH`; the models are bundled, so nothing is downloaded at run
time. To pin a version, `git checkout` a tag from the
[releases page](https://github.com/moravianlibrary/TuzkaOCR/releases) before installing.

### Verify

```console
$ tuzkaocr page.jpg --format txt --out page.txt
Done in 2.4s — 37 lines → page.txt
```

## Running without installing

From the repository root:

```bash
pip install -r requirements.txt
python cli.py page.jpg --out result.alto.xml
```

Only this mode reads `tuzkaocr.env` — see
[Configuration](configuration.md#where-settings-come-from).

## Upgrading

`pip install .` does nothing if the version number is unchanged, so force it:

```bash
git pull
pip install --force-reinstall --no-deps .
```

## Development install

`pip install -e .` makes code and model edits take effect without reinstalling.
