#!/usr/bin/env python3
"""Check every notebook installs the same cvenv version, and say if it is stale.

Each tutorial installs cvenv from GitHub at a pinned tag, written by hand in one
cell. Nothing keeps those pins in step, and they have drifted badly: while three
notebooks sat on v0.1.15, ``ribeiro_bundle_adjustment_pytorch3d.ipynb`` was still
pinned to v0.1.4 — eleven releases back, predating wheel provenance entirely — and
nobody noticed, because a stale pin installs perfectly well. It just installs the
wrong code, and the failure surfaces later as something that looks like a notebook
bug.

Two different checks, deliberately weighted differently:

* Notebooks disagreeing with each other is an **error**. There is no reason for
  two tutorials in one repository to pin different versions, and it is the
  failure that actually happened.
* Being behind the newest cvenv release is a **warning**. Pinning an older tag on
  purpose is legitimate — the point of pinning is that a tutorial keeps working
  across a semester — so this must not break CI every time cvenv ships. Pass
  ``--require-latest`` to make it an error when that is what you want.

Usage
-----
    python3 tools/check_cvenv_pin.py                   # agree? and is it current?
    python3 tools/check_cvenv_pin.py --require-latest  # also fail when behind
    python3 tools/check_cvenv_pin.py --offline         # skip the upstream lookup

Standard library only, so it runs on a bare machine with nothing installed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CVENV_REPO = "https://github.com/ribeiro-computer-vision/cvenv.git"
PIN_RE = re.compile(r"cvenv@v(\d+\.\d+\.\d+)")


def _as_tuple(version: str) -> tuple:
    return tuple(int(p) for p in version.split("."))


def pins() -> dict[str, set[str]]:
    """``{filename: {versions pinned in it}}`` for every notebook and the README.

    Reads notebooks as JSON rather than importing nbformat, so this needs nothing
    installed. A file pinning two different versions is itself a problem, hence a
    set per file rather than one value.
    """
    found: dict[str, set[str]] = {}
    targets = sorted(ROOT.glob("*.ipynb")) + [ROOT / "README.md"]
    for path in targets:
        if not path.exists():
            continue
        if path.suffix == ".ipynb":
            cells = json.loads(path.read_text()).get("cells", [])
            text = "\n".join("".join(c.get("source", [])) for c in cells)
        else:
            text = path.read_text()
        versions = set(PIN_RE.findall(text))
        if versions:
            found[path.name] = versions
    return found


def latest_release() -> str | None:
    """The newest ``vX.Y.Z`` tag in the cvenv repository, or None if unreachable.

    Network failures return None rather than raising: an offline machine should
    still be able to check that the pins agree with each other.
    """
    try:
        out = subprocess.run(["git", "ls-remote", "--tags", CVENV_REPO],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        tags = re.findall(r"refs/tags/v(\d+\.\d+\.\d+)$", out.stdout, re.M)
        return max(tags, key=_as_tuple) if tags else None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    require_latest = "--require-latest" in argv
    offline = "--offline" in argv

    found = pins()
    if not found:
        print("No cvenv@vX.Y.Z pin found in any notebook or the README.")
        return 1

    width = max(len(name) for name in found)
    for name, versions in found.items():
        print(f"{name:<{width}}  {', '.join(sorted(versions, key=_as_tuple))}")

    problems: list[str] = []

    multi = {n: v for n, v in found.items() if len(v) > 1}
    for name, versions in multi.items():
        problems.append(f"{name} pins more than one version: "
                        f"{', '.join(sorted(versions, key=_as_tuple))}")

    distinct = {v for versions in found.values() for v in versions}
    if len(distinct) > 1:
        problems.append("notebooks disagree: " +
                        ", ".join(sorted(distinct, key=_as_tuple)))

    newest = None if offline else latest_release()
    if newest:
        pinned_max = max(distinct, key=_as_tuple)
        if _as_tuple(pinned_max) < _as_tuple(newest):
            note = (f"pinned v{pinned_max}, but cvenv's newest release is "
                    f"v{newest}")
            if require_latest:
                problems.append(note)
            else:
                print(f"\n⚠️  {note}.\n"
                      "   Fine if deliberate; bump every pin together if not.")
        else:
            print(f"\nup to date with cvenv v{newest}")
    elif not offline:
        print("\n(could not reach cvenv to check for newer releases)")

    if problems:
        print()
        for p in problems:
            print(f"MISMATCH: {p}")
        print("\nEvery notebook should install the same cvenv version.")
        return 1

    print("all pins agree" if len(found) > 1 else "pin is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
