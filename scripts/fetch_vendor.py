#!/usr/bin/env python3
"""Download the pinned Plotly bundles into site/vendor/ and verify them.

    uv run scripts/fetch_vendor.py

The bundles are committed to the repo (no CDN at runtime). This script only
exists to reproduce or upgrade them: bump VERSION, set each INTEGRITY to the npm
``dist.integrity`` of the new version, run it, and update the file names in
site/js/main.js (PLOTLY_BUNDLES).

  cartesian  SVG scatter only; used by surveys with a few hundred points
  gl2d       adds WebGL scatter (scattergl); used by surveys with thousands of
             points, where SVG gets slow. Exports still render SVG scatter.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "3.7.0"
VENDOR = Path(__file__).resolve().parent.parent / "site" / "vendor"
# name -> (npm package, npm dist.integrity of the tarball (sha512), file inside the tarball)
BUNDLES = {
    "cartesian": (
        "plotly.js-cartesian-dist-min",
        "sha512-kfQjcAXXByRlskP8JsrlGYRPOvB+kOehZk/qBNewgr9SOMEnBN5nwEnUGOAQ0EEgYdQX51z3VZ81aBnDgSr9Bw==",
        "package/plotly-cartesian.min.js",
    ),
    "gl2d": (
        "plotly.js-gl2d-dist-min",
        "sha512-3vR+89HBYkuLNIx/DmIQbIG6JW+4Ok6MMuWVNYO4OsghZR2n0fB+XL241buc0YsvNWKtLV1YtRrPUiZmOdvBsw==",
        "package/plotly-gl2d.min.js",
    ),
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "survey-viz-fetch-vendor"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(name: str, package: str, pinned: str, member: str) -> bool:
    meta = json.loads(_get(f"https://registry.npmjs.org/{package}/{VERSION}"))
    integrity = meta["dist"]["integrity"]
    if integrity != pinned:
        print(f"{package}: registry integrity {integrity} != pinned {pinned}", file=sys.stderr)
        return False

    tgz = _get(meta["dist"]["tarball"])
    algo, b64 = integrity.split("-", 1)
    if base64.b64encode(hashlib.new(algo, tgz).digest()).decode() != b64:
        print(f"{package}: tarball does not match its integrity hash", file=sys.stderr)
        return False

    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tar:
        f = tar.extractfile(member)
        if f is None:
            print(f"{package}: {member} not found in the tarball", file=sys.stderr)
            return False
        data = f.read()
    out = VENDOR / f"plotly-{name}-{VERSION}.min.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{package}@{VERSION}: {len(data):,} bytes -> {out}")
    print(f"  tarball integrity {integrity}")
    print(f"  file sha256 {hashlib.sha256(data).hexdigest()}")
    return True


def main() -> int:
    ok = [fetch(name, *spec) for name, spec in BUNDLES.items()]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
