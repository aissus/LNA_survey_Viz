#!/usr/bin/env python3
"""Download the pinned Plotly bundle into site/vendor/ and verify it.

    uv run scripts/fetch_vendor.py

The bundle is committed to the repo (no CDN at runtime). This script only
exists to reproduce or upgrade it: bump VERSION, set INTEGRITY to the npm
``dist.integrity`` of the new version, run it, and update the <script> tag in
site/index.html.
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

PACKAGE = "plotly.js-cartesian-dist-min"
VERSION = "3.7.0"
# npm dist.integrity of the tarball (sha512); the download is rejected if it differs
INTEGRITY = "sha512-kfQjcAXXByRlskP8JsrlGYRPOvB+kOehZk/qBNewgr9SOMEnBN5nwEnUGOAQ0EEgYdQX51z3VZ81aBnDgSr9Bw=="
MEMBER = "package/plotly-cartesian.min.js"
OUT = Path(__file__).resolve().parent.parent / "site" / "vendor" / f"plotly-cartesian-{VERSION}.min.js"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "survey-viz-fetch-vendor"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    meta = json.loads(_get(f"https://registry.npmjs.org/{PACKAGE}/{VERSION}"))
    integrity = meta["dist"]["integrity"]
    if INTEGRITY and integrity != INTEGRITY:
        print(f"registry integrity {integrity} != pinned {INTEGRITY}", file=sys.stderr)
        return 1

    tgz = _get(meta["dist"]["tarball"])
    algo, b64 = integrity.split("-", 1)
    if base64.b64encode(hashlib.new(algo, tgz).digest()).decode() != b64:
        print("tarball does not match its integrity hash", file=sys.stderr)
        return 1

    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tar:
        data = tar.extractfile(MEMBER).read()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"{PACKAGE}@{VERSION}: {len(data):,} bytes -> {OUT}")
    print(f"  tarball integrity {integrity}")
    print(f"  file sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
