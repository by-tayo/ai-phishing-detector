"""Download the Tranco list and save the top 10,000 domains as ml/data/top_domains.txt.

Tranco (https://tranco-list.eu/) is a research ranking of popular websites,
built to be harder to manipulate than older lists like Alexa. URLs whose real
domain is on this list skip the model and are marked safe (see
preprocess.known_domain).

Free-hosting and file-sharing domains are left out on purpose: anyone can put
a page on blogspot.com or github.io, and attackers do.

Run:  python3 phishing-detector/ml/data/build_top_domains.py
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess import SHARED_HOSTING, registered_domain  # noqa: E402

LIST_URL = "https://tranco-list.eu/top-1m.csv.zip"
LIST_ID_URL = "https://tranco-list.eu/top-1m-id"
TOP_N = 10_000
OUT = Path(__file__).resolve().parent / "top_domains.txt"


def main() -> None:
    with urllib.request.urlopen(LIST_URL, timeout=120) as resp:
        data = resp.read()
    try:
        with urllib.request.urlopen(LIST_ID_URL, timeout=30) as resp:
            list_id = resp.read().decode().strip()
    except OSError:
        list_id = "unknown"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        rows = csv.reader(io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8"))
        domains, skipped = [], 0
        for rank, domain in rows:
            if len(domains) >= TOP_N:
                break
            reg = registered_domain(domain.strip().lower())
            if not reg or reg in SHARED_HOSTING:
                skipped += 1
                continue
            if reg not in domains:
                domains.append(reg)

    header = [
        f"# Top {len(domains):,} domains from the Tranco list, ID {list_id}",
        f"# https://tranco-list.eu/list/{list_id}  (downloaded {date.today().isoformat()})",
        f"# {skipped} free-hosting/file-sharing domains removed (see preprocess.SHARED_HOSTING)",
    ]
    OUT.write_text("\n".join(header + domains) + "\n")
    print(f"Saved {len(domains):,} domains to {OUT} (Tranco list {list_id}, {skipped} removed)")


if __name__ == "__main__":
    main()
