"""Download the Tranco lists used by the known-site check (see preprocess.known_domain).

Tranco (https://tranco-list.eu/) is a research ranking of popular websites,
built to be harder to manipulate than older lists like Alexa. Two lists:

  ml/data/top_domains.txt  top 10,000 registered domains (paypal.com, python.org)
                           trusted only for the bare domain and www.
  ml/data/top_hosts.txt    top 50,000 full host names (docs.python.org,
                           accounts.google.com), so a popular subdomain is
                           trusted but a random customer subdomain is not

Free-hosting and file-sharing domains are left out of both on purpose: anyone
can put a page on blogspot.com or github.io, and attackers do.

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
from preprocess import SHARED_HOSTING, SHARED_HOSTS, registered_domain  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
LISTS = [
    # (zip URL, ID URL, how many, output file, keep full host names?)
    ("https://tranco-list.eu/top-1m.csv.zip", "https://tranco-list.eu/top-1m-id",
     10_000, DATA_DIR / "top_domains.txt", False),
    ("https://tranco-list.eu/top-1m-incl-subdomains.csv.zip",
     "https://tranco-list.eu/top-1m-id?subdomains=true",
     50_000, DATA_DIR / "top_hosts.txt", True),
]


def _download(url: str, timeout: int = 180) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def build(zip_url: str, id_url: str, top_n: int, out: Path, keep_hosts: bool) -> None:
    data = _download(zip_url)
    try:
        list_id = _download(id_url, 30).decode().strip()
    except OSError:
        list_id = "unknown"

    kept, seen, skipped = [], set(), 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        rows = csv.reader(io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8"))
        for _rank, name in rows:
            if len(kept) >= top_n:
                break
            name = name.strip().lower()
            if name.startswith("www."):
                name = name[4:]
            reg = registered_domain(name)
            shared = any(name == h or name.endswith("." + h) for h in SHARED_HOSTS)
            if not reg or reg in SHARED_HOSTING or shared:
                skipped += 1
                continue
            entry = name if keep_hosts else reg
            if entry not in seen:
                seen.add(entry)
                kept.append(entry)

    kind = "host names" if keep_hosts else "domains"
    header = [
        f"# Top {len(kept):,} {kind} from the Tranco list, ID {list_id}",
        f"# https://tranco-list.eu/list/{list_id}  (downloaded {date.today().isoformat()})",
        f"# {skipped} free-hosting/file-sharing entries removed (see preprocess.SHARED_HOSTING)",
    ]
    out.write_text("\n".join(header + kept) + "\n")
    print(f"Saved {len(kept):,} {kind} to {out} (Tranco list {list_id}, {skipped} removed)")


def main() -> None:
    for args in LISTS:
        build(*args)


if __name__ == "__main__":
    main()
