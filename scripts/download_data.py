"""Fetch the commit-pinned IBM Telco CSV (if absent) and record its provenance."""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "https://github.com/IBM/telco-customer-churn-on-icp4d"
COMMIT = "1fd6fd70906479be1712e64cfc0ea89f58466629"
URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{COMMIT}/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

target = Path("data/raw/telco.csv")
metadata = Path("data/raw/telco.source.json")


def main() -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"{target} already present: not downloading.")
    else:
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as response:
            target.write_bytes(response.read())

    data = target.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Source hash mismatch for {target}: got {sha256}, "
            f"expected {EXPECTED_SHA256}. Do not continue with these bytes."
        )

    if metadata.exists():
        existing = json.loads(metadata.read_text(encoding="utf-8"))
        if existing.get("sha256") == sha256 and existing.get("url") == URL:
            print(f"{metadata} already records these bytes: keeping it.")
            print(json.dumps(existing, indent=2))
            return

    record = {
        "dataset": "IBM Telco Customer Churn",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "url": URL,
        "local_path": target.as_posix(),
        "bytes": len(data),
        "sha256": sha256,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
