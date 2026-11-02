"""data/raw/MANIFEST.json: where each raw file came from, so a run can be audited."""
import hashlib
import json
from datetime import date

from .config import RAW

PATH = RAW / "MANIFEST.json"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def record(name, *, source, path, rows=None, note=None):
    data = json.loads(PATH.read_text()) if PATH.exists() else {}
    data[name] = {"source": source, "file": str(path.relative_to(RAW.parent.parent)),
                  "sha256": sha256(path), "bytes": path.stat().st_size, "rows": rows,
                  "downloaded": date.today().isoformat(), **({"note": note} if note else {})}
    PATH.write_text(json.dumps(data, indent=2, sort_keys=True))
