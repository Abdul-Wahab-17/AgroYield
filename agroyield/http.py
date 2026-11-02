"""HTTP helpers: retries with backoff and resumable downloads."""
import time
from pathlib import Path

import requests

USER_AGENT = "AgroYield/1.0 (research; https://github.com/Abdul-Wahab-17/AgroYield)"


def get(url, *, retries=6, backoff=5, timeout=120, **kwargs):
    """GET with exponential backoff on network errors and 5xx / 429 responses."""
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(backoff * 2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")


def post(url, *, retries=6, backoff=5, timeout=300, **kwargs):
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, timeout=timeout, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(backoff * 2 ** attempt)
    raise RuntimeError(f"POST failed after {retries} attempts: {url}: {last}")


def download_file(url: str, dest: Path, *, chunk=1 << 20):
    """Resumable streaming download; loops until complete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.with_name(dest.name + ".done")
    if done.exists() and dest.exists():
        return dest  # finished earlier; no network needed
    total = int(requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=60,
                              allow_redirects=True).headers.get("content-length", 0))
    for attempt in range(500):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have >= total:
            done.write_text(str(total))
            return dest
        try:
            with requests.get(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={have}-"},
                              stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "ab" if have else "wb") as f:
                    for part in r.iter_content(chunk):
                        f.write(part)
        except requests.RequestException:
            time.sleep(min(60, 5 + attempt))
    raise RuntimeError(f"could not finish downloading {url}")
