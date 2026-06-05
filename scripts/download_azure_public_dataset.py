from __future__ import annotations

import argparse
import gzip
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2"

FILES = {
    "schema": "schema.csv",
    "vmtable": "trace_data/vmtable/vmtable.csv.gz",
    "vm_cpu_1": "trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz",
    "cpu_stats": "azure2019_data/cpu.txt",
    "memory_stats": "azure2019_data/memory.txt",
    "cores_stats": "azure2019_data/cores.txt",
    "lifetime_stats": "azure2019_data/lifetime.txt",
}


def url_for(name: str) -> str:
    if name not in FILES:
        raise ValueError(f"Unknown Azure file key: {name}. Known keys: {sorted(FILES)}")
    return f"{BASE_URL}/{FILES[name]}"


def request_size(url: str, timeout: int) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = response.headers.get("Content-Length")
        return int(value) if value else None


def download(url: str, output: Path, timeout: int, chunk_size: int, retries: int, resume: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        start = output.stat().st_size if resume and output.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start > 0 else {}
        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if start > 0 else "wb"
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, output.open(mode) as handle:
                status = getattr(response, "status", None)
                if start > 0 and status != 206:
                    raise RuntimeError(
                        f"Server did not honor Range request for resume: status={status}. "
                        "Use a fresh output path instead of appending."
                    )
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
            return
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            if attempt >= retries:
                raise
            print(f"Download failed on attempt {attempt}/{retries}: {exc}. Retrying in 5s.")
            time.sleep(5)


def gzip_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return True
    except (EOFError, OSError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Azure Public Dataset files with resume support.")
    parser.add_argument("--file", choices=sorted(FILES), required=True)
    parser.add_argument("--output-dir", default="data/raw/azure_v2")
    parser.add_argument("--output", default=None, help="Optional explicit output file path.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    args = parser.parse_args()

    rel_path = FILES[args.file]
    url = url_for(args.file)
    output = Path(args.output) if args.output else Path(args.output_dir) / Path(rel_path).name
    print(f"url: {url}")
    print(f"output: {output}")

    size = request_size(url, args.timeout)
    print(f"content_length: {size if size is not None else 'unknown'}")
    if args.head_only:
        return

    download(url, output, args.timeout, args.chunk_size, args.retries, args.resume)
    print(f"downloaded_bytes: {output.stat().st_size}")
    if size is not None and output.stat().st_size != size:
        print(f"size_matches_content_length: False expected={size} actual={output.stat().st_size}")
    elif size is not None:
        print("size_matches_content_length: True")
    if output.suffix == ".gz":
        print(f"gzip_ok: {gzip_ok(output)}")


if __name__ == "__main__":
    main()
