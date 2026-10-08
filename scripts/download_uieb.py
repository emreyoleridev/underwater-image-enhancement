"""Download a reproducible random subset of the UIEB dataset (raw + reference pairs)
from the Hugging Face mirror `Edddddd8787/UIEB`."""
import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "Edddddd8787/UIEB"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/raw-890"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{folder}}/{{name}}"


def fetch(name: str, out: Path) -> None:
    for folder, sub in (("raw-890", "raw"), ("reference-890", "reference")):
        dst = out / sub / name
        if not dst.exists():
            urllib.request.urlretrieve(FILE_URL.format(folder=folder, name=name), dst)


def download(n: int = 100, seed: int = 42, out: Path = DEFAULT_OUT) -> list[str]:
    with urllib.request.urlopen(API) as r:
        names = sorted(Path(x["path"]).name for x in json.load(r))
    random.Random(seed).shuffle(names)
    subset = sorted(names[:n])

    for sub in ("raw", "reference"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda name: fetch(name, out), subset))
    return subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of image pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    subset = download(args.n, args.seed, args.out)
    print(f"Downloaded {len(subset)} pairs to {args.out}")


if __name__ == "__main__":
    main()
