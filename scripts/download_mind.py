'''
Download and unpack the MIND-large news dataset (train, dev, test) from its Hugging Face mirror.

The official site (msnews.github.io) links to https://huggingface.co/datasets/yjw1029/MIND. The dataset is
gated: log in to Hugging Face, open that page and accept the conditions (Microsoft Research License Terms),
then create a read token (Settings -> Access Tokens) and put it in .env as HF_TOKEN=...

Result (the layout the news scripts expect):
    data/Mind_Large_Data/MINDlarge_train/MINDlarge_train/{news.tsv, behaviors.tsv, entity_embedding.vec, relation_embedding.vec}
    data/Mind_Large_Data/MINDlarge_dev/MINDlarge_dev/...
    data/Mind_Large_Data/MINDlarge_test/MINDlarge_test/...

Usage (from the repository root):
    python scripts/download_mind.py                 all three splits
    python scripts/download_mind.py train dev       only these splits
    python scripts/download_mind.py --keep-zips     keep the downloaded .zip files
    python scripts/download_mind.py --zips ~/Downloads
                                                    no download: unpack MINDlarge_*.zip files you downloaded
                                                    yourself from msnews.github.io (no token needed; the zips
                                                    in that folder are left untouched)

Interrupted downloads resume where they stopped.
'''
import os
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "Mind_Large_Data"
URL = "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDlarge_{split}.zip"
SPLITS = ("train", "dev", "test")


def download(url, target, token):
    """Stream url to target, resuming a partial file."""
    headers = {"Authorization": f"Bearer {token}"}
    done = target.stat().st_size if target.exists() else 0
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:          # already complete
            return
        if r.status_code in (401, 403):
            sys.exit(f"Access denied ({r.status_code}). Accept the dataset conditions on "
                     "https://huggingface.co/datasets/yjw1029/MIND and check HF_TOKEN in .env.")
        r.raise_for_status()
        if done and r.status_code == 200:  # server ignored the range: start over
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done
        with open(target, "ab" if done else "wb") as f, \
                tqdm(total=total or None, initial=done, unit="B", unit_scale=True, desc=target.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def unpack(zip_path, folder):
    """Extract into folder; drop a single top-level folder inside the zip if there is one."""
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/")[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        for name in tqdm(names, desc=f"unpack {zip_path.name}"):
            rel = name.split("/", 1)[1] if strip else name
            dest = (folder / rel).resolve()
            if folder.resolve() not in dest.parents:   # refuse paths that leave the folder
                raise ValueError(f"unsafe path in zip: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)


def main():
    argv = sys.argv[1:]
    zips_dir = None
    if "--zips" in argv:
        i = argv.index("--zips")
        if i + 1 >= len(argv):
            sys.exit('--zips needs a folder, e.g. --zips ~/Downloads')
        zips_dir = Path(argv[i + 1]).expanduser()
        del argv[i:i + 2]
        if not zips_dir.is_dir():
            sys.exit(f"Folder not found: {zips_dir}")
    args = [a for a in argv if not a.startswith("--")]
    keep = "--keep-zips" in argv
    splits = args or list(SPLITS)
    bad = [s for s in splits if s not in SPLITS]
    if bad:
        sys.exit(f"Unknown split(s): {bad}. Choose from {SPLITS}.")

    token = None
    if zips_dir is None:
        load_dotenv(ROOT / ".env")
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN is missing from .env (see the instructions at the top of this script), "
                     'or download the zips yourself and use --zips "<folder>".')

    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        name = f"MINDlarge_{split}"
        folder = OUT / name / name
        if (folder / "behaviors.tsv").exists() and (folder / "news.tsv").exists():
            print(f"{name}: already unpacked in {folder.relative_to(ROOT)}, skipped")
            continue
        if zips_dir is not None:
            zip_path = zips_dir / f"{name}.zip"
            if not zip_path.exists():
                print(f"{name}: {zip_path} not found, skipped")
                continue
            unpack(zip_path, folder)          # the user's own zip is kept
        else:
            zip_path = OUT / f"{name}.zip"
            download(URL.format(split=split), zip_path, token)
            unpack(zip_path, folder)
            if not keep:
                zip_path.unlink()
        files = sorted(p.name for p in folder.iterdir())
        print(f"{name}: {files}")
    print(f"Done. Data is in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
