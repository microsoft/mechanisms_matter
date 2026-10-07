"""Download raw inputs for the K562-anchored Replogle datasets."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py

DEFAULT_DATA_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DownloadSpec:
    """Download location and local filename for one source dataset."""

    filename: str
    url: str


DOWNLOAD_SPECS = {
    "K562": DownloadSpec(
        filename="replogle22_K562_essential.h5ad",
        url=(
            "https://zenodo.org/records/13350497/files/"
            "ReplogleWeissman2022_K562_essential.h5ad?download=1"
        ),
    ),
    "RPE1": DownloadSpec(
        filename="replogle22_RPE1.h5ad",
        url=("https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1"),
    ),
    "Jurkat": DownloadSpec(
        filename="GSE264667_jurkat_raw_singlecell_01.h5ad",
        url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/"
            "GSE264667/suppl/GSE264667_jurkat_raw_singlecell_01.h5ad"
        ),
    ),
    "HepG2": DownloadSpec(
        filename="GSE264667_hepg2_raw_singlecell_01.h5ad",
        url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/"
            "GSE264667/suppl/GSE264667_hepg2_raw_singlecell_01.h5ad"
        ),
    ),
}
DEFAULT_DATASETS = tuple(DOWNLOAD_SPECS)


def download_datasets(
    datasets: Sequence[str],
    data_dir: Path = DEFAULT_DATA_DIR,
    force: bool = False,
) -> None:
    """
    Download selected source datasets and validate each H5AD file.

    Args:
        datasets: Dataset names to download.
        data_dir: Destination directory.
        force: Download even when the destination is already valid.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        spec = DOWNLOAD_SPECS[dataset]
        destination = data_dir / spec.filename
        if destination.exists() and not force:
            valid, reason = validate_h5ad(destination)
            if valid:
                print(f"Using existing valid file: {destination}")
                continue
            print(f"Replacing invalid file {destination}: {reason}")
        elif destination.exists():
            print(f"Replacing existing file due to --force: {destination}")
        download_file(spec.url, destination)
        print(f"Downloaded {dataset} to {destination}")


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a temporary file and atomically replace its target."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "diversity-by-design-data-downloader"},
            )
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, temporary_file)

        valid, reason = validate_h5ad(temporary_path)
        if not valid:
            raise ValueError(f"Downloaded file is not a valid H5AD: {reason}")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_h5ad(path: Path) -> tuple[bool, str]:
    """Check that a path is a readable H5AD with core AnnData groups."""
    if not path.is_file():
        return False, "file does not exist"
    try:
        with h5py.File(path, "r") as handle:
            missing_groups = [group for group in ("X", "obs", "var") if group not in handle]
            if missing_groups:
                return False, f"missing groups: {missing_groups}"
    except OSError as exc:
        return False, str(exc)
    return True, "valid"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse downloader command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download raw Replogle, Jurkat, and HepG2 H5AD inputs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
        help="Source datasets to download (default: all four).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace source files even when they pass validation.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory in which to store source files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Download selected source datasets from the command line."""
    args = parse_args(argv)
    download_datasets(
        datasets=args.datasets,
        data_dir=args.data_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
