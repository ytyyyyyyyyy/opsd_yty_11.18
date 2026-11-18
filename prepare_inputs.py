"""Stage the pinned OPSD training corpus and AIME25 for an experiment source root."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset
from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

TRAIN_REPOSITORY = "siyanzhao/Openthoughts_math_30k_opsd"
TRAIN_REVISION = "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
TRAIN_SHARDS = {
    "data/train-00000-of-00002.parquet": "d0cb1cfc402f211bc3527c6c23cac0561b461dce0bbdd2679c5b9c045ee1cc21",
    "data/train-00001-of-00002.parquet": "da396c34d82405c1aeda1f5fa4531140d717a68bbdcdc5ddc99f079d6e13cd26",
}


def verified_shards(repository, revision, hashes):
    for filename, expected in sorted(hashes.items()):
        path = Path(hf_hub_download(repository, filename, repo_type="dataset", revision=revision))
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {repository}/{filename}")
        yield path


def write_dataset(dataset, target, expected_count):
    if len(dataset) != expected_count:
        raise ValueError(f"Expected {expected_count} rows for {target}, got {len(dataset)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staging = Path(temporary) / "dataset"
        dataset.save_to_disk(staging)
        if target.exists():
            raise FileExistsError(target)
        staging.rename(target)
    print(f"Staged {len(dataset)} rows at {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    manifest_path = root / "reproduction/full_eval_manifest.json"
    validation = json.loads(manifest_path.read_text())["aime25"]
    training_target = root / "data/formal_train"
    evaluation_target = root / validation["path"]

    if not training_target.exists():
        paths = verified_shards(TRAIN_REPOSITORY, TRAIN_REVISION, TRAIN_SHARDS)
        tables = [pq.read_table(path, columns=["problem", "solution"]).replace_schema_metadata(None)
                  for path in paths]
        write_dataset(Dataset(pa.concat_tables(tables)), training_target, 29434)
    else:
        print(f"Using existing training data at {training_target}")

    if not evaluation_target.exists():
        paths = verified_shards(validation["repository"], validation["revision"], validation["source_sha256"])
        rows = []
        for path in paths:
            for row in pq.read_table(path).to_pylist():
                index = len(rows)
                rows.append({
                    "index": index,
                    "id": str(row.get("id", row.get("problem_idx", index))),
                    "problem": row["problem"],
                    "answer": str(row["answer"]),
                })
        if len({row["problem"] for row in rows}) != len(rows) or not all(
            row["problem"] and row["answer"] for row in rows
        ):
            raise ValueError("AIME25 contains duplicate questions or empty fields")
        write_dataset(Dataset.from_list(rows), evaluation_target, validation["rows"])
    else:
        print(f"Using existing validation data at {evaluation_target}")


if __name__ == "__main__":
    main()
