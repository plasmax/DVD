import argparse
import random
from pathlib import Path, PurePosixPath


def normalize_entry(entry):
    normalized = entry.strip().replace("\\", "/").strip("/")
    if normalized in ("", "."):
        return None
    return normalized


def infer_sequence_key(image_path, data_root):
    rel_parts = image_path.relative_to(data_root).parts
    if "Image" in rel_parts:
        image_idx = rel_parts.index("Image")
        if image_idx > 0:
            return PurePosixPath(*rel_parts[:image_idx]).as_posix()
        return "Image"

    parent = image_path.parent.relative_to(data_root).as_posix()
    return "." if parent == "." else parent


def discover_sequences(data_root):
    data_root = Path(data_root)
    sequences = set()
    for image_path in sorted(data_root.rglob("Image*.png")):
        depth_name = image_path.name.replace("Image", "Depth", 1).rsplit(".", 1)[0] + ".npy"
        if "/Image/" in str(image_path):
            depth_path = Path(str(image_path).replace("/Image/", "/Depth/")).with_name(depth_name)
        else:
            depth_path = image_path.with_name(depth_name)
        if depth_path.exists():
            sequences.add(infer_sequence_key(image_path, data_root))
    return sorted(sequences)


def write_manifest(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Relative Infinigen sequence paths. Lines beginning with # are ignored.\n"
        )
        for entry in entries:
            normalized = normalize_entry(entry)
            if normalized is not None:
                handle.write(f"{normalized}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build sequence-level train/val manifests for Infinigen."
    )
    parser.add_argument("--data-root", required=True, help="Root directory of the Infinigen dataset.")
    parser.add_argument("--train-manifest", help="Path to write the train manifest.")
    parser.add_argument("--val-manifest", required=True, help="Path to write the val manifest.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of sequences to put in validation when not using --all-to-val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    parser.add_argument(
        "--all-to-val",
        action="store_true",
        help="Write every discovered sequence into the validation manifest.",
    )
    args = parser.parse_args()

    sequences = discover_sequences(args.data_root)
    if not sequences:
        raise RuntimeError(f"No Infinigen sequences found under {args.data_root}")

    if args.all_to_val:
        train_sequences = []
        val_sequences = sequences
    else:
        if not 0.0 < args.val_fraction < 1.0:
            raise ValueError("--val-fraction must be between 0 and 1.")
        shuffled = sequences[:]
        random.Random(args.seed).shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * args.val_fraction)))
        val_sequences = sorted(shuffled[:val_count])
        train_sequences = sorted(shuffled[val_count:])

    if args.train_manifest:
        write_manifest(args.train_manifest, train_sequences)
    write_manifest(args.val_manifest, val_sequences)

    print(f"Discovered {len(sequences)} sequences under {args.data_root}")
    if args.all_to_val:
        print(f"Wrote {len(val_sequences)} validation sequences to {args.val_manifest}")
    else:
        if args.train_manifest:
            print(f"Wrote {len(train_sequences)} training sequences to {args.train_manifest}")
        print(f"Wrote {len(val_sequences)} validation sequences to {args.val_manifest}")


if __name__ == "__main__":
    main()
