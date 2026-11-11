"""Part-2 entrypoint, use --stage to prevent accidental GPU/paid operations."""
import argparse
import sys
from . import prepare_dataset, inference, train_lora, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "baseline", "train", "evaluate"])
    args, remainder = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    # Each script also supports rich flags via python -m <module> --help.
    if args.stage == "prepare":
        prepare_dataset.main()
    elif args.stage == "baseline":
        inference.main()
    elif args.stage == "train":
        train_lora.main()
    else:
        evaluate.main()

if __name__ == "__main__":
    main()
