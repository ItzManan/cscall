"""Helper to fetch the Svarah Indian-accented English eval set.

Svarah is published by AI4Bharat. This script only prints/setups the steps and
verifies the target dir; the actual large download is gated behind --execute so
CI/tests never pull gigabytes. Verify the current URL in data/README.md before use.
"""
import argparse
import os

SVARAH_INFO_URL = "https://github.com/AI4Bharat/Svarah"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch the Svarah eval set")
    p.add_argument("--out", required=True, help="target directory")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--execute", action="store_true", help="actually download")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.dry_run or not args.execute:
        print(f"[dry-run] Would download Svarah into {args.out}")
        print(f"See {SVARAH_INFO_URL} for the current dataset link and license.")
        return
    raise SystemExit(
        "Automated download not implemented: follow data/README.md to obtain "
        "Svarah (license acceptance required), then place files in " + args.out
    )


if __name__ == "__main__":
    main()
