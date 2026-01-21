#!/usr/bin/env python
"""CLI for sentence search in Tesserae corpora.

Supports multiple search modes:
- Lemma search: Slower but catches all inflected forms
- Form search: Fast exact word matching
- Pattern search: Fast regex matching

Example usage:
    python reader_search.py --lemmas Caesar --limit 100
    python reader_search.py --forms Thebas Thebarum --limit 100
    python reader_search.py --pattern "\\bTheb\\w+" --no-save
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from latincyreaders import TesseraeReader, LatinLibraryReader, CamenaReader

# Output directory relative to this script
CLI_OUTPUT_DIR = Path(__file__).parent / "cli_output"

if TYPE_CHECKING:
    from typing import TextIO


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search for sentences in Tesserae corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lemma search (slower, catches all forms)
  %(prog)s --lemmas Caesar
  %(prog)s --lemmas bellum pax --fileids "cicero.*"

  # Form search (fast, exact match)
  %(prog)s --forms Thebas Thebarum --limit 50
  %(prog)s --forms Caesar Caesaris --output caesar.tsv

  # Pattern search (fast, regex)
  %(prog)s --pattern "\\bTheb\\w+"
        """,
    )

    # Search parameters (mutually exclusive group)
    search_group = parser.add_mutually_exclusive_group(required=True)
    search_group.add_argument(
        "--lemmas", "-l",
        nargs="+",
        help="Lemma(s) to search for (slower, but finds all inflected forms)",
    )
    search_group.add_argument(
        "--forms", "-f",
        nargs="+",
        help="Exact word form(s) to search for (fast)",
    )
    search_group.add_argument(
        "--pattern", "-p",
        help="Regex pattern to search for (fast)",
    )

    # Corpus parameters
    parser.add_argument(
        "--reader",
        choices=["tesserae", "latinlibrary", "camena"],
        default="tesserae",
        help="Corpus reader to use (default: tesserae)",
    )
    parser.add_argument(
        "--root", "-r",
        type=Path,
        default=None,
        help="Root directory of corpus (default: auto-detect or download)",
    )
    parser.add_argument(
        "--fileids",
        help="Glob pattern or regex to filter files (e.g., 'cicero.*')",
    )
    parser.add_argument(
        "--min-date",
        type=int,
        help="Minimum date (inclusive). Use negative for BCE (e.g., -50)",
    )
    parser.add_argument(
        "--max-date",
        type=int,
        help="Maximum date (inclusive). Use negative for BCE (e.g., 50)",
    )

    # Output parameters
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: auto-generated in cli_output/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Output to stdout instead of saving to file",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "csv", "jsonl"],
        default="tsv",
        help="Output format (default: tsv)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Maximum number of results",
    )

    # Search options
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make search case-sensitive (default: case-insensitive)",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Include surrounding sentences in output",
    )

    # Verbosity
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show timing information",
    )

    return parser.parse_args(argv)


def write_results(
    results: list[dict],
    output: TextIO,
    fmt: str,
) -> None:
    """Write results to output in specified format."""
    if fmt == "jsonl":
        import json

        for r in results:
            output.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        sep = "\t" if fmt == "tsv" else ","
        # Header
        if results:
            output.write(sep.join(results[0].keys()) + "\n")
        # Rows
        for r in results:
            values = []
            for v in r.values():
                if isinstance(v, list):
                    v = ";".join(str(x) for x in v)
                else:
                    v = str(v) if v is not None else ""
                if sep == "," and ("," in v or '"' in v or "\n" in v):
                    v = f'"{v.replace(chr(34), chr(34) + chr(34))}"'
                values.append(v)
            output.write(sep.join(values) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    start_time = time.time()

    # Initialize reader
    if not args.quiet:
        print(f"Initializing {args.reader} reader...", file=sys.stderr)

    reader_classes = {
        "tesserae": TesseraeReader,
        "latinlibrary": LatinLibraryReader,
        "camena": CamenaReader,
    }

    # Warn about limited support for non-Tesserae readers
    if args.reader != "tesserae":
        print(
            f"Warning: find_sents() is optimized for Tesserae format. "
            f"Results with {args.reader} reader may vary.",
            file=sys.stderr,
        )

    reader = reader_classes[args.reader](args.root)

    # Get fileids if filter specified
    fileids = None
    if args.fileids or args.min_date is not None or args.max_date is not None:
        fileids = reader.fileids(
            match=args.fileids,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        if not fileids:
            print("Warning: No files match the specified filters", file=sys.stderr)

    # Determine search mode and run
    if args.lemmas:
        if not args.quiet:
            print(f"Searching for lemmas: {', '.join(args.lemmas)}", file=sys.stderr)
        results_iter = reader.find_sents(
            lemma=args.lemmas,
            fileids=fileids,
            context=args.context,
        )
        search_type = "lemma"
        first_term = args.lemmas[0]
    elif args.forms:
        if not args.quiet:
            print(f"Searching for forms: {', '.join(args.forms)}", file=sys.stderr)
        results_iter = reader.find_sents(
            forms=args.forms,
            fileids=fileids,
            ignore_case=not args.case_sensitive,
            context=args.context,
        )
        search_type = "form"
        first_term = args.forms[0]
    else:
        if not args.quiet:
            print(f"Searching for pattern: {args.pattern}", file=sys.stderr)
        results_iter = reader.find_sents(
            pattern=args.pattern,
            fileids=fileids,
            ignore_case=not args.case_sensitive,
            context=args.context,
        )
        search_type = "pattern"
        first_term = "pattern"

    # Apply limit if specified
    if args.limit is not None:
        results_iter = islice(results_iter, args.limit)

    # Collect results with progress
    results: list[dict] = []
    pbar = tqdm(results_iter, desc="Searching", unit=" matches", disable=args.quiet)
    for result in pbar:
        results.append(result)
    pbar.close()

    search_time = time.time() - start_time

    # Determine output destination
    if args.no_save:
        write_results(results, sys.stdout, args.format)
    else:
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ext = "jsonl" if args.format == "jsonl" else args.format
            filename = f"{timestamp}-{first_term.lower()}-{search_type}-search.{ext}"
            CLI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = CLI_OUTPUT_DIR / filename

        with open(output_path, "w", encoding="utf-8") as f:
            write_results(results, f, args.format)
        if not args.quiet:
            print(f"Wrote {len(results)} results to {output_path}", file=sys.stderr)

    # Timing info
    if args.timing:
        msg = f"\nTiming: {len(results)} results in {search_time:.2f}s"
        print(msg, file=sys.stderr)
        if results:
            rate = len(results) / search_time
            print(f"Rate: {rate:.1f} results/second", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
