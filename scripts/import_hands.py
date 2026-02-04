#!/usr/bin/env python3
"""Import hand histories into the database."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moid.parser import IgnitionParser
from moid.db import create_database, HandRepository


def main():
    parser = argparse.ArgumentParser(
        description="Import Ignition/Bovada hand histories into database"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Hand history files or directories to import",
    )
    parser.add_argument(
        "-d", "--database",
        default="hands.db",
        help="Database file path (default: hands.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate database (deletes existing data)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    console = Console()

    # Create/open database
    console.print(f"[bold]Opening database:[/] {args.database}")
    conn = create_database(args.database, force=args.force)
    repo = HandRepository(conn)

    # Collect all files to import
    files_to_import = []
    for path_str in args.files:
        path = Path(path_str)
        if path.is_dir():
            # Import all .txt files in directory
            files_to_import.extend(path.glob("*.txt"))
            files_to_import.extend(path.glob("**/*.txt"))
        elif path.is_file():
            files_to_import.append(path)
        else:
            console.print(f"[yellow]Warning: {path} not found[/]")

    if not files_to_import:
        console.print("[red]No files to import[/]")
        return 1

    console.print(f"[bold]Found {len(files_to_import)} file(s) to import[/]")

    # Parse and import
    parser_obj = IgnitionParser()
    total_hands = 0
    total_errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Importing...", total=len(files_to_import))

        for filepath in files_to_import:
            progress.update(task, description=f"Processing {filepath.name}")

            try:
                hands = list(parser_obj.parse_file(filepath))
                imported = repo.insert_hands(iter(hands))
                total_hands += imported

                if args.verbose:
                    console.print(f"  {filepath.name}: {imported} hands")

            except Exception as e:
                total_errors += 1
                console.print(f"[red]Error processing {filepath}: {e}[/]")

            progress.advance(task)

    # Final stats
    console.print()
    console.print(f"[bold green]Import complete![/]")
    console.print(f"  Total hands imported: {total_hands}")
    console.print(f"  Total hands in database: {repo.count_hands()}")
    if total_errors > 0:
        console.print(f"  [yellow]Files with errors: {total_errors}[/]")

    # Optimize database
    console.print("\nOptimizing database...")
    repo.analyze()
    repo.vacuum()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
