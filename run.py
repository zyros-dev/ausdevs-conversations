"""
Batch process multiple Discord channels and generate aggregated visualization.

Usage:
    python run.py <channel1.json> <channel2.json> ...
    python run.py ausdevs_again/*.json
    python run.py --all              (processes all .json files in ausdevs_again/)
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_channel(channel_file: str) -> Tuple[str, bool]:
    """
    Process a single channel file.

    Args:
        channel_file: Path to Discord channel JSON export

    Returns:
        Tuple of (channel_file, success: bool)
    """
    channel_path = Path(channel_file)

    if not channel_path.exists():
        print(f"✗ File not found: {channel_file}")
        return (channel_file, False)

    # Check if output already exists (output filename = input filename stem + _embeddings.json)
    output_file = Path("output") / f"{channel_path.stem}_embeddings.json"
    if output_file.exists():
        print(f"⊘ {channel_path.name} (skipped, already processed)")
        return (channel_file, True)  # Consider as success since it's already done

    print(f"→ Starting: {channel_path.name}")
    try:
        result = subprocess.run(
            [sys.executable, "process_channel.py", str(channel_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Completed: {channel_path.name}")
        return (channel_file, True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {channel_path.name}")
        if e.stderr:
            print(f"  Error: {e.stderr[:200]}")
        return (channel_file, False)


def process_channels(channel_files: List[str], max_workers: int = 4) -> bool:
    """
    Process multiple channel files through the feature engineering pipeline in parallel.

    Args:
        channel_files: List of paths to Discord channel JSON exports
        max_workers: Number of parallel processes (default: 4)

    Returns:
        True if all channels processed successfully, False otherwise
    """
    if not channel_files:
        print("Error: No channel files provided")
        return False

    print("=" * 80)
    print(f"Processing {len(channel_files)} channels (sequential, 1 at a time)")
    print("=" * 80)
    print()

    success_count = 0
    failed_channels = []

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_single_channel, cf): cf for cf in channel_files}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            channel_file, success = future.result()
            if success:
                success_count += 1
            else:
                failed_channels.append(channel_file)
            print(f"[{completed}/{len(channel_files)}]", end=" ")

    # Summary
    print("\n" + "=" * 80)
    print(f"Processing Summary")
    print("=" * 80)
    print(f"  Successful: {success_count}/{len(channel_files)}")
    if failed_channels:
        print(f"  Failed: {len(failed_channels)}")
        for ch in failed_channels:
            print(f"    - {ch}")

    if success_count == 0:
        print("\nNo channels processed successfully. Skipping visualization.")
        return False

    # Generate aggregated visualization
    print("\n" + "=" * 80)
    print(f"Generating aggregated visualization...")
    print("=" * 80)

    try:
        subprocess.run(
            [sys.executable, "visualizations.py"],
            check=True,
            capture_output=False,
        )
        print("\n✓ Visualization complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate visualization")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Handle --all flag
    if sys.argv[1] == "--all":
        channel_files = sorted(glob.glob("ausdevs_again/*.json"))
        if not channel_files:
            print("Error: No .json files found in ausdevs_again/")
            sys.exit(1)
        print(f"Found {len(channel_files)} channel files to process\n")
    else:
        channel_files = sys.argv[1:]

    success = process_channels(channel_files)
    sys.exit(0 if success else 1)
