from pathlib import Path
from collections import Counter

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]  # project root (this file is in news/)
base_dir = PROJECT / "data" / "Mind_Large_Data"

train_dir = base_dir / "MINDlarge_train" / "MINDlarge_train"
dev_dir = base_dir / "MINDlarge_dev" / "MINDlarge_dev"

behavior_columns = [
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
]


def inspect_behavior_dates(split_name: str, split_dir: Path):
    file_path = split_dir / "behaviors.tsv"

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_rows = 0
    earliest_timestamp = None
    latest_timestamp = None
    impressions_per_day = Counter()

    behavior_chunks = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=behavior_columns,
        usecols=["timestamp"],
        dtype=str,
        keep_default_na=False,
        chunksize=50_000,
    )

    print(f"\nInspecting MIND-large {split_name}...")
    print("Source:", file_path)

    for chunk_number, chunk in enumerate(behavior_chunks, start=1):
        if chunk.empty:
            continue

        timestamps = pd.to_datetime(
            chunk["timestamp"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="raise",
        )

        # Reject missing timestamps rather than silently excluding rows.
        if timestamps.isna().any():
            raise ValueError(
                f"Missing timestamps in {split_name}, chunk {chunk_number}"
            )

        total_rows += len(chunk)

        chunk_earliest = timestamps.min()
        chunk_latest = timestamps.max()

        if (
            earliest_timestamp is None
            or chunk_earliest < earliest_timestamp
        ):
            earliest_timestamp = chunk_earliest

        if (
            latest_timestamp is None
            or chunk_latest > latest_timestamp
        ):
            latest_timestamp = chunk_latest

        # Counter.update adds counts to existing daily totals.
        daily_counts = timestamps.dt.date.value_counts()
        impressions_per_day.update(daily_counts.to_dict())

        if chunk_number % 10 == 0:
            print(f"  Processed {total_rows:,} impressions...")

    if sum(impressions_per_day.values()) != total_rows:
        raise ValueError("Daily counts do not match the total row count.")

    sorted_daily_counts = dict(sorted(impressions_per_day.items()))

    print(f"\nMIND-large {split_name} summary")
    print("Total impressions:", f"{total_rows:,}")
    print("Earliest timestamp:", earliest_timestamp)
    print("Latest timestamp:", latest_timestamp)
    print("Number of dates:", len(sorted_daily_counts))

    print("\nImpressions per day:")
    for date, count in sorted_daily_counts.items():
        print(f"  {date}: {count:,}")

    return {
        "split": split_name,
        "total_impressions": total_rows,
        "earliest_timestamp": earliest_timestamp,
        "latest_timestamp": latest_timestamp,
        "impressions_per_day": sorted_daily_counts,
    }


def main():
    train_summary = inspect_behavior_dates("train", train_dir)
    dev_summary = inspect_behavior_dates("dev", dev_dir)

    # Check whether all training events precede development events.
    train_end = train_summary["latest_timestamp"]
    dev_start = dev_summary["earliest_timestamp"]

    if train_end is not None and dev_start is not None:
        print("\nChronological check:")
        if train_end < dev_start:
            print("All training events occur before development events.")
        else:
            print(
                "Training and development date ranges overlap "
                "or are not strictly ordered. Inspect before splitting."
            )


if __name__ == "__main__":
    main()