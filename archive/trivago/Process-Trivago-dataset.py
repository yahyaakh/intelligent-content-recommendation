import pandas as pd
import numpy as np
import os
from pathlib import Path

# Project root, relative to this script in Scripts/
data_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "trivago"


def inspect_csv(path):
    total_rows = 0

    with pd.read_csv(path, chunksize=50_000) as reader:
        for index, chunk in enumerate(reader):
            if index == 0:
                print(f"\n{path.name}")
                print("Columns:", chunk.columns.tolist())
                print(chunk.head().to_string(index=False))

            total_rows += len(chunk)

    print(f"Total rows: {total_rows:,}")

inspect_csv(data_dir / "train.csv")
inspect_csv(data_dir / "item_metadata.csv")