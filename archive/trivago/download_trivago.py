import kagglehub
from pathlib import Path

path = Path(
    kagglehub.dataset_download(
        "pranavmahajan725/trivagorecsyschallengedata2019"
    )
)

print("Dataset location:", path)

for file in path.rglob("*"):
    if file.is_file():
        print(f"{file.relative_to(path)} — {file.stat().st_size / 1024**2:.2f} MB")