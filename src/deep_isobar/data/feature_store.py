from pathlib import Path

import pandas as pd

from deep_isobar.config import get_project_root, get_setting


def _dataset_root() -> Path:
    data_dir = get_setting("paths.data_dir", "data")
    return get_project_root() / data_dir


def dataset_exists(dataset_name: str) -> bool:
    path = _dataset_root() / dataset_name
    return path.exists()


def save_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    partition_cols: list[str] | None = None,
    mode: str = "append",
) -> str:
    if mode not in {"append", "overwrite"}:
        raise ValueError("mode must be 'append' or 'overwrite'")

    path = _dataset_root() / dataset_name
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / "data.parquet"

    if mode == "overwrite" and file_path.exists():
        file_path.unlink()

    df.to_parquet(file_path, index=False)
    return str(file_path)


def load_dataframe(
    dataset_name: str,
    filters: dict | None = None,
) -> pd.DataFrame:
    path = _dataset_root() / dataset_name / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    if filters:
        for key, value in filters.items():
            df = df[df[key] == value]

    return df