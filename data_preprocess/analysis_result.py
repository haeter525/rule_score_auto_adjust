import os
from pathlib import Path
import json
from typing import Tuple
import polars as pl

SCHEMA = {
    "rule_name": pl.String(),
    "passing_stage": pl.Int8(),
}

def get_folder() -> Path:
    return Path(os.getenv("ANALYSIS_RESULT_FOLDER"))

def get_file_old(sha256:str) -> Path:
    return get_folder() / f"{sha256}.apk_progress.json"

def get_file_new(sha256: str) -> Path:
    return get_folder() / f"{sha256}.csv"

get_file = get_file_new
        

def load_old(sha256: str) -> list[Tuple[str, int]]:
    def parse_item(item: list[str, int] | str) -> list[str, int]:
        return (item, -1) if isinstance(item, str) else item

    file = get_file_old(sha256=sha256)
    if not file.exists():
        return []

    try:
        with get_file_old(sha256=sha256).open("r") as file:
            content = json.load(file)
            return [parse_item(item) for item in content]
    except json.JSONDecodeError as e:
        print(f"Failed to parser JSON file: {file}")
        return []
    
def load_new(sha256:str) -> list[str, int]:
    file = get_file_new(sha256=sha256)
    if not file.exists():
        return []

    table = pl.read_csv(file, schema=SCHEMA, has_header=True)

    return [r for r in table.rows()]

def load(sha256:str) -> list[str,int]:
    combined = [tuple(item) for item in load_old(sha256=sha256) + load_new(sha256=sha256)]
    return list(set(combined))

def load_as_dataframe(sha256:str) -> pl.DataFrame:
    content = load_new(sha256=sha256)
    dataframe = pl.DataFrame(content, schema=SCHEMA, orient="row")
    return dataframe

def save(sha256:str, analysis_result: list[str, int]) -> Path:
    file = get_file_new(sha256=sha256)
    pl.DataFrame(analysis_result, schema=SCHEMA).write_csv(file, include_header=True)
    return file