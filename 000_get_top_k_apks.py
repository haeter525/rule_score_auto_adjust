# %%
# 將 SOURCE_APK_LISTS 的 APK 清單，
# 依照 SORT_BY_COLUMN 排序，取出前 PORTION % 的 APK，
# 各自輸出為新的 APK List
# 命名為 {source_apk_list_name}_top_{PORTION}_{SORT_BY_COLUMN}.csv

from typing import Callable
from pathlib import Path


SOURCE_APK_LISTS = [
    "data/lists/benignAPKs.csv",
    "data/lists/maliciousAPKs.csv"
]

SORT_BY_COLUMN = "vt_scan_date"

PORTION = 0.4

EXTRA = 10 # 多曲幾個 APK 作為備用，以防後續部分 APK 無法分析

GET_OUTPUT_APK_LIST: Callable[[Path], Path] = lambda source: source.parent / f"{source.stem}_top_{PORTION}_{SORT_BY_COLUMN}.csv"

# %%
# 將 String Path 轉換成 Path Object，並確認是否存在
from pathlib import Path

SOURCE_APK_LISTS: list[Path] = [
    Path(apk_list) for apk_list in SOURCE_APK_LISTS
]

for p in SOURCE_APK_LISTS:
    assert p.exists(), f"Source APK list {str(p)} is not exists."

# %%
# 讀取 CSV 檔
import polars as pl
source_data_frames = [
    pl.read_csv(source, has_header=True, schema_overrides={"vt_scan_date": pl.Datetime})
    for source in SOURCE_APK_LISTS
]

for source, dataframe in zip(SOURCE_APK_LISTS, source_data_frames):
    print(f"Source apk list: {source}, len: {len(dataframe)}")
# %%
# 逐一讀取每個 source Apk list，排序並取出一定數量 APK
import polars as pl
import math

def get_top_k_apks(source_apk_list: pl.DataFrame, sort_by_column: str, portion: int, extra: int) -> pl.DataFrame:
    assert sort_by_column in source_apk_list.columns, f"Target column {sort_by_column} is not in the dataframe. Existing Columns: {source_apk_list.columns}"

    sorted_list = source_apk_list.sort(pl.col(sort_by_column), descending=True)

    number_of_apk_to_select = math.floor(len(source_apk_list) * portion) + extra
    top_k_apk = sorted_list.head(number_of_apk_to_select)

    return top_k_apk


selected_apk_lists = [
    get_top_k_apks(
        source,
        SORT_BY_COLUMN,
        PORTION,
        EXTRA
    )
    for source in source_data_frames
]

for source, dataframe, selected in zip(SOURCE_APK_LISTS, source_data_frames, selected_apk_lists):
    print(f"Source apk list: {source}, len: {len(dataframe)}, selected: {len(selected)}")

selected_apk_lists[0][:5]

# %%
# 準備輸出路徑

OUTPUT_PATHS = [
    GET_OUTPUT_APK_LIST(source)
    for source in SOURCE_APK_LISTS
]

# %%
# 輸出

for source, output_path in zip(selected_apk_lists, OUTPUT_PATHS):
    print(f"Output to {output_path}")
    source.write_csv(output_path, include_header=True)

# %%
