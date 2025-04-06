# %%
# 將 SOURCE_APK_LISTS CSV 檔轉換成 Dataset 支援的格式（sha256,apk_path,is_malicious）
# 輸出檔名為 {p.stem}_{calculate_sha256(p)}.csv
from pathlib import Path

SOURCE_APK_LISTS = [
    Path("data/lists/benignAPKs_top_0.4_vt_scan_date.csv"),
    Path("data/lists/maliciousAPKs_top_0.4_vt_scan_date.csv")
]

IS_MALICIOUS = [
    False,
    True
]

DATASET_FOLDER = Path("data/dataset")
assert DATASET_FOLDER.exists()

APK_FOLDER = Path("/Volumes/SF_Storage/apks") # APK 檔案的存放位置
assert APK_FOLDER.exists()

all(p.exists() for p in SOURCE_APK_LISTS)

# %%
# 計算每個 APK List 的 SHA256，以準備輸出檔案路徑
import hashlib

def calculate_sha256(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
        sha256_digest = file_hash.hexdigest()
    return sha256_digest

output_paths = [
    DATASET_FOLDER / f"{p.stem}_{calculate_sha256(p)}.csv"
    for p in SOURCE_APK_LISTS
]

for source, output in zip(SOURCE_APK_LISTS, output_paths, strict=True):
    print(f"Source: {source} is going to output to {output}")

# %%
# 載入每個 CSV 檔
import polars as pl

apk_lists = [
    pl.read_csv(p, has_header=True)
    for p in SOURCE_APK_LISTS
]

# %%
# 將 APK List 轉換成 Dataset 格式
from typing import Callable
import polars as pl

get_apk_path: Callable[[str], str] = lambda sha256: str(APK_FOLDER / f"{sha256}.apk")

def convert_to_dataset(apk_list: pl.DataFrame, is_malicious: bool):
    is_malicious_value = 1 if is_malicious else 0

    dataset = apk_list.with_columns(
        pl.col("sha256").map_elements(get_apk_path, return_dtype=str).alias("apk_path"),
        pl.lit(is_malicious_value).alias("is_malicious")
    ).select(
        pl.col("sha256"),
        pl.col("apk_path"),
        pl.col("is_malicious"),
        pl.col("vt_scan_date")
    )

    return dataset

datasets = [
    convert_to_dataset(apk_list, is_malicious)
    for apk_list, is_malicious in zip(apk_lists, IS_MALICIOUS, strict=True)
]

datasets[0].head(5)

# %%
# 確認 Dataset 中每個 APK 都存在

non_exists_apks = [
    p
    for dataset in datasets
    for apk_path in dataset["apk_path"].to_list()
    if not (p := Path(apk_path)).exists()
]

if non_exists_apks:
    print(f"! Warning: {len(non_exists_apks)} apk is not exists.\n{"\n".join(map(str, non_exists_apks))}")

# %%
# 過濾掉不存在的 APK，並從存在的 APK 中取 240 個出來

datasets = [
    dataset.filter(pl.col("apk_path").map_elements(lambda x: Path(x).exists(), return_dtype=bool)).sort(pl.col("vt_scan_date"), descending=True).head(300)
    for dataset in datasets
]

print([len(d) for d in datasets])

# %%
# 將 Dataset 寫入檔案

for dataset, output_path in zip(datasets, output_paths, strict=True):
    print(f"Write dataset to {output_path}")
    dataset.write_csv(output_path, include_header=True)

# %%