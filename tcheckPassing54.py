import os
import shutil
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from quark.script import Rule

# 全域變數，每個 worker 的 Quark 實體
quark = None

# Worker 初始化：建立 Quark 實體
def init_worker(apk_path):
    import resource
    global quark
    from quark.core.quark import Quark  # 確保在每個 worker 裡都可以 import

    mem_bytes = 10 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    quark = Quark(apk_path)

def calculate_sha256(file_path: str) -> str:
    """計算檔案的 SHA256 雜湊值"""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 每次讀取 4KB 以避免大檔案占用過多記憶體
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

PATH_TO_REPORTS = "data/reports"
PATH_TO_RULES = "data/rule_top_1000"

def runQuarkAnalysisAndWriteReport(apkPath) -> dict | None:
    from quark.report import Report
    if not os.path.exists(apkPath):
        print(f"Skip {apkPath} since the apk didn't exist.")
        return

    sha256 = calculate_sha256(apkPath)
    report_path = os.path.join(PATH_TO_REPORTS, f"{sha256}.json")
    if os.path.exists(report_path):
        print(f"Skip {apkPath} since the report exists.")
        return
    else:
        print(f"Run Quark analysis on {apkPath}.")

    quark = Report()
    quark.analysis(apk=apkPath, rule=PATH_TO_RULES)
    report = quark.get_report("json")

    # 將結果寫入檔案

    # 確保目錄存在
    os.makedirs(PATH_TO_REPORTS, exist_ok=True)

    # 寫入 JSON 檔案
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path

# Worker 工作函式
def expensive_worker(rule_path_str):
    global quark
    try:
        result = quark.run(Rule(rule_path_str))
        return rule_path_str, result
    except Exception as e:
        return rule_path_str, f"error: {e}"

# 設定路徑
SOURCE_DIR = Path("data/rule_top_1000")
# DEST_DIR = Path("data/final_rules")
PROGRESS_FOLDER = Path("data/analysis_results")
PROGRESS_FILE = None

def load_progress_and_result(apk_path):
    global PROGRESS_FILE
    PROGRESS_FOLDER.mkdir(exist_ok=True)
    PROGRESS_FILE = PROGRESS_FOLDER / Path(f"{os.path.basename(apk_path)}_progress.json")

    if PROGRESS_FILE.exists():
        print(f"Found progress file: {PROGRESS_FILE}")
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)

            if len(progress) == 0:
                return progress
            elif isinstance(progress[0], str):
                # old progress
                return {
                    (file, -1) for file in progress
                }
            else:
                return progress
    return list()

def save_progress(processed_files):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed_files), f)

def move_file_if_needed(rule_path_str, result):
    rule_path = Path(rule_path_str)
    if result is None or result < 5:
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    target_path = DEST_DIR / rule_path.name
    if not target_path.exists():
        shutil.move(str(rule_path), target_path)
    else:
        # 檔名衝突處理，加上遞增數字
        stem = rule_path.stem
        suffix = rule_path.suffix
        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_target_path = DEST_DIR / new_name
            if not new_target_path.exists():
                shutil.move(str(rule_path), new_target_path)
                break
            counter += 1

def main(apk_path, max_workers=None, skip_unfinished=False):
    processed_files_and_result = load_progress_and_result(apk_path)
    if skip_unfinished and len(processed_files_and_result) > 1:
        return
    processed_apks = set(file for file, _ in processed_files_and_result)
    # RULE_FOLDER = SOURCE_DIR / (os.path.splitext(os.path.basename(apk_path))[0])
    # all_rules = sorted(RULE_FOLDER.glob("*.json"))
    all_rules = sorted(SOURCE_DIR.rglob("*.json")) # Analysis with all rules
    print(f"before: {len(all_rules)}")
    files_to_process = [str(p) for p in all_rules if (os.path.realpath(str(p)) not in processed_apks)]

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers or os.cpu_count(),
            initializer=init_worker,
            initargs=(apk_path,)
        ) as executor:
            futures = {executor.submit(expensive_worker, f): f for f in files_to_process}

            counter = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rules"):
                rule_path_str, result = future.result()
                filename = Path(rule_path_str).name

                try:
                    if isinstance(result, str) and result.startswith("error"):
                        print(f"[!] Error processing {filename}: {result}")
                    else:
                        # move_file_if_needed(rule_path_str, result)
                        processed_files_and_result.append((os.path.realpath(rule_path_str), result))
                        counter += 1
                        save_progress(processed_files_and_result)
                except Exception as e:
                    print(f"Error handling result for {rule_path_str}: {e}")
                    save_progress(processed_files_and_result)

                
    except KeyboardInterrupt as e:
        print("Terminated by user.")
        save_progress(processed_files_and_result)

    save_progress(processed_files_and_result)


if __name__ == "__main__":
    import resource

    mem_bytes = 20 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    # apk_dir = Path("data/apks")
    # apk_files = sorted(
    #     apk_dir.rglob("*.apk"),
    #     key=lambda p: p.stat().st_size,
    #     reverse=False# sort by file size
    # )

    PATH_TO_DATASET = [
        "data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv",
        "data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv"
    ]

    import polars
    # apk_files = polars.read_csv("apk_list.csv", has_header=True)["apk_path"]
    combined = polars.DataFrame()
    for dataset_path in PATH_TO_DATASET:
        combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["apk_path"]))
    apk_files = combined.to_series().to_list()
    
    for apk_path in tqdm(apk_files, desc="APKs"):
        try:
            print(f"\n[+] Processing {apk_path}...")
            main(str(apk_path), max_workers=7, skip_unfinished=True) # TODO - Run again with skip_unfinished == False and larger max_worker to finish analysis on small APKs
        except KeyboardInterrupt as e:
            break
        except BaseException as e:
            print(e)
