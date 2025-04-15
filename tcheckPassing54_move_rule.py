import os
import shutil
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import psutil
from tqdm import tqdm
from quark.script import Rule

# 全域變數，每個 worker 的 Quark 實體
quark = None

# Worker 初始化：建立 Quark 實體
def init_worker(apk_path):
    import resource
    global quark
    from quark.core.quark import Quark  # 確保在每個 worker 裡都可以 import

    mem_bytes = 15 * 1024 * 1024 * 1024
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


# Worker 工作函式
def expensive_worker(rule_path_str):
    global quark

    try:
        result = quark.run(Rule(rule_path_str))
        return rule_path_str, result
    except BaseException as e:
        return rule_path_str, f"error: {e}"

# 設定路徑
SOURCE_DIR = Path("/mnt/storage/data/generated_rules")
# DEST_DIR = Path("data/final_rules")
PROGRESS_FOLDER = Path("/mnt/storage/data/analysis_results")
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
                return list()
            else:
                return progress
    return list()

def save_progress(processed_files):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed_files), f)


def main(apk_path, rule_set: set[str], max_workers=None, skip_unfinished=False):

    processed_files_and_result = load_progress_and_result(apk_path)
    if skip_unfinished and len(processed_files_and_result) > 1:
        return rule_set
    processed_rules = set(file for file, _ in processed_files_and_result)

    rules_reach_stage_5 = set(rule for rule, result in processed_files_and_result if result == 5)    

    rules_to_run = rule_set.difference(processed_rules)

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers or os.cpu_count(),
            initializer=init_worker,
            initargs=(apk_path,)
        ) as executor:
            futures = {executor.submit(expensive_worker, f): f for f in rules_to_run}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Rules"):
                rule_path_str, result = future.result()
                filename = Path(rule_path_str).name

                try:
                    if isinstance(result, str) and result.startswith("error"):
                        print(f"[!] Error processing {filename}: {result}")
                    else:
                        # move_file_if_needed(rule_path_str, result)
                        processed_files_and_result.append((os.path.realpath(rule_path_str), result))
                        save_progress(processed_files_and_result)

                        if result == 5 and rule_path_str in rule_set:
                            rule_set.remove(rule_path_str)

                except Exception as e:
                    print(f"Error handling result for {rule_path_str}: {e}")
                    save_progress(processed_files_and_result)

                
    except KeyboardInterrupt as e:
        print("Terminated by user.")
        save_progress(processed_files_and_result)

    save_progress(processed_files_and_result)

    return rule_set.difference(rules_reach_stage_5)


if __name__ == "__main__":
    import resource

    mem_bytes = 20 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    PATH_TO_DATASET = [
        # "data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv",
        "data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv"
    ]

    import polars
    # apk_files = polars.read_csv("apk_list.csv", has_header=True)["apk_path"]
    combined = polars.DataFrame()
    for dataset_path in PATH_TO_DATASET:
        combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["apk_path"]))
    apk_files = combined.to_series().to_list()
    
    from dotenv import load_dotenv
    load_dotenv()

    import os
    APK_FOLDER = Path(os.getenv("APK_FOLDER"))
    
    apk_files = [
        apk
        for apk_name in apk_files
        if (apk:=(APK_FOLDER / apk_name)).exists
    ]

    all_rules = set([str(p.resolve()) for p in sorted(SOURCE_DIR.rglob("*.json"))][:100])
    print(f"Total Rules: {len(all_rules)}")

    for apk_path in tqdm(apk_files, desc="APKs"):
        try:
            print(f"\n[+] Processing {apk_path} with {len(all_rules)} rules")
            all_rules = main(str(apk_path), all_rules, max_workers=5, skip_unfinished=False) # TODO - Run again with skip_unfinished == False and larger max_worker to finish analysis on small APKs
        except KeyboardInterrupt as e:
            break
        except json.JSONDecodeError as e:
            pass
        except BaseException as e:
            print(e)

    polars.DataFrame(list(all_rules), schema={"sha256": str}).write_csv("benign_not_5.csv", include_header=True)