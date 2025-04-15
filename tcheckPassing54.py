import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from quark.script import Rule
from data_preprocess import apk, analysis_result, rule
import polars
import resource

from dotenv import load_dotenv
load_dotenv()

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

# Worker 工作函式
def expensive_worker(rule_path_str):
    global quark
    try:
        result = quark.run(Rule(rule_path_str))
        return rule_path_str, result
    except Exception as e:
        return rule_path_str, f"error: {e}"

def main(apk_name, max_workers=None, skip_unfinished=False):
    processed_rules_and_result = analysis_result.load(apk_name)
    if skip_unfinished and len(processed_rules_and_result) > 1:
        return
    processed_rules = set(file for file, _ in processed_rules_and_result)

    all_rules = [
        rule.get(rule_name)
        for rule_name in polars.read_csv("/mnt/storage/data/rule_list/selected_rules_result_on_benign.csv", has_header=True, columns=["rule"]).to_series().to_list()
    ]
    print(f"before: {len(all_rules)}")
    rules_to_process = [str(p) for p in all_rules if (os.path.realpath(str(p)) not in processed_rules)][:1000]

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers or os.cpu_count(),
            initializer=init_worker,
            initargs=(apk.get(apk_name),)
        ) as executor:
            futures = {executor.submit(expensive_worker, f): f for f in rules_to_process}

            counter = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rules"):
                rule_path_str, result = future.result()
                filename = Path(rule_path_str).name

                try:
                    if isinstance(result, str) and result.startswith("error"):
                        print(f"[!] Error processing {filename}: {result}")
                    else:
                        # move_file_if_needed(rule_path_str, result)
                        processed_rules_and_result.append((os.path.realpath(rule_path_str), result))
                        counter += 1
                        analysis_result.save(apk_name, processed_rules_and_result)
                except Exception as e:
                    print(f"Error handling result for {rule_path_str}: {e}")
                    analysis_result.save(apk_name, processed_rules_and_result)

                
    except KeyboardInterrupt as e:
        print("Terminated by user.")
        analysis_result.save(apk_name, processed_rules_and_result)

    analysis_result.save(apk_name, processed_rules_and_result)


if __name__ == "__main__":
    

    mem_bytes = 20 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    PATH_TO_DATASET = [
        "/mnt/storage/rule_score_auto_adjust/data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv",
        "/mnt/storage/rule_score_auto_adjust/data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv",
    ]

    combined = polars.DataFrame()
    for dataset_path in PATH_TO_DATASET:
        combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["sha256"]))
    apk_names = combined.to_series().to_list()
    
    apk_names = [
        apk_name
        for apk_name in apk_names
        if (apk.get(apk_name)).exists()
    ]

    for apk_name in tqdm(apk_names, desc="APKs"):
        try:
            print(f"\n[+] Processing {apk_name}...")
            main(apk_name, max_workers=2, skip_unfinished=False)
        except KeyboardInterrupt as e:
            break
        except BaseException as e:
            print(e)
