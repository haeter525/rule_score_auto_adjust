import os
import json
from pathlib import Path
from tqdm import tqdm
from quark.script import Rule
import dask
from dask.distributed import Client, LocalCluster
import polars
import resource

def calculate_sha256(file_path: str) -> str:
    """計算檔案的 SHA256 雜湊值"""
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

PATH_TO_REPORTS = "data/reports"
PATH_TO_RULES = "data/rule_top_1000"

@dask.delayed
def run_quark_analysis(apk_path, rule_path):
    """使用 dask.delayed 包裝 Quark 分析任務"""
    from quark.core.quark import Quark
    
    try:
        # 設定記憶體限制
        mem_bytes = 10 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        
        quark = Quark(apk_path)
        result = quark.run(Rule(rule_path))
        return rule_path, result
    except Exception as e:
        return rule_path, f"error: {e}"

def process_apk(apk_path, client):
    """處理單個 APK 檔案"""
    print(f"\n[+] Processing {apk_path}...")
    
    # 載入進度
    progress_folder = Path("data/analysis_results")
    progress_folder.mkdir(exist_ok=True)
    progress_file = progress_folder / f"{os.path.basename(apk_path)}_progress.json"
    
    processed_files = set()
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
            if progress:
                processed_files = set(file for file, _ in progress)
    
    # 獲取所有規則檔案
    all_rules = sorted(Path(PATH_TO_RULES).rglob("*.json"))
    files_to_process = [str(p) for p in all_rules if os.path.realpath(str(p)) not in processed_files]
    
    if not files_to_process:
        print(f"No new rules to process for {apk_path}")
        return
    
    # 建立延遲任務
    tasks = [run_quark_analysis(apk_path, rule_path) for rule_path in files_to_process]
    
    # 使用 dask 的 compute 來執行任務
    results = dask.compute(*tasks)
    
    # 處理結果
    processed_results = []
    for rule_path, result in results:
        try:
            if isinstance(result, str) and result.startswith("error"):
                print(f"[!] Error processing {rule_path}: {result}")
            else:
                processed_results.append((os.path.realpath(rule_path), result))
        except Exception as e:
            print(f"Error handling result for {rule_path}: {e}")
    
    # 更新進度檔案
    if progress_file.exists():
        with open(progress_file, "r") as f:
            existing_progress = json.load(f)
            processed_results.extend(existing_progress)
    
    with open(progress_file, "w") as f:
        json.dump(processed_results, f)

def main():
    # 設定記憶體限制
    mem_bytes = 20 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    
    # 設定 dask 集群
    cluster = LocalCluster(n_workers=7, threads_per_worker=1)
    client = Client(cluster)

    print(f"Dashboard Link: {cluster.dashboard_link}")
    
    # 讀取 APK 檔案列表
    PATH_TO_DATASET = [
        "data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv",
        "data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv"
    ]
    
    combined = polars.DataFrame()
    for dataset_path in PATH_TO_DATASET:
        combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["apk_path"]))
    apk_files = combined.to_series().to_list()
    
    # 處理每個 APK
    for apk_path in tqdm(apk_files, desc="APKs"):
        try:
            process_apk(apk_path, client)
        except KeyboardInterrupt:
            print("Terminated by user.")
            break
        except Exception as e:
            print(f"Error processing {apk_path}: {e}")
    
    # 關閉 dask 集群
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
