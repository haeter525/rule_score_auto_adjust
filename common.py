import json
from pathlib import Path
import resource
import dask
import dask.delayed
import dask.distributed
import requests
from tqdm import tqdm
from quark.core.quark import Quark
from quark.core.struct.ruleobject import RuleObject as Rule
import os
from typing import Tuple
import dotenv
dotenv.load_dotenv()

def get_analysis_result_file(sha256: str) -> str:
    analysis_result_folder = os.getenv("ANALYSIS_RESULT_FOLDER")
    return Path(analysis_result_folder) / f"{sha256}.apk_progress.json"


from deprecated import deprecated
@deprecated(action="Use analysis_result.load instead")
def load_analysis_result(sha256: str) -> list[Tuple[str, int]]:
    def parse_item(item: list[str, int] | str) -> list[str, int]:
        return (item, -1) if isinstance(item, str) else item

    analysis_result_folder = os.getenv("ANALYSIS_RESULT_FOLDER")

    analysis_result = Path(analysis_result_folder) / f"{sha256}.apk_progress.json"

    try:
        with analysis_result.open("r") as file:
            content = json.load(file)
            return [parse_item(item) for item in content]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parser JSON file: {analysis_result}")
        return []



@dask.delayed
def run_quark_analysis_with_exception_handling(apk_path: str, rule_path_list: list[str], analysis_result_path: str, force: bool = False) -> Path:
    try:
        return run_quark_analysis(apk_path=apk_path, rule_path_list=rule_path_list, analysis_result_path=analysis_result_path, force=force)
    except KeyboardInterrupt:
        print("Terminated by user.")
    except BaseException as e:
        print(f"Error in computation: {e}")

def run_quark_analysis(apk_path: str, rule_path_list: list[str], analysis_result_path: str, force: bool = False) -> Path:
    assert Path(apk_path).exists()
    assert all(Path(rule).exists() for rule in rule_path_list)

    print(f"{rule_path_list[0]=}")
    print(f"{analysis_result_path=}")

    apk_name = Path(apk_path).name

    # 載入進度
    progress = get_analysis_result_file(apk_name)

    processed_rules = [rule for rule, _ in load_analysis_result(apk_name)]

    # 載入規則
    rule_paths = [
        full_path for rule_path in rule_path_list
        if (full_path:=str(Path(rule_path).resolve())) not in set(processed_rules)
    ]

    # 初始化 Quark
    quark = Quark(apk_path)
        
    results = (
        (rule_path, quark.run(Rule(rule_path))) for rule_path in rule_paths
    )

    def save_result(progress, processed_rules):
        with progress.open("w") as file:
            json.dump(processed_rules, file)
        
    # 執行 Quark 分析，並逐次寫入進度
    for rule_path, result in tqdm(results, desc=apk_name, total=len(rule_paths)):
        try:
            if isinstance(result, str) and result.startswith("error"):
                print(f"[!] Error processing {rule_path}: {result}")
                continue

            # 更新進度
            processed_rules.append((rule_path, result))

            # 寫入進度檔
            save_result(progress, processed_rules)

        except Exception as e:
            print(f"Error handling result for {rule_path}: {e}")
    
    return progress


import polars

def main():
    import dask
    from dask.distributed import Client, LocalCluster

    # 設定 dask 集群
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, memory_limit="20Gib", host="", dashboard_address=":8080")
    client = Client(cluster)

    dask.config.set({"distributed.workers.memory.terminate": 0.9})
    

    print(f"Dashboard Link: {cluster.dashboard_link}")
    
    # 讀取 APK 檔案列表
    PATH_TO_DATASET = [
        "/mnt/storage/rule_score_auto_adjust/data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv",
        "/mnt/storage/rule_score_auto_adjust/data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv",
    ]
    
    combined = polars.DataFrame()
    for dataset_path in PATH_TO_DATASET:
        combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["apk_path"]))
    
    # 讀取 APK 目錄位置
    from dotenv import load_dotenv
    load_dotenv()

    import os
    APK_FOLDER = Path(os.getenv("APK_FOLDER"))
    assert APK_FOLDER.exists() and APK_FOLDER.is_dir()
    
    apk_files_from_list = combined.to_series().to_list()

    # 組合出 APK 路徑
    apk_files = [
        apk
        for apk_name in apk_files_from_list
        if (apk:=(APK_FOLDER / apk_name)).exists
    ][:2] # TODO - 補完剩下的規則

    apk_files.sort(key=lambda x: Path(x).stat().st_size)

    assert len(apk_files) > 0, f"No APK to process."
    if (actual:=len(apk_files)) != (expected:=len(apk_files_from_list)):
        print(f"{expected - actual} apks are missing.")
        if input("Continue?") != "Y":
            return
    
    # 取得 Rule 清單
    rule_folder = Path(os.getenv("GENERATED_RULE_FOLDER"))
    assert rule_folder.exists()

    rule_path_list = [
        str(rule.resolve()) for rule in rule_folder.glob("*.json")
    ][:10]

    rule_path_list = [
        str(rule_folder / rule_name)
        for rule_name in polars.read_csv("/mnt/storage/data/rule_list/selected_rules_result_on_benign.csv", has_header=True, columns=["rule"]).to_series().to_list()
    ]

    rule_path_list = list(set(rule_path_list))

    analysis_result_path = Path(os.getenv("ANALYSIS_RESULT_FOLDER"))
    assert analysis_result_path.exists()
    
    import dask.diagnostics
    with dask.diagnostics.ProgressBar():
        tasks = []
        for apk in apk_files:
            sha256 = Path(apk).stem
            analysis_result = str(analysis_result_path / f"{sha256}.apk_progress.json")

            result = run_quark_analysis_with_exception_handling(apk, rule_path_list, analysis_result, force = True)
            # result = run_quark_analysis_with_exception_handling(apk, rule_path_list, analysis_result)
            tasks.append(result)

        from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
        for task in tasks:
            with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
                    
                dask.compute(task)
                print(prof.results)
                print()
                print(rprof.results)
                print()
                print(cprof.results)
                print()
        
    
    # 關閉 dask 集群
    client.close()
    cluster.close()


if __name__ == '__main__':
    main()
