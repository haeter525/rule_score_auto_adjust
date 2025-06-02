import polars as pl
import data_preprocess.apk as apk_lib
import data_preprocess.rule as rule_lib
import data_preprocess.analysis_result as analysis_result_lib
import ray
import ray.data
import resource
from pathlib import Path
import pandas as pd
from typing import Dict, Any
import functools

def download_apk(row: Dict[str, Any]) -> Dict[str, Any]:
    row["apk_path"] = str(apk_lib.download(row["sha256"]))
    return row

def analyze_apk(row: Dict[str, Any], rules: list[Path]) -> Dict[str, Any]:
    row["analysis_result"] = analysis_result_lib.analyze_rules(
        row["sha256"], Path(row["apk_path"]), rules
    )
    return row


def main(sha256s: list[str], rules: list[Path]):
    ray.init()

    sha256s_pl = pd.DataFrame({"sha256": sha256s})
    sha256s_pl = sha256s_pl.assign(size=sha256s_pl["sha256"].apply(lambda x: apk_lib._get_path(x).stat().st_size))
    sha256s_pl = sha256s_pl.sort_values("size")

    dataset = ray.data.from_pandas(sha256s_pl, override_num_blocks=len(sha256s_pl))
    dataset = dataset.map(download_apk)

    dataset = dataset.filter(lambda row: row["apk_path"] is not None)
    
    partial_analyze_apk = functools.partial(analyze_apk, rules=rules)
    dataset = dataset.map(partial_analyze_apk)

    success_task_results = []
    for result in dataset.iter_rows():
        if result["analysis_result"] is not None:
            success_task_results.append(result)

    success_sha256s = [result["sha256"] for result in success_task_results]
    print(f"Complete analysis {len(sha256s)} APKs on {len(rules)} rules")
    print(f"with {len(success_task_results)} success")

    failed_sha256s = list(set(sha256s) - set(success_sha256s))
    failed_out_file = "failed_apks.csv"
    pl.DataFrame(failed_sha256s, schema=["sha256"]).write_csv(
        failed_out_file, include_header=True
    )

    if failed_sha256s:
        print(
            f"and {len(failed_sha256s)} failed due to out of memory, "
            f"please refer to {failed_out_file}"
        )

    ray.shutdown()


if __name__ == "__main__":
    mem_bytes = 22 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    PATH_TO_DATASET = [
        # "data/lists/family/droidkungfu.csv",
        "/mnt/storage/rule_score_auto_adjust/data/lists/family/apk-sample.csv",
        # "data/lists/benignAPKs_top_0.4_vt_scan_date.csv",
    ]

    sha256s = pl.concat(
        [apk_lib.load_list(dataset) for dataset in PATH_TO_DATASET]
    )["sha256"].to_list()

    rules = [
        rule_lib.get(rule_name)
        for rule_name in rule_lib.load_list(
            "/mnt/storage/data/rule_to_release/0611/all_rules.csv"
        )["rule"].to_list()
    ]

    main(sha256s, rules)
