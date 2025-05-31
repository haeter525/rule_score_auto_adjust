import polars as pl
import data_preprocess.apk as apk_lib
import data_preprocess.rule as rule_lib
import data_preprocess.analysis_result as analysis_result_lib
import ray
import ray.data
import ray.core
import resource

@ray.remote
def download_apk(sha256):
    return {"sha256": sha256, "apk_path": apk_lib.download(sha256)}

@ray.remote(num_cpus=4)
def analyze_apk(sha256, apk_path, rules):
    return {"sha256": sha256, "analysis_result": analysis_result_lib.analyze_rules(sha256, apk_path, rules)}


def main(dataset: str):
    ray.init()

    sha256s = apk_lib.load_list(dataset)["sha256"].to_list()
    sha256s.sort(key=lambda sha256: apk_lib._get_path(sha256).stat().st_size)
    
    rules = [
        rule_lib.get(rule_name)
        for rule_name in rule_lib.load_list("data/rule_top_1000.csv")["rule"].to_list()
    ]

    download_apk_results = ray.get([download_apk.remote(sha256) for sha256 in sha256s])

    success_task_results = [
        result
        for result in download_apk_results
        if result["apk_path"] is not None
    ]

    analyze_apk_tasks = [
        analyze_apk.remote(result["sha256"], result["apk_path"], rules)
        for result in success_task_results
    ]

    analyze_apk_results = ray.get(analyze_apk_tasks)

    success_task_results = [
        result
        for result in analyze_apk_results
        if result["analysis_result"] is not None
    ]

    success_sha256s = [result["sha256"] for result in success_task_results]
    print(f"Complete analysis {len(sha256s)} APKs on {len(rules)} rules")
    print(f"with {len(success_task_results)} success")

    failed_sha256s = list(set(sha256s) - set(success_sha256s))
    failed_out_file = "failed_apks.csv"
    pl.DataFrame(failed_sha256s, schema=["sha256"]).write_csv(
        failed_out_file, include_header=True
    )

    print(
        f"and {len(failed_sha256s)} failed due to out of memory, please refer to {failed_out_file}"
    )

    ray.shutdown()

if __name__ == "__main__":
    mem_bytes = 22 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    
    main("data/lists/benignAPKs_top_0.4_vt_scan_date.csv")
        
