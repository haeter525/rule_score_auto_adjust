import polars as pl
import data_preprocess.apk as apk_lib
import data_preprocess.rule as rule_lib
import data_preprocess.analysis_result as analysis_result_lib
import dask
import dask.diagnostics
from dask.distributed import Client, LocalCluster, as_completed


def download_apk(sha256):
    return sha256, apk_lib.download(sha256)


def analyze_apk(sha256, apk_path, rules):
    return sha256, analysis_result_lib.analyze_rules(sha256, apk_path, rules)


cluster_setting = [
    {"n_workers": 8, "threads_per_worker": 2, "memory_limit": "2Gib"},
    {"n_workers": 4, "threads_per_worker": 2, "memory_limit": "5Gib"},
    {"n_workers": 2, "threads_per_worker": 4, "memory_limit": "10Gib"},
    {"n_workers": 1, "threads_per_worker": 1, "memory_limit": "20Gib"},
]


def main(dataset: str, n_workers, threads_per_worker, memory_limit):
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=":8080",
    )
    client = Client(cluster)
    dask.config.set({"distributed.workers.memory.terminate": 0.9})
    dask.config.set({"distributed.scheduler.allowed-failures": 1})
    print(f"Dashboard Link: {cluster.dashboard_link}")

    sha256s = apk_lib.load_list(dataset)["sha256"].to_list()
    rules = [
        rule_lib.get(rule_name)
        for rule_name in rule_lib.load_list("data/rule_top_1000.csv")["rule"].to_list()
    ]

    download_apk_tasks = client.map(
        download_apk, sha256s, key=[f"download-{sha256}" for sha256 in sha256s]
    )

    success_task_results = [
        task.result()
        for task in as_completed(download_apk_tasks)
        if task.status != "error"
    ]

    analyze_apk_tasks = client.map(
        analyze_apk,
        [sha256 for sha256, _ in success_task_results],
        [apk_path for _, apk_path in success_task_results],
        [rules] * len(success_task_results),
        key=[f"analyze-{sha256}" for sha256, _ in success_task_results],
    )

    # Wait for all analyze_apk_tasks finish and use tqdm to show the progress

    success_task_results = [
        task.result()
        for task in as_completed(analyze_apk_tasks)
        if task.status != "error"
    ]

    print(f"Complete analysis {len(sha256s)} APKs on {len(rules)} rules")
    print(f"with {len(success_task_results)} success")

    success_sha256s = [sha256 for sha256, _ in success_task_results]

    failed_sha256s = list(set(sha256s) - set(success_sha256s))
    failed_out_file = "failed_apks.csv"
    pl.DataFrame(failed_sha256s, schema=["sha256"]).write_csv(
        failed_out_file, include_header=True
    )

    print(
        f"and {len(failed_sha256s)} failed due to out of memory, please refer to {failed_out_file}"
    )

    client.wait_for_workers(len(client.scheduler_info()["workers"]))
    client.close()
    cluster.close()

if __name__ == "__main__":
    mem_bytes = 22 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    
    for setting in cluster_setting:
        print(f"{setting=}")
        main("data/lists/benignAPKs_top_0.4_vt_scan_date.csv", **setting)
        
        
        
        