from pathlib import Path
import resource
from typing import Any, Generator
from click.testing import CliRunner
from quark.cli import entry_point
import ray
import polars as pl
from prefect_ray import RayTaskRunner
from prefect import flow, task, create_progress_artifact, update_progress_artifact
import data_preprocess.apk as apk_lib

@task
def run_quark_analysis(apk_path: str) -> tuple[str, str, float]:
    print(f"Run Quark with {apk_path}")
    runner = CliRunner()
    result = runner.invoke(
        entry_point,
        [
            "-r", "/mnt/storage/quark-rules/rules",
            "-a", apk_path,
            "-s",
            "-t", "100",
        ]
    )

    print("Exit Code:", result.exit_code)
    print("Output:", result.output)
    assert result.exit_code == 0, result.output
    
    outputs = result.output.splitlines()
    risk_level = next(line.split("WARNING: ")[-1] for line in outputs if r"WARNING:" in line)
    apk_score = float(next(line.split("APK Score: ")[-1] for line in outputs if r"APK Score:" in line))
    
    return apk_path, risk_level, apk_score

@flow(
    task_runner=RayTaskRunner(),
    max_concurrency=10,
    log_prints=True,
) # type: ignore
def run_quark_analysis_async(apk_paths: list[str]) -> Generator[tuple[str, str, float]]:
    
    # Submit tasks
    futures = [
        run_quark_analysis.submit(apk_path)
        for apk_path in apk_paths
    ]
    
    # Create progress artifact
    progress = create_progress_artifact(
        progress=0.0,
        description="Indicates the progress of Quark analysis on APKs"
    )
    step = 100 / len(futures)
    
    for idx, future in enumerate(futures):
        apk_path, risk_level, apk_score = future.result()
        
        print(f"APK: {apk_path}, Risk Level: {risk_level}, Score: {apk_score}")
        
        # Update progress artifact
        update_progress_artifact(
            artifact_id=progress, 
            progress=step * (idx + 1)
        )
        
        yield apk_path, risk_level, apk_score
    
    
@task
def verify_classification(expected_risk_levels: list[str], actual_risk_levels: list[str]) -> list[bool]:
    return [expected == actual for expected, actual in zip(expected_risk_levels, actual_risk_levels)]
    
@flow
def verify_scores(
    apk_prediction: Path,
    builtin_rule_set: Path
    ) -> Path:
    
    apk_prediction_df = pl.read_csv(apk_prediction, columns=["sha256", "y_truth", "y_score", "y_pred"], skip_rows_after_header=2)
    
    apk_prediction_df = apk_prediction_df.with_columns(
        pl.col("sha256").map_elements(lambda sha: apk_lib.download(sha, dry_run=True), return_dtype=pl.String).alias("apk_path")
    )
    
    analysis_results = run_quark_analysis_async(
        apk_paths=apk_prediction_df["apk_path"].to_list()
    )
    
    result_df = pl.DataFrame(
        list(analysis_results),
        orient="row",
        schema={
            "apk_path": pl.String(),
            "risk_level": pl.String(),
            "apk_score": pl.Float64(),
        }
    )
    
    expected_actual_df = apk_prediction_df.join(
        result_df,
        on="apk_path",
        how="left"
    ).rename(
        {
            "y_truth": "expected_risk_level",
            "y_score": "expected_apk_score",
            "": "predicted_risk_level"
        }
    )
    
    
    
    
    
    
def ray_wrapper_run_quark(row: dict[str, Any]) -> dict[str, Any]:
    apk_path = str(Path("/mnt/storage/data/apks") / f"{row['sha256']}.apk")
    risk_level, apk_score = run_quark_analysis(apk_path)
    row["y_from_quark"] = 1 if "High Risk" in risk_level else 0
    row["y_score_from_quark"] = apk_score
    
    return row
    
if __name__ == "__main__":
    mem_bytes = 22 * 1024 * 1024 * 1024  # 20 GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    
    ray.init()
    # ray.init(local_mode=True)
    # ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}})
    
    df = pl.read_csv("/mnt/storage/rule_score_auto_adjust_haeter/2025-07-05T10:00:04.csv", columns=["sha256", "y_truth", "y_score", "y_pred"])[2:]
    df = df.filter(pl.col("y_truth").eq(pl.lit(1)))
    
    expected = ray.data.from_arrow(df.to_arrow())
    expected = expected.repartition(10)
    expected = expected.map(ray_wrapper_run_quark)
    expected = expected.add_column("check_pred", lambda row: row["y_pred"] == row["y_from_quark"])
    expected = expected.add_column("check_score", lambda row: abs(row["y_score"] - row["y_score_from_quark"]) < 0.1)
    expected = expected.repartition(1)
    expected.write_csv("verify_result.csv")
    
    # print("Check if all true")
    # df = pl.from_arrow(expected.to_arrow_refs())
    # print(f"Is risk level all true? {df["y_from_quark"].all()}")
    # print(f"Is score all true? {df["y_score_from_quark"].all()}")
    