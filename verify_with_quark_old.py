from pathlib import Path
import resource
from typing import Any
from click.testing import CliRunner
from quark.cli import entry_point
import ray
import polars as pl

def run_quark_analysis(apk_path: str):
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
    
    return risk_level, apk_score
    
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
    
    df = pl.read_csv("/mnt/storage/data/rule_to_release/golddream/golddream.csv", columns=["sha256", "y_truth", "y_score", "y_pred"])[2:]
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
    